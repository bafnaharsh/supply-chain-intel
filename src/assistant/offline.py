"""Offline mode: a fixed set of question shapes answered from the tools with no LLM.

A demo path for a reviewer without a Gemini key, not the assistant. Design: tokenize once, then
look things up rather than pattern-match them — port names come from the API's own port list,
windows are parsed as dates, the shipment id uses the tools' normaliser, and the intent is a
keyword-set test on the tokens. No regular expressions.
"""

from collections.abc import Callable
from datetime import date

from assistant import tools
from assistant.result import AnswerResult

HELP = (
    "Offline mode answers a fixed set of question shapes (no LLM):\n"
    '- Route stats: "What is the on-time rate from Shanghai to Rotterdam?" '
    '(also average delay, "between X and Y")\n'
    '- Rankings: "Which routes have the highest average delay this quarter?" '
    "(also worst/best on-time rate, busiest; this quarter, last quarter, in 2024-06, all time)\n"
    '- Delay risk: "What is the delay risk for shipment 1?"\n'
    '- Shipment status: "Status of shipment 3" or "Was SHP-00002 late?"\n'
    '- Ports: "Which ports are in the data?"\n'
    '- Counts: "How many shipments from Shanghai to Rotterdam in 2024-06?" '
    "(also status delivered/delayed/cancelled)\n"
    "Set GEMINI_API_KEY in .env to ask free-form questions."
)
NEED_ID = "Which shipment? Give an id like SHP-00421 (or just 421)."

_PUNCT = str.maketrans("", "", "?!.,;:()\"'")
_ORIGIN_WORDS = {"from", "between", "for", "at", "of", "origin"}
_DEST_WORDS = {"to", "and", "into", "destination"}
_STATUSES = {"delivered": "DELIVERED", "delayed": "DELAYED", "cancelled": "CANCELLED"}
_STATUSES["canceled"] = "CANCELLED"
_LOW = {"worst", "lowest", "least", "shortest"}
# Only after these does an unknown word get the fuzzy treatment ("from Rotterdm"); after "of"
# or "for" it is usually just English ("number of shipments").
_FUZZY_AFTER = {"from", "to", "between", "and"}

Calls = list[str]


class Q:
    """The question, tokenized once. `raw` keeps the user's spelling for messages."""

    def __init__(self, text: str):
        norm = text.replace("on-time", "ontime").replace("On-time", "ontime")
        norm = norm.replace("on time", "ontime").translate(_PUNCT)
        self.raw = norm.split()
        self.low = [t.lower() for t in self.raw]
        self.words = set(self.low)

    def has(self, *words: str) -> bool:
        return not self.words.isdisjoint(words)


def _call(calls: Calls, fn: Callable[..., dict], *args, **kwargs) -> dict:
    calls.append(fn.__name__)
    return fn(*args, **kwargs)


def _cite(calls: Calls) -> str:
    return " [" + ", ".join(dict.fromkeys(calls)) + "]"


def _fail(r: dict) -> str:
    return f"I could not get that from the data service: {r['message']}"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _hours(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1f} h"


# --- entities -----------------------------------------------------------------------------


def _shipment_id(q: Q) -> str | None:
    return next((s for t in q.raw if (s := tools.normalise_shipment_id(t))), None)


def _month(q: Q) -> tuple[int, int] | None:
    for t in q.low:
        try:
            d = date.fromisoformat(f"{t}-01") if len(t) == 7 else None
        except ValueError:
            continue
        if d:
            return d.year, d.month
    return None


def _quarter(q: Q) -> str | None:
    if "quarter" not in q.words:
        return None
    return "previous_quarter" if q.has("last", "previous") else "current_quarter"


def _window(q: Q, calls: Calls) -> tuple[str, str | None, str | None] | str:
    """(label, booked_from, booked_to) from the question, or an error string."""
    if ym := _month(q):
        y, m = ym
        end = date(y + m // 12, m % 12 + 1, 1)
        return f"booked in {y}-{m:02d}", f"{y}-{m:02d}-01", end.isoformat()
    if which := _quarter(q):
        p = _call(calls, tools.data_period)
        if p["status"] != "ok":
            return _fail(p)
        start, end = p[which]["start"], p[which]["end"]
        d = date.fromisoformat(start)
        return f"booked in Q{(d.month - 1) // 3 + 1} {d.year} ({start} to {end})", start, end
    return "all time", None, None


def _ports(q: Q) -> dict[str, tuple[str, str]] | str:
    """{'o': (code, label), 'd': ...} for the ports named in the question, or a message to show.

    Exact names and codes are a dictionary lookup over 1- and 2-word spans; an unmatched word
    right after 'from'/'to' goes through resolve_port, which corrects one obvious typo or asks.
    """
    listing = tools.list_ports()
    if listing["status"] != "ok":
        return _fail(listing)
    known = {p["port_name"].lower(): p for p in listing["ports"]}
    known |= {p["port_code"].lower(): p for p in listing["ports"]}
    found: list[tuple[int, dict, str]] = []  # (position, port, note)
    i = 0
    while i < len(q.low):
        for n in (2, 1):
            span = " ".join(q.low[i : i + n])
            if span in known:
                found.append((i, known[span], ""))
                i += n
                break
        else:
            prev = q.low[i - 1] if i else ""
            if prev in _FUZZY_AFTER and q.low[i] not in _ORIGIN_WORDS | _DEST_WORDS:
                word, r = q.raw[i], tools.resolve_port(q.raw[i])
                if r["status"] == "ok":
                    found.append((i, r, r.get("note", "")))
                elif r["status"] == "ambiguous":
                    opts = ", ".join(
                        f"{c['port_name']} ({c['port_code']})" for c in r["candidates"]
                    )
                    return f"{word!r} could mean: {opts}. Which one do you mean?"
                elif r["status"] == "unknown":
                    return (
                        f"{word!r} is not in the data. Ask 'which ports' to see the ones that are."
                    )
                else:
                    return _fail(r)
            i += 1
    out: dict[str, tuple[str, str]] = {}
    for pos, port, note in found:
        prev = q.low[pos - 1] if pos else ""
        key = "d" if prev in _DEST_WORDS or ("o" in out and prev not in _ORIGIN_WORDS) else "o"
        label = f"{port['port_name']} ({port['port_code']})"
        out.setdefault(key, (port["port_code"], label + (f"\n{note}" if note else "")))
    return out


def _ask_other(known: tuple[str, str], ask_destination: bool, calls: Calls) -> str:
    code, known_label = known
    listing = _call(calls, tools.list_ports)
    if listing["status"] != "ok":
        return _fail(listing)
    others = [p for p in listing["ports"] if p["port_code"] != code]
    opts = ", ".join(f"{p['port_name']} ({p['port_code']})" for p in others[:5])
    if ask_destination:
        return f"{known_label} to which destination? For example: {opts}."
    return f"To {known_label} from which origin? For example: {opts}."


def _label(entry: tuple[str, str]) -> tuple[str, str]:
    """Split a label with an attached typo note into (label, note line)."""
    label, _, note = entry[1].partition("\n")
    return label, f"{note}\n" if note else ""


# --- handlers: (question, calls) -> text ---------------------------------------------------


def _route_stats(q: Q, calls: Calls) -> str:
    ports = _ports(q)
    if isinstance(ports, str):
        return ports
    if not ports:
        return "Which route? Say e.g. 'from Shanghai to Rotterdam'. 'Which ports' lists the ports."
    if len(ports) == 1:
        key, entry = next(iter(ports.items()))
        return _ask_other((entry[0], _label(entry)[0]), key == "o", calls)
    (o, ol, on), (d, dl, dn) = ((e[0], *_label(e)) for e in (ports["o"], ports["d"]))
    r = _call(calls, tools.get_route_stats, o, d)
    if r["status"] != "ok":
        return f"No data for {ol} to {dl}: {r['message']}"
    ignored = (
        " (route stats are all-time; the window in your question was ignored)"
        if _month(q) or _quarter(q)
        else ""
    )
    return (
        f"{on}{dn}{ol} to {dl}: on-time rate {_pct(r['on_time_rate'])}, average delay "
        f"{_hours(r['avg_delay_hours'])} (n={r['completed_count']} completed of "
        f"{r['shipment_count']} shipments, all time){ignored}.{_cite(calls)}"
    )


def _rank(q: Q, calls: Calls) -> str:
    if "ontime" in q.words:
        sort, order, what = "on_time_rate", "asc" if q.has(*_LOW) else "desc", "on-time rate"
    elif q.has("busiest", "count") or ("most" in q.words and "shipments" in q.words):
        sort, order, what = "shipment_count", "desc", "shipment count"
    elif q.has("delay", "delays", "delayed"):
        asc = q.has("lowest", "least", "best", "shortest")
        sort, order, what = "avg_delay_hours", "asc" if asc else "desc", "average delay"
    else:
        return "Worst or best by what: average delay, on-time rate, or number of shipments?"
    win = _window(q, calls)
    if isinstance(win, str):
        return win
    label, start, end = win
    period = _call(calls, tools.data_period)
    if period["status"] != "ok":
        return _fail(period)
    for floor in (3, 1):  # prefer routes with enough completed shipments; fall back on tiny data
        r = _call(
            calls,
            tools.top_routes,
            booked_from=start,
            booked_to=end,
            min_completed=floor,
            sort=sort,
            order=order,
        )
        if r["status"] != "ok":
            return _fail(r)
        if r["routes"]:
            break
    covers = f"data covers {period['min_booking_date']} to {period['max_booking_date']}"
    if not r["routes"]:
        return f"No routes found for {label}. The {covers}."
    lines = [
        f"{i}. {x['route_key']}: avg delay {_hours(x['avg_delay_hours'])}, on-time "
        f"{_pct(x['on_time_rate'])}, n={x['completed_count']} completed of "
        f"{x['shipment_count']} shipments"
        for i, x in enumerate(r["routes"], 1)
    ]
    direction = "Lowest" if order == "asc" else "Highest"
    head = f"{direction} {what}, {label}, routes with {floor}+ completed shipments ({covers}):"
    return head + "\n" + "\n".join(lines) + _cite(calls)


def _outcome(r: dict) -> str:
    delay = r["actual_delay_hours"]
    if delay is None:
        return "no actual arrival recorded"
    return f"actual delay {_hours(delay)} ({'on time' if r['on_time_flag'] else 'late'})"


def _get(sid: str, calls: Calls) -> dict:
    r = _call(calls, tools.get_shipment, sid)
    if r["status"] != "ok" and "404" in r["message"]:
        r["message"] = f"{sid} was not found; check the id."
        r["not_found"] = True
    return r


def _risk(q: Q, calls: Calls) -> str:
    if not (sid := _shipment_id(q)):
        return NEED_ID
    s = _get(sid, calls)
    if s["status"] != "ok":
        return s["message"] if s.get("not_found") else _fail(s)
    p = _call(calls, tools.predict_delay, sid)
    if p["status"] != "ok":
        return _fail(p)
    text = (
        f"{sid} ({s['route_key']}): booking-time delay risk {p['delay_risk'] * 100:.0f}% "
        f"({p['risk_label']}; chance of arriving more than 24 h late, model {p['model_version']})."
    )
    if s["shipment_status"] in ("DELIVERED", "DELAYED", "CANCELLED"):
        text += (
            f" It is already {s['shipment_status']}: {_outcome(s)} "
            "(the risk is what the model said at booking)."
        )
    return text + _cite(calls)


def _status(q: Q, calls: Calls) -> str:
    if not (sid := _shipment_id(q)):
        return NEED_ID
    r = _get(sid, calls)
    if r["status"] != "ok":
        return r["message"] if r.get("not_found") else _fail(r)
    return (
        f"{r['shipment_id']} ({r['route_key']}) is {r['shipment_status']}; {_outcome(r)}"
        f"; booked {r['booking_date'][:10]}, planned arrival {r['planned_arrival'][:10]}, "
        f"actual arrival {(r['actual_arrival'] or 'none')[:10]}.{_cite(calls)}"
    )


def _list_ports(q: Q, calls: Calls) -> str:
    r = _call(calls, tools.list_ports)
    if r["status"] != "ok":
        return _fail(r)
    lines = [f"{p['port_code']}: {p['port_name']}, {p['country']}" for p in r["ports"]]
    return f"{len(lines)} ports in the data:\n" + "\n".join(lines) + _cite(calls)


def _count(q: Q, calls: Calls) -> str:
    ports = _ports(q)
    if isinstance(ports, str):
        return ports
    note = "".join(_label(e)[1] for e in ports.values())
    filters = [f"{w} {_label(ports[k])[0]}" for k, w in (("o", "from"), ("d", "to")) if k in ports]
    status = next((_STATUSES[t] for t in q.low if t in _STATUSES), None)
    if status:
        filters.append(f"with status {status}")
    win = _window(q, calls)
    if isinstance(win, str):
        return win
    label, start, end = win
    r = _call(
        calls,
        tools.query_shipments,
        origin=ports.get("o", (None,))[0],
        destination=ports.get("d", (None,))[0],
        status=status,
        booked_from=start,
        booked_to=end,
        limit=1,
    )
    if r["status"] != "ok":
        return _fail(r)
    n = r["total"]
    what = " ".join(["shipment" if n == 1 else "shipments", *filters])
    return f"{note}{n} {what}, {label}.{_cite(calls)}"


# First match wins, so the specific intents come before the broad ones.
INTENTS: list[tuple[set[str], Callable[[Q, Calls], str]]] = [
    ({"risk", "predict", "prediction", "probability", "likely", "likelihood", "chance"}, _risk),
    ({"many", "count", "number"}, _count),
    ({"status", "track", "where", "late"}, _status),
    ({"highest", "worst", "lowest", "best", "busiest", "longest", "top", "most"}, _rank),
    ({"ontime", "delay", "delays", "stats", "statistics", "rate", "average"}, _route_stats),
    ({"ports", "port"}, _list_ports),
]


def answer_offline(question: str) -> AnswerResult:
    q, calls = Q(question), []
    text = next((h(q, calls) for words, h in INTENTS if q.has(*words)), HELP)
    return AnswerResult(
        text=text,
        tools_called=list(dict.fromkeys(calls)),
        llm_calls=0,
        usage={"prompt_tokens": 0, "output_tokens": 0, "est_usd": 0.0},
        guardrail=None,
    )
