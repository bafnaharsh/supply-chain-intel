"""Tools the assistant may call. Each wraps one HTTP API call and never raises to the model.

Every tool returns `{"status": "ok", ...}` or `{"status": "error", "message": ..., "hint": ...}`.
"""

import difflib
from datetime import date

import httpx

from core.config import ASSISTANT_TOOL_ROW_LIMIT

http: httpx.Client | None = None  # set by configure(); tests inject the FastAPI TestClient

_ID_SEPARATORS = str.maketrans("", "", " _-#")


def configure(client: httpx.Client) -> None:
    """Point the tools at an API client (base_url already set)."""
    global http
    http = client


def _error(message: str, hint: str = "") -> dict:
    return {"status": "error", "message": message, "hint": hint}


def _call(method: str, path: str, hint: str = "", **kwargs) -> tuple[object, dict | None]:
    """(json, None) on 2xx, else (None, error dict). Network and API errors are both caught."""
    if http is None:
        return None, _error("Assistant tools are not connected to the API.")
    try:
        r = http.request(method, path, **kwargs)
    except httpx.HTTPError as e:
        return None, _error(f"API unreachable: {e}", "Tell the user the data service is down.")
    if r.is_success:
        return r.json(), None
    try:
        message = r.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = r.text[:200]
    return None, _error(f"API error {r.status_code}: {message}", hint)


def _drop_none(**params) -> dict:
    return {k: v for k, v in params.items() if v is not None}


def normalise_shipment_id(raw: str) -> str | None:
    """'421', 'shp 421', 'SHP-421', 'SHP-00421' -> 'SHP-00421'; anything else -> None."""
    digits = (raw or "").translate(_ID_SEPARATORS).lower().removeprefix("shp")
    return f"SHP-{int(digits):05d}" if digits.isdigit() and len(digits) <= 5 else None


def list_ports() -> dict:
    """List every port in the data: code (5-letter UN/LOCODE), name, country and region.

    Use when the user asks which ports exist or you need to show port options.
    """
    data, err = _call("GET", "/ports")
    if err:
        return err
    keys = ("port_code", "port_name", "country", "region")
    return {"status": "ok", "ports": [{k: p[k] for k in keys} for p in data]}


def resolve_port(name_or_code: str) -> dict:
    """Turn a port name or code the user typed (e.g. 'Shanghai', 'rotterdam', 'CNSHA') into its
    5-letter port code. ALWAYS call this before using a port in another tool.

    Returns status 'ok' with `port_code`; 'ambiguous' with `candidates` (ask the user which one);
    or 'unknown' with possible `candidates` (tell the user the port is not in the data).
    """
    listing = list_ports()
    if listing["status"] != "ok":
        return listing
    ports = listing["ports"]
    q = (name_or_code or "").strip()
    by_code = {p["port_code"]: p for p in ports}
    by_name = {p["port_name"].casefold(): p for p in ports}
    hit = by_code.get(q.upper()) or by_name.get(q.casefold())
    if hit is None:
        contains = [p for p in ports if q and q.casefold() in p["port_name"].casefold()]
        hit = contains[0] if len(contains) == 1 else None
    if hit:
        return {"status": "ok", "port_code": hit["port_code"], "port_name": hit["port_name"]}
    keys = list(by_name) + [c.casefold() for c in by_code]
    close = difflib.get_close_matches(q.casefold(), keys, n=3)
    found = {(by_name.get(k) or by_code[k.upper()])["port_code"]: None for k in close}
    candidates = [{"port_code": c, "port_name": by_code[c]["port_name"]} for c in found][:3]
    if (
        len(candidates) == 1
        and difflib.SequenceMatcher(None, q.casefold(), close[0]).ratio() >= 0.8
    ):
        # One obvious typo ('Rotterdm'): correct it and say so rather than ask.
        c = candidates[0]
        return {
            "status": "ok",
            **c,
            "corrected_from": q,
            "note": f"Interpreted {q!r} as {c['port_name']}.",
        }
    return {
        "status": "ambiguous" if candidates else "unknown",
        "query": q,
        "candidates": candidates,
        "hint": "Ask the user which port they mean." if candidates else "Port is not in the data.",
    }


def query_shipments(
    origin: str | None = None,
    destination: str | None = None,
    status: str | None = None,
    booked_from: str | None = None,
    booked_to: str | None = None,
    limit: int = ASSISTANT_TOOL_ROW_LIMIT,
) -> dict:
    """Find shipments, newest booking first. All filters are optional.

    origin/destination: 5-letter port codes (resolve names first with resolve_port).
    status: one of DELIVERED, DELAYED, CANCELLED (the only values in the data).
    booked_from (inclusive) / booked_to (exclusive): ISO dates 'YYYY-MM-DD'.
    Returns `total` matching shipments and at most a few compact rows; use `total` for counts.
    """
    limit = max(1, min(limit or ASSISTANT_TOOL_ROW_LIMIT, ASSISTANT_TOOL_ROW_LIMIT))
    params = _drop_none(
        origin=origin,
        destination=destination,
        status=status,
        booked_from=booked_from,
        booked_to=booked_to,
        limit=limit,
    )
    data, err = _call("GET", "/shipments", "Check port codes, status and dates.", params=params)
    if err:
        return err
    keys = ("shipment_id", "route_key", "booking_date", "status")
    keys += ("actual_delay_hours", "on_time_flag")
    rows = [{k: s[k] for k in keys} for s in data["items"]]
    return {"status": "ok", "total": data["total"], "returned": len(rows), "shipments": rows}


def get_shipment(shipment_id: str) -> dict:
    """Status, route, planned/actual dates and actual delay of ONE shipment by id.

    Use for 'status of', 'where is', 'was it late' questions, and BEFORE predict_delay so you
    can tell the user if the shipment is already DELIVERED, DELAYED or CANCELLED.
    shipment_id: as the user wrote it, e.g. 'SHP-00001', 'shp 1' or '1'.
    """
    sid = normalise_shipment_id(shipment_id)
    if sid is None:
        return _error(f"{shipment_id!r} is not a shipment id.", "Ask the user for the id.")
    data, err = _call("GET", f"/shipments/{sid}", "Shipment not found; ask the user to check it.")
    if err:
        return err
    keys = (
        "shipment_id",
        "route_key",
        "origin_port",
        "destination_port",
        "cargo_type",
        "booking_date",
        "planned_departure",
        "actual_departure",
        "planned_arrival",
        "actual_arrival",
        "actual_delay_hours",
        "on_time_flag",
    )
    # The row's own `status` (DELIVERED/DELAYED/CANCELLED) must not shadow the envelope's.
    return {"status": "ok", "shipment_status": data["status"], **{k: data[k] for k in keys}}


def get_route_stats(origin: str, destination: str) -> dict:
    """All-time stats for one route: shipment_count, completed_count, avg_delay_hours and
    on_time_rate (share of completed shipments arriving <= 24h late).

    origin/destination: 5-letter port codes, e.g. 'CNSHA', 'NLRTM'.
    """
    data, err = _call(
        "GET",
        f"/routes/{origin}/{destination}/stats",
        "No shipments on that route; check the port codes with resolve_port.",
    )
    return err or {"status": "ok", **data}


def top_routes(
    booked_from: str | None = None,
    booked_to: str | None = None,
    min_completed: int = 3,
    sort: str = "avg_delay_hours",
    order: str = "desc",
    limit: int = 5,
) -> dict:
    """Rank routes by stats in a booking window, in ONE call. Use for questions like 'which
    routes had the highest average delay this quarter' or 'worst on-time rate'.

    booked_from (inclusive) / booked_to (exclusive): ISO dates 'YYYY-MM-DD' (get quarter bounds
    from data_period). min_completed: drop routes with fewer completed shipments.
    sort: 'avg_delay_hours' | 'on_time_rate' | 'shipment_count'.
    order: 'desc' (highest first) or 'asc' (lowest first). For 'worst on-time rate' use
    sort='on_time_rate', order='asc'. Never page through results to find a minimum.
    """
    params = _drop_none(
        booked_from=booked_from,
        booked_to=booked_to,
        min_completed=min_completed,
        sort=sort,
        order=order if order in ("asc", "desc") else "desc",
        limit=max(1, min(limit, ASSISTANT_TOOL_ROW_LIMIT)),
    )
    data, err = _call("GET", "/routes/stats", "Check the dates and sort field.", params=params)
    if err:
        return err
    return {"status": "ok", "window": {"from": booked_from, "to": booked_to}, "routes": data}


def predict_delay(shipment_id: str) -> dict:
    """Predict the probability (delay_risk, 0-1) that a shipment arrives more than 24h late, with
    a risk_label (low/medium/high) and the model_version.

    shipment_id: as the user wrote it, e.g. 'SHP-00421', 'shp 421' or '421'.
    """
    sid = normalise_shipment_id(shipment_id)
    if sid is None:
        return _error(f"{shipment_id!r} is not a shipment id.", "Ask the user for the id.")
    data, err = _call(
        "POST",
        "/predict-delay",
        "Shipment not found; ask the user to check the id.",
        json={"shipment_id": sid},
    )
    return err or {"status": "ok", **data}


def _quarter_start(d: date, shift: int = 0) -> date:
    q = (d.year * 12 + (d.month - 1) // 3 * 3) + shift * 3
    return date(q // 12, q % 12 + 1, 1)


def data_period() -> dict:
    """Date range of the shipment data and the 'current' and 'previous' quarter bounds.

    The current quarter is the one holding the LATEST booking in the data, not today's date.
    Call this before any question about 'this quarter', 'last quarter', 'recent', etc.
    Quarter `end` dates are exclusive: pass start/end as booked_from/booked_to.
    """
    newest, err = _call("GET", "/shipments", params={"limit": 1})
    if err:
        return err
    if not newest["total"]:
        return _error("There are no shipments in the data.")
    oldest, err = _call("GET", "/shipments", params={"limit": 1, "offset": newest["total"] - 1})
    if err:
        return err
    latest = date.fromisoformat(newest["items"][0]["booking_date"][:10])
    cur, prev, nxt = (_quarter_start(latest, s) for s in (0, -1, 1))
    return {
        "status": "ok",
        "min_booking_date": oldest["items"][0]["booking_date"][:10],
        "max_booking_date": latest.isoformat(),
        "current_quarter": {"start": cur.isoformat(), "end": nxt.isoformat()},
        "previous_quarter": {"start": prev.isoformat(), "end": cur.isoformat()},
    }


TOOLS = [
    list_ports,
    resolve_port,
    query_shipments,
    get_shipment,
    get_route_stats,
    top_routes,
    predict_delay,
    data_period,
]
