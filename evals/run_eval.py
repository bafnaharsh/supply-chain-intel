"""Accuracy eval of the assistant against evals/questions.json, live Gemini + running API.

    uv run python evals/run_eval.py                 # all questions
    uv run python evals/run_eval.py --ids A3,B4     # a subset (e.g. re-run failures)

Each question runs in a fresh in-memory session through assistant.runner.ask with the real
agent. Writes evals/results_<timestamp>.json and prints a per-group table.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from google.genai import types

from assistant import tools
from assistant.agent import LLM_CAP_TEXT, UNGROUNDED_TEXT, build_agent
from assistant.runner import ask
from assistant.session import get_or_create_session, make_session_service
from core.config import ON_TIME_THRESHOLD_HOURS, settings

HERE = Path(__file__).parent
DATA_TOOLS = {"query_shipments", "get_route_stats", "top_routes", "predict_delay"}
TOOL_NAMES = {t.__name__ for t in tools.TOOLS}
RPM_SLEEP_S = 4.5  # per LLM call: stays under the free tier's 15 requests/min
DIMS = ("tool", "value", "behaviour", "grounded", "citation")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_REFUSAL = re.compile(
    r"can't|cannot|can not|unable|only (?:help|answer|assist)|not (?:in|found|available|able)"
    r"|don't|do not|isn't|is not|no (?:shipment|data|tool|such)|couldn't|could not",
    re.IGNORECASE,
)


#  Not factual claims: list ordinals "1. ", quarter labels "Q3", years/months, shipment ids.
_ASKS = re.compile(r"\bplease (?:provide|give|tell|specify|share|let me know)\b", re.IGNORECASE)
_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
_NOT_CLAIMS = re.compile(
    r"(?m)^\s*\d+\.\s|\bQ\d\b|\b(?:19|20)\d\d\b|\b\d{4}-\d\d(?:-\d\d)?\b|SHP-?\d+"
    rf"|\b{_MONTHS}\s+\d{{1,2}}\b|\b\d{{1,2}}\s+{_MONTHS}\b",  # "June 1", "14 October"
    re.IGNORECASE,
)


def nums(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUM.findall(_NOT_CLAIMS.sub(" ", text or ""))]


class Capture(logging.Handler):
    """Collects the assistant's tool_call and llm_call log records for one question."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def tool_calls(self) -> list[dict]:
        return [
            {"tool": r.tool, "args": r.tool_args, "status": r.status}
            for r in self.records
            if r.getMessage() == "tool_call"
        ]

    def tool_results(self) -> list:
        return [
            p["result"]
            for r in self.records
            if r.getMessage() == "llm_call"
            for p in r.new_contents
            if "function_response" in p
        ]


def value_ok(ev: dict, text: str) -> bool:
    if ev["kind"] == "str":
        return any(a.casefold() in text.casefold() for a in ev["any"])
    found, v = nums(text), ev["value"]
    if ev["kind"] == "int":
        return v in found
    if ev["kind"] == "hours":
        return any(abs(n - v) <= 0.5 for n in found)
    # pct: "75%", "75.0" or "0.75"
    return any(abs(n - v * 100) <= 0.5 or abs(n - v) <= 0.005 for n in found)


def score(q: dict, text: str, called: list[str], calls: list[dict], results: list) -> dict:
    called_set = set(called)
    data_ok = any(c["tool"] in DATA_TOOLS and c["status"] == "ok" for c in calls)
    any_ok = any(c["status"] == "ok" for c in calls)
    canned = text.strip() in (UNGROUNDED_TEXT, LLM_CAP_TEXT)

    tool = set(q["expected_tools"]) <= called_set and not (set(q["forbidden_tools"]) & called_set)

    checks = [value_ok(ev, text) for ev in q["expected_values"]]
    checks += [f.casefold() not in text.casefold() for f in q.get("forbidden_values", [])]
    value = all(checks) if checks else None

    b = q["expected_behaviour"]
    if b == "clarify":
        # A clarification is a question or an imperative request for the missing detail.
        asks = "?" in text or bool(_ASKS.search(text))
        behaviour = asks and not (DATA_TOOLS & called_set)
    elif b == "refuse":
        behaviour = not data_ok and not canned and bool(_REFUSAL.search(text))
    else:
        behaviour = any_ok and not canned

    # Numbers the user typed are not claims; everything else must trace to a tool result.
    q_nums = set(nums(q["question"]))
    claims = [n for n in nums(text) if n not in q_nums]
    # 24 h is the on-time threshold stated in the tool docstrings the model sees.
    pool = nums(json.dumps(results, default=str)) + [ON_TIME_THRESHOLD_HOURS]
    pool += [p * 100 for p in pool]
    grounded = all(any(abs(n - p) <= 0.51 for p in pool) for n in claims)
    citation = any(t in text for t in TOOL_NAMES) if claims else None
    return {
        "tool": tool,
        "value": value,
        "behaviour": behaviour,
        "grounded": grounded,
        "citation": citation,
        "ungrounded_claims": [n for n in claims if not any(abs(n - p) <= 0.51 for p in pool)],
    }


async def run(questions: list[dict]) -> list[dict]:
    os.environ.setdefault("GOOGLE_API_KEY", settings.gemini_api_key or "")
    tools.configure(httpx.Client(base_url=settings.assistant_api_url, timeout=15))
    agent = build_agent()
    # build_agent sets temperature 0.1; pin to 0 (+ seed) here without touching src/.
    agent.generate_content_config = types.GenerateContentConfig(temperature=0, seed=0)
    svc = make_session_service(persistent=False)
    cap = Capture()
    log = logging.getLogger("assistant")
    log.handlers[:] = [cap]
    log.setLevel(logging.INFO)
    log.propagate = False

    out = []
    for q in questions:
        for _attempt in range(3):
            cap.records.clear()
            session = await get_or_create_session(svc, "eval")  # fresh session per attempt
            t0 = time.perf_counter()
            try:
                r = await ask(
                    q["question"],
                    agent=agent,
                    session_service=svc,
                    user_id="eval",
                    session_id=session.id,
                )
                err = None
            except Exception as e:  # record and keep going; one API hiccup should not sink the run
                r, err = None, f"{type(e).__name__}: {str(e)[:300]}"
            latency = time.perf_counter() - t0
            if not (err and "RESOURCE_EXHAUSTED" in err):
                break
            print(f"{q['id']} 429, waiting 30s", flush=True)  # rejected calls are not billed
            await asyncio.sleep(30)
        # shortcut: fixed pacing for the free tier's 15 requests/min; drop on a paid key.
        await asyncio.sleep(RPM_SLEEP_S * max(1, r.llm_calls if r else 1))
        text = r.text if r else ""
        calls = cap.tool_calls()
        s = score(q, text, r.tools_called if r else [], calls, cap.tool_results())
        s["passed"] = err is None and all(s[d] is not False for d in DIMS)
        out.append(
            {
                "id": q["id"],
                "group": q["group"],
                "question": q["question"],
                "answer": text,
                "error": err,
                "tools_called": r.tools_called if r else [],
                "tool_calls": calls,
                "guardrail": r.guardrail if r else None,
                "guardrail_replaced_text": [
                    getattr(x, "text", None)
                    for x in cap.records
                    if x.getMessage() == "guardrail_triggered"
                ],
                "tool_results": cap.tool_results(),
                "llm_calls": r.llm_calls if r else 0,
                "prompt_tokens": r.usage["prompt_tokens"] if r else 0,
                "output_tokens": r.usage["output_tokens"] if r else 0,
                "est_usd": r.usage["est_usd"] if r else 0.0,
                "latency_s": round(latency, 2),
                **s,
            }
        )
        mark = "PASS" if s["passed"] else "FAIL"
        print(f"{q['id']} {mark} {latency:5.1f}s tools={r.tools_called if r else err}", flush=True)
    return out


def rate(rows: list[dict], dim: str) -> str:
    vals = [r[dim] for r in rows if r[dim] is not None]
    return f"{sum(vals)}/{len(vals)}" if vals else "-"


def report(rows: list[dict]) -> None:
    hdr = ("group", "n", *DIMS, "passed")
    print("\n| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for g in sorted({r["group"] for r in rows}) + ["ALL"]:
        rs = [r for r in rows if g in ("ALL", r["group"])]
        cells = [g, str(len(rs)), *(rate(rs, d) for d in DIMS), rate(rs, "passed")]
        print("| " + " | ".join(cells) + " |")
    passed = sum(r["passed"] for r in rows)
    print(
        f"\nOverall accuracy {passed}/{len(rows)} = {passed / len(rows):.0%} | "
        f"LLM calls {sum(r['llm_calls'] for r in rows)} | "
        f"tokens {sum(r['prompt_tokens'] for r in rows)} in / "
        f"{sum(r['output_tokens'] for r in rows)} out | "
        f"cost ${sum(r['est_usd'] for r in rows):.4f} | "
        f"mean latency {statistics.mean(r['latency_s'] for r in rows):.1f}s"
    )
    failed = [r["id"] for r in rows if not r["passed"]]
    if failed:
        print("failed: " + ",".join(failed))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--ids", help="comma-separated question ids to run")
    args = p.parse_args()
    if not settings.gemini_api_key:
        sys.exit("GEMINI_API_KEY is not set (.env).")
    qs = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))["questions"]
    if args.ids:
        wanted = set(args.ids.split(","))
        qs = [q for q in qs if q["id"] in wanted]
    rows = asyncio.run(run(qs))
    path = HERE / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
    meta = {"model": settings.assistant_model, "temperature": 0, "seed": 0, "ids": args.ids}
    path.write_text(json.dumps({"meta": meta, "results": rows}, indent=2), encoding="utf-8")
    report(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
