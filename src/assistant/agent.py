"""The ADK agent: instruction, tools and the guardrail/logging callbacks."""

import logging
import sys
import time
from datetime import date

from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

from app.logging_config import JsonFormatter
from assistant import tools
from core.config import GEMINI_PRICES_PER_MTOK, settings

log = logging.getLogger("assistant")

LLM_CAP_TEXT = "I've hit my reasoning limit for this question; please ask something narrower."
UNGROUNDED_TEXT = (
    "I can only answer from shipment data and I couldn't retrieve any for that question."
)


def _is_port_code(value: object) -> bool:
    s = str(value)
    return len(s) == 5 and s.isascii() and s.isalpha() and s.isupper()


# Per-turn state keys, reset by _start_turn.
K_LLM, K_TOOLS, K_OK, K_GUARD, K_T0, K_MODEL = (
    "turn_llm_calls",
    "turn_tool_calls",
    "turn_tool_ok",
    "turn_guardrail",
    "turn_llm_t0",
    "turn_model",
)

INSTRUCTION = """\
You are a supply-chain analyst assistant for an ocean-shipping dataset. You answer ONLY questions \
about shipments, routes, ports and shipment delay risk, using the tools. For anything else \
(general knowledge, coding, chit-chat beyond a greeting) refuse politely in one sentence and say \
what you can help with.

Rules:
1. Port names: ALWAYS call resolve_port to turn a port name into its 5-letter code before using \
it. If it returns 'ok' with `corrected_from`, mention the correction; if 'ambiguous', ask the \
user which port; if 'unknown', say the port is not in the data.
2. Shipment ids: pass the id exactly as the user wrote it to get_shipment / predict_delay; they \
normalise '421', 'shp 421' and 'SHP-00421'. Never guess an id. For 'status of' / 'where is' \
questions use get_shipment. When asked for a delay risk, call get_shipment AND predict_delay: \
report the booking-time risk from predict_delay, and if the shipment is already DELIVERED, \
DELAYED or CANCELLED say so and give the actual outcome too (the risk is what the model would \
have said at booking). Never say a shipment is 'not found' unless get_shipment or \
predict_delay returned that error.
3. Time: the data is historical. 'This quarter', 'last quarter' and 'recent' are relative to the \
LATEST booking in the data, never today's date: call data_period first and use its quarter \
bounds as booked_from/booked_to. If the user names a window that does not exist (e.g. 'Q5', a \
year outside the data), say what range the data covers and ask which window they want.
4. When information is missing or ambiguous (a route question with no destination, an ambiguous \
port, no shipment id, an unclear time window, or a ranking with no metric such as 'worst route' \
without saying delay or on-time rate) do NOT guess and do NOT call data tools: ask ONE concise \
clarifying question offering the likely options, e.g. "Shanghai (CNSHA) to which destination? \
Rotterdam (NLRTM) and Hamburg (DEHAM) both have data."
5. Rankings: use top_routes once with the right sort and order ('worst on-time rate' = \
sort='on_time_rate', order='asc'; 'highest delay' = sort='avg_delay_hours', order='desc'). \
Never page through results to find an extreme.
6. If a ranking for 'this quarter' returns fewer than 5 routes, say the quarter is thin (the data \
ends mid-quarter) and offer the previous quarter.
7. Every number must come from a tool result in this conversation. State the window and the \
number of shipments (n) behind it, and cite the tool in brackets, e.g. "On-time rate 62% \
(n=118 completed, all time) (get_route_stats)". Never invent numbers; if no tool returned data, \
say so.
8. Be brief: a sentence or a short list. Rates as percentages, delays in hours.

Ports in the data:
{ports}
"""


def configure_logging() -> None:
    """JSON lines to settings.assistant_log_path and stderr (stdout stays for answers)."""
    settings.assistant_log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(settings.assistant_log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ]
    for h in handlers:
        h.setFormatter(JsonFormatter())
    log.handlers[:] = handlers
    log.setLevel(logging.INFO)
    log.propagate = False


def estimate_usd(model: str, prompt_tokens: int, output_tokens: int) -> float:
    price_in, price_out = GEMINI_PRICES_PER_MTOK.get(model, (0.0, 0.0))
    return (prompt_tokens * price_in + output_tokens * price_out) / 1e6


def _ids(ctx) -> dict:
    return {"session_id": ctx.session.id, "user_id": ctx.user_id}


def _parts(contents: list[types.Content]) -> list[dict]:
    """Compact view of content parts for the log."""
    out = []
    for c in contents:
        for p in c.parts or []:
            if p.text:
                out.append({"role": c.role, "text": p.text})
            elif p.function_call:
                fc = p.function_call
                out.append({"role": c.role, "function_call": fc.name, "args": fc.args})
            elif p.function_response:
                fr = p.function_response
                out.append({"role": c.role, "function_response": fr.name, "result": fr.response})
    return out


def _start_turn(callback_context) -> None:
    s = callback_context.state
    s["turn_index"] = s.get("turn_index", 0) + 1
    for k, v in ((K_LLM, 0), (K_TOOLS, 0), (K_OK, False), (K_GUARD, None)):
        s[k] = v


def _guard(ctx, name: str, **extra) -> None:
    ctx.state[K_GUARD] = name
    log.warning("guardrail_triggered", extra={"guardrail": name, **_ids(ctx), **extra})


def _before_model(callback_context, llm_request: LlmRequest) -> LlmResponse | None:
    s = callback_context.state
    s[K_LLM] = s.get(K_LLM, 0) + 1
    if s[K_LLM] > settings.assistant_max_llm_calls:
        _guard(callback_context, "max_llm_calls", limit=settings.assistant_max_llm_calls)
        text = types.Content(role="model", parts=[types.Part(text=LLM_CAP_TEXT)])
        return LlmResponse(content=text)
    contents = llm_request.contents or []
    last_model = max((i for i, c in enumerate(contents) if c.role == "model"), default=-1)
    s[K_T0], s[K_MODEL] = time.perf_counter(), llm_request.model
    if s[K_LLM] == 1:
        # Once per turn, the complete prompt as sent: system instruction, tool declarations and
        # the full conversation so far. Later calls in the turn log only what is new.
        cfg = llm_request.config
        log.info(
            "llm_prompt",
            extra={
                **_ids(callback_context),
                "model": llm_request.model,
                "turn": s.get("turn_index"),
                "system_instruction": str(cfg.system_instruction) if cfg else None,
                "tools": sorted(llm_request.tools_dict or {}),
                "contents": _parts(contents),
            },
        )
    log.info(
        "llm_call",
        extra={
            **_ids(callback_context),
            "model": llm_request.model,
            "turn": s.get("turn_index"),
            "llm_call": s[K_LLM],
            "new_contents": _parts(contents[last_model + 1 :]),
        },
    )
    return None


def _after_model(callback_context, llm_response: LlmResponse) -> LlmResponse | None:
    s = callback_context.state
    usage = llm_response.usage_metadata
    p_tok = (usage and usage.prompt_token_count) or 0
    o_tok = (usage and usage.candidates_token_count) or 0
    model = s.get(K_MODEL) or ""
    content = llm_response.content
    parts = _parts([content]) if content else []
    log.info(
        "llm_response",
        extra={
            **_ids(callback_context),
            "model": model,
            "turn": s.get("turn_index"),
            "parts": parts,
            "prompt_tokens": p_tok,
            "output_tokens": o_tok,
            "est_usd": round(estimate_usd(model, p_tok, o_tok), 6),
            "latency_ms": round((time.perf_counter() - s.get(K_T0, time.perf_counter())) * 1000),
        },
    )
    text = "".join(p.get("text", "") for p in parts)
    is_final = text and not any("function_call" in p for p in parts)
    if is_final and _has_unsupported_numbers(text) and not s.get(K_OK):
        _guard(callback_context, "ungrounded_numbers", text=text)
        new = types.Content(role="model", parts=[types.Part(text=UNGROUNDED_TEXT)])
        return llm_response.model_copy(update={"content": new})
    return None


_PUNCT = str.maketrans("", "", "().,;:!?*\"'")


def _is_claim(token: str) -> bool:
    """A token with a digit counts as a factual claim unless it is a shipment id, a quarter
    label, the 24 h threshold or a word like '5-letter'."""
    t = token.translate(_PUNCT).lower()
    if not any(c.isdigit() for c in t):
        return False
    if t.startswith("shp") or t in ("24h", "24") or t.endswith("-letter"):
        return False
    return not (len(t) == 2 and t[0] == "q" and t[1] in "1234")


def _has_unsupported_numbers(text: str) -> bool:
    """Any claim-like number, unless the text is a clarifying question (which may quote an
    example id)."""
    if text.rstrip().endswith("?"):
        return False
    return any(_is_claim(tok) for tok in text.split())


def _check_args(args: dict) -> str | None:
    """Cheap validation so a bad call never reaches the API."""
    for key in ("origin", "destination"):
        if args.get(key) is not None and not _is_port_code(args[key]):
            return f"{key}={args[key]!r} is not a 5-letter port code; call resolve_port first."
    try:
        start, end = (
            date.fromisoformat(args[k][:10]) if args.get(k) else None
            for k in ("booked_from", "booked_to")
        )
    except (TypeError, ValueError):
        return "booked_from/booked_to must be ISO dates 'YYYY-MM-DD'."
    if start and end and start >= end:
        return "booked_from must be before booked_to."
    return None


_tool_t0: dict[str, float] = {}  # function_call_id -> start, for tool latency


def _before_tool(tool, args: dict, tool_context) -> dict | None:
    s = tool_context.state
    _tool_t0[tool_context.function_call_id or ""] = time.perf_counter()
    s[K_TOOLS] = s.get(K_TOOLS, 0) + 1
    if s[K_TOOLS] > settings.assistant_max_tool_calls:
        _guard(tool_context, "max_tool_calls", tool=tool.name)
        msg = f"Tool-call limit ({settings.assistant_max_tool_calls}) reached for this question."
        return {"status": "error", "message": msg, "hint": "Answer with what you have."}
    if problem := _check_args(args):
        _guard(tool_context, "invalid_tool_args", tool=tool.name, tool_args=args)
        return {"status": "error", "message": problem, "hint": "Fix the arguments and retry."}
    if tool.name == "data_period" and s.get("data_period"):
        return s["data_period"]  # fixed once per session so 'this quarter' never moves
    return None


def _after_tool(tool, args: dict, tool_context, tool_response) -> dict | None:
    s = tool_context.state
    t0 = _tool_t0.pop(tool_context.function_call_id or "", time.perf_counter())
    status = tool_response.get("status") if isinstance(tool_response, dict) else None
    if status == "ok":
        s[K_OK] = True
        if tool.name == "data_period":
            s["data_period"] = tool_response
    log.info(
        "tool_call",
        extra={
            **_ids(tool_context),
            "tool": tool.name,
            "tool_args": args,  # "args" is a reserved LogRecord attribute
            "status": status,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "result_chars": len(str(tool_response)),
        },
    )
    return None


def _port_table() -> str:
    listing = tools.list_ports() if tools.http is not None else {"status": "error"}
    if listing["status"] != "ok":
        return "(not loaded; use list_ports)"
    return "\n".join(
        f"- {p['port_code']}: {p['port_name']}, {p['country']} ({p['region']})"
        for p in listing["ports"]
    )


def build_agent(model: str | BaseLlm | None = None) -> LlmAgent:
    """The assistant agent; call tools.configure() first so the port table is embedded."""
    return LlmAgent(
        name="supply_chain_assistant",
        description="Answers shipment, route, port and delay-risk questions from the API.",
        model=model or settings.assistant_model,  # read at call time so tests can override
        instruction=INSTRUCTION.format(ports=_port_table()),
        tools=tools.TOOLS,
        generate_content_config=types.GenerateContentConfig(temperature=0.1),
        before_agent_callback=_start_turn,
        before_model_callback=_before_model,
        after_model_callback=_after_model,
        before_tool_callback=_before_tool,
        after_tool_callback=_after_tool,
    )
