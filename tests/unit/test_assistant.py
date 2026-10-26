"""Assistant tools, guardrails and the ADK loop, against the fixture API with a scripted LLM."""

import asyncio
import logging
from types import SimpleNamespace

import pytest
from google.adk.models import BaseLlm, LlmResponse
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.genai import types

from assistant import agent as agent_mod
from assistant import tools
from assistant.runner import ask
from assistant.session import get_or_create_session
from core.config import ASSISTANT_APP_NAME, ASSISTANT_TOOL_ROW_LIMIT, settings


@pytest.fixture
def api(client):
    tools.configure(client)
    yield client
    tools.configure(None)


def _ctx(**state):
    """Stand-in for an ADK callback/tool context: only what the callbacks touch."""
    return SimpleNamespace(
        state=dict(state), session=SimpleNamespace(id="s1"), user_id="u1", function_call_id="f1"
    )


class FakeLlm(BaseLlm):
    """Replays scripted responses, one per model call."""

    script: list[LlmResponse]

    async def generate_content_async(self, llm_request, stream=False):
        yield self.script.pop(0)


def _script() -> list[LlmResponse]:
    call = types.Part(
        function_call=types.FunctionCall(
            name="get_route_stats", args={"origin": "CNSHA", "destination": "NLRTM"}
        )
    )
    text = types.Part(text="On-time rate is 50% (get_route_stats)")
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=1000, candidates_token_count=20
    )
    return [
        LlmResponse(content=types.Content(role="model", parts=[p]), usage_metadata=usage)
        for p in (call, text)
    ]


async def _one_turn(svc, question="On-time rate Shanghai to Rotterdam?"):
    session = await get_or_create_session(svc, "u1")
    agent = agent_mod.build_agent(FakeLlm(model="gemini-3.5-flash-lite", script=_script()))
    result = await ask(
        question, agent=agent, session_service=svc, user_id="u1", session_id=session.id
    )
    return session.id, result


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("421", "SHP-00421"),
        ("shp 421", "SHP-00421"),
        ("SHP-00421", "SHP-00421"),
        ("SHP-421", "SHP-00421"),
        ("bogus", None),
    ],
)
def test_normalise_shipment_id(raw, expected):
    assert tools.normalise_shipment_id(raw) == expected


def test_resolve_port_exact(api):
    assert tools.resolve_port("CNSHA")["port_code"] == "CNSHA"
    assert tools.resolve_port("shanghai")["port_code"] == "CNSHA"


def test_resolve_port_ambiguous(api):
    r = tools.resolve_port("Ham")
    assert r.get("port_code") == "DEHAM" or "DEHAM" in [c["port_code"] for c in r["candidates"]]
    r = tools.resolve_port(
        "Rotterdm"
    )  # one obvious typo: corrected, and the correction is announced
    assert r["status"] == "ok" and r["port_code"] == "NLRTM"
    assert r["corrected_from"] == "Rotterdm" and "Rotterdam" in r["note"]
    r = tools.resolve_port("Atlantis")  # nothing close: must not be auto-picked
    assert r["status"] == "unknown"


def test_get_shipment_tool(api):
    r = tools.get_shipment("1")
    assert r["status"] == "ok" and r["shipment_id"] == "SHP-00001"
    assert r["shipment_status"] == "DELIVERED"
    assert r["route_key"] == "CNSHA → NLRTM" and r["actual_delay_hours"] == 10.0
    assert tools.get_shipment("SHP-99999")["status"] == "error"
    assert tools.get_shipment("bogus")["status"] == "error"


def test_top_routes_order_asc(api):
    r = tools.top_routes(min_completed=1, sort="on_time_rate", order="asc")
    assert r["status"] == "ok"
    assert [x["route_key"] for x in r["routes"]][0] == "CNSHA → NLRTM"  # 0.5 < 1.0
    assert tools.top_routes(min_completed=1, order="sideways")["status"] == "ok"  # falls back


@pytest.mark.parametrize(
    ("text", "flagged"),
    [
        ("On-time rate is 87%.", True),
        ("Could you give the shipment id (e.g. SHP-00421)?", False),
        ("Shipment SHP-99999 was not found. Please check the id.", False),
        ("Q3 2025 is the previous quarter; which one do you want?", False),
        ("There were 12 shipments.", True),
    ],
)
def test_unsupported_number_detection(text, flagged):
    from assistant.agent import _has_unsupported_numbers

    assert _has_unsupported_numbers(text) is flagged


def test_resolve_port_unknown(api):
    r = tools.resolve_port("Atlantis")
    assert r["status"] == "unknown"
    assert len(r["candidates"]) <= 3


def test_get_route_stats_tool(api):
    r = tools.get_route_stats("CNSHA", "NLRTM")
    assert r["status"] == "ok"
    assert r["shipment_count"] == 3
    assert r["on_time_rate"] == 0.5


def test_get_route_stats_404_is_error_dict(api):
    r = tools.get_route_stats("CNSHA", "USLAX")
    assert r["status"] == "error"
    assert "404" in r["message"] and "resolve_port" in r["hint"]


def test_top_routes_tool(api):
    r = tools.top_routes(min_completed=1)
    assert r["routes"][0]["route_key"] == "CNSHA → NLRTM"
    assert r["routes"][0]["avg_delay_hours"] == 29.0


def test_query_shipments_caps_limit(api):
    r = tools.query_shipments(limit=999)
    assert r["total"] == 4
    assert len(r["shipments"]) <= ASSISTANT_TOOL_ROW_LIMIT
    assert set(r["shipments"][0]) == {
        "shipment_id",
        "route_key",
        "booking_date",
        "status",
        "actual_delay_hours",
        "on_time_flag",
    }


def test_predict_delay_tool_hits_endpoint(api):
    missing = tools.predict_delay("421")
    assert missing["status"] == "error" and "404" in missing["message"]
    assert tools.predict_delay("bogus")["status"] == "error"
    r = tools.predict_delay("SHP-00001")
    assert r["status"] == "ok"
    assert 0 <= r["delay_risk"] <= 1 and r["model_version"]


def test_tools_report_unreachable_api():
    tools.configure(None)
    assert tools.list_ports()["status"] == "error"


def test_data_period_quarters(api):
    r = tools.data_period()
    assert r["max_booking_date"] == "2024-07-10"
    assert r["min_booking_date"] == "2024-06-01"
    assert r["current_quarter"] == {"start": "2024-07-01", "end": "2024-10-01"}
    assert r["previous_quarter"] == {"start": "2024-04-01", "end": "2024-07-01"}


def test_before_tool_blocks_bad_port_without_http(caplog):
    tool = SimpleNamespace(name="get_route_stats")  # http is unset: a real call would error
    r = agent_mod._before_tool(tool, {"origin": "Shanghai", "destination": "NLRTM"}, _ctx())
    assert r["status"] == "error" and "resolve_port" in r["message"]
    bad_dates = {"booked_from": "2024-10-01", "booked_to": "2024-07-01"}
    assert agent_mod._before_tool(tool, bad_dates, _ctx())["status"] == "error"
    assert agent_mod._before_tool(tool, {"origin": "CNSHA"}, _ctx()) is None
    assert "guardrail_triggered" in caplog.messages


def test_tool_cap():
    ctx = _ctx(turn_tool_calls=settings.assistant_max_tool_calls)
    r = agent_mod._before_tool(SimpleNamespace(name="list_ports"), {}, ctx)
    assert r["status"] == "error" and "limit" in r["message"]
    assert ctx.state[agent_mod.K_GUARD] == "max_tool_calls"


def test_llm_cap():
    ctx = _ctx(turn_llm_calls=settings.assistant_max_llm_calls)
    r = agent_mod._before_model(ctx, SimpleNamespace(contents=[], model="m"))
    assert r.content.parts[0].text == agent_mod.LLM_CAP_TEXT
    assert ctx.state[agent_mod.K_GUARD] == "max_llm_calls"


def test_grounding_guard():
    resp = LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Rate is 73%")]))
    replaced = agent_mod._after_model(_ctx(), resp)
    assert replaced.content.parts[0].text == agent_mod.UNGROUNDED_TEXT
    assert agent_mod._after_model(_ctx(turn_tool_ok=True), resp) is None
    clarify = types.Content(role="model", parts=[types.Part(text="Which destination?")])
    assert agent_mod._after_model(_ctx(), LlmResponse(content=clarify)) is None


def test_data_period_cached_in_session_state():
    cached = {"status": "ok", "max_booking_date": "2024-07-10"}
    tool = SimpleNamespace(name="data_period")
    assert agent_mod._before_tool(tool, {}, _ctx(data_period=cached)) is cached


def test_agent_loop_with_fake_llm_logs_everything(api, caplog):
    caplog.set_level(logging.INFO, logger="assistant")
    session_id, r = asyncio.run(_one_turn(InMemorySessionService()))
    assert r.text == "On-time rate is 50% (get_route_stats)"
    assert r.tools_called == ["get_route_stats"]
    assert r.llm_calls == 2 and r.guardrail is None
    assert r.usage["prompt_tokens"] == 2000 and r.usage["est_usd"] > 0
    by_msg = {rec.getMessage(): rec for rec in caplog.records if rec.name == "assistant"}
    assert {"llm_call", "llm_response", "tool_call"} <= set(by_msg)
    assert all(rec.session_id == session_id for rec in by_msg.values())
    assert by_msg["tool_call"].status == "ok"


def test_session_persists_and_resumes(api, tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}"

    async def run():
        session_id, _ = await _one_turn(DatabaseSessionService(db_url=url))
        fresh = DatabaseSessionService(db_url=url)
        return await fresh.get_session(
            app_name=ASSISTANT_APP_NAME, user_id="u1", session_id=session_id
        )

    resumed = asyncio.run(run())
    assert len(resumed.events) > 0
    assert resumed.state["turn_index"] == 1


def test_off_domain_instruction_present():
    text = agent_mod.INSTRUCTION.lower()
    assert "refuse politely" in text
    assert "clarifying question" in text
    assert "latest booking" in text
