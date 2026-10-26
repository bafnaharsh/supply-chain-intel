"""Live Gemini + running API. Needs GEMINI_API_KEY (env or .env) and BASE_URL."""

import asyncio
import logging
import os

import httpx
import pytest

from core.config import settings

KEY = settings.gemini_api_key
BASE_URL = os.getenv("BASE_URL")
pytestmark = pytest.mark.skipif(not (KEY and BASE_URL), reason="needs a Gemini key and BASE_URL")


@pytest.fixture(scope="module")
def run():
    from assistant import tools
    from assistant.agent import build_agent
    from assistant.runner import ask
    from assistant.session import get_or_create_session, make_session_service

    os.environ.setdefault("GOOGLE_API_KEY", KEY)
    tools.configure(httpx.Client(base_url=BASE_URL, timeout=15))
    agent, svc = build_agent(), make_session_service(persistent=False)

    def _ask(question: str):
        async def go():
            session = await get_or_create_session(svc, "live-test")  # fresh session per question
            return await ask(
                question,
                agent=agent,
                session_service=svc,
                user_id="live-test",
                session_id=session.id,
            )

        return asyncio.run(go())

    return _ask


def test_top_routes_this_quarter(run):
    r = run("Which routes have had the highest average delay this quarter?")
    assert "top_routes" in r.tools_called
    assert r.guardrail is None


def test_on_time_rate_route(run, caplog):
    caplog.set_level(logging.INFO, logger="assistant")
    r = run("What is the on-time rate for shipments from Shanghai to Rotterdam?")
    assert "get_route_stats" in r.tools_called
    calls = [
        rec.tool_args for rec in caplog.records if getattr(rec, "tool", None) == "get_route_stats"
    ]
    assert {"origin": "CNSHA", "destination": "NLRTM"} in calls


def test_delay_risk_for_shipment(run):
    r = run("What is the delay risk for shipment SHP-00421?")
    assert "predict_delay" in r.tools_called
    assert "421" in r.text


def test_missing_destination_asks_back(run):
    r = run("What is the on-time rate from Shanghai?")
    assert "get_route_stats" not in r.tools_called
    assert "?" in r.text
