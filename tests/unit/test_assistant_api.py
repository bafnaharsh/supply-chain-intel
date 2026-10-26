"""The assistant's HTTP front, driven by the FakeLlm against the in-process data API."""

import pytest
from fastapi.testclient import TestClient
from google.adk.sessions import InMemorySessionService

from assistant import agent as agent_mod
from assistant.api import create_app
from tests.unit.test_assistant import FakeLlm, _script


@pytest.fixture
def chat(client):
    """Assistant API over a FakeLlm with enough scripted turns for two questions."""
    fake = FakeLlm(model="gemini-3.5-flash-lite", script=_script() + _script())
    app = create_app(
        agent=agent_mod.build_agent(fake),
        session_service=InMemorySessionService(),
        api_client=client,
        log=False,
    )
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health_reports_data_api_and_model(chat):
    r = chat.get("/health")
    assert r.status_code == 200 and r.headers["x-request-id"]
    assert r.json() == {"status": "ok", "data_api": "ok", "model": "gemini-3.5-flash-lite"}


def test_chat_answers_and_returns_session(chat):
    r = chat.post("/chat", json={"question": "On-time rate Shanghai to Rotterdam?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "On-time rate is 50% (get_route_stats)"
    assert body["tools_called"] == ["get_route_stats"]
    assert body["usage"]["prompt_tokens"] == 2000 and body["usage"]["est_usd"] > 0
    assert body["guardrail"] is None
    assert body["session_id"]


def test_chat_resumes_session_and_lists_it(chat):
    first = chat.post("/chat", json={"question": "first"}).json()
    second = chat.post("/chat", json={"question": "second", "session_id": first["session_id"]})
    assert second.status_code == 200
    assert second.json()["session_id"] == first["session_id"]
    sessions = chat.get("/sessions").json()
    assert [s["session_id"] for s in sessions] == [first["session_id"]]


def test_concurrent_turns_on_one_session_are_serialised(client):
    """Two simultaneous turns on the same session must not interleave."""
    import asyncio

    import httpx

    fake = FakeLlm(model="gemini-3.5-flash-lite", script=_script() * 3)
    app = create_app(
        agent=agent_mod.build_agent(fake),
        session_service=InMemorySessionService(),
        api_client=client,
        log=False,
    )

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=transport, base_url="http://t") as c,
        ):
            sid = (await c.post("/chat", json={"question": "first"})).json()["session_id"]
            body = {"question": "again", "session_id": sid}
            return await asyncio.gather(c.post("/chat", json=body), c.post("/chat", json=body))

    a, b = asyncio.run(run())
    assert a.status_code == b.status_code == 200
    # Each turn ran its full scripted tool call + answer: nothing was interleaved or dropped.
    assert a.json()["tools_called"] == b.json()["tools_called"] == ["get_route_stats"]


def test_without_key_is_degraded_not_dead(client, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "gemini_api_key", None)
    app = create_app(session_service=InMemorySessionService(), api_client=client, log=False)
    with TestClient(app, raise_server_exceptions=False) as c:
        h = c.get("/health").json()
        assert h["status"] == "degraded" and h["model"] == "unconfigured"
        r = c.post("/chat", json={"question": "hi"})
        assert r.status_code == 503 and r.json()["error"]["code"] == "llm_unconfigured"


@pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": "x" * 2001}])
def test_chat_validates_input_with_structured_error(chat, body):
    r = chat.post("/chat", json=body)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"
