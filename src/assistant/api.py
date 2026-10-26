"""HTTP front for the assistant: the same agent, sessions, guardrails and logs as the CLI.

Runs on the host next to the Gemini key, talking to the data API over HTTP:

    uv run uvicorn assistant.api:app --port 8001      # then open http://localhost:8001/docs
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Query
from google.adk.agents import LlmAgent
from google.adk.sessions import BaseSessionService
from pydantic import BaseModel, Field

from app.errors import APIError, register_error_handlers
from app.middleware import add_request_logging
from app.schemas import ErrorResponse
from assistant import tools
from assistant.agent import build_agent, configure_logging
from assistant.runner import ask
from assistant.session import get_or_create_session, list_sessions, make_session_service
from core.config import settings


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, description="Resume this session; omit for new")
    user_id: str = Field(default="local", min_length=1, max_length=64)


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    tools_called: list[str]
    llm_calls: int
    usage: dict[str, Any]
    guardrail: str | None


class SessionInfo(BaseModel):
    session_id: str
    last_update_time: float


def create_app(
    *,
    agent: LlmAgent | None = None,
    session_service: BaseSessionService | None = None,
    api_client: httpx.Client | None = None,
    log: bool = True,
) -> FastAPI:
    """Keyword arguments exist so tests can inject a fake model and an in-process data API."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tools.configure(api_client or httpx.Client(base_url=settings.assistant_api_url, timeout=15))
        if log:
            configure_logging()
        if agent is None and not settings.gemini_api_key:
            # Degraded, not dead: /health says so and /chat answers 503, instead of a crash loop
            # under compose's restart policy on a clone that has no .env yet.
            logging.getLogger("assistant").error(
                "llm_unconfigured", extra={"hint": "set GEMINI_API_KEY in .env (see .env.example)"}
            )
            app.state.agent = None
        else:
            if agent is None:
                os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key
            app.state.agent = agent or build_agent()
        app.state.sessions = session_service or make_session_service()
        yield
        tools.configure(None)

    app = FastAPI(
        title="Supply Chain Assistant API",
        description="Ask questions about shipments, routes, ports and delay risk.",
        lifespan=lifespan,
        root_path=settings.assistant_root_path,  # the gateway strips /assistant before proxying
    )
    register_error_handlers(app)
    add_request_logging(app, logging.getLogger("assistant.http"))
    # shortcut: one lock per session for the process lifetime; fine for a single worker, needs a
    # shared lock (or a DB row lock) once there are several replicas.
    locks: dict[tuple[str, str], asyncio.Lock] = {}

    @app.get("/health")
    def health() -> dict[str, str]:
        try:
            data_api = "ok" if tools.http.get("/health").is_success else "fail"
        except (httpx.HTTPError, AttributeError):
            data_api = "fail"
        agent_ = app.state.agent
        model = agent_.model if agent_ is not None else None
        return {
            "status": "ok" if data_api == "ok" and agent_ is not None else "degraded",
            "data_api": data_api,
            "model": "unconfigured"
            if model is None
            else (model if isinstance(model, str) else model.model),
        }

    @app.post("/chat", response_model=ChatResponse, responses={503: {"model": ErrorResponse}})
    async def chat(body: ChatRequest) -> ChatResponse:
        """One user turn. Pass back `session_id` to continue the conversation."""
        if app.state.agent is None:
            raise APIError(503, "llm_unconfigured", "GEMINI_API_KEY is not set; see .env.example")
        session = await get_or_create_session(app.state.sessions, body.user_id, body.session_id)
        # Two concurrent turns on one session would interleave the per-turn counters and race on
        # appending events; serialise per session (different sessions still run in parallel).
        lock = locks.setdefault((body.user_id, session.id), asyncio.Lock())
        async with lock:
            r = await ask(
                body.question,
                agent=app.state.agent,
                session_service=app.state.sessions,
                user_id=body.user_id,
                session_id=session.id,
            )
        return ChatResponse(
            answer=r.text,
            session_id=session.id,
            tools_called=r.tools_called,
            llm_calls=r.llm_calls,
            usage=r.usage,
            guardrail=r.guardrail,
        )

    @app.get("/sessions", response_model=list[SessionInfo])
    async def sessions(user_id: str = Query(default="local")) -> list[SessionInfo]:
        return [
            SessionInfo(session_id=s.id, last_update_time=s.last_update_time)
            for s in await list_sessions(app.state.sessions, user_id)
        ]

    return app


app = create_app()
