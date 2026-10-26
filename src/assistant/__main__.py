"""CLI: `python -m assistant "question"` (one-shot) or `python -m assistant` (REPL)."""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from typing import Literal

import httpx

from core.config import settings


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="assistant", description=__doc__)
    p.add_argument("question", nargs="?", help="ask once and exit; omit for an interactive REPL")
    p.add_argument("--session", help="resume this session id")
    p.add_argument("--new", action="store_true", help="force a new session")
    p.add_argument("--list-sessions", action="store_true", help="list your sessions and exit")
    p.add_argument("--user", default="local", help="user id (default: local)")
    p.add_argument("--model", default=settings.assistant_model)
    p.add_argument("--api-url", default=settings.assistant_api_url)
    p.add_argument(
        "--offline",
        action="store_true",
        help="no LLM: answer a fixed set of question shapes from the tools (needs no key)",
    )
    p.add_argument("--no-persist", action="store_true", help="keep the session in memory only")
    return p.parse_args()


def _fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def run_mode(args: argparse.Namespace, has_key: bool) -> Literal["online", "offline"]:
    return "offline" if args.offline or not has_key else "online"


def _connect(api_url: str):
    """Check /health and point the tools at the API; exits with a hint when it is down."""
    from assistant import tools

    client = httpx.Client(base_url=api_url, timeout=15)
    try:
        client.get("/health").raise_for_status()
    except httpx.HTTPError as e:
        _fail(f"API at {api_url} is not reachable ({e}); start it with `docker compose up`.")
    tools.configure(client)


def _footer(r, session_id: str | None = None, prefix: str = "") -> str:
    u = r.usage
    return (
        f"{prefix}[tools: {', '.join(r.tools_called) or 'none'} | tokens {u['prompt_tokens']} in / "
        f"{u['output_tokens']} out | ~${u['est_usd']:.5f}"
        + (f" | session {session_id}" if session_id else "")
        + (f" | guardrail: {r.guardrail}" if r.guardrail else "")
        + "]"
    )


async def _repl(turn) -> None:
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if q.lower() in ("exit", "quit"):
            return
        if q:
            await turn(q)


async def _main_offline(args: argparse.Namespace) -> None:
    """No ADK, no agent, no sessions: the rule-based router over the same tools."""
    from assistant.offline import answer_offline

    _connect(args.api_url)

    async def turn(question: str) -> None:
        r = answer_offline(question)
        print(r.text)
        print(_footer(r, prefix="[offline mode: no LLM] "), file=sys.stderr)

    if args.question:
        await turn(args.question)
        return
    print("offline mode (fixed question shapes, no LLM); 'exit' to quit")
    await _repl(turn)


async def _main(args: argparse.Namespace) -> None:
    # Heavy ADK imports only after the cheap checks have passed.
    from assistant.agent import build_agent, configure_logging
    from assistant.runner import ask
    from assistant.session import get_or_create_session, list_sessions, make_session_service

    svc = make_session_service(persistent=not args.no_persist)
    if args.list_sessions:
        for s in await list_sessions(svc, args.user):
            print(f"{s.id}\tlast used {datetime.fromtimestamp(s.last_update_time):%Y-%m-%d %H:%M}")
        return

    _connect(args.api_url)
    configure_logging()
    agent = build_agent(args.model)
    session = await get_or_create_session(svc, args.user, None if args.new else args.session)

    async def turn(question: str) -> None:
        r = await ask(
            question, agent=agent, session_service=svc, user_id=args.user, session_id=session.id
        )
        print(r.text)
        print(_footer(r, session.id), file=sys.stderr)

    if args.question:
        await turn(args.question)
        return
    print(f"session {session.id} (resume with --session {session.id}); 'exit' to quit")
    await _repl(turn)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = _parse()
    if args.list_sessions:
        asyncio.run(_main(args))
    elif run_mode(args, bool(settings.gemini_api_key)) == "offline":
        if not args.offline:
            print(
                "no GEMINI_API_KEY in .env: running in offline mode "
                "(fixed question shapes, no LLM); see .env.example",
                file=sys.stderr,
            )
        asyncio.run(_main_offline(args))
    else:
        os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key
        asyncio.run(_main(args))


if __name__ == "__main__":
    main()
