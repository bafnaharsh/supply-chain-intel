"""Session storage: SQLite (resumable across runs) or in-memory."""

from google.adk.sessions import BaseSessionService, DatabaseSessionService, InMemorySessionService
from google.adk.sessions.session import Session

from core.config import ASSISTANT_APP_NAME, settings


def make_session_service(persistent: bool = True) -> BaseSessionService:
    if not persistent:
        return InMemorySessionService()
    settings.assistant_session_db.parent.mkdir(parents=True, exist_ok=True)
    return DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{settings.assistant_session_db}")


async def get_or_create_session(
    svc: BaseSessionService, user_id: str, session_id: str | None = None
) -> Session:
    """Resume `session_id` if it exists, else start a new session (with that id if given)."""
    if session_id:
        found = await svc.get_session(
            app_name=ASSISTANT_APP_NAME, user_id=user_id, session_id=session_id
        )
        if found:
            return found
    return await svc.create_session(
        app_name=ASSISTANT_APP_NAME, user_id=user_id, session_id=session_id
    )


async def list_sessions(svc: BaseSessionService, user_id: str) -> list[Session]:
    resp = await svc.list_sessions(app_name=ASSISTANT_APP_NAME, user_id=user_id)
    return sorted(resp.sessions, key=lambda s: s.last_update_time, reverse=True)
