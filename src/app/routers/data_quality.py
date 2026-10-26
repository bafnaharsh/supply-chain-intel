import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import Response

from app.errors import APIError
from app.schemas import ErrorResponse
from core.config import settings

router = APIRouter(prefix="/data-quality", tags=["data-quality"])

_cache: tuple[Path, float, bytes] | None = None  # (path, mtime, body): re-read only on change


def _report_bytes(path: Path) -> bytes:
    global _cache
    mtime = path.stat().st_mtime
    if _cache is None or _cache[:2] != (path, mtime):
        body = path.read_bytes()
        json.loads(body)  # a half-written or corrupt file must fail here, not reach the client
        _cache = (path, mtime, body)
    return _cache[2]


@router.get("/report", responses={503: {"model": ErrorResponse}})
def dq_report() -> Response:
    """Serve the report written by the standalone DQ step (`python -m dq check`)."""
    path = settings.dq_report_path
    if not path.exists():
        raise APIError(
            503,
            "dq_report_unavailable",
            "Data-quality report has not been generated; run `python -m dq check` first.",
            {"path": str(path)},
        )
    return Response(_report_bytes(path), media_type="application/json")
