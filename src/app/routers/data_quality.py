import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.errors import APIError
from app.schemas import ErrorResponse
from core.config import settings

router = APIRouter(prefix="/data-quality", tags=["data-quality"])


@router.get("/report", responses={503: {"model": ErrorResponse}})
def dq_report() -> JSONResponse:
    """Serve the report written by the standalone DQ step (`python -m dq check`)."""
    if not settings.dq_report_path.exists():
        raise APIError(
            503,
            "dq_report_unavailable",
            "Data-quality report has not been generated; run `python -m dq check` first.",
            {"path": str(settings.dq_report_path)},
        )
    return JSONResponse(json.loads(settings.dq_report_path.read_text(encoding="utf-8")))
