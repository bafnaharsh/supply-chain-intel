from fastapi import APIRouter, Request, Response

from app import __version__
from app.schemas import HealthResponse
from core.config import settings
from ml.predictor import PlaceholderPredictor

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def health(request: Request, response: Response) -> HealthResponse:
    predictor = request.app.state.predictor
    checks = {
        "data_store": "ok" if request.app.state.db.ping() else "fail",
        # The placeholder keeps the API up, but a deploy without the artefact is not ready.
        "model": "fail" if isinstance(predictor, PlaceholderPredictor) else "ok",
        "dq_report": "ok" if settings.dq_report_path.exists() else "fail",
    }
    # dq_report is informational: the service can serve shipments without it.
    ready = checks["data_store"] == "ok" and checks["model"] == "ok"
    response.status_code = 200 if ready else 503
    return HealthResponse(status="ok" if ready else "degraded", version=__version__, checks=checks)
