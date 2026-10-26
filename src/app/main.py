import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.db import Database
from app.errors import register_error_handlers
from app.logging_config import configure_logging
from app.middleware import add_request_logging
from app.routers import data_quality, health, ports, predict, routes, shipments
from app.schemas import ErrorResponse
from core.config import settings
from ml.predictor import BOOKING_TIME_FEATURES, load_predictor

log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.perf_counter()
    app.state.db = Database(settings.db_path)
    app.state.predictor = load_predictor(settings.model_path)
    # Warm-up: the first sklearn call pays for lazy imports and thread-pool setup (~4 s under
    # load in the load test); do it before the health check turns green.
    warm = app.state.db.one(f"SELECT {', '.join(BOOKING_TIME_FEATURES)} FROM shipments LIMIT 1")
    if warm:
        app.state.predictor.predict_proba(warm)
    log.info(
        "startup",
        extra={
            "version": __version__,
            "db_path": str(settings.db_path),
            "model_version": app.state.predictor.version,
            "startup_ms": round((time.perf_counter() - t) * 1000),
        },
    )
    yield
    app.state.db.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Supply Chain Intelligence API",
        version=__version__,
        lifespan=lifespan,
        responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    register_error_handlers(app)
    add_request_logging(app, log)
    for r in (health, shipments, ports, routes, predict, data_quality):
        app.include_router(r.router)
    return app


configure_logging(settings.log_level)
app = create_app()
