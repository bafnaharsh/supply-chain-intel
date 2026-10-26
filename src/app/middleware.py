"""Request logging shared by the data API and the assistant API."""

import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request


def add_request_logging(app: FastAPI, logger: logging.Logger) -> None:
    """Request id on every response and one JSON log line per request (method, path, status,
    latency). Shared by the data API and the assistant API."""

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        # A client id is honoured only if it looks like one; anything else is replaced.
        client_id = request.headers.get("x-request-id", "")
        request.state.request_id = (
            client_id if 0 < len(client_id) <= 128 and client_id.isprintable() else uuid4().hex
        )
        start = time.perf_counter()
        status = 500  # stays 500 if the handler raises
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            logger.info(
                "request",
                extra={
                    "request_id": request.state.request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                },
            )
        response.headers["x-request-id"] = request.state.request_id
        return response
