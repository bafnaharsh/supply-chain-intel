"""Every error leaves the API as {"error": {code, message, request_id, details}}."""

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas import ErrorBody, ErrorResponse

log = logging.getLogger(__name__)


class APIError(Exception):
    """Expected failure raised from route code, e.g. APIError(404, "shipment_not_found", ...)."""

    def __init__(self, status_code: int, code: str, message: str, details: Any = None):
        super().__init__(message)
        self.status_code, self.code, self.message, self.details = (
            status_code,
            code,
            message,
            details,
        )


def _error(request: Request, status: int, code: str, message: str, details: Any = None):
    body = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            request_id=getattr(request.state, "request_id", None),
            details=jsonable_encoder(details),
        )
    )
    # mode=json: a rejected NaN/Infinity echoed in `details` becomes null instead of a 500.
    # The header is set here too: an unhandled exception is answered by Starlette's outermost
    # middleware, which bypasses the request-logging middleware that normally adds it.
    headers = {"x-request-id": body.error.request_id} if body.error.request_id else None
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"), headers=headers)


def _snake(status: int) -> str:
    return HTTPStatus(status).phrase.lower().replace(" ", "_").replace("-", "_")


async def _api_error(request: Request, exc: APIError):
    return _error(request, exc.status_code, exc.code, exc.message, exc.details)


async def _http_error(request: Request, exc: StarletteHTTPException):
    return _error(request, exc.status_code, _snake(exc.status_code), str(exc.detail))


async def _validation_error(request: Request, exc: RequestValidationError):
    return _error(request, 422, "validation_error", "Request validation failed", exc.errors())


async def _unhandled(request: Request, exc: Exception):
    log.exception("unhandled_exception", extra={"path": request.url.path})
    return _error(request, 500, "internal_error", "Internal server error")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(APIError, _api_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled)
