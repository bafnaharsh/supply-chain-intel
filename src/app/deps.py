"""FastAPI dependencies: app-state singletons created in the lifespan, and shared query params."""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import Query, Request

from app.db import Database
from app.errors import APIError
from ml.predictor import Predictor


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor


def _naive_utc(t: datetime | None) -> datetime | None:
    # Warehouse timestamps are naive UTC; DuckDB would compare an aware one in its local TimeZone.
    return t.astimezone(UTC).replace(tzinfo=None) if t and t.tzinfo else t


def booking_window(
    booked_from: Annotated[datetime | None, Query(description="booking_date >= this")] = None,
    booked_to: Annotated[datetime | None, Query(description="booking_date < this")] = None,
    start_date: Annotated[datetime | None, Query(description="alias of booked_from")] = None,
    end_date: Annotated[datetime | None, Query(description="alias of booked_to")] = None,
) -> tuple[list[str], list[datetime]]:
    """WHERE clauses and params for a booking_date window; 422 if the window is empty.

    `start_date`/`end_date` are accepted as aliases because that is what most clients try first.
    """
    booked_from, booked_to = (
        _naive_utc(booked_from or start_date),
        _naive_utc(booked_to or end_date),
    )
    if booked_from and booked_to and booked_from >= booked_to:
        raise APIError(
            422, "invalid_date_range", "booked_from/start_date must be before booked_to/end_date"
        )
    clauses, params = [], []
    if booked_from:
        clauses.append("booking_date >= ?")
        params.append(booked_from)
    if booked_to:
        clauses.append("booking_date < ?")
        params.append(booked_to)
    return clauses, params
