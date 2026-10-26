from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Path, Query

from app.db import Database
from app.deps import booking_window, get_db
from app.errors import APIError
from app.schemas import ErrorResponse, RouteStats
from core.config import ON_TIME_THRESHOLD_HOURS

router = APIRouter(prefix="/routes", tags=["routes"])

_AGG = f"""
       count(*)                                   AS shipment_count,
       count(actual_arrival)                      AS completed_count,
       avg(actual_delay_hours)                    AS avg_delay_hours,
       avg(CASE WHEN actual_delay_hours <= {ON_TIME_THRESHOLD_HOURS} THEN 1.0 ELSE 0.0 END)
           FILTER (actual_arrival IS NOT NULL)    AS on_time_rate
"""


@router.get("/stats", response_model=list[RouteStats])
def list_route_stats(
    db: Annotated[Database, Depends(get_db)],
    window: Annotated[tuple[list[str], list], Depends(booking_window)],
    min_completed: Annotated[int, Query(ge=0, description="routes with fewer are dropped")] = 3,
    sort: Literal["avg_delay_hours", "on_time_rate", "shipment_count"] = "avg_delay_hours",
    order: Literal["desc", "asc"] = "desc",
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> list[RouteStats]:
    """Per-route aggregates, e.g. the routes with the highest average delay in a window."""
    clauses, params = window
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return db.query(
        f"""
        SELECT origin_port, destination_port, route_key, {_AGG}
        FROM shipments {where}
        GROUP BY 1, 2, 3
        HAVING count(actual_arrival) >= ?
        ORDER BY {sort} {order} NULLS LAST, route_key
        LIMIT ?
        """,
        [*params, min_completed, limit],
    )


@router.get(
    "/{origin}/{destination}/stats",
    response_model=RouteStats,
    responses={404: {"model": ErrorResponse}},
)
def route_stats(
    origin: Annotated[str, Path(pattern=r"^[A-Z]{5}$")],
    destination: Annotated[str, Path(pattern=r"^[A-Z]{5}$")],
    db: Annotated[Database, Depends(get_db)],
) -> RouteStats:
    row = db.one(
        f"SELECT {_AGG} FROM shipments WHERE origin_port = ? AND destination_port = ?",
        [origin, destination],
    )
    if row["shipment_count"] == 0:
        raise APIError(404, "route_not_found", f"No shipments for route {origin} → {destination}")
    return RouteStats(
        origin_port=origin,
        destination_port=destination,
        route_key=f"{origin} → {destination}",
        **row,
    )
