from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from app.db import Database
from app.deps import booking_window, get_db
from app.errors import APIError
from app.schemas import ErrorResponse, Shipment, ShipmentPage
from core.config import API_PAGE_SIZE_DEFAULT, API_PAGE_SIZE_MAX

router = APIRouter(prefix="/shipments", tags=["shipments"])


@router.get("", response_model=ShipmentPage)
def list_shipments(
    db: Annotated[Database, Depends(get_db)],
    window: Annotated[tuple[list[str], list], Depends(booking_window)],
    origin: Annotated[str | None, Query(pattern=r"^[A-Z]{5}$")] = None,
    destination: Annotated[str | None, Query(pattern=r"^[A-Z]{5}$")] = None,
    status: Literal["DELIVERED", "DELAYED", "CANCELLED"] | None = None,
    limit: Annotated[int, Query(ge=1, le=API_PAGE_SIZE_MAX)] = API_PAGE_SIZE_DEFAULT,
    offset: Annotated[int, Query(ge=0, le=2**63 - 1)] = 0,  # DuckDB OFFSET is a BIGINT
) -> ShipmentPage:
    clauses, params = [], []
    for col, val in (
        ("origin_port", origin),
        ("destination_port", destination),
        ("status", status),
    ):
        if val is not None:
            clauses.append(f"{col} = ?")
            params.append(val)
    clauses += window[0]
    params += window[1]
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    total = db.one(f"SELECT count(*) AS n FROM shipments {where}", params)["n"]
    items = db.query(
        f"SELECT * FROM shipments {where} ORDER BY booking_date DESC, shipment_id LIMIT ? OFFSET ?",
        [*params, limit, offset],
    )
    return ShipmentPage(items=items, total=total, limit=limit, offset=offset)


@router.get("/{shipment_id}", response_model=Shipment, responses={404: {"model": ErrorResponse}})
def get_shipment(shipment_id: str, db: Annotated[Database, Depends(get_db)]) -> Shipment:
    row = db.one("SELECT * FROM shipments WHERE shipment_id = ?", [shipment_id])
    if row is None:
        raise APIError(404, "shipment_not_found", f"Shipment {shipment_id!r} not found")
    return Shipment(**row)
