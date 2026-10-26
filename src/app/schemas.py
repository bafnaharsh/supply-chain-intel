"""Pydantic request/response models shared across routers."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PortCode = Field(pattern=r"^[A-Z]{5}$", description="5-letter UN/LOCODE, e.g. CNSHA")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    checks: dict[str, Literal["ok", "fail"]]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class Shipment(BaseModel):
    shipment_id: str
    customer_id: str
    origin_port: str
    destination_port: str
    route_key: str
    vessel_id: str
    booking_date: datetime
    planned_departure: datetime
    actual_departure: datetime | None
    planned_arrival: datetime
    actual_arrival: datetime | None
    container_count: int | None
    cargo_type: str | None
    weight_tons: float | None
    status: str | None
    actual_delay_hours: float | None
    on_time_flag: bool | None
    transit_days_planned: int
    transit_days_actual: int | None
    dq_flags: list[str] = Field(default_factory=list, description="Fixes/flags applied in curation")


class ShipmentPage(BaseModel):
    items: list[Shipment]
    total: int
    limit: int
    offset: int


class Port(BaseModel):
    port_code: str
    port_name: str
    country: str | None
    region: str
    timezone: str
    avg_congestion_score: float
    dq_flags: list[str] = Field(default_factory=list)


class RouteStats(BaseModel):
    origin_port: str
    destination_port: str
    route_key: str
    shipment_count: int
    completed_count: int = Field(description="Shipments with an actual arrival")
    avg_delay_hours: float | None
    on_time_rate: float | None = Field(description="Share of completed shipments ≤ 24h late")


class ShipmentFeatures(BaseModel):
    """Only fields known at booking time. Extra fields are rejected, so a client cannot slip
    `actual_arrival` or `status` in and have it silently ignored."""

    model_config = ConfigDict(extra="forbid")

    origin_port: str = PortCode
    destination_port: str = PortCode
    cargo_type: str
    container_count: int = Field(ge=1, le=10_000)  # sanity bound; the model nulls > 100 itself
    weight_tons: float = Field(gt=0, le=1e6, allow_inf_nan=False)
    planned_departure: datetime
    planned_arrival: datetime

    @model_validator(mode="after")
    def _schedule_is_possible(self):
        if self.planned_arrival <= self.planned_departure:
            raise ValueError("planned_arrival must be after planned_departure")
        return self


class PredictDelayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shipment_id: str | None = Field(default=None, pattern=r"^SHP-\d{5}$")
    features: ShipmentFeatures | None = None

    @model_validator(mode="after")
    def _one_source(self):
        if (self.shipment_id is None) == (self.features is None):
            raise ValueError("provide exactly one of shipment_id or features")
        return self


class PredictDelayResponse(BaseModel):
    shipment_id: str | None
    delay_risk: float = Field(ge=0, le=1, description="P(arrival > 24h late)")
    risk_label: Literal["low", "medium", "high"]
    model_version: str
    features_used: dict[str, Any]
