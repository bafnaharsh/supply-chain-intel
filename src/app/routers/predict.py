import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.encoders import jsonable_encoder

from app.db import Database
from app.deps import get_db, get_predictor
from app.errors import APIError
from app.schemas import ErrorResponse, PredictDelayRequest, PredictDelayResponse
from ml.predictor import BOOKING_TIME_FEATURES, Predictor, risk_label

router = APIRouter(tags=["ml"])
log = logging.getLogger("app.predict")


@router.post(
    "/predict-delay",
    response_model=PredictDelayResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def predict_delay(
    request: Request,
    body: PredictDelayRequest,
    db: Annotated[Database, Depends(get_db)],
    predictor: Annotated[Predictor, Depends(get_predictor)],
) -> PredictDelayResponse:
    if body.features is not None:
        features = body.features.model_dump()
    else:
        cols = ", ".join(BOOKING_TIME_FEATURES)
        features = db.one(
            f"SELECT {cols} FROM shipments WHERE shipment_id = ? LIMIT 1", [body.shipment_id]
        )
        if features is None:
            raise APIError(404, "shipment_not_found", f"Shipment {body.shipment_id!r} not found")
    p = predictor.predict_proba(features)
    label = risk_label(p)
    # Monitoring hook: every score is logged with its inputs so drift and realised accuracy
    # (once arrivals land) can be computed from the logs alone. See models/model_card.md.
    log.info(
        "prediction",
        extra={
            "request_id": request.state.request_id,
            "shipment_id": body.shipment_id,
            "model_version": predictor.version,
            "delay_risk": round(p, 4),
            "risk_label": label,
            "features": jsonable_encoder(features),
        },
    )
    return PredictDelayResponse(
        shipment_id=body.shipment_id,
        delay_risk=round(p, 4),
        risk_label=label,
        model_version=predictor.version,
        features_used=features,
    )
