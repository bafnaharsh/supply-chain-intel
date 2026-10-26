"""Delay-risk predictor loaded once at API startup.

`load_predictor` loads the bundle written by `python -m ml.train` (a fitted sklearn pipeline that
does its own cleaning), or falls back to `PlaceholderPredictor` (the ~40% late base rate) when
no model has been trained yet, so the API can still start.
"""

import logging
from pathlib import Path
from typing import Any, Protocol

import joblib
import pandas as pd
import sklearn

from core.config import RISK_LABEL_THRESHOLDS

log = logging.getLogger(__name__)

BOOKING_TIME_FEATURES = (
    "origin_port",
    "destination_port",
    "cargo_type",
    "container_count",
    "weight_tons",
    "planned_departure",
    "planned_arrival",
)


class Predictor(Protocol):
    version: str

    def predict_proba(self, features: dict[str, Any]) -> float:
        """Probability that the shipment arrives more than 24h late."""


class PlaceholderPredictor:
    version = "0.0.0-placeholder"

    def predict_proba(self, features: dict[str, Any]) -> float:
        return 0.40


class SklearnPredictor:
    """Wraps the trained bundle; the pipeline cleans the raw feature dict itself."""

    def __init__(self, bundle: dict[str, Any]):
        self.pipeline = bundle["pipeline"]
        self.version: str = bundle["version"]

    def predict_proba(self, features: dict[str, Any]) -> float:
        row = pd.DataFrame([{k: features.get(k) for k in BOOKING_TIME_FEATURES}])
        return float(self.pipeline.predict_proba(row)[0, 1])


def load_predictor(path: Path) -> Predictor:
    """Missing artefact -> placeholder with a warning; a present but broken one raises."""
    if not path.exists():
        log.warning("model_artefact_missing", extra={"path": str(path), "using": "placeholder"})
        return PlaceholderPredictor()
    bundle = joblib.load(path)
    if bundle["sklearn_version"] != sklearn.__version__:
        log.warning(
            "model_sklearn_version_mismatch",
            extra={"trained_with": bundle["sklearn_version"], "running": sklearn.__version__},
        )
    return SklearnPredictor(bundle)


def risk_label(p: float) -> str:
    low, high = RISK_LABEL_THRESHOLDS
    return "high" if p >= high else "medium" if p >= low else "low"
