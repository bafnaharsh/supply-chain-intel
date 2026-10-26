"""Booking-time features: one cleaning function shared by training and serving, and the pipeline.

`clean_and_derive` runs inside the fitted pipeline, so the API's raw feature dict (values may be
None, cargo may be free text) goes through exactly the cleaning the model was trained on.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from core.config import CONTAINER_RANGE, LABEL_THRESHOLD_HOURS
from ml.predictor import BOOKING_TIME_FEATURES

FEATURES = BOOKING_TIME_FEATURES
CATEGORICAL = ["origin_port", "destination_port", "cargo_type"]
NUMERIC = ["container_count", "transit_days_planned", "dep_month", "dep_dow"]
LOG_NUMERIC = ["weight_tons"]


def clean_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    """Raw FEATURES columns -> CATEGORICAL + NUMERIC + LOG_NUMERIC, with invalid values as NaN."""
    out = pd.DataFrame(index=df.index)
    for col in ("origin_port", "destination_port"):
        out[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    cargo = df["cargo_type"].astype("string").str.strip().str.title()
    out["cargo_type"] = cargo.replace("", pd.NA).astype(object).where(cargo.notna(), np.nan)

    low, high = CONTAINER_RANGE
    containers = pd.to_numeric(df["container_count"], errors="coerce").astype(float)
    out["container_count"] = containers.where(containers.between(low, high))
    weight = pd.to_numeric(df["weight_tons"], errors="coerce").astype(float)
    out["weight_tons"] = weight.where(weight > 0)

    dep = pd.to_datetime(df["planned_departure"], errors="coerce", utc=True)
    arr = pd.to_datetime(df["planned_arrival"], errors="coerce", utc=True)
    out["transit_days_planned"] = (arr - dep).dt.total_seconds() / 86400
    out["dep_month"] = dep.dt.month.astype(float)
    out["dep_dow"] = dep.dt.day_of_week.astype(float)
    return out


def make_label(df: pd.DataFrame) -> pd.Series:
    """True when the shipment arrived more than LABEL_THRESHOLD_HOURS after plan."""
    late = pd.to_datetime(df["actual_arrival"]) - pd.to_datetime(df["planned_arrival"])
    return (late > pd.Timedelta(hours=LABEL_THRESHOLD_HOURS)).rename("late")


def build_pipeline(model) -> Pipeline:
    """clean_and_derive -> one-hot categoricals / imputed, scaled numerics -> model."""
    encode = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False
                ),
                CATEGORICAL,
            ),
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                ),
                NUMERIC,
            ),
            (
                "log",
                make_pipeline(
                    FunctionTransformer(np.log1p),
                    SimpleImputer(strategy="median", add_indicator=True),
                    StandardScaler(),
                ),
                LOG_NUMERIC,
            ),
        ]
    )
    return Pipeline(
        [("clean", FunctionTransformer(clean_and_derive)), ("encode", encode), ("model", model)]
    )
