import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.features import FEATURES, build_pipeline, clean_and_derive, make_label
from ml.train import RouteBaseline, time_split


def _rows(n: int = 1, **overrides) -> pd.DataFrame:
    row = {
        "origin_port": "CNSHA",
        "destination_port": "NLRTM",
        "cargo_type": "Electronics",
        "container_count": 12,
        "weight_tons": 800.0,
        "planned_departure": "2025-01-01T00:00:00",
        "planned_arrival": "2025-01-20T00:00:00",
    }
    return pd.DataFrame([{**row, **overrides}] * n)


def test_cargo_type_normalised():
    df = pd.concat([_rows(cargo_type=c) for c in ("machinery ", " food & beverage", "", None)])
    out = clean_and_derive(df.reset_index(drop=True))["cargo_type"].tolist()
    assert out[:2] == ["Machinery", "Food & Beverage"]
    assert pd.isna(out[2]) and pd.isna(out[3])


def test_invalid_numerics_become_nan():
    df = pd.concat(
        [
            _rows(weight_tons=-5.0, container_count=0),
            _rows(weight_tons=0, container_count=996),
            _rows(weight_tons=None, container_count=None),
            _rows(weight_tons=10.0, container_count=100),
        ]
    ).reset_index(drop=True)
    out = clean_and_derive(df)
    assert out["weight_tons"].isna().tolist() == [True, True, True, False]
    assert out["container_count"].isna().tolist() == [True, True, True, False]


def test_transit_days_derived():
    out = clean_and_derive(_rows())
    assert out["transit_days_planned"].iloc[0] == 19
    assert out["dep_month"].iloc[0] == 1 and out["dep_dow"].iloc[0] == 2  # 2025-01-01 = Wed


def test_label_boundary():
    planned = pd.Timestamp("2025-01-20")
    df = pd.DataFrame(
        {
            "planned_arrival": [planned] * 2,
            "actual_arrival": [planned + pd.Timedelta(hours=24), planned + pd.Timedelta("24h1min")],
        }
    )
    assert make_label(df).tolist() == [False, True]


def test_time_split_disjoint():
    dates = pd.date_range("2024-06-01", "2025-10-01", freq="7D")
    df = pd.DataFrame({"shipment_id": [f"S{i}" for i in range(len(dates))], "booking_date": dates})
    splits = time_split(df)
    assert splits["train"]["booking_date"].max() < splits["validation"]["booking_date"].min()
    assert splits["validation"]["booking_date"].max() < splits["test"]["booking_date"].min()
    assert sum(len(s) for s in splits.values()) == len(df)


def test_route_baseline_uses_train_only():
    train = pd.concat([_rows(10), _rows(10, origin_port="DEHAM")]).reset_index(drop=True)
    y = np.array([1] * 10 + [0] * 10)
    model = RouteBaseline(smoothing=0.0).fit(train, y)
    test = pd.concat(
        [
            _rows(),
            _rows(origin_port="DEHAM"),
            _rows(destination_port="USLAX"),
            _rows(origin_port="X"),
        ]
    ).reset_index(drop=True)
    p = model.predict_proba(test)[:, 1]
    # seen route -> its train rate; unseen route -> origin rate; unseen origin -> global rate
    assert p.tolist() == [1.0, 0.0, 1.0, 0.5]


def test_no_leakage():
    leaky = {
        "status",
        "actual_departure",
        "actual_arrival",
        "actual_delay_hours",
        "transit_days_actual",
        "on_time_flag",
        "customer_id",
        "vessel_id",
        "dq_flags",
    }
    assert not set(FEATURES) & leaky
    rng = np.random.default_rng(0)
    df = pd.concat([_rows(20, cargo_type="Machinery"), _rows(20)]).reset_index(drop=True)
    df["weight_tons"] = rng.uniform(10, 1000, len(df))
    pipe = build_pipeline(LogisticRegression()).fit(df[list(FEATURES)], np.arange(len(df)) % 2)
    p = pipe.predict_proba(df[list(FEATURES)])[:, 1]
    assert ((p > 0) & (p < 1)).all()


def test_pipeline_handles_api_style_single_row():
    df = pd.concat([_rows(20, cargo_type="Machinery"), _rows(20)]).reset_index(drop=True)
    pipe = build_pipeline(LogisticRegression()).fit(df, np.arange(len(df)) % 2)
    row = {**_rows().iloc[0].to_dict(), "weight_tons": None, "cargo_type": "machinery "}
    row["origin_port"] = "XXTST"
    p = pipe.predict_proba(pd.DataFrame([row]))[0, 1]
    assert 0 < p < 1
