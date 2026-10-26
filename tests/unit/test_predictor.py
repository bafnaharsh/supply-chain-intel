from core.config import RISK_LABEL_THRESHOLDS, settings
from ml.predictor import PlaceholderPredictor, SklearnPredictor, load_predictor, risk_label

README_FEATURES = {
    "origin_port": "CNSHA",
    "destination_port": "NLRTM",
    "cargo_type": "Electronics",
    "container_count": 12,
    "weight_tons": 800.0,
    "planned_departure": "2025-01-01T00:00:00",
    "planned_arrival": "2025-01-20T00:00:00",
}


def test_real_model_predicts_a_probability():
    p = load_predictor(settings.model_path)
    assert isinstance(p, SklearnPredictor)
    assert 0 < p.predict_proba(README_FEATURES) < 1
    assert 0 < p.predict_proba({**README_FEATURES, "weight_tons": None, "cargo_type": "x "}) < 1


def test_missing_model_falls_back_to_placeholder(tmp_path):
    assert isinstance(load_predictor(tmp_path / "nope.joblib"), PlaceholderPredictor)


def test_risk_label_uses_config_thresholds():
    low, high = RISK_LABEL_THRESHOLDS
    assert risk_label(low - 1e-9) == "low"
    assert risk_label(low) == "medium"
    assert risk_label(high) == "high"
