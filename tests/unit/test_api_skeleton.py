import json
import logging

import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.errors import APIError
from app.logging_config import JsonFormatter
from app.main import create_app
from core.config import settings


@pytest.fixture
def skel(client):
    app = client.app

    @app.get("/_test/items/{item_id}")
    def item(item_id: int):
        raise APIError(404, "item_not_found", f"No item {item_id}", {"item_id": item_id})

    @app.get("/_test/boom")
    def boom():
        raise RuntimeError("kaboom")

    return client


def test_health_returns_status_version_and_checks(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["version"] == __version__
    assert body["checks"] == {"data_store": "ok", "model": "ok", "dq_report": "ok"}
    assert r.headers["x-request-id"]


def test_health_is_503_when_data_store_fails(client):
    client.app.state.db.ping = lambda: False
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "degraded"


def test_health_missing_dq_report_is_informational(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "dq_report_path", tmp_path / "missing.json")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["checks"]["dq_report"] == "fail"


def test_request_id_is_propagated(client):
    r = client.get("/health", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"


@pytest.mark.parametrize("bad", ["x" * 129, "tab\there", ""])
def test_unusable_request_id_is_replaced_not_echoed(client, bad):
    r = client.get("/health", headers={"x-request-id": bad})
    rid = r.headers["x-request-id"]
    assert rid != bad and len(rid) == 32


def test_health_reports_missing_model_as_not_ready(client):
    from ml.predictor import PlaceholderPredictor

    client.app.state.predictor = PlaceholderPredictor()
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["checks"]["model"] == "fail"


def test_unknown_route_is_structured_404(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"
    assert r.json()["error"]["request_id"]


def test_api_error_maps_to_status_and_code(skel):
    r = skel.get("/_test/items/7")
    assert r.status_code == 404
    assert r.json()["error"] == {
        "code": "item_not_found",
        "message": "No item 7",
        "request_id": r.headers["x-request-id"],
        "details": {"item_id": 7},
    }


def test_validation_error_is_422(skel):
    r = skel.get("/_test/items/not-an-int")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"
    assert r.json()["error"]["details"]


def test_unhandled_exception_is_500_without_leaking_detail(skel):
    r = skel.get("/_test/boom", headers={"x-request-id": "boom-1"})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "kaboom" not in r.text
    # The promise "every response carries x-request-id" must hold for unhandled errors too.
    assert r.headers["x-request-id"] == "boom-1" == r.json()["error"]["request_id"]


def test_every_request_is_logged_with_required_fields(client, caplog):
    caplog.set_level(logging.INFO, logger="app")
    client.get("/health")
    rec = next(r for r in caplog.records if r.name == "app" and r.getMessage() == "request")
    assert (rec.method, rec.path, rec.status) == ("GET", "/health", 200)
    assert rec.latency_ms >= 0


def test_json_formatter_emits_extra_fields():
    rec = logging.makeLogRecord({"name": "t", "msg": "hi", "levelname": "INFO", "status": 200})
    out = json.loads(JsonFormatter().format(rec))
    assert out["msg"] == "hi" and out["status"] == 200


def test_app_fails_fast_when_warehouse_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "db_path", tmp_path / "nowhere.duckdb")
    with pytest.raises(FileNotFoundError, match="nowhere"), TestClient(create_app()):
        pass
