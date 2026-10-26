"""Real HTTP against a running stack (`docker compose up --wait`). Target via BASE_URL."""

import os

import httpx
import pytest

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def http():
    with httpx.Client(base_url=BASE_URL, timeout=10) as c:
        yield c


def test_health(http):
    r = http.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["version"]
    assert body["checks"] == {"data_store": "ok", "model": "ok", "dq_report": "ok"}


def test_shipment_lookup(http):
    r = http.get("/shipments/SHP-00421")
    assert r.status_code == 200
    body = r.json()
    assert body["shipment_id"] == "SHP-00421"
    assert body["route_key"] == f"{body['origin_port']} → {body['destination_port']}"
    for derived in ("actual_delay_hours", "on_time_flag", "transit_days_planned", "dq_flags"):
        assert derived in body


def test_shipment_unknown_id_is_404(http):
    r = http.get("/shipments/SHP-99999")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "shipment_not_found"
    assert r.headers["x-request-id"]


def test_list_and_route_stats_agree(http):
    page = http.get("/shipments", params={"origin": "CNSHA", "destination": "NLRTM"}).json()
    stats = http.get("/routes/CNSHA/NLRTM/stats").json()
    assert stats["shipment_count"] == page["total"] > 0
    assert 0 <= stats["on_time_rate"] <= 1 and stats["completed_count"] <= stats["shipment_count"]


def test_dq_report(http):
    r = http.get("/data-quality/report")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total_checks"] == len(body["checks"]) >= 12
    assert {c["file"] for c in body["checks"]} == {"ports.csv", "shipments.csv", "port_events.csv"}
    required = {"name", "rows_affected", "pct_affected", "severity", "action", "recommendation"}
    assert all(required <= c.keys() for c in body["checks"])


def test_predict_delay(http):
    r = http.post("/predict-delay", json={"shipment_id": "SHP-00421"})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["delay_risk"] <= 1
    assert body["risk_label"] in {"low", "medium", "high"}
    assert body["model_version"] and body["model_version"] != "0.0.0-placeholder"


def test_unknown_route_returns_structured_404(http):
    r = http.get("/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"
