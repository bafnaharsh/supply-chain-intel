"""Business endpoints over the fixture CSV (see tests/fixtures/shipments.csv)."""

import pytest

from ml.predictor import PlaceholderPredictor

# --- shipments ---------------------------------------------------------------


def test_get_shipment_returns_full_details_with_derived_fields(client):
    r = client.get("/shipments/SHP-00001")
    assert r.status_code == 200
    s = r.json()
    assert s["route_key"] == "CNSHA → NLRTM"
    assert s["actual_delay_hours"] == 10.0
    assert s["on_time_flag"] is True
    assert s["transit_days_planned"] == 26
    assert s["transit_days_actual"] == 26


def test_get_shipment_is_null_safe_for_cancelled(client):
    s = client.get("/shipments/SHP-00003").json()
    assert s["actual_arrival"] is None
    assert s["actual_delay_hours"] is None
    assert s["on_time_flag"] is None
    assert s["transit_days_actual"] is None


def test_get_shipment_404(client):
    r = client.get("/shipments/SHP-99999")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "shipment_not_found"


def test_list_shipments_default_page(client):
    body = client.get("/shipments").json()
    assert body["total"] == 4 and len(body["items"]) == 4
    assert body["items"][0]["shipment_id"] == "SHP-00004"  # newest booking first


@pytest.mark.parametrize(
    ("params", "expected_ids"),
    [
        ({"origin": "CNSHA"}, {"SHP-00001", "SHP-00002", "SHP-00003"}),
        ({"destination": "USLAX"}, {"SHP-00004"}),
        ({"status": "DELAYED"}, {"SHP-00002"}),
        ({"booked_from": "2024-06-02T00:00:00"}, {"SHP-00002", "SHP-00003", "SHP-00004"}),
        ({"booked_to": "2024-06-02T00:00:00"}, {"SHP-00001"}),
        ({"origin": "CNSHA", "status": "DELIVERED"}, {"SHP-00001"}),
    ],
)
def test_list_shipments_filters(client, params, expected_ids):
    body = client.get("/shipments", params=params).json()
    assert {i["shipment_id"] for i in body["items"]} == expected_ids
    assert body["total"] == len(expected_ids)


def test_list_shipments_pagination(client):
    p1 = client.get("/shipments", params={"limit": 3, "offset": 0}).json()
    p2 = client.get("/shipments", params={"limit": 3, "offset": 3}).json()
    assert len(p1["items"]) == 3 and len(p2["items"]) == 1
    assert p1["total"] == p2["total"] == 4
    assert not {i["shipment_id"] for i in p1["items"]} & {i["shipment_id"] for i in p2["items"]}


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 501},
        {"offset": -1},
        {"offset": 2**64},  # was a 500: DuckDB OFFSET is a BIGINT
        {"origin": "sha"},
        {"booked_from": "yesterday"},
    ],
)
def test_list_shipments_rejects_bad_params(client, params):
    r = client.get("/shipments", params=params)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_list_shipments_accepts_start_end_date_aliases(client):
    canonical = client.get(
        "/shipments", params={"booked_from": "2024-06-02", "booked_to": "2024-07-01"}
    )
    alias = client.get("/shipments", params={"start_date": "2024-06-02", "end_date": "2024-07-01"})
    assert alias.json() == canonical.json() and alias.json()["total"] == 2
    r = client.get("/shipments", params={"start_date": "2024-07-01", "end_date": "2024-06-01"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_date_range"


def test_list_shipments_rejects_inverted_date_range(client):
    r = client.get("/shipments", params={"booked_from": "2024-07-01", "booked_to": "2024-06-01"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_date_range"


@pytest.mark.parametrize("path", ["/shipments", "/routes/stats"])
def test_offset_aware_dates_compare_as_utc(client, path):
    # SHP-00001 is booked 2024-06-01 00:00 UTC. Mixing an aware and a naive bound was a 500, and
    # DuckDB compared aware bounds in the machine's local TimeZone.
    params = {"booked_from": "2024-06-01T05:30:00+05:30", "booked_to": "2024-06-01T01:00:00"}
    if path == "/shipments":
        ids = {i["shipment_id"] for i in client.get(path, params=params).json()["items"]}
        assert ids == {"SHP-00001"}
    else:
        rows = client.get(path, params={**params, "min_completed": 1}).json()
        assert [r["shipment_count"] for r in rows] == [1]
    inverted = {"booked_from": "2024-06-02T00:00:00Z", "booked_to": "2024-06-01T00:00:00"}
    assert client.get(path, params=inverted).json()["error"]["code"] == "invalid_date_range"


# --- routes ------------------------------------------------------------------


def test_route_stats_aggregates(client):
    r = client.get("/routes/CNSHA/NLRTM/stats")
    assert r.status_code == 200
    assert r.json() == {
        "origin_port": "CNSHA",
        "destination_port": "NLRTM",
        "route_key": "CNSHA → NLRTM",
        "shipment_count": 3,
        "completed_count": 2,
        "avg_delay_hours": 29.0,  # (10 + 48) / 2, cancelled excluded
        "on_time_rate": 0.5,
    }


def test_route_stats_404_for_unknown_route(client):
    r = client.get("/routes/CNSHA/USLAX/stats")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "route_not_found"


def test_route_stats_validates_port_codes(client):
    assert client.get("/routes/shanghai/NLRTM/stats").status_code == 422


def test_list_route_stats_ranks_routes(client):
    rows = client.get("/routes/stats", params={"min_completed": 1}).json()
    assert [r["route_key"] for r in rows] == ["CNSHA → NLRTM", "DEHAM → USLAX"]
    assert rows[0]["avg_delay_hours"] == 29.0 and rows[1]["avg_delay_hours"] == -24.0


@pytest.mark.parametrize(
    ("sort", "order", "expected"),
    [
        ("avg_delay_hours", "asc", ["DEHAM → USLAX", "CNSHA → NLRTM"]),
        ("on_time_rate", "desc", ["DEHAM → USLAX", "CNSHA → NLRTM"]),
        ("shipment_count", "desc", ["CNSHA → NLRTM", "DEHAM → USLAX"]),
    ],
)
def test_list_route_stats_sort_and_order(client, sort, order, expected):
    params = {"min_completed": 1, "sort": sort, "order": order}
    assert [r["route_key"] for r in client.get("/routes/stats", params=params).json()] == expected


def test_list_route_stats_rejects_unknown_sort(client):
    assert client.get("/routes/stats", params={"sort": "route_key"}).status_code == 422


def test_list_route_stats_window_and_min_completed(client):
    rows = client.get("/routes/stats", params={"booked_from": "2024-07-01"}).json()
    assert rows == []  # DEHAM → USLAX has only 1 completed shipment, below the default 3
    rows = client.get(
        "/routes/stats", params={"booked_from": "2024-07-01", "min_completed": 1}
    ).json()
    assert [r["route_key"] for r in rows] == ["DEHAM → USLAX"]


def test_list_ports(client):
    ports = client.get("/ports").json()
    assert [p["port_code"] for p in ports] == ["CNSHA", "DEHAM", "NLRTM", "USLAX"]
    assert ports[0]["port_name"] == "Shanghai"


# --- predict-delay -----------------------------------------------------------


def test_predict_by_shipment_id(client):
    r = client.post("/predict-delay", json={"shipment_id": "SHP-00001"})
    assert r.status_code == 200
    body = r.json()
    assert body["shipment_id"] == "SHP-00001"
    assert 0 <= body["delay_risk"] <= 1
    assert body["risk_label"] in {"low", "medium", "high"}
    served = client.app.state.predictor.version
    assert body["model_version"] == served != PlaceholderPredictor.version
    assert set(body["features_used"]) == {
        "origin_port",
        "destination_port",
        "cargo_type",
        "container_count",
        "weight_tons",
        "planned_departure",
        "planned_arrival",
    }


def test_predict_by_features(client):
    r = client.post(
        "/predict-delay",
        json={
            "features": {
                "origin_port": "CNSHA",
                "destination_port": "NLRTM",
                "cargo_type": "Electronics",
                "container_count": 12,
                "weight_tons": 800.0,
                "planned_departure": "2025-01-01T00:00:00",
                "planned_arrival": "2025-01-20T00:00:00",
            }
        },
    )
    assert r.status_code == 200
    assert r.json()["shipment_id"] is None


def test_predict_unknown_shipment_404(client):
    r = client.post("/predict-delay", json={"shipment_id": "SHP-99999"})
    assert r.status_code == 404


_NAN_BODY = (
    '{"features": {"origin_port": "CNSHA", "destination_port": "NLRTM", "cargo_type": "Machinery",'
    ' "container_count": 1, "weight_tons": %s, "planned_departure": "2025-01-01T00:00:00",'
    ' "planned_arrival": "2025-01-20T00:00:00"}}'
)


@pytest.mark.parametrize("weight", ["NaN", "Infinity"])
def test_predict_rejects_non_finite_weight(client, weight):
    # Python's json accepts these literals; they used to pass gt=0 and crash the response (500).
    r = client.post(
        "/predict-delay",
        content=_NAN_BODY % weight,
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"shipment_id": "bad-id"},
        {"shipment_id": "SHP-00001", "features": {"origin_port": "CNSHA"}},
        {"features": {"origin_port": "CNSHA", "container_count": 0}},
        {"shipment_id": "SHP-00001", "actual_arrival": "2025-01-01T00:00:00"},  # extra field
        {
            "features": {  # post-booking field smuggled in, and arrival before departure
                "origin_port": "CNSHA",
                "destination_port": "NLRTM",
                "cargo_type": "Electronics",
                "container_count": 1,
                "weight_tons": 1.0,
                "planned_departure": "2025-01-20T00:00:00",
                "planned_arrival": "2025-01-01T00:00:00",
            }
        },
    ],
)
def test_predict_rejects_bad_input(client, body):
    assert client.post("/predict-delay", json=body).status_code == 422


def test_list_shipments_rejects_unknown_status(client):
    r = client.get("/shipments", params={"status": "delayed"})  # must be upper-case canonical
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


# --- data quality ------------------------------------------------------------


def test_dq_report_serves_file(client):
    r = client.get("/data-quality/report")
    assert r.status_code == 200
    assert r.json() == {"status": "test", "checks": []}


def test_dq_report_503_when_missing(client, monkeypatch, tmp_path):
    from core.config import settings

    monkeypatch.setattr(settings, "dq_report_path", tmp_path / "missing.json")
    r = client.get("/data-quality/report")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "dq_report_unavailable"
