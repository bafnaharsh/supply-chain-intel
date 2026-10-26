"""Pipeline behaviour on the clean and dirty fixtures (each build takes a few hundred ms)."""

import shutil
from pathlib import Path

import duckdb
import pytest

from pipeline import build as build_module
from pipeline.build import build

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
HEADER = (FIXTURES / "shipments.csv").read_text(encoding="utf-8").splitlines()[0]


def _db(tmp_path_factory, input_dir: Path, name: str):
    path = tmp_path_factory.mktemp(name) / "w.duckdb"
    counts = build(input_dir, path)
    return duckdb.connect(str(path), read_only=True), counts


@pytest.fixture(scope="module")
def clean_db(tmp_path_factory):
    con, _ = _db(tmp_path_factory, FIXTURES, "clean")
    yield con
    con.close()


@pytest.fixture(scope="module")
def dirty_built(tmp_path_factory):
    con, counts = _db(tmp_path_factory, FIXTURES / "dirty", "dirty")
    yield con, counts
    con.close()


@pytest.fixture(scope="module")
def dirty_db(dirty_built):
    return dirty_built[0]


def one(con, sql, *params):
    return con.execute(sql, list(params)).fetchone()


def shipment(con, sid, table="shipments"):
    cur = con.execute(f"SELECT * FROM {table} WHERE shipment_id = ?", [sid])
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    assert len(rows) == 1, f"{sid}: {len(rows)} rows in {table}"
    return dict(zip(cols, rows[0], strict=True))


def test_derived_fields_match_api_semantics(clean_db):
    s = shipment(clean_db, "SHP-00001")
    assert s["actual_delay_hours"] == 10.0
    assert s["on_time_flag"] is True
    assert s["transit_days_planned"] == 26
    assert s["transit_days_actual"] == 26
    assert s["route_key"] == "CNSHA → NLRTM"


def test_on_time_boundary(tmp_path):
    for name in ("ports.csv", "port_events.csv"):
        shutil.copy(FIXTURES / name, tmp_path / name)

    def row(sid, actual_arrival):
        return (
            f"{sid},CUST-1,CNSHA,NLRTM,VSL-0001,2024-06-01 00:00:00,2024-06-05 00:00:00,"
            f"2024-06-05 00:00:00,2024-07-01 00:00:00,{actual_arrival},1,Apparel,1,DELIVERED"
        )

    rows = [
        row("SHP-00001", "2024-07-02 00:00:00"),
        row("SHP-00002", "2024-07-02 01:00:00"),
        row("SHP-00003", "2024-06-30 12:00:00"),
    ]
    (tmp_path / "shipments.csv").write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    db = tmp_path / "w.duckdb"
    build(tmp_path, db)
    with duckdb.connect(str(db), read_only=True) as con:
        got = {
            r[0]: r[1:]
            for r in con.execute(
                "SELECT shipment_id, actual_delay_hours, on_time_flag FROM shipments"
            ).fetchall()
        }
    assert got["SHP-00001"] == (24.0, True)
    assert got["SHP-00002"] == (25.0, False)
    assert got["SHP-00003"] == (-12.0, True)


def test_null_safe_for_cancelled(clean_db):
    s = shipment(clean_db, "SHP-00003")
    assert s["actual_delay_hours"] is None
    assert s["on_time_flag"] is None
    assert s["transit_days_actual"] is None


def test_placeholder_ports_quarantined(dirty_db):
    expected = {
        "SHP-00006": "unknown_origin_port",
        "SHP-00007": "unknown_destination_port",
        "SHP-00008": "origin_equals_destination",
        "SHP-00018": "planned_arrival_before_departure",
        "SHP-00024": "unparseable_value",
        "SHP-00025": "missing_required",
    }
    got = dict(
        dirty_db.execute(
            "SELECT shipment_id, quarantine_reason FROM curated.quarantine_shipments"
        ).fetchall()
    )
    assert got == expected
    assert not dirty_db.execute(
        "SELECT 1 FROM shipments WHERE shipment_id IN (SELECT shipment_id FROM "
        "curated.quarantine_shipments)"
    ).fetchall()


def test_duplicates_resolved(dirty_db):
    shipment(dirty_db, "SHP-00001")
    s = shipment(dirty_db, "SHP-00005")
    assert s["cargo_type"] == "Chemicals"
    assert "conflicting_duplicate" in s["dq_flags"]
    assert "conflicting_duplicate" not in shipment(dirty_db, "SHP-00001")["dq_flags"]


def test_normalisation(dirty_db):
    s = shipment(dirty_db, "SHP-00009")
    assert s["status"] == "DELIVERED"
    assert {"status_non_canonical", "completed_without_actuals"} <= set(s["dq_flags"])
    for sid in ("SHP-00010", "SHP-00011"):
        s = shipment(dirty_db, sid)
        assert s["status"] is None and "status_missing" in s["dq_flags"]
    assert shipment(dirty_db, "SHP-00012")["cargo_type"] == "Furniture"
    assert "cargo_type_non_canonical" in shipment(dirty_db, "SHP-00012")["dq_flags"]
    assert shipment(dirty_db, "SHP-00013")["cargo_type"] == "Chemicals"
    s = shipment(dirty_db, "SHP-00014")
    assert s["weight_tons"] is None and "weight_negative" in s["dq_flags"]
    s = shipment(dirty_db, "SHP-00015")
    assert s["container_count"] is None and "container_count_zero" in s["dq_flags"]
    s = shipment(dirty_db, "SHP-00016")
    assert s["container_count"] is None and "container_count_outlier" in s["dq_flags"]
    s = shipment(dirty_db, "SHP-00027")
    assert s["status"] == "DELIVERED" and "status_non_canonical" in s["dq_flags"]
    assert "bad_id_format" in shipment(dirty_db, "SHIP-1")["dq_flags"]
    assert "weight_missing" in shipment(dirty_db, "SHP-00026")["dq_flags"]
    assert "planned_departure_before_booking" in shipment(dirty_db, "SHP-00019")["dq_flags"]
    assert "actual_arrival_before_departure" in shipment(dirty_db, "SHP-00020")["dq_flags"]


def test_departure_before_booking_nulled(dirty_db):
    s = shipment(dirty_db, "SHP-00017")
    assert s["actual_departure"] is None
    assert s["transit_days_actual"] is None
    assert s["actual_delay_hours"] is not None
    assert "departure_before_booking" in s["dq_flags"]


def test_ports_fixed(dirty_db):
    country, flags = one(dirty_db, "SELECT country, dq_flags FROM ports WHERE port_code = 'BEANR'")
    assert country == "Belgium"
    assert {"missing_country", "congestion_out_of_range"} <= set(flags)


def test_port_events(dirty_db):
    assert one(dirty_db, "SELECT count(*), count(DISTINCT event_key) FROM port_events") == (12, 12)

    def event(ts):
        return one(
            dirty_db,
            "SELECT vessel_id, event_type, delay_minutes, dq_flags FROM port_events "
            "WHERE event_timestamp = ?",
            ts,
        )

    vessel, _, _, flags = event("2024-06-05 06:00:00")  # VSL-9999
    assert vessel is None and "placeholder_vessel" in flags
    _, _, delay, flags = event("2024-06-05 11:00:00")  # delay -5: nulled AND flagged
    assert delay is None and "delay_negative" in flags
    _, etype, _, flags = event("2024-06-05 05:00:00")  # FOO
    assert etype is None and "event_type_unknown" in flags
    _, etype, _, flags = event("2024-06-05 04:00:00")  # blank
    assert etype is None and "event_type_missing" in flags
    _, _, delay, _ = event("2024-06-05 11:00:00")  # -5
    assert delay is None
    assert "event_id_reused" in event("2024-06-05 01:00:00")[3]
    assert "notes_missing" in event("2024-06-05 07:00:00")[3]
    assert "notes_contradict_delay" in event("2024-06-05 08:00:00")[3]
    assert "delayed_zero_minutes" in event("2024-06-05 09:00:00")[3]


def test_row_reconciliation(dirty_built):
    counts = dirty_built[1]
    assert counts["raw_shipments"] == 28
    assert counts["dropped_exact_duplicates"] == 1
    assert counts["dropped_conflicting_duplicates"] == 1
    assert counts["raw_shipments"] == (
        counts["curated_shipments"] + counts["quarantine_shipments"] + 1 + 1
    )
    assert counts["curated_port_events"] == 12 and counts["curated_ports"] == 5


def test_idempotent(tmp_path):
    db = tmp_path / "w.duckdb"
    first = build(FIXTURES / "dirty", db)

    def dump():
        with duckdb.connect(str(db), read_only=True) as con:
            return {
                t: con.execute(f"SELECT * FROM curated.{t} ORDER BY {k}").fetchall()
                for t, k in [
                    ("shipments", "shipment_id"),
                    ("quarantine_shipments", "_source_row"),
                    ("ports", "port_code"),
                    ("port_events", "event_key"),
                ]
            }

    before = dump()
    assert build(FIXTURES / "dirty", db) == first
    assert dump() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["w.duckdb"]


def test_invariant_failure_leaves_no_file(tmp_path, monkeypatch):
    def boom(con, counts):
        raise RuntimeError("invariant failed: forced")

    db = tmp_path / "w.duckdb"
    monkeypatch.setattr(build_module, "_check_invariants", boom)
    with pytest.raises(RuntimeError, match="forced"):
        build(FIXTURES, db)
    assert list(tmp_path.iterdir()) == []

    monkeypatch.undo()
    build(FIXTURES, db)
    good = db.read_bytes()
    monkeypatch.setattr(build_module, "_check_invariants", boom)
    with pytest.raises(RuntimeError):
        build(FIXTURES / "dirty", db)
    assert db.read_bytes() == good
    assert sorted(p.name for p in tmp_path.iterdir()) == ["w.duckdb"]
