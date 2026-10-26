"""Build the warehouse: raw -> staging -> curated -> serving views, in one DuckDB file.

The build goes to a temp file and is renamed over `db_path` only after every invariant holds,
so a failed or interrupted run never leaves a half-built warehouse behind.
"""

import logging
import os
import time
from pathlib import Path

import duckdb

from core.config import CONTAINER_RANGE, ON_TIME_THRESHOLD_HOURS
from dq.rules import (
    CANONICAL_STATUS,
    CARGO_TYPES,
    EVENT_TYPES,
    EXPECTED_COLUMNS,
    INVALID_VESSELS,
    PORT_FIXES,
    QUARANTINE_RULES,
    STATUS_MAP,
    TIMESTAMP_FORMAT,
    load_raw,
)

logger = logging.getLogger(__name__)


def _case(expr: str, mapping: dict[str, str]) -> str:
    whens = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in mapping.items())
    return f"CASE {expr} {whens} END"


def _in(values) -> str:
    return ", ".join(f"'{v}'" for v in values)


def _cols(table: str) -> str:
    return ", ".join(EXPECTED_COLUMNS[table])


_TS = f"'{TIMESTAMP_FORMAT}'"
_STATUS = _case("upper(trim(status))", {**{s: s for s in CANONICAL_STATUS}, **STATUS_MAP})
_CARGO = _case("lower(trim(cargo_type))", {c.lower(): c for c in CARGO_TYPES})
_COUNTRY = _case("port_code", {p: f["country"] for p, f in PORT_FIXES.items() if "country" in f})
_QUARANTINE_CASE = (
    "CASE "
    + " ".join(f"WHEN ({pred}) THEN '{reason}'" for reason, pred in QUARANTINE_RULES.items())
    + " END"
)
_MIN_CONTAINERS, _MAX_CONTAINERS = CONTAINER_RANGE

_STAGING_SHIPMENTS = f"""
CREATE TABLE staging.shipments AS
WITH dedup AS (
    SELECT * FROM raw.shipments
    QUALIFY row_number() OVER (PARTITION BY {_cols("shipments")} ORDER BY _source_row) = 1
), typed AS (
    SELECT _source_row,
           trim(shipment_id) AS shipment_id,
           trim(customer_id) AS customer_id,
           trim(origin_port) AS origin_port,
           trim(destination_port) AS destination_port,
           trim(vessel_id) AS vessel_id,
           try_strptime(booking_date, {_TS}) AS booking_date,
           try_strptime(planned_departure, {_TS}) AS planned_departure,
           try_strptime(actual_departure, {_TS}) AS actual_departure,
           try_strptime(planned_arrival, {_TS}) AS planned_arrival,
           try_strptime(actual_arrival, {_TS}) AS actual_arrival,
           try_cast(container_count AS INTEGER) AS container_count,
           {_CARGO} AS cargo_type,
           trim(cargo_type) AS cargo_raw,
           try_cast(weight_tons AS DOUBLE) AS weight_tons,
           {_STATUS} AS status,
           trim(status) AS status_raw
    FROM dedup
)
SELECT shipment_id, customer_id, origin_port, destination_port, vessel_id, booking_date,
       planned_departure,
       CASE WHEN actual_departure >= booking_date OR booking_date IS NULL
            THEN actual_departure END AS actual_departure,
       planned_arrival, actual_arrival,
       CASE WHEN container_count BETWEEN {_MIN_CONTAINERS} AND {_MAX_CONTAINERS}
            THEN container_count END AS container_count,
       cargo_type,
       CASE WHEN weight_tons > 0 THEN weight_tons END AS weight_tons,
       status,
       list_filter([
           CASE WHEN status IS NOT NULL AND status_raw <> status THEN 'status_non_canonical' END,
           CASE WHEN status IS NULL THEN 'status_missing' END,
           CASE WHEN cargo_type IS NOT NULL AND cargo_raw <> cargo_type
                THEN 'cargo_type_non_canonical' END,
           CASE WHEN cargo_type IS NULL THEN 'cargo_type_missing' END,
           CASE WHEN weight_tons IS NULL THEN 'weight_missing' END,
           CASE WHEN weight_tons < 0 THEN 'weight_negative' END,
           CASE WHEN weight_tons = 0 THEN 'weight_zero' END,
           CASE WHEN container_count = 0 THEN 'container_count_zero' END,
           CASE WHEN container_count < 0 OR container_count > {_MAX_CONTAINERS}
                THEN 'container_count_outlier' END,
           CASE WHEN actual_departure < booking_date THEN 'departure_before_booking' END,
           CASE WHEN status = 'DELIVERED' AND actual_arrival IS NULL
                THEN 'completed_without_actuals' END,
           CASE WHEN planned_departure < booking_date THEN 'planned_departure_before_booking' END,
           CASE WHEN actual_arrival < actual_departure THEN 'actual_arrival_before_departure' END,
           CASE WHEN NOT regexp_matches(shipment_id, '^SHP-[0-9]{{5}}$') THEN 'bad_id_format' END
       ], lambda x: x IS NOT NULL) AS dq_flags,
       _source_row
FROM typed
"""

_STAGING_PORTS = f"""
CREATE TABLE staging.ports AS
SELECT trim(port_code) AS port_code,
       trim(port_name) AS port_name,
       coalesce({_COUNTRY}, trim(country)) AS country,
       trim(region) AS region,
       trim(timezone) AS timezone,
       try_cast(avg_congestion_score AS DOUBLE) AS avg_congestion_score,
       list_filter([
           CASE WHEN country IS NULL THEN 'missing_country' END,
           CASE WHEN try_cast(avg_congestion_score AS DOUBLE) NOT BETWEEN 0 AND 1
                THEN 'congestion_out_of_range' END
       ], lambda x: x IS NOT NULL) AS dq_flags,
       _source_row
FROM raw.ports
-- one row per port_code (first in file order): the DQ check `duplicate_port_code` says drop
QUALIFY row_number() OVER (PARTITION BY trim(port_code) ORDER BY _source_row) = 1
"""

_STAGING_EVENTS = f"""
CREATE TABLE staging.port_events AS
WITH dedup AS (
    SELECT md5(concat_ws('|', {_cols("port_events")})) AS event_key, * FROM raw.port_events
    QUALIFY row_number() OVER (PARTITION BY {_cols("port_events")} ORDER BY _source_row) = 1
), typed AS (
    SELECT event_key, _source_row, trim(event_id) AS event_id,
           try_strptime(event_timestamp, {_TS}) AS event_timestamp,
           trim(port_code) AS port_code,
           trim(vessel_id) AS vessel_id_raw,
           trim(event_type) AS event_type_raw,
           try_cast(delay_minutes AS INTEGER) AS delay_raw,
           trim(notes) AS notes
    FROM dedup
), cleaned AS (
    SELECT *, CASE WHEN delay_raw >= 0 THEN delay_raw END AS delay_minutes,
           CASE WHEN event_type_raw IN ({_in(EVENT_TYPES)}) THEN event_type_raw END AS event_type
    FROM typed
)
SELECT event_key, event_id, event_timestamp, port_code,
       CASE WHEN vessel_id_raw NOT IN ({_in(INVALID_VESSELS)}) THEN vessel_id_raw END AS vessel_id,
       event_type, delay_minutes, notes,
       list_filter([
           CASE WHEN vessel_id_raw IN ({_in(INVALID_VESSELS)}) THEN 'placeholder_vessel' END,
           CASE WHEN event_type_raw IS NULL THEN 'event_type_missing' END,
           CASE WHEN event_type_raw IS NOT NULL AND event_type IS NULL
                THEN 'event_type_unknown' END,
           CASE WHEN notes IS NULL THEN 'notes_missing' END,
           CASE WHEN notes = 'Normal operations' AND delay_minutes > 0
                THEN 'notes_contradict_delay' END,
           CASE WHEN event_type = 'DELAYED' AND delay_minutes = 0 THEN 'delayed_zero_minutes' END,
           CASE WHEN delay_raw < 0 THEN 'delay_negative' END,  -- nulled above; keep it traceable
           CASE WHEN port_code NOT IN (SELECT trim(port_code) FROM raw.ports
                                       WHERE port_code IS NOT NULL) THEN 'unknown_port' END,
           CASE WHEN count(*) OVER (PARTITION BY event_id) > 1 THEN 'event_id_reused' END
       ], lambda x: x IS NOT NULL) AS dq_flags,
       _source_row
FROM cleaned
"""

# Runs with search_path = raw: the predicates say `FROM ports`, meaning the raw ports table.
_QUARANTINE = f"""
CREATE TABLE curated.quarantine_shipments AS
SELECT s.*, r.quarantine_reason
FROM staging.shipments s
JOIN (SELECT _source_row, {_QUARANTINE_CASE} AS quarantine_reason FROM raw.shipments) r
  ON r._source_row = s._source_row
WHERE r.quarantine_reason IS NOT NULL
"""

_CURATED_SHIPMENTS = f"""
CREATE TABLE curated.shipments AS
WITH ok AS (
    SELECT *, count(*) OVER (PARTITION BY shipment_id) AS n_rows
    FROM staging.shipments
    WHERE _source_row NOT IN (SELECT _source_row FROM curated.quarantine_shipments)
), picked AS (
    SELECT * FROM ok
    QUALIFY row_number() OVER (
        PARTITION BY shipment_id ORDER BY cargo_type IS NULL, weight_tons IS NULL, _source_row
    ) = 1
)
SELECT * EXCLUDE (n_rows, dq_flags, _source_row),
       origin_port || ' → ' || destination_port AS route_key,
       epoch(actual_arrival - planned_arrival) / 3600.0 AS actual_delay_hours,
       -- elapsed whole days, not calendar midnights crossed (date_diff would say 11 for 10.4 days)
       floor(epoch(planned_arrival - planned_departure) / 86400)::INTEGER AS transit_days_planned,
       floor(epoch(actual_arrival - actual_departure) / 86400)::INTEGER AS transit_days_actual,
       epoch(actual_arrival - planned_arrival) / 3600.0
           <= {ON_TIME_THRESHOLD_HOURS} AS on_time_flag,
       CASE WHEN n_rows > 1 THEN list_append(dq_flags, 'conflicting_duplicate')
            ELSE dq_flags END AS dq_flags,
       _source_row
FROM picked
"""

_COUNTS = """
SELECT
    (SELECT count(*) FROM raw.shipments),
    (SELECT count(*) FROM staging.shipments),
    (SELECT count(*) FROM curated.shipments),
    (SELECT count(*) FROM curated.quarantine_shipments),
    (SELECT count(*) - count(DISTINCT shipment_id) FROM staging.shipments
      WHERE _source_row NOT IN (SELECT _source_row FROM curated.quarantine_shipments)),
    (SELECT count(*) FROM curated.ports),
    (SELECT count(*) FROM curated.port_events)
"""

_ORPHAN_ROUTES = """
SELECT count(*) FROM curated.shipments
WHERE origin_port NOT IN (SELECT port_code FROM curated.ports)
   OR destination_port NOT IN (SELECT port_code FROM curated.ports)
"""


def _layers(con: duckdb.DuckDBPyConnection, input_dir: Path) -> dict[str, int]:
    load_raw(con, input_dir, schema="raw")
    con.execute("CREATE SCHEMA staging")
    con.execute("CREATE SCHEMA curated")
    for sql in (_STAGING_SHIPMENTS, _STAGING_PORTS, _STAGING_EVENTS):
        con.execute(sql)

    con.execute("CREATE TABLE curated.ports AS SELECT * FROM staging.ports")
    con.execute("ALTER TABLE curated.ports ADD PRIMARY KEY (port_code)")
    con.execute("SET schema = 'raw'")
    con.execute(_QUARANTINE)
    con.execute("RESET schema")
    con.execute(_CURATED_SHIPMENTS)
    con.execute("ALTER TABLE curated.shipments ADD PRIMARY KEY (shipment_id)")
    con.execute("CREATE TABLE curated.port_events AS SELECT * FROM staging.port_events")
    con.execute("ALTER TABLE curated.port_events ADD PRIMARY KEY (event_key)")

    for table in ("shipments", "ports", "port_events"):
        con.execute(f"CREATE OR REPLACE VIEW main.{table} AS SELECT * FROM curated.{table}")

    raw, staged, curated, quarantined, conflicting, ports, events = con.execute(_COUNTS).fetchone()
    return {
        "raw_shipments": raw,
        "staging_shipments": staged,
        "curated_shipments": curated,
        "quarantine_shipments": quarantined,
        "dropped_exact_duplicates": raw - staged,
        "dropped_conflicting_duplicates": conflicting,
        "curated_ports": ports,
        "curated_port_events": events,
    }


def _check_invariants(con: duckdb.DuckDBPyConnection, counts: dict[str, int]) -> None:
    """Raise RuntimeError if the built warehouse breaks a contract the API relies on."""
    if con.execute(
        "SELECT count(*) - count(DISTINCT shipment_id) FROM curated.shipments"
    ).fetchone()[0]:
        raise RuntimeError("invariant failed: shipment_id is not unique in curated.shipments")
    if con.execute(
        "SELECT count(*) - count(DISTINCT event_key) FROM curated.port_events"
    ).fetchone()[0]:
        raise RuntimeError("invariant failed: event_key is not unique in curated.port_events")
    orphans = con.execute(_ORPHAN_ROUTES).fetchone()[0]
    if orphans:
        raise RuntimeError(f"invariant failed: {orphans} shipments reference unknown ports")
    accounted = (
        counts["curated_shipments"]
        + counts["quarantine_shipments"]
        + counts["dropped_exact_duplicates"]
        + counts["dropped_conflicting_duplicates"]
    )
    if accounted != counts["raw_shipments"]:
        raise RuntimeError(
            f"invariant failed: raw shipments {counts['raw_shipments']} != "
            f"curated + quarantine + dropped duplicates ({accounted})"
        )


def build(input_dir: Path, db_path: Path) -> dict[str, int]:
    """Rebuild the warehouse at `db_path` from the CSVs in `input_dir`; return row counts.

    Raises ValueError (from load_raw) on a missing file/column and RuntimeError on a broken
    invariant. In both cases `db_path` is left exactly as it was.
    """
    start = time.perf_counter()
    db_path = Path(db_path)
    tmp = db_path.with_suffix(".duckdb.tmp")
    for stale in (tmp, Path(f"{tmp}.wal")):
        stale.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        counts = _layers(con, Path(input_dir))
        _check_invariants(con, counts)
        con.execute("CHECKPOINT")
    except BaseException as e:
        con.close()
        tmp.unlink(missing_ok=True)
        Path(f"{tmp}.wal").unlink(missing_ok=True)
        if isinstance(e, duckdb.ConstraintException):
            raise RuntimeError(f"invariant failed: {e}") from e
        raise
    con.close()
    os.replace(tmp, db_path)
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    logger.info(
        "pipeline_built", extra={**counts, "db_path": str(db_path), "elapsed_ms": elapsed_ms}
    )
    return counts
