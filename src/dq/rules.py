"""Shared facts about the raw data: file layout, canonical values, and the quarantine rules.

Both the standalone DQ checks and the ingestion pipeline import from here so they can never
disagree about what "bad" means (numeric thresholds such as CONTAINER_RANGE live in
core.config). This module must not import from `pipeline`.
"""

from pathlib import Path

import duckdb

FILES = {
    "shipments": "shipments.csv",
    "ports": "ports.csv",
    "port_events": "port_events.csv",
}

EXPECTED_COLUMNS = {
    "shipments": [
        "shipment_id",
        "customer_id",
        "origin_port",
        "destination_port",
        "vessel_id",
        "booking_date",
        "planned_departure",
        "actual_departure",
        "planned_arrival",
        "actual_arrival",
        "container_count",
        "cargo_type",
        "weight_tons",
        "status",
    ],
    "ports": ["port_code", "port_name", "country", "region", "timezone", "avg_congestion_score"],
    "port_events": [
        "event_id",
        "event_timestamp",
        "port_code",
        "vessel_id",
        "event_type",
        "delay_minutes",
        "notes",
    ],
}

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

INVALID_VESSELS = ("VSL-0000", "VSL-9999")

CANONICAL_STATUS = ("DELIVERED", "DELAYED", "CANCELLED")
# Non-canonical spellings that mean the same thing. Anything else (N/A, blank) becomes NULL.
STATUS_MAP = {"COMPLETED": "DELIVERED", "COMPLETE": "DELIVERED"}

CARGO_TYPES = (
    "Apparel",
    "Automotive Parts",
    "Chemicals",
    "Consumer Goods",
    "Electronics",
    "Food & Beverage",
    "Furniture",
    "Machinery",
    "Pharmaceuticals",
    "Raw Materials",
)
EVENT_TYPES = (
    "ARRIVAL",
    "BERTHED",
    "LOADING_COMPLETE",
    "CUSTOMS_CLEARED",
    "INSPECTION",
    "DEPARTURE",
    "DELAYED",
)

PORT_FIXES = {"BEANR": {"country": "Belgium"}}

# Rows that cannot be served or trained on. Predicates run on the raw (all-varchar) table.
# Order matters: the first matching reason is the one recorded.
QUARANTINE_RULES = {
    "missing_required": (
        "shipment_id IS NULL OR origin_port IS NULL OR destination_port IS NULL "
        "OR booking_date IS NULL OR planned_departure IS NULL OR planned_arrival IS NULL"
    ),
    "unparseable_value": (
        f"try_strptime(booking_date, '{TIMESTAMP_FORMAT}') IS NULL "
        f"OR try_strptime(planned_departure, '{TIMESTAMP_FORMAT}') IS NULL "
        f"OR try_strptime(planned_arrival, '{TIMESTAMP_FORMAT}') IS NULL "
        "OR (actual_departure IS NOT NULL "
        f"AND try_strptime(actual_departure, '{TIMESTAMP_FORMAT}') IS NULL) "
        "OR (actual_arrival IS NOT NULL "
        f"AND try_strptime(actual_arrival, '{TIMESTAMP_FORMAT}') IS NULL) "
        "OR try_cast(container_count AS INTEGER) IS NULL "
        "OR (weight_tons IS NOT NULL AND try_cast(weight_tons AS DOUBLE) IS NULL)"
    ),
    # The IS NOT NULL guard matters: NOT IN against a set containing NULL is never true.
    "unknown_origin_port": (
        "origin_port NOT IN (SELECT port_code FROM ports WHERE port_code IS NOT NULL)"
    ),
    "unknown_destination_port": (
        "destination_port NOT IN (SELECT port_code FROM ports WHERE port_code IS NOT NULL)"
    ),
    "origin_equals_destination": "origin_port = destination_port",
    "planned_arrival_before_departure": (
        f"try_strptime(planned_arrival, '{TIMESTAMP_FORMAT}') "
        f"< try_strptime(planned_departure, '{TIMESTAMP_FORMAT}')"
    ),
}

SOURCE_ROW = "_source_row"  # 1-based CSV line order, added to every raw table


def resolve_input_dir(input_dir: Path) -> Path:
    """The brief's example is `--input ./data/`; the CSVs live one level down in data/raw/."""
    input_dir = Path(input_dir)
    if not (input_dir / FILES["shipments"]).exists() and (input_dir / "raw").is_dir():
        return input_dir / "raw"
    return input_dir


def load_raw(con: duckdb.DuckDBPyConnection, input_dir: Path, schema: str = "main") -> None:
    """Load the three CSVs as all-varchar tables so nothing fails to parse at load time.

    Blank fields become NULL (DuckDB's default `nullstr`). Each table gets a `_source_row`
    column with the original line order. Raises ValueError on a missing file or column so
    callers fail before any check or layer runs.
    """
    input_dir = resolve_input_dir(input_dir)
    if schema != "main":
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    for table, filename in FILES.items():
        path = input_dir / filename
        if not path.exists():
            raise ValueError(f"missing input file: {path}")
        con.execute(
            f"CREATE OR REPLACE TABLE {schema}.{table} AS "
            f"SELECT row_number() OVER () AS {SOURCE_ROW}, * "
            f"FROM read_csv({_lit(path)}, header=true, all_varchar=true)"
        )
        cols = [r[0] for r in con.execute(f"DESCRIBE {schema}.{table}").fetchall()]
        missing = [c for c in EXPECTED_COLUMNS[table] if c not in cols]
        if missing:
            raise ValueError(f"{filename}: missing columns {missing}")


def _lit(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"
