"""Standalone data-quality checks over the raw CSVs, and the report built from them.

Each check is one SELECT that returns the key of every offending row. Checks that mirror a
quarantine rule take their predicate from `rules.QUARANTINE_RULES`, so the report and the
pipeline cannot disagree about what is bad. Unparseable values never match a comparison
(`try_strptime` / `try_cast` give NULL); they are reported by the `unparseable_value` checks.
"""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import duckdb

from core.config import CONTAINER_RANGE, ON_TIME_THRESHOLD_HOURS
from dq.rules import (
    CANONICAL_STATUS,
    CARGO_TYPES,
    EVENT_TYPES,
    EXPECTED_COLUMNS,
    FILES,
    INVALID_VESSELS,
    QUARANTINE_RULES,
    SOURCE_ROW,
    STATUS_MAP,
    TIMESTAMP_FORMAT,
    load_raw,
    resolve_input_dir,
)

SAMPLE_SIZE = 5


@dataclass(frozen=True)
class Check:
    id: str
    file: str
    name: str
    detection: str
    sql: str
    severity: Literal["critical", "warning", "info"]
    action: Literal["drop", "fix", "flag", "quarantine", "none"]
    recommendation: str


def _in(values) -> str:
    return "(" + ", ".join("'" + v.replace("'", "''") + "'" for v in values) + ")"


def _ts(col: str) -> str:
    return f"try_strptime({col}, '{TIMESTAMP_FORMAT}')"


def _int(col: str) -> str:
    return f"try_cast({col} AS INTEGER)"


def _surplus(table: str, key: str) -> str:
    """Rows beyond the first of each group of fully identical rows."""
    cols = ", ".join(EXPECTED_COLUMNS[table])
    return (
        f"SELECT {key} FROM (SELECT {key}, row_number() OVER "
        f"(PARTITION BY {cols} ORDER BY {SOURCE_ROW}) AS rn FROM {table}) WHERE rn > 1"
    )


def _conflicting(table: str, key: str) -> str:
    """All rows whose key appears on more than one distinct row."""
    cols = ", ".join(EXPECTED_COLUMNS[table])
    return (
        f"SELECT {key} FROM {table} WHERE {key} IN (SELECT {key} FROM "
        f"(SELECT DISTINCT {cols} FROM {table}) GROUP BY 1 HAVING count(*) > 1)"
    )


def _rule(rule: str) -> str:
    return f"SELECT shipment_id FROM shipments WHERE {QUARANTINE_RULES[rule]}"


_DELAY_H = f"(epoch({_ts('actual_arrival')}) - epoch({_ts('planned_arrival')})) / 3600"
_COMPLETED = _in(sorted(set(STATUS_MAP) | set(STATUS_MAP.values())))
_STATUS_VARIANT = _in(sorted(set(STATUS_MAP) | set(CANONICAL_STATUS)))

_CHECKS = [
    # id, name, detection, sql, severity, action, recommendation
    (
        "ports.missing_country",
        "Port with no country",
        "country IS NULL",
        "SELECT port_code FROM ports WHERE country IS NULL",
        "warning",
        "fix",
        "Fix: fill from the PORT_FIXES lookup (BEANR is Antwerp, Belgium). One known port, "
        "the value is certain, and country is a reporting attribute.",
    ),
    (
        "ports.duplicate_port_code",
        "Duplicate port code",
        "port_code appears on more than one row",
        "SELECT port_code FROM ports WHERE port_code IN "
        "(SELECT port_code FROM ports GROUP BY 1 HAVING count(*) > 1)",
        "critical",
        "drop",
        "Drop: ports is the join target for every other file, so the key must be unique.",
    ),
    (
        "ports.congestion_out_of_range",
        "Congestion score outside 0-1",
        "avg_congestion_score does not parse as a number, or is not between 0 and 1",
        "SELECT port_code FROM ports WHERE try_cast(avg_congestion_score AS DOUBLE) IS NULL "
        "OR try_cast(avg_congestion_score AS DOUBLE) NOT BETWEEN 0 AND 1",
        "warning",
        "flag",
        "Flag: the score is documented as 0-1, but the correct value is unknowable, so keep "
        "the row and exclude the score from features until it is confirmed.",
    ),
    (
        "shipments.exact_duplicate",
        "Exact duplicate row",
        "surplus rows: count minus count distinct over all columns",
        _surplus("shipments", "shipment_id"),
        "warning",
        "drop",
        "Drop: byte-identical copies add no information and would double-count the booking.",
    ),
    (
        "shipments.conflicting_duplicate_id",
        "Same shipment_id, different values",
        "shipment_id appears on more than one distinct row",
        _conflicting("shipments", "shipment_id"),
        "critical",
        "fix",
        "Fix: keep one row per id (most complete, then first in file order). The copies differ "
        "only in cargo_type or weight_tons, and no source says which is right.",
    ),
    (
        "shipments.missing_required",
        "Missing required field",
        "shipment_id, origin/destination port or a planned/booking date is NULL",
        _rule("missing_required"),
        "critical",
        "quarantine",
        "Quarantine: a booking without a key, route or schedule cannot be joined, served or "
        "labelled, and nothing can be inferred for the blank.",
    ),
    (
        "shipments.unparseable_value",
        "Unparseable timestamp or number",
        "a required timestamp, container_count or weight does not parse in its format",
        _rule("unparseable_value"),
        "critical",
        "quarantine",
        "Quarantine: guessing a date format (2024/06/23 could be day-first) risks silently "
        "wrong dates; keep the raw row for manual repair.",
    ),
    (
        "shipments.bad_id_format",
        "shipment_id not SHP-NNNNN",
        "shipment_id does not match ^SHP-\\d{5}$",
        "SELECT shipment_id FROM shipments WHERE NOT regexp_matches(shipment_id, '^SHP-\\d{5}$')",
        "warning",
        "flag",
        "Flag: the id may still be unique and usable, but it is outside the documented key "
        "format, so surface it rather than rewrite a business key.",
    ),
    (
        "shipments.unknown_origin_port",
        "Origin port not in ports",
        "origin_port NOT IN (SELECT port_code FROM ports)",
        _rule("unknown_origin_port"),
        "critical",
        "quarantine",
        "Quarantine: placeholder codes (XXTST, ZZZZZ, UNKNW) have no real port behind them, so "
        "route features and the port join are impossible.",
    ),
    (
        "shipments.unknown_destination_port",
        "Destination port not in ports",
        "destination_port NOT IN (SELECT port_code FROM ports)",
        _rule("unknown_destination_port"),
        "critical",
        "quarantine",
        "Quarantine: same reasoning as the origin port; the destination cannot be recovered.",
    ),
    (
        "shipments.origin_equals_destination",
        "Origin equals destination",
        "origin_port = destination_port",
        _rule("origin_equals_destination"),
        "critical",
        "quarantine",
        "Quarantine: a container shipment cannot start and end at the same port, so one of the "
        "two fields is wrong and neither can be trusted.",
    ),
    (
        "shipments.status_non_canonical",
        "Status spelling variant",
        "status not canonical but its upper-case form is a known status or alias",
        "SELECT shipment_id FROM shipments WHERE status IS NOT NULL "
        f"AND status NOT IN {_in(CANONICAL_STATUS)} AND upper(status) IN {_STATUS_VARIANT}",
        "warning",
        "fix",
        "Fix: map COMPLETED/Complete/delivered to DELIVERED via STATUS_MAP; same meaning, "
        "different spelling.",
    ),
    (
        "shipments.status_missing",
        "Status missing or N/A",
        "status IS NULL or status = 'N/A'",
        "SELECT shipment_id FROM shipments WHERE status IS NULL OR status = 'N/A'",
        "warning",
        "fix",
        "Fix: set to NULL (unknown) and keep the row; do not guess a status. Excluded from "
        "status-based analysis, still usable where the label comes from the dates.",
    ),
    (
        "shipments.cargo_type_non_canonical",
        "Cargo type case/whitespace variant",
        "cargo_type not canonical but trim/lower matches a canonical value",
        "SELECT shipment_id FROM shipments WHERE cargo_type IS NOT NULL "
        f"AND cargo_type NOT IN {_in(CARGO_TYPES)} "
        f"AND lower(trim(cargo_type)) IN {_in(c.lower() for c in CARGO_TYPES)}",
        "warning",
        "fix",
        "Fix: trim and map to the canonical casing; the match is unambiguous.",
    ),
    (
        "shipments.cargo_type_missing",
        "Cargo type missing",
        "cargo_type IS NULL",
        "SELECT shipment_id FROM shipments WHERE cargo_type IS NULL",
        "warning",
        "flag",
        "Flag: keep the row with cargo_type NULL rather than impute; cargo type cannot be "
        "derived from other columns.",
    ),
    (
        "shipments.weight_missing",
        "Weight missing",
        "weight_tons IS NULL",
        "SELECT shipment_id FROM shipments WHERE weight_tons IS NULL",
        "warning",
        "flag",
        "Flag: keep NULL and let the model handle missing values; imputing a weight would "
        "invent data.",
    ),
    (
        "shipments.weight_negative",
        "Negative weight",
        "weight_tons cast to DOUBLE < 0",
        "SELECT shipment_id FROM shipments WHERE try_cast(weight_tons AS DOUBLE) < 0",
        "warning",
        "fix",
        "Fix: set to NULL; a negative weight is impossible and the sign flip is not provably "
        "the only error.",
    ),
    (
        "shipments.weight_zero",
        "Zero weight",
        "weight_tons cast to DOUBLE = 0",
        "SELECT shipment_id FROM shipments WHERE try_cast(weight_tons AS DOUBLE) = 0",
        "warning",
        "fix",
        "Fix: set to NULL and flag; a loaded container weighs something, and zero is a "
        "placeholder, not a measurement.",
    ),
    (
        "shipments.container_count_zero",
        "Container count is zero",
        f"container_count cast to INTEGER = 0 (valid range starts at {CONTAINER_RANGE[0]})",
        f"SELECT shipment_id FROM shipments WHERE {_int('container_count')} = 0",
        "warning",
        "fix",
        "Fix: set to NULL and flag. A booking has at least one container, so the count is wrong, "
        "but the true value is unknown; the row stays, the count stays out of features.",
    ),
    (
        "shipments.container_count_outlier",
        "Container count implausible (negative or very high)",
        f"container_count cast to INTEGER < 0 or > {CONTAINER_RANGE[1]} (zero has its own check)",
        f"SELECT shipment_id FROM shipments WHERE {_int('container_count')} < 0 "
        f"OR {_int('container_count')} > {CONTAINER_RANGE[1]}",
        "warning",
        "fix",
        "Fix: set to NULL and flag. The bulk is 1-60 and the outliers (510-996) look like entry "
        "errors with no way to correct them; a cap would be a guess.",
    ),
    (
        "shipments.planned_arrival_before_departure",
        "Planned arrival before planned departure",
        "planned_arrival < planned_departure",
        _rule("planned_arrival_before_departure"),
        "critical",
        "quarantine",
        "Quarantine: an impossible schedule makes planned transit time negative, and the "
        "delay label is measured against planned arrival.",
    ),
    (
        "shipments.planned_departure_before_booking",
        "Planned departure before booking",
        "planned_departure < booking_date",
        f"SELECT shipment_id FROM shipments WHERE {_ts('planned_departure')} "
        f"< {_ts('booking_date')}",
        "warning",
        "flag",
        "Flag: a departure planned before the booking exists is odd but does not break the "
        "label; keep the row and watch the count.",
    ),
    (
        "shipments.departure_before_booking",
        "Actual departure before booking",
        "actual_departure < booking_date",
        f"SELECT shipment_id FROM shipments WHERE {_ts('actual_departure')} "
        f"< {_ts('booking_date')}",
        "warning",
        "fix",
        "Fix: set actual_departure to NULL and flag. The timeline is impossible, but the arrival "
        "is plausible, so the row and its delay label are kept.",
    ),
    (
        "shipments.actual_arrival_before_departure",
        "Actual arrival before actual departure",
        "actual_arrival < actual_departure",
        f"SELECT shipment_id FROM shipments WHERE {_ts('actual_arrival')} "
        f"< {_ts('actual_departure')}",
        "warning",
        "flag",
        "Flag: an impossible transit; the label uses actual_arrival, so treat the row's label "
        "as suspect.",
    ),
    (
        "shipments.completed_without_actuals",
        "Completed status but no actual arrival",
        f"a status that maps to DELIVERED (upper(status) IN {_COMPLETED}) with "
        f"actual_arrival IS NULL — the same predicate the pipeline flags",
        f"SELECT shipment_id FROM shipments WHERE upper(trim(status)) IN {_COMPLETED} "
        f"AND actual_arrival IS NULL",
        "warning",
        "flag",
        "Flag: the status claims delivery but there is no arrival to label from. Keep the row "
        "unlabelled instead of trusting either field.",
    ),
    (
        "shipments.status_contradicts_delay",
        "Status contradicts measured delay",
        f"DELIVERED with delay > {ON_TIME_THRESHOLD_HOURS}h, or DELAYED with delay <= "
        f"{ON_TIME_THRESHOLD_HOURS}h (actual_arrival - planned_arrival)",
        f"SELECT shipment_id FROM shipments WHERE (status = 'DELIVERED' "
        f"AND {_DELAY_H} > {ON_TIME_THRESHOLD_HOURS}) "
        f"OR (status = 'DELAYED' AND {_DELAY_H} <= {ON_TIME_THRESHOLD_HOURS})",
        "warning",
        "flag",
        "Flag: status is a leaky copy of the label, so it must not be a feature; where the "
        "two disagree, trust the timestamps.",
    ),
    (
        "port_events.event_id_reused",
        "event_id reused for different events",
        "event_id appears on more than one distinct row",
        _conflicting("port_events", "event_id"),
        "warning",
        "flag",
        "Flag: nothing joins on event_id, so keep every distinct event and key on the row "
        "rather than the id.",
    ),
    (
        "port_events.exact_duplicate",
        "Exact duplicate event",
        "surplus rows: count minus count distinct over all columns",
        _surplus("port_events", "event_id"),
        "warning",
        "drop",
        "Drop: identical copies double-count the event in port-level aggregates.",
    ),
    (
        "port_events.unknown_port",
        "Event port not in ports",
        "port_code NOT IN (SELECT port_code FROM ports)",
        "SELECT event_id FROM port_events WHERE port_code NOT IN "
        "(SELECT port_code FROM ports WHERE port_code IS NOT NULL)",
        "critical",
        "flag",
        "Flag: the event cannot be joined to a port, so it is kept for the event log but "
        "excluded from any per-port aggregate (no FK is enforced on port_events).",
    ),
    (
        "port_events.unparseable_value",
        "Unparseable or negative event value",
        "event_timestamp does not parse, or delay_minutes is not an INTEGER or is negative",
        f"SELECT event_id FROM port_events WHERE {_ts('event_timestamp')} IS NULL "
        f"OR {_int('delay_minutes')} IS NULL OR {_int('delay_minutes')} < 0",
        "critical",
        "fix",
        "Fix: a negative or unparseable delay becomes NULL and the event is kept; an unparseable "
        "timestamp cannot be fixed and would surface here before ingestion.",
    ),
    (
        "port_events.event_type_missing",
        "Event type missing",
        "event_type IS NULL",
        "SELECT event_id FROM port_events WHERE event_type IS NULL",
        "warning",
        "flag",
        "Flag: keep the event with event_type NULL; delay_minutes is still valid for port "
        "delay aggregates.",
    ),
    (
        "port_events.event_type_unknown",
        "Event type not recognised",
        "event_type is not NULL and not in EVENT_TYPES",
        f"SELECT event_id FROM port_events WHERE event_type IS NOT NULL "
        f"AND event_type NOT IN {_in(EVENT_TYPES)}",
        "warning",
        "fix",
        "Fix: set to NULL alongside the blanks (flagged event_type_unknown) rather than guess.",
    ),
    (
        "port_events.placeholder_vessel",
        "Placeholder vessel id",
        f"vessel_id IN {_in(INVALID_VESSELS)}",
        f"SELECT event_id FROM port_events WHERE vessel_id IN {_in(INVALID_VESSELS)}",
        "warning",
        "fix",
        "Fix: set vessel_id to NULL; the event is real port activity, only the vessel is "
        "unknown, and the feed is used at port level.",
    ),
    (
        "port_events.notes_missing",
        "Notes missing",
        "notes IS NULL",
        "SELECT event_id FROM port_events WHERE notes IS NULL",
        "info",
        "none",
        "None: notes are free text, nullable by design, and are not used by any feature.",
    ),
    (
        "port_events.notes_contradict_delay",
        "'Normal operations' with a delay",
        "notes = 'Normal operations' and delay_minutes > 0",
        "SELECT event_id FROM port_events WHERE notes = 'Normal operations' "
        f"AND {_int('delay_minutes')} > 0",
        "info",
        "none",
        "None: the 0-60 minute delays are in the normal range for any event type and the note "
        "is a canned label; delay_minutes is the measurement, so trust it over the note.",
    ),
    (
        "port_events.delayed_zero_minutes",
        "DELAYED event with zero delay",
        "event_type = 'DELAYED' and delay_minutes = 0",
        f"SELECT event_id FROM port_events WHERE event_type = 'DELAYED' "
        f"AND {_int('delay_minutes')} = 0",
        "info",
        "flag",
        "Flag: a handful of rows; the type and the measurement disagree, so keep the row and "
        "use delay_minutes as measured.",
    ),
    (
        "port_events.out_of_order_timestamp",
        "Timestamp earlier than previous row",
        "event_timestamp < the previous row's (file order)",
        f"SELECT event_id FROM (SELECT event_id, {_ts('event_timestamp')} AS ts, "
        f"lag({_ts('event_timestamp')}) OVER (ORDER BY {SOURCE_ROW}) AS prev "
        "FROM port_events) WHERE ts < prev",
        "warning",
        "flag",
        "Flag: the feed should arrive sorted; consumers must sort by timestamp rather than "
        "rely on file order.",
    ),
]

CHECKS = [
    Check(i, FILES[i.split(".")[0]], *rest)
    for i, *rest in _CHECKS  # tuple order matches Check fields after (id, file)
]


def run_check(con: duckdb.DuckDBPyConnection, check: Check, total_rows: int) -> dict[str, Any]:
    n = con.execute(f"SELECT count(*) FROM ({check.sql})").fetchone()[0]
    keys = con.execute(
        f"SELECT DISTINCT * FROM ({check.sql}) ORDER BY 1 LIMIT {SAMPLE_SIZE}"
    ).fetchall()
    return {
        "id": check.id,
        "file": check.file,
        "name": check.name,
        "detection": check.detection,
        "status": "pass" if n == 0 else "fail",
        "rows_affected": n,
        "total_rows": total_rows,
        "pct_affected": round(100 * n / total_rows, 2) if total_rows else 0.0,
        "severity": check.severity,
        "action": check.action,
        "recommendation": check.recommendation,
        "sample_keys": [k[0] for k in keys],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(input_dir: Path) -> dict[str, Any]:
    input_dir = resolve_input_dir(input_dir)
    con = duckdb.connect()
    load_raw(con, input_dir)
    totals = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in FILES}
    results = [run_check(con, c, totals[c.id.split(".")[0]]) for c in CHECKS]
    failed = [r for r in results if r["status"] == "fail"]
    any_rule = " OR ".join(f"({p})" for p in QUARANTINE_RULES.values())
    quarantined = con.execute(f"SELECT count(*) FROM shipments WHERE {any_rule}").fetchone()[0]
    return {
        "report_version": "1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "files": {
            FILES[t]: {"rows": totals[t], "sha256": _sha256(input_dir / FILES[t])} for t in FILES
        },
        "summary": {
            "total_checks": len(results),
            "failed_checks": len(failed),
            "failed_by_severity": {
                s: sum(r["severity"] == s for r in failed) for s in ("critical", "warning", "info")
            },
            "rows_quarantined_estimate": quarantined,
        },
        "checks": results,
    }


def to_markdown(report: dict[str, Any]) -> str:
    failed = [c for c in report["checks"] if c["status"] == "fail"]
    passed = [c["id"] for c in report["checks"] if c["status"] == "pass"]
    lines = [
        "| Check | File | Rows | % | Severity | Action | Decision |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in failed:
        lines.append(
            f"| `{c['id']}` | {c['file']} | {c['rows_affected']:,} | {c['pct_affected']} "
            f"| {c['severity']} | {c['action']} | {c['recommendation']} |"
        )
    lines.append("")
    lines.append(f"{len(passed)} further checks passed ({', '.join(passed)}).")
    return "\n".join(lines)
