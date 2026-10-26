# Data details

The three raw files in [`raw/`](raw/) came with the assignment, unchanged apart from the file names (the originals were called `ports 2.csv`, `shipments 2.csv` and `port_events 2.csv`). Treat them as read-only: every later stage reads from here and writes somewhere else.

| File | Rows | Grain | Role |
|---|---|---|---|
| [`ports.csv`](raw/ports.csv) | 25 | one port | Reference / dimension |
| [`shipments.csv`](raw/shipments.csv) | 5,028 (5,000 unique IDs) | one container booking | Main fact table; source of the ML label |
| [`port_events.csv`](raw/port_events.csv) | 25,000 | one event at one port | Streaming-style event feed |

All three files are UTF-8, comma-separated, with CRLF line endings and a header row. Timestamps have the form `YYYY-MM-DD HH:MM:SS` and **carry no timezone**.

The figures in the "Observed quality issues" sections come from the first profiling pass and are what the DQ checks were written from. The authoritative, re-runnable outputs are:

- [`../reports/data_profile.json`](../reports/data_profile.json): every column of every file — inferred type, nulls, distinct count, range, top values (`python -m dq profile`).
- [`../reports/dq_report.json`](../reports/dq_report.json): one entry per check with rows affected, severity, action and reasoning (`python -m dq check`). Served unchanged by `GET /data-quality/report`.
- The README "Data quality summary" and "Data model" sections describe how each issue is handled in the curated layer.

---

## `ports.csv`: port reference data

| Column | Type | Description |
|---|---|---|
| `port_code` | string (5-char UN/LOCODE) | **Primary key**, e.g. `CNSHA`, `NLRTM` |
| `port_name` | string | Human name, e.g. Shanghai, Rotterdam |
| `country` | string | Country name |
| `region` | enum | `APAC`, `EUR`, `AMER`, `MEA` |
| `timezone` | string | Fixed UTC offset, e.g. `UTC+8`, `UTC+5:30` (no DST) |
| `avg_congestion_score` | float 0–1 | Static average congestion; higher means more congested (0.55–0.85) |

**Relationships:** `shipments.origin_port`, `shipments.destination_port` and `port_events.port_code` all join to `port_code`.

**Observed quality issues**
- `BEANR` (Antwerp) has no `country`; it should be Belgium.

---

## `shipments.csv`: container bookings

| Column | Type | Description |
|---|---|---|
| `shipment_id` | string `SHP-NNNNN` | Business key. **Not unique in the raw file** (see below) |
| `customer_id` | string `CUST-NNNN` | 251 distinct customers |
| `origin_port` | string | FK to `ports.port_code` |
| `destination_port` | string | FK to `ports.port_code`. There are 671 distinct origin→destination routes |
| `vessel_id` | string `VSL-NNNN` | 80 distinct vessels |
| `booking_date` | timestamp | When the booking was made, from 2024-06-01 to 2025-10-14 |
| `planned_departure` | timestamp | Scheduled departure, 3–14 days after booking |
| `actual_departure` | timestamp, nullable | Observed departure. **Not known at booking time** |
| `planned_arrival` | timestamp | Scheduled arrival, giving 3–35 planned transit days |
| `actual_arrival` | timestamp, nullable | Observed arrival. **Not known at booking time** |
| `container_count` | int | Containers on the booking (typically 1–60) |
| `cargo_type` | enum | 10 categories, e.g. Electronics, Pharmaceuticals, Raw Materials |
| `weight_tons` | float | Total cargo weight |
| `status` | enum | `DELIVERED`, `DELAYED`, `CANCELLED`, plus variants (below) |

**Derived later (curated layer):** `actual_delay_hours = actual_arrival − planned_arrival`, `on_time_flag = delay ≤ 24h`, `route_key = origin → destination`, `transit_days_planned`, `transit_days_actual`.

**ML label:** `actual_delay_hours > 24`, with a base rate of about 40% among shipments that have an actual arrival.
**Leakage warning:** `status = DELAYED` matches that label exactly (1,901 of 1,901 rows), and `actual_*` are post-booking. None of these may be used as features.

**Observed quality issues**
| Issue | Rows |
|---|---|
| Exact duplicate rows | 8 |
| Same `shipment_id` with conflicting values (15 differ only in `cargo_type`, 5 in `weight_tons`) | 20 IDs |
| Placeholder port codes `XXTST` / `ZZZZZ` / `UNKNW` not in `ports.csv` | 42 origin, 55 destination |
| Non-canonical `status`: `COMPLETED` 34, `Complete` 30, `delivered` 28, `N/A` 34, blank 20 | 146 |
| `cargo_type` case/whitespace variants (`FURNITURE`, `' Chemicals'`, `'machinery '`, …) | ~260 |
| `cargo_type` blank | 15 |
| `weight_tons` blank | 73 |
| `weight_tons` negative | 36 |
| `container_count` = 0 | 23 |
| `container_count` implausibly high (510–996) | 17 |
| `actual_departure` before `booking_date` (impossible timeline) | 39 |
| Completed-type status (`COMPLETED`/`Complete`/`delivered`) but no actual dates | 92 |

`CANCELLED` rows (203) have null actuals, which is expected and not an error.

---

## `port_events.csv`: port event feed

| Column | Type | Description |
|---|---|---|
| `event_id` | string `EVT-NNNNNN` | Intended event key. **Not unique** (see below) |
| `event_timestamp` | timestamp | 2024-06-01 to 2025-10-14, already sorted ascending |
| `port_code` | string | FK to `ports.port_code`; every value is valid. About 1,000 events per port |
| `vessel_id` | string `VSL-NNNN` | Vessel at the port; joins loosely to `shipments.vessel_id` |
| `event_type` | enum | `ARRIVAL`, `BERTHED`, `LOADING_COMPLETE`, `CUSTOMS_CLEARED`, `INSPECTION`, `DEPARTURE`, `DELAYED` (about 3.5k each) |
| `delay_minutes` | int ≥ 0 | 0–60 for normal events; 0–300 (mean ~149) for `DELAYED` events |
| `notes` | free text, nullable | One of 12 canned reasons, e.g. "Weather delay – fog", "Berth congestion – 3 vessels waiting" |

**Relationship to shipments:** there is no `shipment_id`. Matching on vessel + origin port within ±3 days of planned departure links only about 680 of 5,000 shipments. So this file is treated as a **port-level** feed (e.g. trailing congestion and delay per port as of a date), not as per-shipment tracking.

**Observed quality issues**
| Issue | Rows |
|---|---|
| `event_id` reused for different events | 103 IDs (206 rows) |
| `event_type` blank | 140 |
| `vessel_id` not a real vessel (`VSL-0000`, `VSL-9999`) | 207 |
| `notes` blank | 1,913 |
| `notes` = "Normal operations" while `delay_minutes` > 0 | 671 |
| `event_type` = `DELAYED` with `delay_minutes` = 0 | 12 |

The notes contain a Unicode en dash (–). It is valid UTF-8; the Windows console just renders it as `�`.

---

## Open assumptions

- Timestamps are treated as UTC. The `timezone` column in `ports.csv` is kept as a reference attribute only.
- Relative-time questions such as "this quarter" are anchored to the latest date in the data (2025-10), not the wall clock.
