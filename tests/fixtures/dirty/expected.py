"""Planted issues in tests/fixtures/dirty/*.csv: check id -> rows_affected.

Every check in dq.checks.CHECKS must have an entry here (0 is allowed). The counts are
by hand from the fixture rows, so a change to either file must update the other.
"""

EXPECTED = {
    # ports.csv (5 rows)
    "ports.missing_country": 1,  # BEANR
    "ports.duplicate_port_code": 0,
    "ports.congestion_out_of_range": 1,  # BEANR 1.7
    # shipments.csv (28 rows, 26 distinct ids + SHIP-1)
    "shipments.exact_duplicate": 1,  # SHP-00001 repeated (surplus rows)
    "shipments.conflicting_duplicate_id": 2,  # SHP-00005 x2, cargo differs
    "shipments.missing_required": 1,  # SHP-00025 blank origin
    "shipments.unparseable_value": 1,  # SHP-00024 booking_date 2024/06/23
    "shipments.bad_id_format": 1,  # SHIP-1
    "shipments.unknown_origin_port": 1,  # SHP-00006 XXTST
    "shipments.unknown_destination_port": 1,  # SHP-00007 ZZZZZ
    "shipments.origin_equals_destination": 1,  # SHP-00008
    "shipments.status_non_canonical": 2,  # SHP-00009 COMPLETED, SHP-00027 delivered
    "shipments.status_missing": 2,  # SHP-00010 N/A, SHP-00011 blank
    "shipments.cargo_type_non_canonical": 2,  # SHP-00012 FURNITURE, SHP-00013 ' Chemicals'
    "shipments.cargo_type_missing": 1,  # second SHP-00005 row
    "shipments.weight_missing": 2,  # SHP-00003, SHP-00026
    "shipments.weight_negative": 1,  # SHP-00014
    "shipments.weight_zero": 0,
    "shipments.container_count_zero": 1,  # SHP-00015
    "shipments.container_count_outlier": 1,  # SHP-00016 (996)
    "shipments.planned_arrival_before_departure": 1,  # SHP-00018
    "shipments.planned_departure_before_booking": 1,  # SHP-00019
    "shipments.departure_before_booking": 1,  # SHP-00017
    "shipments.actual_arrival_before_departure": 1,  # SHP-00020
    "shipments.completed_without_actuals": 1,  # SHP-00009
    "shipments.status_contradicts_delay": 2,  # SHP-00021 DELIVERED 48h, SHP-00022 DELAYED 5h
    # port_events.csv (13 rows)
    "port_events.event_id_reused": 2,  # EVT-000001 on two different events
    "port_events.exact_duplicate": 1,  # EVT-000002 repeated
    "port_events.unknown_port": 1,  # ZZZZZ
    "port_events.unparseable_value": 1,  # delay -5
    "port_events.event_type_missing": 1,
    "port_events.event_type_unknown": 1,  # FOO
    "port_events.placeholder_vessel": 1,  # VSL-9999
    "port_events.notes_missing": 1,
    "port_events.notes_contradict_delay": 1,  # Normal operations with 30 min
    "port_events.delayed_zero_minutes": 1,
    "port_events.out_of_order_timestamp": 1,  # last row earlier than previous
}
