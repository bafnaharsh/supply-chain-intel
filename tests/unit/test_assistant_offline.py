"""Offline mode: the rule-based router over the fixture API (4 shipments, no LLM, no ADK)."""

from types import SimpleNamespace

import pytest

from assistant.__main__ import run_mode
from assistant.offline import HELP, answer_offline
from tests.unit.test_assistant import api  # noqa: F401


def test_route_stats(api):  # noqa: F811
    r = answer_offline("What is the on-time rate from Shanghai to Rotterdam?")
    assert "on-time rate 50.0%" in r.text and "average delay 29.0 h" in r.text
    assert "n=2 completed" in r.text and "all time" in r.text
    assert r.tools_called == ["get_route_stats"] and "[get_route_stats]" in r.text


@pytest.mark.parametrize(
    "q",
    [
        "average delay CNSHA to NLRTM",
        "stats between Shanghai and Rotterdam",
        "what is the on-time rate Shanghai to Rotterdam",
    ],
)
def test_route_stats_phrasings(api, q):  # noqa: F811
    assert "CNSHA" in answer_offline(q).text and "29.0 h" in answer_offline(q).text


def test_top_routes_all_time(api):  # noqa: F811
    r = answer_offline("Which routes have the highest average delay?")
    assert r.text.index("1. CNSHA → NLRTM") < r.text.index("DEHAM → USLAX")
    assert "29.0 h" in r.text and "all time" in r.text
    assert r.tools_called == ["data_period", "top_routes"]


def test_top_routes_worst_on_time_and_window(api):  # noqa: F811
    r = answer_offline("worst on-time rate in 2024-06")
    assert "Lowest on-time rate" in r.text and "booked in 2024-06" in r.text
    assert "1. CNSHA → NLRTM" in r.text  # 50% < 100%
    # the fixture's latest booking is 2024-07-10, so this quarter is Q3 and last quarter is Q2
    r = answer_offline("busiest routes this quarter")
    assert "Q3 2024" in r.text and "1. DEHAM → USLAX" in r.text and "CNSHA" not in r.text
    r = answer_offline("busiest routes last quarter")
    assert "Q2 2024" in r.text and "1. CNSHA → NLRTM" in r.text


def test_rank_without_metric_asks(api):  # noqa: F811
    r = answer_offline("what is the worst route?")
    assert "average delay, on-time rate, or number of shipments" in r.text
    assert r.tools_called == []


def test_delay_risk(api):  # noqa: F811
    r = answer_offline("delay risk for shipment 1")
    assert "SHP-00001" in r.text and "DELIVERED" in r.text and "%" in r.text
    assert "actual delay 10.0 h" in r.text
    assert r.tools_called == ["get_shipment", "predict_delay"]


def test_delay_risk_not_found(api):  # noqa: F811
    r = answer_offline("predict 99")
    assert "SHP-00099 was not found" in r.text


def test_status(api):  # noqa: F811
    r = answer_offline("status of shipment 3")
    assert "SHP-00003" in r.text and "CANCELLED" in r.text and r.tools_called == ["get_shipment"]
    assert "DELAYED" in answer_offline("Was SHP-00002 late?").text
    assert "Which shipment" in answer_offline("status of the shipment").text


def test_ports(api):  # noqa: F811
    t = answer_offline("which ports are there?").text
    assert all(c in t for c in ("CNSHA", "NLRTM", "DEHAM", "USLAX")) and "Shanghai" in t


def test_count(api):  # noqa: F811
    t = answer_offline("how many shipments from CNSHA").text
    assert "3 shipments from Shanghai (CNSHA), all time" in t and "[query_shipments]" in t
    assert "1 shipment from Shanghai (CNSHA) to Rotterdam (NLRTM) with status DELAYED" in (
        answer_offline("How many shipments from Shanghai to Rotterdam delayed?").text
    )
    assert "3 shipments, booked in 2024-06" in answer_offline("number of shipments in 2024-06").text
    assert "0 shipments, booked in 2024-05" in answer_offline("how many shipments in 2024-05").text


def test_typo_is_corrected_and_said(api):  # noqa: F811
    t = answer_offline("on-time rate from Shanghai to Rotterdm").text
    assert "Interpreted 'Rotterdm' as Rotterdam" in t and "50.0%" in t


def test_single_port_asks_back(api):  # noqa: F811
    t = answer_offline("What is the on-time rate from Shanghai?").text
    assert "to which destination" in t and "Rotterdam (NLRTM)" in t and "Shanghai (CNSHA)," not in t
    assert "from which origin" in answer_offline("on-time rate to Rotterdam").text
    assert "to which destination" in answer_offline("average delay for Shanghai").text


def test_unknown_port(api):  # noqa: F811
    assert (
        "'Atlantis' is not in the data" in answer_offline("delay from Atlantis to Rotterdam").text
    )


def test_gibberish_gets_help(api):  # noqa: F811
    r = answer_offline("nonsense")
    assert (
        r.text == HELP and "fixed set of question shapes" in r.text and "GEMINI_API_KEY" in r.text
    )
    assert r.tools_called == []


def test_no_llm_and_no_cost(api):  # noqa: F811
    r = answer_offline("which ports")
    assert r.llm_calls == 0 and r.usage["est_usd"] == 0.0 and r.guardrail is None
    assert r.usage == {"prompt_tokens": 0, "output_tokens": 0, "est_usd": 0.0}


@pytest.mark.parametrize(
    ("offline", "has_key", "mode"),
    [(True, True, "offline"), (False, False, "offline"), (False, True, "online")],
)
def test_run_mode(offline, has_key, mode):
    assert run_mode(SimpleNamespace(offline=offline), has_key) == mode
