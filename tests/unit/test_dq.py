import functools
import importlib.util
import json
from pathlib import Path

import duckdb
import pytest

from dq.__main__ import main
from dq.checks import CHECKS, build_report, run_check
from dq.rules import FILES, load_raw

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
DIRTY = FIXTURES / "dirty"

_spec = importlib.util.spec_from_file_location("dirty_expected", DIRTY / "expected.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EXPECTED = _mod.EXPECTED


@functools.cache
def _run(input_dir: Path) -> dict[str, dict]:
    con = duckdb.connect()
    load_raw(con, input_dir)
    totals = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in FILES}
    return {c.id: run_check(con, c, totals[c.id.split(".")[0]]) for c in CHECKS}


def test_clean_fixture_passes_every_check():
    failed = {i: r["rows_affected"] for i, r in _run(FIXTURES).items() if r["rows_affected"]}
    assert failed == {}


@pytest.mark.parametrize("check_id", list(EXPECTED))
def test_dirty_fixture_counts(check_id):
    assert _run(DIRTY)[check_id]["rows_affected"] == EXPECTED[check_id]


def test_every_check_has_expected_count():
    ids = [c.id for c in CHECKS]
    assert set(ids) == set(EXPECTED)
    assert len(ids) == len(set(ids))
    assert ids == list(EXPECTED)


def test_report_shape():
    r = build_report(DIRTY)
    assert list(r) == [
        "report_version",
        "generated_at",
        "input_dir",
        "files",
        "summary",
        "checks",
    ]
    assert r["files"]["shipments.csv"]["rows"] == 28
    assert len(r["files"]["ports.csv"]["sha256"]) == 64
    for c in r["checks"]:
        assert list(c) == [
            "id",
            "file",
            "name",
            "detection",
            "status",
            "rows_affected",
            "total_rows",
            "pct_affected",
            "severity",
            "action",
            "recommendation",
            "sample_keys",
        ]
        assert c["status"] == ("pass" if c["rows_affected"] == 0 else "fail")
        assert c["pct_affected"] == round(100 * c["rows_affected"] / c["total_rows"], 2)
        assert c["severity"] in ("critical", "warning", "info")
        assert c["action"] in ("drop", "fix", "flag", "quarantine", "none")
        assert len(c["sample_keys"]) <= 5
    failed = [c for c in r["checks"] if c["status"] == "fail"]
    s = r["summary"]
    assert s["total_checks"] == len(r["checks"])
    assert s["failed_checks"] == len(failed)
    assert s["failed_by_severity"] == {
        sev: sum(c["severity"] == sev for c in failed) for sev in ("critical", "warning", "info")
    }
    assert 0 < s["rows_quarantined_estimate"] <= 28


def test_report_is_deterministic():
    a, b = build_report(DIRTY), build_report(DIRTY)
    a.pop("generated_at"), b.pop("generated_at")
    assert a == b


def test_cli_writes_json(tmp_path, capsys):
    out = tmp_path / "nested" / "dq.json"
    assert main(["check", "--input", str(DIRTY), "--output", str(out), "--markdown"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["checks"]
    assert "| Check | File | Rows |" in capsys.readouterr().out
    assert main(["--input", str(DIRTY), "--output", str(out)]) == 0  # default subcommand


def test_input_may_be_the_parent_of_raw(tmp_path):
    """The brief's example is `--input ./data/`; the CSVs live in data/raw/."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for f in DIRTY.glob("*.csv"):
        (raw / f.name).write_bytes(f.read_bytes())
    assert build_report(tmp_path / "data")["files"] == build_report(raw)["files"]


def test_cli_missing_column_exits_2(tmp_path, capsys):
    for name in FILES.values():
        (tmp_path / name).write_text((FIXTURES / name).read_text(encoding="utf-8"))
    (tmp_path / "shipments.csv").write_text("shipment_id\nSHP-00001\n", encoding="utf-8")
    assert main(["check", "--input", str(tmp_path), "--output", str(tmp_path / "o.json")]) == 2
    assert "missing columns" in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


def test_cli_profile_writes_json(tmp_path):
    out = tmp_path / "profile.json"
    assert main(["profile", "--input", str(FIXTURES), "--output", str(out), "--markdown"]) == 0
    assert set(json.loads(out.read_text(encoding="utf-8"))["files"]) == set(FILES.values())
