"""The pipeline CLI: exit codes and the counts JSON on stdout."""

import json

from pipeline.__main__ import main
from tests.unit.conftest import FIXTURES


def test_cli_builds_and_prints_counts(tmp_path, capsys):
    db = tmp_path / "w.duckdb"
    assert main(["--input", str(FIXTURES), "--db", str(db)]) == 0
    counts = json.loads(capsys.readouterr().out)
    assert counts["curated_shipments"] == 4 and db.exists()


def test_cli_missing_input_exits_2(tmp_path, capsys):
    assert main(["--input", str(tmp_path / "nowhere"), "--db", str(tmp_path / "w.duckdb")]) == 2
    assert "input error" in capsys.readouterr().err
