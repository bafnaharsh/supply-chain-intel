from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as app_main
from app.main import create_app
from core.config import settings
from ml.predictor import load_predictor
from pipeline.build import build

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="session")
def warehouse(tmp_path_factory) -> Path:
    """Warehouse built once from the 4-row clean fixture through the real pipeline."""
    db = tmp_path_factory.mktemp("wh") / "warehouse.duckdb"
    build(FIXTURES, db)
    return db


@pytest.fixture(scope="session")
def predictor():
    """The committed model, unpickled once per session instead of once per test."""
    return load_predictor(settings.model_path)


@pytest.fixture
def client(monkeypatch, tmp_path, warehouse, predictor):
    """App over the fixture warehouse and a tiny DQ report; no network, no real data."""
    report = tmp_path / "dq_report.json"
    report.write_text('{"status": "test", "checks": []}', encoding="utf-8")
    monkeypatch.setattr(settings, "db_path", warehouse)
    monkeypatch.setattr(settings, "dq_report_path", report)
    monkeypatch.setattr(app_main, "load_predictor", lambda path: predictor)
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        yield c
