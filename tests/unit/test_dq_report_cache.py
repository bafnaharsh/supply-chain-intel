"""GET /data-quality/report serves the file, re-reads it only when it changes."""

import os


def test_report_cache_follows_file_changes(client, tmp_path, monkeypatch):
    from core.config import settings

    report = tmp_path / "r.json"
    report.write_text('{"v": 1}', encoding="utf-8")
    monkeypatch.setattr(settings, "dq_report_path", report)
    assert client.get("/data-quality/report").json() == {"v": 1}
    report.write_text('{"v": 2}', encoding="utf-8")
    st = report.stat()
    os.utime(report, (st.st_atime, st.st_mtime + 5))  # coarse filesystem clocks
    assert client.get("/data-quality/report").json() == {"v": 2}


def test_corrupt_report_is_500_not_garbage(client, tmp_path, monkeypatch):
    from core.config import settings

    report = tmp_path / "bad.json"
    report.write_text('{"half": ', encoding="utf-8")
    monkeypatch.setattr(settings, "dq_report_path", report)
    r = client.get("/data-quality/report")
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error"
