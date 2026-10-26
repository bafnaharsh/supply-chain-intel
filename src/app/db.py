"""Read-only DuckDB access to the warehouse built by `python -m pipeline`.

The serving layer exposes `shipments`, `ports` and `port_events` views in the default schema,
so routers query plain table names and never see the raw/staging/curated layers.
"""

from pathlib import Path
from typing import Any

import duckdb


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        if not db_path.exists():
            raise FileNotFoundError(f"warehouse not found: {db_path} (run `python -m pipeline`)")
        self._conn = duckdb.connect(str(db_path), read_only=True)
        self.ping(raise_on_error=True)  # fail fast at startup if the file is malformed

    def query(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        # DuckDB connections are not thread-safe; FastAPI runs sync routes in a threadpool.
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params or [])
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
        finally:
            cur.close()

    def one(self, sql: str, params: list[Any] | None = None) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        self._conn.close()

    def ping(self, raise_on_error: bool = False) -> bool:
        try:
            self.query("SELECT 1 FROM shipments LIMIT 1")
            return True
        except duckdb.Error:
            if raise_on_error:
                raise
            return False
