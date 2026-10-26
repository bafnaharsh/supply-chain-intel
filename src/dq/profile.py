"""Column-level profile of the raw CSVs: types, nulls, distinct values, ranges, top values.

This is descriptive, not judgemental: it says what is in the files so the checks in
`checks.py` can be written (and defended) from evidence rather than assumptions.
"""

from pathlib import Path
from typing import Any

import duckdb

from dq.rules import FILES, SOURCE_ROW, TIMESTAMP_FORMAT, load_raw

TOP_N = 12


def profile_column(con: duckdb.DuckDBPyConnection, table: str, col: str) -> dict[str, Any]:
    q = con.execute(
        f"""
        SELECT count(*)                                              AS rows,
               count({col})                                         AS non_null,
               count(DISTINCT {col})                                AS distinct_values,
               min(length({col})), max(length({col})),
               count(try_cast({col} AS DOUBLE)),
               count(try_strptime({col}, '{TIMESTAMP_FORMAT}')),
               sum(CASE WHEN {col} <> trim({col}) THEN 1 ELSE 0 END),
               sum(CASE WHEN {col} <> upper({col}) THEN 1 ELSE 0 END)
        FROM {table}
        """
    ).fetchone()
    rows, non_null, distinct, min_len, max_len, as_num, as_ts, untrimmed, has_lower = q
    p: dict[str, Any] = {
        "rows": rows,
        "nulls": rows - non_null,
        "null_pct": round(100 * (rows - non_null) / rows, 2) if rows else 0.0,
        "distinct": distinct,
        "length": {"min": min_len, "max": max_len},
        "untrimmed": untrimmed,
    }
    # Inferred type: the strictest type that every non-null value parses as.
    if non_null and as_ts == non_null:
        p["inferred_type"] = "timestamp"
        lo, hi = con.execute(
            f"SELECT min(try_strptime({col}, '{TIMESTAMP_FORMAT}')), "
            f"max(try_strptime({col}, '{TIMESTAMP_FORMAT}')) FROM {table}"
        ).fetchone()
        p["range"] = {"min": str(lo), "max": str(hi)}
    elif non_null and as_num == non_null:
        p["inferred_type"] = "number"
        stats = con.execute(
            f"SELECT min(x), max(x), round(avg(x), 3), round(median(x), 3), "
            f"sum(CASE WHEN x <= 0 THEN 1 ELSE 0 END) "
            f"FROM (SELECT try_cast({col} AS DOUBLE) AS x FROM {table}) WHERE x IS NOT NULL"
        ).fetchone()
        p["range"] = {"min": stats[0], "max": stats[1], "mean": stats[2], "median": stats[3]}
        p["non_positive"] = stats[4]
    else:
        p["inferred_type"] = "string"
        p["mixed_case"] = has_lower
    if distinct <= 50 or p["inferred_type"] == "string":
        p["top_values"] = [
            {"value": v, "count": n}
            for v, n in con.execute(
                f"SELECT {col}, count(*) AS n FROM {table} WHERE {col} IS NOT NULL "
                f"GROUP BY 1 ORDER BY n DESC, 1 LIMIT {TOP_N}"
            ).fetchall()
        ]
    return p


def profile_table(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, Any]:
    cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall() if r[0] != SOURCE_ROW]
    col_list = ", ".join(cols)
    rows, exact_dups = con.execute(
        f"SELECT count(*), "
        f"count(*) - (SELECT count(*) FROM (SELECT DISTINCT {col_list} FROM {table})) "
        f"FROM {table}"
    ).fetchone()
    key = cols[0]
    key_dups = con.execute(
        f"SELECT count(*) FROM (SELECT {key} FROM {table} GROUP BY 1 HAVING count(*) > 1)"
    ).fetchone()[0]
    return {
        "rows": rows,
        "columns": len(cols),
        "exact_duplicate_rows": exact_dups,
        "first_column": key,
        "first_column_duplicated_values": key_dups,
        "column_profiles": {c: profile_column(con, table, c) for c in cols},
    }


def build_profile(input_dir: Path) -> dict[str, Any]:
    con = duckdb.connect()
    load_raw(con, input_dir)
    return {
        "input_dir": str(input_dir),
        "files": {FILES[t]: profile_table(con, t) for t in FILES},
    }


def to_markdown(profile: dict[str, Any]) -> str:
    out = []
    for fname, t in profile["files"].items():
        out.append(
            f"### {fname}: {t['rows']:,} rows, {t['columns']} columns, "
            f"{t['exact_duplicate_rows']} exact duplicate rows, "
            f"{t['first_column_duplicated_values']} duplicated `{t['first_column']}` values\n"
        )
        out.append("| column | type | nulls | distinct | range / top values |")
        out.append("|---|---|---|---|---|")
        for c, p in t["column_profiles"].items():
            if "range" in p:
                r = p["range"]
                detail = f"{r['min']} → {r['max']}"
                if "mean" in r:
                    detail += f" (mean {r['mean']}, median {r['median']})"
            else:
                detail = ", ".join(f"`{v['value']}` ×{v['count']}" for v in p["top_values"][:6])
            nulls = f"{p['nulls']} ({p['null_pct']}%)" if p["nulls"] else "0"
            out.append(f"| `{c}` | {p['inferred_type']} | {nulls} | {p['distinct']} | {detail} |")
        out.append("")
    return "\n".join(out)
