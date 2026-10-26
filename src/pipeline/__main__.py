"""CLI: python -m pipeline --input data/raw --db data/warehouse.duckdb"""

import argparse
import json
import logging
import sys
from pathlib import Path

from app.logging_config import JsonFormatter
from core.config import settings
from pipeline.build import build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline", description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--db", type=Path, default=settings.db_path)
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8")
    handler = logging.StreamHandler(sys.stderr)  # stdout is reserved for the counts JSON
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    try:
        counts = build(args.input, args.db)
    except ValueError as e:
        print(f"input error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"pipeline error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
