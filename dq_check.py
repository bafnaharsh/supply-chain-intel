"""Standalone data-quality check, as named in the brief: `python dq_check.py --input ./data/raw`.

Thin entry point over `python -m dq check`; needs the project's dependencies (`uv sync`, or
`uv run python dq_check.py ...`). Writes reports/dq_report.json unless --output says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dq.__main__ import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["check", *argv]
    sys.exit(main(argv))
