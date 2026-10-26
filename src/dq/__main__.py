"""CLI: `python -m dq check|profile --input data/raw --output <json> [--markdown]`.

`check` is the default subcommand, so `python -m dq --input ... --output ...` works too.
The report is the output: exit 0 even when checks fail, exit 2 when the input is unusable.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from core.config import ROOT, settings
from dq import checks, profile

COMMANDS = {
    "check": (checks.build_report, checks.to_markdown, settings.dq_report_path),
    "profile": (profile.build_profile, profile.to_markdown, ROOT / "reports/data_profile.json"),
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dq", description="Data-quality checks and profiling.")
    sub = p.add_subparsers(dest="command")
    for name, (_, _, default_out) in COMMANDS.items():
        s = sub.add_parser(name, help=f"write {default_out}")
        s.add_argument("--input", type=Path, default=Path("data/raw"), help="folder with the CSVs")
        s.add_argument("--output", type=Path, default=default_out, help="JSON path")
        s.add_argument("--markdown", action="store_true", help="also print a markdown table")
    return p


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console defaults to cp1252
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in (*COMMANDS, "-h", "--help"):
        argv.insert(0, "check")
    args = _parser().parse_args(argv)
    build, to_markdown, _ = COMMANDS[args.command]
    try:
        result = build(args.input)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Write then swap, so an API serving this file never reads a half-written report.
    tmp = args.output.with_name(args.output.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, args.output)
    if args.command == "check":
        s = result["summary"]
        print(
            f"dq check: {s['failed_checks']}/{s['total_checks']} checks failed "
            f"{s['failed_by_severity']}, ~{s['rows_quarantined_estimate']} shipment rows "
            f"to quarantine -> {args.output}",
            file=sys.stderr,
        )
    else:
        print(f"dq profile: {len(result['files'])} files -> {args.output}", file=sys.stderr)
    if args.markdown:
        print(to_markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
