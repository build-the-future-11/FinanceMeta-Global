from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_csv, load_config, render_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="financemeta-data-audit")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="audit one OHLCV CSV without modifying it")
    check.add_argument("csv")
    check.add_argument("--config", required=True)
    check.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        config = load_config(args.config)
        report = audit_csv(args.csv, config)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "summary.md").write_text(render_markdown(report), encoding="utf-8")
        print(json.dumps({"overall_status": report["overall_status"], "report": str(out / "report.json")}, sort_keys=True))
        return 0 if report["overall_status"] == "PASS" else 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
