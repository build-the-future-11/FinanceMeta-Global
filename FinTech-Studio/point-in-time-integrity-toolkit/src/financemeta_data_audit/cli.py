from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_csv, load_config, render_markdown
from .backtest import audit_backtest, render_backtest_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="financemeta-data-audit")
    sub = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("check", "audit one OHLCV CSV without modifying it"),
        ("backtest", "audit timing and costs of a frozen single-asset prediction ledger"),
    ):
        check = sub.add_parser(command, help=help_text)
        check.add_argument("csv")
        check.add_argument("--config", required=True)
        check.add_argument("--out", required=True, help="new output directory (existing directories are refused)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"check", "backtest"}:
        config = load_config(args.config)
        audit, render = (audit_csv, render_markdown) if args.command == "check" else (audit_backtest, render_backtest_markdown)
        report = audit(args.csv, config)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)
        (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "summary.md").write_text(render(report), encoding="utf-8")
        print(json.dumps({"overall_status": report["overall_status"], "report": str(out / "report.json")}, sort_keys=True))
        return 0 if report["overall_status"] == "PASS" else 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
