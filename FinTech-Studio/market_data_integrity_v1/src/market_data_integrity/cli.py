from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import markdown_summary, validate_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only OHLCV integrity checker")
    parser.add_argument("csv")
    parser.add_argument("config")
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    report = validate_csv(Path(args.csv), Path(args.config))
    encoded = json.dumps(report, indent=2, sort_keys=True)

    if args.json_out:
        Path(args.json_out).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)

    if args.md_out:
        Path(args.md_out).write_text(markdown_summary(report), encoding="utf-8")

    raise SystemExit(0 if report["overall"] == "PASS" else 2)


if __name__ == "__main__":
    main()
