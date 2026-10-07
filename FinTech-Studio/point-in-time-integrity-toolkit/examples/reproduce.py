#!/usr/bin/env python3
"""Replay the fixed arithmetic/availability examples; no market fetch or fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from financemeta_data_audit.audit import load_config
from financemeta_data_audit.backtest import (
    audit_backtest,
    render_backtest_markdown,
)


def reproduce(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(ROOT / "examples/ledger_config.json")
    cases = []
    for filename, expected in (("ledger_clean.csv", "PASS"), ("ledger_future_feature.csv", "FAIL")):
        report = audit_backtest(ROOT / "examples" / filename, config)
        target = output / Path(filename).stem
        target.mkdir()
        (target / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        (target / "summary.md").write_text(render_backtest_markdown(report))
        cases.append({"input": filename, "expected": expected, "actual": report["overall_status"],
                      "metrics_withheld": report["performance"] is None, "source_sha256": report["source_sha256"]})
    status = "PASS" if all(row["actual"] == row["expected"] for row in cases) and cases[1]["metrics_withheld"] else "FAIL"
    summary = {"data_kind": "synthetic", "status": status, "cases": cases,
               "claim": "Engineering arithmetic and declared timestamp checks only; no empirical financial result."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    artifacts = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "SHA256SUMS.json").write_text(json.dumps(artifacts, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory; existing results are never overwritten")
    result = reproduce(parser.parse_args().output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 2)
