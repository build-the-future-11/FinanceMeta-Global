#!/usr/bin/env python3
import json
import sys
from pathlib import Path

REQUIRED = [
    "benchmark_id",
    "title",
    "task",
    "data",
    "splits",
    "baselines",
    "metrics",
    "leakage_controls",
    "reproducibility",
    "result_status",
    "release_decision",
]

ALLOWED_RESULTS = {"UNTESTED", "POSITIVE", "NEGATIVE", "INCONCLUSIVE"}
ALLOWED_DECISIONS = {"CONTINUE", "FREEZE", "RELEASE", "RELEASE_WITH_LIMITATIONS", "ARCHIVE"}

def fail(msg: str) -> None:
    raise SystemExit(f"FinanceBench contract invalid: {msg}")

def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: validate_contract.py <benchmark.json>")
    path = Path(sys.argv[1])
    payload = json.loads(path.read_text())
    missing = [k for k in REQUIRED if k not in payload]
    if missing:
        fail(f"missing required fields: {', '.join(missing)}")
    if not payload["baselines"]:
        fail("at least one baseline is required")
    if not payload["metrics"]:
        fail("at least one metric is required")
    if not any(m.get("primary") is True for m in payload["metrics"]):
        fail("one metric must be marked primary")
    if not payload["leakage_controls"]:
        fail("at least one leakage control is required")
    if payload["result_status"] not in ALLOWED_RESULTS:
        fail("unknown result_status")
    if payload["release_decision"] not in ALLOWED_DECISIONS:
        fail("unknown release_decision")
    repro = payload["reproducibility"]
    for key in ("commit_sha", "exact_command", "raw_outputs_retained"):
        if key not in repro:
            fail(f"reproducibility.{key} is required")
    if payload["release_decision"] in {"RELEASE", "RELEASE_WITH_LIMITATIONS"}:
        if payload["result_status"] == "UNTESTED":
            fail("untested benchmark cannot be released")
        if repro["raw_outputs_retained"] is not True:
            fail("release requires retained raw outputs")
        if payload["splits"].get("protected_test_locked") is not True:
            fail("release requires a protected-test lock")
    print(f"FinanceBench contract OK: {payload['benchmark_id']}")

if __name__ == "__main__":
    main()
