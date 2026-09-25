#!/usr/bin/env python3
"""Validate a FinanceBench v0 evaluation manifest.

This validates structure and fail-closed chronology/claim requirements.
It does not validate the underlying scientific result.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

VALID_CLAIMS = {
    "FORECAST_ONLY",
    "SIMULATION_GROSS",
    "SIMULATION_NET",
    "PAPER_TRADING",
    "LIVE",
}


def fail(message: str) -> None:
    raise SystemExit(f"FINANCEBENCH VALIDATION: FAIL\n- {message}")


def require(obj: dict, key: str, where: str) -> object:
    if key not in obj:
        fail(f"missing {where}.{key}")
    return obj[key]


def parse_date(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        fail(f"{field} must be a non-empty ISO date/time string")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        fail(f"{field} is not valid ISO date/time: {value!r}")
        raise exc


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: validate_financebench_manifest.py <manifest.json>")

    path = Path(sys.argv[1])
    if not path.exists():
        fail(f"manifest not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        fail(f"invalid JSON: {exc}")

    if not isinstance(data, dict):
        fail("manifest root must be an object")

    if require(data, "version", "manifest") != "financebench-v0":
        fail("manifest.version must be 'financebench-v0'")

    project_id = require(data, "project_id", "manifest")
    if not isinstance(project_id, str) or not project_id.strip():
        fail("manifest.project_id must be a non-empty string")

    dataset = require(data, "dataset", "manifest")
    chronology = require(data, "chronology", "manifest")
    evaluation = require(data, "evaluation", "manifest")
    reproduction = require(data, "reproduction", "manifest")

    for name, obj in [
        ("dataset", dataset),
        ("chronology", chronology),
        ("evaluation", evaluation),
        ("reproduction", reproduction),
    ]:
        if not isinstance(obj, dict):
            fail(f"manifest.{name} must be an object")

    for key in ("name", "version", "as_of"):
        value = require(dataset, key, "dataset")
        if not isinstance(value, str) or not value.strip():
            fail(f"dataset.{key} must be a non-empty string")

    train_end = parse_date(require(chronology, "train_end", "chronology"), "chronology.train_end")
    validation_end = parse_date(require(chronology, "validation_end", "chronology"), "chronology.validation_end")
    test_end = parse_date(require(chronology, "test_end", "chronology"), "chronology.test_end")

    if not train_end < validation_end < test_end:
        fail("chronology must satisfy train_end < validation_end < test_end")

    if require(chronology, "point_in_time_verified", "chronology") is not True:
        fail("chronology.point_in_time_verified must be true")

    walk_forward = require(evaluation, "walk_forward", "evaluation")
    if not isinstance(walk_forward, bool):
        fail("evaluation.walk_forward must be boolean")

    regimes = require(evaluation, "regimes", "evaluation")
    if not isinstance(regimes, list) or not regimes:
        fail("evaluation.regimes must be a non-empty list")

    costs = require(evaluation, "transaction_cost_bps", "evaluation")
    if not isinstance(costs, list) or not costs:
        fail("evaluation.transaction_cost_bps must be a non-empty list")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0 for v in costs):
        fail("transaction-cost values must be non-negative numbers")

    baselines = require(data, "baselines", "manifest")
    metrics = require(data, "metrics", "manifest")
    if not isinstance(baselines, list) or not baselines:
        fail("manifest.baselines must be a non-empty list")
    if not isinstance(metrics, list) or not metrics:
        fail("manifest.metrics must be a non-empty list")

    claim = require(data, "claim_boundary", "manifest")
    if claim not in VALID_CLAIMS:
        fail(f"manifest.claim_boundary must be one of {sorted(VALID_CLAIMS)}")

    if claim in {"SIMULATION_NET", "PAPER_TRADING", "LIVE"}:
        if not any(v > 0 for v in costs):
            fail(f"{claim} requires at least one positive transaction-cost assumption")

    if claim in {"PAPER_TRADING", "LIVE"} and not walk_forward:
        fail(f"{claim} requires evaluation.walk_forward=true")

    command = require(reproduction, "command", "reproduction")
    revision = require(reproduction, "source_revision", "reproduction")
    if not isinstance(command, str) or not command.strip():
        fail("reproduction.command must be non-empty")
    if not isinstance(revision, str) or not revision.strip():
        fail("reproduction.source_revision must be non-empty")

    print(
        "FINANCEBENCH VALIDATION: PASS "
        f"(project={project_id}, claim={claim}, "
        f"regimes={len(regimes)}, costs={len(costs)}, baselines={len(baselines)})"
    )


if __name__ == "__main__":
    main()
