#!/usr/bin/env python3
"""Recompute a descriptive comparison from one hash-pinned, closed FI-JEPA artifact.

This reads retained outcomes. It never trains, retests a model, selects seeds,
changes a study, or converts macroeconomic forecasting into trading returns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SOURCE_COMMIT = "c94e1616eba1b2a7415ead4695def1d3b93094db"
SOURCE_SHA256 = "525fe5f141bba765b3346c1aca3b327303703e5f98142b0f25d664211e10f7c8"
SOURCE_URL = f"https://github.com/Finance-Meta-Research/FI-JEPA/blob/{SOURCE_COMMIT}/experiments/paper_results.json"
PROTOCOL = "FIJEPA_MACRODATA_PAPER_V1_20260909"


def audit(path: Path) -> dict:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("artifact hash differs from the retained canonical v1 source; no substitution allowed")
    data = json.loads(raw)
    if data["protocol_id"] != PROTOCOL or data["status"] != "EXECUTED_CANONICAL_PAPER_EVIDENCE":
        raise ValueError("unexpected canonical protocol or artifact status")
    if data["seed_list"] != [7, 17, 27]:
        raise ValueError("the complete predeclared seed list must be retained")
    rows = []
    for seed in data["seed_list"]:
        values = {}
        for variant in ("full", "raw_context_ridge", "no_operator_split"):
            matches = [r for r in data["runs"] if r["seed"] == seed and r["variant"] == variant]
            if len(matches) != 1:
                raise ValueError(f"expected one retained {variant}/seed{seed}")
            values[variant] = matches[0]["metrics"]["probe_regression_mse"]
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("invalid retained MSE")
        rows.append({"seed": seed, **values,
                     "full_minus_ridge": values["full"] - values["raw_context_ridge"],
                     "full_minus_no_operator_split": values["full"] - values["no_operator_split"]})
    mean = math.fsum(row["full_minus_ridge"] for row in rows) / len(rows)
    reported = data["paired_seed_level_statistics"]["full_latent_minus_raw_context_ridge_mse"]
    if not math.isclose(mean, reported["mean_delta"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("reported aggregate disagrees with retained seed-level values")
    return {
        "audit_status": "PASS", "scientific_state": "CLOSED NEGATIVE / BOUNDARY",
        "source_url": SOURCE_URL, "source_commit": SOURCE_COMMIT, "source_sha256": digest,
        "protocol_id": PROTOCOL, "source_generated_utc": data["generated_utc"],
        "data_kind": "retained outcomes on statsmodels macrodata; not a trading ledger",
        "seed_rows": rows, "mean_full_minus_ridge_mse": mean,
        "reported_bootstrap_95ci_not_recomputed": reported["bootstrap_95ci"],
        "all_retained_full_mse_worse_than_ridge": all(r["full_minus_ridge"] > 0 for r in rows),
        "full_and_no_operator_split_mse_identical_all_seeds": all(r["full_minus_no_operator_split"] == 0 for r in rows),
        "limitations": [
            "Arithmetic/provenance replay of retained evidence; original dataset/model execution is not revalidated.",
            "The operator-removal condition is non-identifying in this one-stage configuration.",
            "Three seeds support a bounded descriptive comparison, not a general significance claim.",
            "FI-JEPA v1 remains closed negative/boundary; this does not authorize or execute a successor.",
            "No observed trading ledger was supplied and no trading profitability is claimed.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"audit_status": report["audit_status"], "scientific_state": report["scientific_state"],
                      "mean_full_minus_ridge_mse": report["mean_full_minus_ridge_mse"]}))
