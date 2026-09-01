#!/usr/bin/env python3
"""Fail-closed validator for the September 2026 FinanceMeta literacy pilot protocol."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "evaluation/september-2026-financial-literacy-pilot/protocol.json"
INTERVENTION = ROOT / "evaluation/september-2026-financial-literacy-pilot/INTERVENTION.md"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_items(items: list[dict[str, object]], expected_count: int, prefix: str) -> None:
    require(len(items) == expected_count, f"{prefix}: expected {expected_count} frozen items")
    ids = [str(item.get("id", "")) for item in items]
    require(len(ids) == len(set(ids)), f"{prefix}: duplicate item IDs")
    require(all(ids), f"{prefix}: missing item ID")

    for item in items:
        choices = item.get("choices")
        require(isinstance(choices, list) and len(choices) >= 4, f"{prefix}: every item needs >=4 choices")
        index = item.get("correct_index")
        require(isinstance(index, int) and 0 <= index < len(choices), f"{prefix}: invalid correct_index")
        prompt = str(item.get("prompt", "")).strip()
        require(bool(prompt), f"{prefix}: missing prompt")


def main() -> None:
    require(PROTOCOL.exists(), "protocol.json missing")
    require(INTERVENTION.exists(), "frozen intervention document missing")

    data = json.loads(PROTOCOL.read_text())

    require(data["protocol_id"] == "FINANCEMETA-LITERACY-SEP2026-v1", "protocol ID drift")
    require(data["status"] == "FROZEN_PRE_PILOT", "protocol must remain frozen pre-pilot")
    require(data["frozen_date"] == "2026-09-01", "freeze date drift")

    claims = data["claim_boundary"]
    allowed = str(claims["allowed_if_gate_passes"]).lower()
    forbidden = [str(value).lower() for value in claims["forbidden"]]
    require("preliminary" in allowed and "non-causal" in allowed, "allowed claim must remain preliminary/non-causal")
    require(any("finra" in value and ("endorsement" in value or "validation" in value) for value in forbidden), "FINRA no-endorsement boundary missing")
    require(any("causal" in value for value in forbidden), "causal-effectiveness claim must remain forbidden")

    population = data["population"]
    require(int(population["minimum_age"]) >= 13, "minimum age unexpectedly weakened")
    require("consent" in str(population["consent"]).lower(), "consent requirement missing")
    privacy = str(population["privacy"]).lower()
    require("pseudonymous" in privacy, "pseudonymous ID requirement missing")
    for sensitive_term in ("household income", "account balances", "bank details", "investment holdings"):
        require(sensitive_term in privacy, f"data-minimization boundary missing: {sensitive_term}")

    intervention = data["intervention"]
    require(intervention["version"] == "FINANCEMETA-LITERACY-MODULE-SEP2026-v1", "intervention version drift")
    require(intervention["content_path"] == "evaluation/september-2026-financial-literacy-pilot/INTERVENTION.md", "intervention path drift")
    require(int(intervention["duration_minutes"]) == 35, "intervention duration drift")
    require("new_protocol_version" in str(intervention["changes_after_first_participant"]), "post-start intervention mutation rule weakened")

    assessment = data["assessment"]
    note = str(assessment["note"]).lower()
    require("original" in note and "not canonical gflec" in note, "instrument-equivalence boundary missing")
    validate_items(assessment["pre_form"], 5, "pre_form")
    validate_items(assessment["post_form"], 5, "post_form")
    validate_items(assessment["scenario_transfer"], 2, "scenario_transfer")
    require("0-100" in str(assessment["confidence_scale"]), "confidence scale drift")

    analysis = data["analysis"]
    require(str(analysis["primary_metric"]).lower().startswith("paired"), "primary metric must remain paired")
    require(analysis["subgroup_analyses"] == [], "unfrozen subgroup analysis added")
    require(int(analysis["minimum_cohort_for_promising_label"]) == 20, "minimum cohort threshold drift")
    missing = analysis["missing_data"]
    require("complete-pair" in str(missing["primary"]).lower(), "complete-pair rule missing")
    require("never impute" in str(missing["primary"]).lower(), "missing post-test fail-closed rule missing")
    require("dropout" in str(missing["reporting"]).lower(), "dropout reporting requirement missing")
    require("report" in str(analysis["exclusions"]).lower(), "exclusion accounting missing")

    decision = data["decision_rule"]
    promising = "\n".join(str(value).lower() for value in decision["promising_preliminary_evidence_requires_all"])
    for required_text in ("20 paired completers", "25%", "+10 percentage points", "+1 of 5", "60%", "integrity or consent"):
        require(required_text.lower() in promising, f"promising-evidence gate drift: {required_text}")
    require(len(decision["inconclusive_if"]) >= 3, "inconclusive decision path weakened")
    require(len(decision["redesign_required_if"]) >= 5, "redesign/failure decision path weakened")

    reporting = data["reporting"]
    required_reporting = [str(value).lower() for value in reporting["required"]]
    for required_term in ("protocol version", "participant flow", "missing-data", "failure cases"):
        require(any(required_term in value for value in required_reporting), f"reporting requirement missing: {required_term}")
    prohibited = str(reporting["prohibited"]).lower()
    require("participant-level identities" in prohibited, "identity-publication prohibition missing")
    require("attendance" in prohibited and "learning outcomes" in prohibited, "outputs-vs-outcomes boundary missing")

    intervention_text = INTERVENTION.read_text().lower()
    for phrase in (
        "not financial advice",
        "do not add a product pitch",
        "do not selectively reteach",
        "pseudonymous participant ids",
    ):
        require(phrase in intervention_text, f"intervention safeguard missing: {phrase}")

    print("PASS: September literacy pilot protocol is frozen, non-causal, privacy-bounded, and pre-outcome")


if __name__ == "__main__":
    main()
