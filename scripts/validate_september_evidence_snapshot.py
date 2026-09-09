#!/usr/bin/env python3
"""Fail-closed validator for FinanceMeta's September external-proof evidence snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "operations/september-2026/evidence_snapshot.json"
LEDGER = ROOT / "operations/september-2026/EVIDENCE_LEDGER.md"

REQUIRED_METRICS = {
    "registrations",
    "active_builders_or_teams",
    "completions",
    "submitted_artifacts",
    "external_reviewers_actually_participated",
    "sponsor_contributions_actually_used",
    "failures_data_issues_or_complaints",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate(data: dict[str, object], ledger_path: Path = LEDGER) -> None:
    require(
        data.get("snapshot_id") == "FINANCEMETA-SEPTEMBER-2026-EVIDENCE-20260909-v1",
        "snapshot ID drift",
    )
    require(data.get("status") == "IN_PROGRESS_FAIL_CLOSED", "snapshot must remain in-progress/fail-closed")

    window = data["reporting_window"]
    require(window["start"] == "2026-09-01", "reporting-window start drift")
    require(window["end"] == "2026-09-30", "reporting-window end drift")
    require(data["snapshot_as_of"] == "2026-09-09", "snapshot date drift")

    policy = data["metric_policy"]
    for key in (
        "unknown_is_zero",
        "planning_target_is_registration",
        "registration_is_active_participant",
        "active_is_completion",
        "provisioned_key_is_usage",
        "sponsor_offer_is_used_contribution",
        "scheduled_reviewer_is_participated_reviewer",
        "prepared_submission_is_external_submission",
        "external_submission_is_acceptance",
        "scheduled_call_is_delivered_collaboration",
        "internal_artifact_is_external_outcome",
    ):
        require(policy[key] is False, f"evidence accounting boundary weakened: {key}")

    metrics = data["metrics"]
    require(set(metrics) == REQUIRED_METRICS, "required metric set drift")
    for metric_id, record in metrics.items():
        value = record["value"]
        refs = record["evidence_refs"]
        rule = str(record["counting_rule"]).strip()
        require(rule, f"missing counting rule: {metric_id}")
        require(isinstance(refs, list), f"evidence refs must be a list: {metric_id}")
        if value is None:
            require(refs == [], f"unsupported metric must not carry misleading evidence refs: {metric_id}")
        else:
            require(isinstance(value, int) and value >= 0, f"metric must be a non-negative integer: {metric_id}")
            require(len(refs) > 0, f"reported metric requires preserved evidence: {metric_id}")
            require("UNRESOLVED" not in rule, f"reported metric cannot use unresolved counting rule: {metric_id}")

    workstreams = data["workstreams"]
    fmp = workstreams["fmp_buildathon"]
    require(fmp["state"] == "TECHNICAL_TERMS_CONFIRMED_MANAGEMENT_APPROVAL_PENDING", "FMP state drift")
    require(fmp["planning_team_estimate"] == {"min": 15, "max": 25, "counted_as_registration": False}, "FMP planning estimate boundary drift")
    require(fmp["external_outcome_counted"] is False, "FMP planning work cannot be counted as external outcome")

    haven = workstreams["haven"]
    require(haven["state"] == "PRECALL_PACKET_PREPARED_CALL_NOT_YET_DELIVERED", "HAVEN state drift")
    require(haven["external_outcome_counted"] is False, "future HAVEN call cannot be counted as delivered")

    resource = workstreams["five_foundations"]
    require(resource["external_submission_completed"] is False, "Jump$tart submission cannot be preclaimed")
    require(resource["accepted_or_listed"] is False, "Jump$tart acceptance/listing cannot be preclaimed")
    require(resource["canonical_production_verified"] is False, "canonical production cannot be preclaimed")
    require(resource["external_outcome_counted"] is False, "resource preparation cannot be counted as external outcome")

    nov1 = workstreams["nov1_stock_pitch"]
    require(
        nov1["state"] == "FINANCEMETA_INTERNAL_POSITION_AND_JUDGING_PROTOCOL_NOT_PARTNER_APPROVED",
        "Nov 1 governance state drift",
    )
    require(nov1["final_partner_rulebook"] is False, "final partner rulebook cannot be preclaimed")
    require(nov1["registrations_or_submissions_counted"] is False, "Nov 1 registrations/submissions cannot be preclaimed")
    require(nov1["judging_activity_counted"] is False, "Nov 1 judging activity cannot be preclaimed")
    require(nov1["external_outcome_counted"] is False, "Nov 1 internal work cannot be counted as external outcome")

    reporting = data["reporting_requirements"]
    for key in (
        "source_window_required",
        "source_timestamp_required",
        "counting_rule_version_required",
        "dedupe_and_exclusions_required",
        "limitations_required",
        "failures_and_complaints_required",
        "authorized_reconstruction_required",
    ):
        require(reporting[key] is True, f"reporting requirement removed: {key}")

    require(ledger_path.is_file(), "evidence ledger missing")
    ledger = ledger_path.read_text().lower()
    for phrase in (
        "unknown is recorded as **unknown**, not `0`",
        "planning target ↔ registration",
        "provisioned api key ↔ actual api usage",
        "scheduled reviewer/judge ↔ reviewer/judge who actually scored work",
        "merged internal documentation ↔ external program outcome",
        "if the source cannot be preserved, the metric remains unsupported",
    ):
        require(phrase.lower() in ledger, f"evidence-ledger safeguard missing: {phrase}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", nargs="?", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    data = json.loads(args.snapshot.read_text())
    validate(data, args.snapshot.parent / "EVIDENCE_LEDGER.md")
    print("PASS: September evidence snapshot remains source-backed, fail-closed, and outcome-safe")


if __name__ == "__main__":
    main()
