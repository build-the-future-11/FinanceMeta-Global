#!/usr/bin/env python3
"""Fail-closed validator for the FinanceMeta FMP buildathon operating contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "operations/fintech-studio-buildathon/fmp_contract.json"
BRIEF = ROOT / "operations/fintech-studio-buildathon/FMP_TECHNICAL_BRIEF.md"

EXPECTED_ATTRIBUTION = "Financial Data Powered by FMP"
EXPECTED_DIMENSIONS = {
    "usefulness",
    "correctness",
    "finance_economics_depth",
    "technical_execution",
    "validation",
    "communication",
    "continuation_potential",
}
EXPECTED_REQUIREMENTS = {
    "user_or_problem",
    "input_data",
    "system_or_output",
    "validation_plan",
    "runnable_or_reviewable_demo_or_artifact",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate(data: dict[str, object], brief_path: Path = BRIEF) -> None:
    require(
        data.get("contract_id") == "FINANCEMETA-FINTECH-BUILDATHON-FMP-SEP2026-v1",
        "contract ID drift",
    )
    require(
        data.get("status") == "PRELAUNCH_MANAGEMENT_APPROVAL_PENDING",
        "contract must remain prelaunch until explicit approval/date gates are resolved",
    )

    event = data["event"]
    require(event["window"] == "2026-11", "event window drift")
    require(event["exact_start_date"] is None, "exact start date may not be invented")
    require(event["exact_end_date"] is None, "exact end date may not be invented")
    require(event["duration_days_min"] == 7 and event["duration_days_max"] == 10, "duration drift")
    require(event["format"] == "ONLINE", "format drift")
    require(event["planning_team_estimate_min"] == 15, "planning minimum drift")
    require(event["planning_team_estimate_max"] == 25, "planning maximum drift")
    require(event["registered_team_count"] is None, "registration count cannot be preclaimed")

    fmp = data["fmp"]
    require(fmp["public_partner_claim_allowed"] is False, "FMP partner/sponsor claim prematurely enabled")
    require(fmp["management_approval_preserved"] is False, "management approval cannot be preclaimed")
    require(fmp["logo_use_allowed"] is False, "brand/logo permission cannot be preclaimed")
    require(fmp["keys_guaranteed_to_participants"] is False, "keys may not be guaranteed before final approval")
    require(fmp["technical_terms_confirmed"] is True, "confirmed technical terms were lost")
    require(fmp["datasets"] == "ALL_FMP_DATASETS_EXCLUDING_REAL_TIME_QUOTES", "dataset boundary drift")
    require(fmp["real_time_quotes_allowed"] is False, "real-time quote access must remain prohibited")
    require(fmp["rate_limit_calls_per_minute"] == 300, "rate-limit drift")
    require(fmp["bandwidth_cap_gb"] == 20, "bandwidth cap drift")
    require(fmp["key_model"] == "ONE_EVENT_ONLY_API_KEY_PER_TEAM", "key model drift")
    require(fmp["access_duration"] == "EVENT_ONLY", "event-only access boundary drift")
    require(fmp["required_attribution"] == EXPECTED_ATTRIBUTION, "required FMP attribution drift")
    require(fmp["endpoint_tier_finalized"] is False, "endpoint tier cannot be preclaimed as final")
    require(fmp["provisioning_process_finalized"] is False, "provisioning process cannot be preclaimed")
    require(fmp["expiry_process_finalized"] is False, "expiry process cannot be preclaimed")

    participant = data["participant_rules"]
    require(participant["cross_team_key_sharing_allowed"] is False, "cross-team key sharing was enabled")
    require(participant["public_key_exposure_allowed"] is False, "public credential exposure was enabled")
    require(
        participant["unrelated_personal_or_commercial_use_allowed"] is False,
        "event key scope was weakened",
    )
    require(
        participant["secret_storage_expected"] == "ENVIRONMENT_VARIABLE_OR_SERVER_SIDE_SECRET",
        "secret-storage expectation drift",
    )
    require(participant["post_event_secret_removal_required"] is True, "post-event secret cleanup was weakened")
    require(
        participant["attribution_required_on_public_fmp_data_outputs"] is True,
        "public attribution requirement was weakened",
    )

    requirements = set(str(value) for value in data["submission_requirements"])
    require(requirements == EXPECTED_REQUIREMENTS, "minimum project-discipline requirements drift")

    judging = data["judging"]
    require(set(str(value) for value in judging["dimensions"]) == EXPECTED_DIMENSIONS, "judging dimensions drift")
    require(judging["weights_frozen"] is False, "judging weights cannot be preclaimed")
    require(judging["conflict_policy_frozen"] is False, "conflict policy cannot be preclaimed")
    require(judging["tie_policy_frozen"] is False, "tie policy cannot be preclaimed")
    require(judging["final_rulebook_frozen"] is False, "final rulebook cannot be preclaimed")

    gates = data["launch_gates"]
    for unresolved in (
        "exact_dates_frozen",
        "public_event_page_frozen",
        "eligibility_and_team_rules_frozen",
        "registered_team_count_frozen",
    ):
        require(gates[unresolved] is False, f"unresolved launch gate prematurely marked complete: {unresolved}")
    require(gates["participant_technical_brief_contains_attribution"] is True, "technical brief attribution gate missing")
    require(gates["fmp_management_approval_required"] is True, "management approval gate removed")
    require(gates["fmp_brand_asset_permission_required"] is True, "brand permission gate removed")
    require(gates["judging_rulebook_required_before_submissions"] is True, "rulebook gate removed")

    outcomes = data["outcome_accounting"]
    require(outcomes["planning_estimate_is_registration"] is False, "planning estimates cannot count as registrations")
    require(outcomes["provisioned_key_is_usage"] is False, "provisioned keys cannot count as actual usage")
    require(outcomes["warm_reply_is_delivered_outcome"] is False, "warm replies cannot count as delivered outcomes")
    for key in (
        "track_registered_teams",
        "track_keys_issued",
        "track_teams_using_fmp",
        "track_active_teams",
        "track_completed_prototypes",
        "track_submitted_artifacts",
        "track_failures_and_incidents",
    ):
        require(outcomes[key] is True, f"required outcome accounting missing: {key}")

    require(brief_path.is_file(), "participant technical brief missing")
    brief = brief_path.read_text()
    require(EXPECTED_ATTRIBUTION in brief, "exact FMP attribution missing from technical brief")
    require("do **not** call FMP a confirmed sponsor" in brief, "public partner-claim boundary missing")
    require("Real-time quotes | Not permitted" in brief, "real-time prohibition missing")
    require("15–25 teams" in brief, "latest bounded planning estimate missing")
    require("Weights are not frozen" in brief, "judging-weight boundary missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", nargs="?", type=Path, default=CONTRACT)
    args = parser.parse_args()
    data = json.loads(args.contract.read_text())
    validate(data, args.contract.parent / "FMP_TECHNICAL_BRIEF.md")
    print(
        "PASS: FMP buildathon contract preserves technical limits, secret handling, "
        "and fail-closed approval/date/outcome boundaries"
    )


if __name__ == "__main__":
    main()
