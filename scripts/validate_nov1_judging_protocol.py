#!/usr/bin/env python3
"""Fail-closed validator for FinanceMeta's internal Nov 1 judging-record protocol."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "operations/nov1-stock-pitch/judging_protocol.json"
SPEC = ROOT / "operations/nov1-stock-pitch/JUDGING_RECORD_SPEC.md"

EXPECTED_WEIGHTS = {
    "thesis_and_reasoning": 30,
    "financial_analysis_and_valuation": 25,
    "evidence_and_citations": 20,
    "risk_assessment": 15,
    "structure_and_writing": 10,
}

REQUIRED_SUBMISSION_EVIDENCE = {
    "submission_id",
    "submission_locked_at_utc",
    "final_artifact_or_platform_snapshot_id",
    "artifact_digest_when_available",
    "cited_source_urls_as_captured_at_submission",
    "ai_assistance_disclosure",
    "integrity_record_id",
    "entry_type_from_final_rulebook",
    "rubric_version",
}

REQUIRED_SCORE_FIELDS = {
    "submission_id",
    "blinded_judge_id",
    "assignment_or_scoring_round_id",
    "rubric_version",
    "calibration_version",
    "conflict_check_status",
    "component_scores",
    "computed_total_score",
    "rationale",
    "integrity_gate_status_at_scoring",
    "scored_at_utc",
    "recusal_or_escalation_state",
    "record_version_or_supersession_state",
}

REQUIRED_RESULTS_RECORDS = {
    "final_partner_approved_brief_and_rubric",
    "submission_ids_and_immutable_artifact_references",
    "all_judge_score_records",
    "conflict_recusal_and_reassignment_records",
    "calibration_version",
    "integrity_decisions_and_reason_codes",
    "aggregation_and_finalist_selection_rule_version",
    "computation_output_used_for_finalist_and_winner_slate",
    "manual_overrides_with_approver_and_reason",
    "final_announced_result_set",
}

REQUIRED_RECUSAL_CATEGORIES = {
    "family_or_household",
    "direct_mentorship_supervision_or_employment",
    "current_close_collaboration",
    "direct_submission_preparation",
    "material_professional_or_financial_conflict",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def normalized_markdown_text(path: Path) -> str:
    """Normalize formatting syntax without weakening the semantic safeguard checks."""
    text = path.read_text().lower()
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate(data: dict[str, object], spec_path: Path = SPEC) -> None:
    require(
        data.get("protocol_id") == "FINANCEMETA-NOV1-JUDGING-RECORDS-2026-v1",
        "judging protocol ID drift",
    )
    require(
        data.get("status") == "FINANCEMETA_INTERNAL_DRAFT_NOT_PARTNER_APPROVED",
        "judging protocol must remain an internal draft until joint partner approval",
    )

    activation = data["activation"]
    require(activation["partner_approved"] is False, "partner approval may not be preclaimed")
    require(activation["active_competition_rule"] is False, "internal draft may not be marked active")
    require(activation["final_partner_brief_required"] is True, "final partner brief gate removed")

    anti = data["anti_lookahead"]
    require(anti["judge_as_of_submission_lock"] is True, "submission-lock judging boundary removed")
    for key in (
        "future_market_performance_scoring_component",
        "post_lock_price_moves_allowed_in_scoring",
        "post_lock_news_or_filings_allowed_in_scoring",
        "post_lock_earnings_or_guidance_allowed_in_scoring",
        "later_consensus_or_analyst_commentary_allowed_in_scoring",
    ):
        require(anti[key] is False, f"anti-lookahead boundary weakened: {key}")
    require(
        anti["unreconstructable_source_action"]
        == "RECORD_LIMITATION_DO_NOT_SUBSTITUTE_LATER_INFORMATION",
        "source reconstruction fail-closed rule drift",
    )

    evidence = data["submission_evidence"]
    require(set(evidence["required_fields"]) == REQUIRED_SUBMISSION_EVIDENCE, "submission evidence fields drift")
    require(evidence["silent_post_deadline_replacement_allowed"] is False, "post-deadline artifact replacement enabled")

    blind = data["blindness_and_conflicts"]
    require(blind["hide_author_identity_from_scorer"] is True, "author blindness removed")
    require(blind["hide_school_from_scorer"] is True, "school blindness removed")
    require(blind["hide_country_from_scorer"] is True, "country blindness removed")
    require(blind["conflict_screen_before_blind_assignment"] is True, "pre-assignment conflict screen removed")
    require(blind["judge_must_stop_if_identity_recognized"] is True, "recognized-identity recusal rule removed")
    require(set(blind["recusal_categories"]) == REQUIRED_RECUSAL_CATEGORIES, "recusal category drift")
    require(
        blind["private_conflict_details_required_in_public_results"] is False,
        "private conflict details cannot be required in public results",
    )
    require(blind["conflict_check_and_recusal_state_must_be_preserved"] is True, "conflict audit trail removed")

    rubric = data["candidate_rubric"]
    require(rubric["partner_approved"] is False, "candidate rubric may not be marked partner-approved")
    require(rubric["weights"] == EXPECTED_WEIGHTS, "candidate scoring weights drift")
    require(sum(rubric["weights"].values()) == 100, "candidate scoring weights must sum to 100")

    integrity = data["integrity_gate"]
    require(integrity["separate_from_merit_score"] is True, "integrity and merit score were conflated")
    require(
        integrity["standalone_ai_detector_sufficient_for_auto_disqualification"] is False,
        "standalone AI detector cannot become sufficient for automatic disqualification",
    )
    require(
        integrity["integrity_hold_may_be_replaced_by_score_penalty"] is False,
        "integrity hold cannot be silently converted into a merit penalty",
    )
    require(integrity["reason_code_required"] is True, "integrity reason code requirement removed")
    require(integrity["reviewer_and_timestamp_required"] is True, "integrity reviewer/timestamp requirement removed")

    scoring = data["scoring_model"]
    require(scoring["preferred_independent_scores_when_capacity_permits"] == 2, "preferred independent score count drift")
    require(scoring["required_independent_scores"] is None, "judge-count requirement cannot be invented before partner decision")
    require(scoring["third_review_for_finalists_or_material_disagreement_preferred"] is True, "third-review preference removed")
    require(scoring["aggregation_rule"] == "UNRESOLVED_PARTNER_DECISION", "aggregation rule was invented")
    require(scoring["finalist_rescore_rule"] == "UNRESOLVED_PARTNER_DECISION", "finalist rescore rule was invented")
    require(scoring["overwrite_individual_scores_with_consensus_allowed"] is False, "individual score audit trail may not be overwritten")

    calibration = data["calibration"]
    for key in (
        "active_rubric_version_required",
        "calibration_version_required",
        "anti_lookahead_instruction_required",
        "conflict_instruction_required",
        "integrity_escalation_instruction_required",
    ):
        require(calibration[key] is True, f"calibration requirement removed: {key}")

    record = data["per_score_record"]
    require(set(record["required_fields"]) == REQUIRED_SCORE_FIELDS, "per-score record fields drift")
    require(record["silent_deletion_or_overwrite_allowed"] is False, "score audit trail may not be silently overwritten")

    packet = data["results_packet"]
    require(set(packet["required_records"]) == REQUIRED_RESULTS_RECORDS, "results reproducibility packet drift")
    require(packet["authorized_reconstruction_must_be_possible"] is True, "results reconstruction requirement removed")

    post = data["post_event_outcomes"]
    require(post["subsequent_returns_may_be_backfilled_into_judging"] is False, "post-outcome leakage into judging enabled")
    require(post["later_market_performance_analysis_requires_separate_frozen_protocol"] is True, "separate post-outcome protocol gate removed")

    require(spec_path.is_file(), "judging record specification missing")
    text = normalized_markdown_text(spec_path)
    for phrase in (
        "not partner approved",
        "future market performance is never a scoring component",
        "do not silently replace the frozen artifact",
        "conflict screening should happen before blind assignment",
        "integrity gate is separate from merit score",
        "do not overwrite one judge's record with a consensus number",
        "freeze a separate post-outcome protocol",
    ):
        require(phrase in text, f"judging specification safeguard missing: {phrase}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", nargs="?", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    data = json.loads(args.protocol.read_text())
    validate(data, args.protocol.parent / "JUDGING_RECORD_SPEC.md")
    print(
        "PASS: Nov 1 judging protocol is anti-lookahead, conflict-auditable, "
        "reproducible, and still fail-closed on partner decisions"
    )


if __name__ == "__main__":
    main()
