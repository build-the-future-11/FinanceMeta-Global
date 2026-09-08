#!/usr/bin/env python3
"""Fail-closed validator for FinanceMeta's internal November 2026 stock-pitch position."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSITION = ROOT / "operations/nov1-stock-pitch/financemeta_position.json"

EXPECTED_OPEN_ITEMS = {
    "entry_fee_or_free_entry",
    "cash_prize_amount_and_funding_source",
    "complete_prize_table",
    "individual_or_team_entry",
    "memo_word_cap",
    "eligibility_scope",
    "judge_count_per_partner",
    "scoring_window_and_results_date",
    "promotion_launch_and_reminder_schedule",
    "escalation_contact_per_partner",
    "final_name_and_cobranding",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate(data: dict[str, object]) -> None:
    require(
        data.get("position_id") == "FINANCEMETA-NOV1-STOCK-PITCH-2026-v1",
        "position ID drift",
    )
    status = str(data.get("status", ""))
    require(
        status == "FINANCEMETA_INTERNAL_POSITION_NOT_PARTNER_APPROVED",
        "internal position must never masquerade as partner-approved",
    )

    source = data["source_brief"]
    require(source["version"] == "v1.0", "source brief version drift")
    require(source["date"] == "2026-08-17", "source brief date drift")
    require(source["open_date"] == "2026-11-01", "November 1 open date drift")
    require(source["submission_window"] == "one_month", "one-month submission window drift")

    boundaries = data["ownership_boundaries"]
    require(boundaries["platform_and_registration"] == "Empiric", "Empiric platform/registration ownership drift")
    require(boundaries["judging_infrastructure"] == "Empiric", "Empiric judging-infrastructure ownership drift")
    require(boundaries["all_partner_signoff_required_before_promotion"] is True, "partner sign-off gate weakened")

    recommendations = data["recommendations"]
    fee = recommendations["entry_fee"]
    require(fee["position"] == "FREE_FIRST_YEAR", "FinanceMeta free-first-year position drift")
    require(fee["requires_partner_signoff"] is True, "fee position must remain explicitly partner-governed")
    require(recommendations["entry_format"] == "INDIVIDUAL", "FinanceMeta individual-entry recommendation drift")
    require(int(recommendations["memo_word_cap"]) == 2000, "FinanceMeta memo-cap recommendation drift")

    scoring = recommendations["scoring"]
    require(sum(int(value) for value in scoring.values()) == 100, "scoring weights must sum to 100")
    require(int(scoring["thesis_and_reasoning"]) == 30, "thesis/reasoning weight drift")
    require(int(scoring["financial_analysis_and_valuation"]) == 25, "financial-analysis weight drift")
    require(int(scoring["evidence_and_citations"]) == 20, "evidence/citations weight drift")
    require(int(scoring["risk_assessment"]) == 15, "risk weight drift")
    require(int(scoring["structure_and_writing"]) == 10, "structure/writing weight drift")

    integrity = data["integrity"]
    for key in (
        "in_platform_authoring_required",
        "paste_events_and_history_auditable",
        "ai_editing_or_proofreading_allowed_only_if_disclosed",
        "standalone_ai_detector_not_sufficient_for_disqualification",
    ):
        require(integrity[key] is True, f"integrity safeguard weakened: {key}")
    require(integrity["post_deadline_edits_allowed"] is False, "post-deadline edits must remain disallowed")
    require(
        integrity["future_market_performance_is_scoring_component"] is False,
        "future market performance must not become a scoring component",
    )

    judging = data["judging"]
    require(judging["blind_scoring"] is True, "blind-scoring boundary weakened")
    require(judging["conflict_disclosure_and_recusal"] is True, "judge conflict safeguard weakened")
    require(
        judging["finance_meta_commitment_rule"] == "only_named_confirmed_judges_count_as_committed",
        "judge commitment must remain evidence-based",
    )

    promotion = data["promotion"]
    not_before = date.fromisoformat(str(promotion["not_before"]))
    require(not_before >= date(2026, 10, 16), "promotion may not begin before the September event closes")
    require(str(promotion["open_date"]) == "2026-11-01", "promotion/open-date mismatch")
    require(promotion["no_unapproved_prize_claims"] is True, "unapproved prize claims must remain blocked")
    require(promotion["final_partner_brief_required"] is True, "final partner brief gate weakened")

    open_items = set(str(item) for item in data["open_partner_decisions"])
    require(open_items == EXPECTED_OPEN_ITEMS, "open partner-decision set drifted or was silently resolved")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("position", nargs="?", type=Path, default=DEFAULT_POSITION)
    args = parser.parse_args()
    data = json.loads(args.position.read_text())
    validate(data)
    print("PASS: FinanceMeta November position remains internal, partner-gated, integrity-bounded, and promotion-safe")


if __name__ == "__main__":
    main()
