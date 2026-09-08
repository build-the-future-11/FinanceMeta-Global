#!/usr/bin/env python3
"""Validate the Five Foundations release and its Jump$tart claim boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIR = ROOT / "resources/five-foundations"
DEFAULT_MANIFEST = RESOURCE_DIR / "resource_manifest.json"

EXPECTED_FILES = {
    "README.md",
    "TEACHER_GUIDE.md",
    "STUDENT_HANDOUT.md",
    "ANSWER_KEY.md",
    "STANDARDS_ALIGNMENT.md",
    "CLEARINGHOUSE_AUDIT.md",
    "SUBMISSION_PACKET.md",
}

REQUIRED_STANDARDS = {
    "Saving 8-5",
    "Saving 12-4",
    "Investing 8-5",
    "Investing 8-7",
    "Investing 12-3",
    "Investing 12-4",
    "Investing 12-6",
    "Managing Credit 8-2",
    "Managing Credit 8-3",
    "Managing Credit 12-1",
    "Managing Credit 12-3",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate(data: dict[str, object], resource_dir: Path = RESOURCE_DIR) -> None:
    require(
        data.get("resource_id") == "FINANCEMETA-FIVE-FOUNDATIONS-2026-v1",
        "resource ID drift",
    )
    require(data.get("version") == "1.0", "resource version drift")
    require(data.get("created_date") == "2026-09-08", "release date drift")
    require(
        data.get("status") == "READY_FOR_PROVIDER_ELIGIBILITY_CONFIRMATION_NOT_SUBMITTED",
        "release must remain explicitly unsubmitted until the provider gate is resolved",
    )

    access = data["access"]
    require(access["price_usd"] == 0, "resource must remain free")
    require(access["account_required"] is False, "resource must not require an account")
    require(
        access["personal_financial_data_required"] is False,
        "resource must not require personal financial data",
    )
    require(
        access["public_main_url_requires_final_logged_out_check"] is True,
        "final public-link verification gate was silently removed",
    )

    boundaries = data["content_boundaries"]
    for key in (
        "financial_advice",
        "individualized_recommendations",
        "specific_security_recommendations",
        "specific_lender_recommendations",
        "specific_account_recommendations",
        "paid_advertising",
        "affiliate_links",
    ):
        require(boundaries[key] is False, f"content boundary weakened: {key}")

    files = set(str(item) for item in data["files"])
    require(files == EXPECTED_FILES, "resource file manifest drift")
    for relative_path in EXPECTED_FILES:
        require((resource_dir / relative_path).is_file(), f"missing resource file: {relative_path}")

    standards = set(str(item) for item in data["national_standards"])
    require(standards == REQUIRED_STANDARDS, "standards mapping drift")

    jumpstart = data["jumpstart"]
    require(
        jumpstart["provider_eligibility"] == "UNRESOLVED_REQUIRES_JUMPSTART_CONFIRMATION",
        "provider eligibility may not be self-approved",
    )
    require(jumpstart["submission_completed"] is False, "submission cannot be claimed before it occurs")
    require(jumpstart["accepted_or_listed"] is False, "Clearinghouse acceptance/listing cannot be preclaimed")
    require(
        jumpstart["submission_receipt_preserved"] is False,
        "receipt cannot be marked preserved before submission",
    )

    criteria = jumpstart["resource_criteria"]
    require(criteria["1_personal_finance_focus"] == "PASS", "personal-finance focus criterion drift")
    require(criteria["2_national_standards_consistency"] == "PASS", "standards criterion drift")
    require(criteria["3_accuracy_and_currency"] == "PASS", "accuracy criterion drift")
    require(criteria["5_balanced_unbiased"] == "PASS", "balance criterion drift")
    require(criteria["6_audience_appropriate"] == "PASS", "audience criterion drift")
    require(criteria["7_respectful_nondiscriminatory"] == "PASS", "respect criterion drift")
    require(
        criteria["8_nationwide_access"] == "CONDITIONAL_FINAL_URL_CHECK",
        "nationwide-access condition may not be silently marked complete",
    )
    require(criteria["9_transparent_access_terms"] == "PASS", "access-terms criterion drift")

    pilot = data["pilot_relationship"]
    require(pilot["frozen_protocol_id"] == "FINANCEMETA-LITERACY-SEP2026-v1", "pilot protocol binding drift")
    require(
        pilot["this_resource_changes_frozen_intervention"] is False,
        "public resource may not mutate the frozen pilot intervention",
    )
    require(
        pilot["substitution_for_pilot_requires_new_protocol_version"] is True,
        "pilot substitution must require a new protocol version",
    )

    readme = (resource_dir / "README.md").read_text()
    audit = (resource_dir / "CLEARINGHOUSE_AUDIT.md").read_text()
    packet = (resource_dir / "SUBMISSION_PACKET.md").read_text()
    require("not financial advice" in readme.lower(), "README advice boundary missing")
    require("NOT SUBMITTED" in audit, "audit must state that the resource is not submitted")
    require("provider eligibility" in audit.lower(), "provider eligibility gate missing from audit")
    require("Preparing this packet is not a submission" in packet, "submission evidence boundary missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    validate(data, args.manifest.parent)
    print(
        "PASS: Five Foundations is complete, standards-mapped, advice-bounded, "
        "and still fail-closed on provider eligibility/submission claims"
    )


if __name__ == "__main__":
    main()
