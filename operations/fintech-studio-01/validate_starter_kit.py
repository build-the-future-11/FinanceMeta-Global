from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"
BRIEF = ROOT / "BRIEF_TEMPLATE.md"
EVIDENCE = ROOT / "EVIDENCE_TEMPLATE.json"
FINDINGS = ROOT / "FINDINGS_TEMPLATE.md"

SHA40 = re.compile(r"^[0-9a-f]{40}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    for path in (README, BRIEF, EVIDENCE, FINDINGS):
        _require(path.is_file(), f"missing starter-kit file: {path.name}")

    readme = README.read_text(encoding="utf-8")
    brief = BRIEF.read_text(encoding="utf-8")
    findings = FINDINGS.read_text(encoding="utf-8")
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    _require("FORMING / BUILDER SHORTLIST OPEN" in readme, "README launch boundary drifted")
    _require("Do **not** call Studio 01 launched" in readme, "README must remain fail-closed on launch status")
    _require("before implementation or primary-outcome inspection" in brief, "brief freeze boundary missing")
    _require("Failure condition" in brief, "brief must retain a stop/falsification condition")
    _require("financial-advice boundary" in brief, "brief must retain the financial-advice guardrail")
    _require("What failed" in findings, "findings must retain failures")
    _require("STOP_INVALID_EVALUATION" in findings, "findings must permit invalid-evaluation stop")
    _require("does **not** establish" in findings, "findings must require an explicit negative claim boundary")

    _require(evidence.get("schema_version") == "1.0.0", "unexpected evidence schema version")
    _require(evidence.get("status") == "template_not_executed", "starter evidence must remain unexecuted")
    _require(evidence["evaluation"]["result"] is None, "template must not contain a fabricated result")
    _require(evidence["evaluation"]["passed_frozen_rule"] is None, "template must not preclaim success")
    _require(evidence["testing"]["automated_test_count"] is None, "template must not preclaim tests")
    _require(evidence["review"]["status"] == "not_reviewed", "template must not preclaim review")

    for field in ("source_commit_sha", "brief_freeze_commit_sha"):
        value = evidence["project"][field]
        _require(value is None or SHA40.fullmatch(value) is not None, f"invalid {field}")

    # Reject affirmative promotional claims while allowing the templates to name
    # those same risks inside explicit guardrails/disclaimers.
    combined = "\n".join((readme, brief, findings)).lower()
    affirmative_claims = (
        "status: launched",
        "studio 01 is launched",
        "guaranteed profit",
        "proven trading strategy",
        "provides personalized investment advice",
        "offers personalized investment advice",
    )
    for phrase in affirmative_claims:
        _require(phrase not in combined, f"prohibited affirmative claim present: {phrase}")

    print("PASS_FINTECH_STUDIO_01_STARTER_KIT")


if __name__ == "__main__":
    main()
