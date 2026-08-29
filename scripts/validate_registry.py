#!/usr/bin/env python3
"""Validate FinanceMeta evidence registries without promoting any claim."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = ROOT / "registry" / "programs.json"
PROJECTS = ROOT / "registry" / "projects.json"

EVIDENCE = {f"E{i}" for i in range(6)}
MATURITY = {f"M{i}" for i in range(6)}
PROGRAM_STATUS = {
    "planned_until_evidence_record",
    "ready",
    "active",
    "at_risk",
    "complete",
    "archived",
}
PROJECT_STATUS = {
    "proposal_stub",
    "executable",
    "internal_evidence",
    "external_data",
    "external_review",
    "external_outcome",
    "archived",
}


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing registry: {path.relative_to(ROOT)}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path.relative_to(ROOT)}: {exc}") from exc


def unique_ids(records: list[dict], label: str) -> None:
    ids = [r.get("id") for r in records]
    if any(not isinstance(v, str) or not v.strip() for v in ids):
        raise SystemExit(f"{label}: every record needs a non-empty string id")
    duplicates = sorted({v for v in ids if ids.count(v) > 1})
    if duplicates:
        raise SystemExit(f"{label}: duplicate ids: {duplicates}")


def main() -> None:
    programs = load(PROGRAMS).get("programs")
    projects = load(PROJECTS).get("projects")
    if not isinstance(programs, list) or not isinstance(projects, list):
        raise SystemExit("registry documents must contain list-valued programs/projects")

    unique_ids(programs, "programs")
    unique_ids(projects, "projects")
    errors: list[str] = []

    for p in programs:
        pid = p["id"]
        if p.get("status") not in PROGRAM_STATUS:
            errors.append(f"program {pid}: invalid status {p.get('status')!r}")
        if p.get("minimum_evidence_level") not in EVIDENCE - {"E0"}:
            errors.append(f"program {pid}: minimum_evidence_level must be E1–E5")
        if not str(p.get("launch_gate", "")).strip():
            errors.append(f"program {pid}: missing launch_gate")
        if p.get("status") == "planned_until_evidence_record" and p.get("evidence_record"):
            errors.append(f"program {pid}: planned status conflicts with linked evidence_record; review promotion explicitly")

    for p in projects:
        pid = p["id"]
        maturity = p.get("maturity")
        evidence = p.get("evidence_level")
        status = p.get("status")
        state = p.get("verified_repository_state")
        if maturity not in MATURITY:
            errors.append(f"project {pid}: invalid maturity {maturity!r}")
        if evidence not in EVIDENCE:
            errors.append(f"project {pid}: invalid evidence level {evidence!r}")
        if status not in PROJECT_STATUS:
            errors.append(f"project {pid}: invalid status {status!r}")
        if not str(p.get("claim_boundary", "")).strip():
            errors.append(f"project {pid}: missing claim_boundary")
        if not str(p.get("next_gate", "")).strip():
            errors.append(f"project {pid}: missing next_gate")
        if not isinstance(state, dict):
            errors.append(f"project {pid}: verified_repository_state must be an object")
            continue
        required_state = {
            "readme_present",
            "license_present",
            "implementation_present",
            "tests_present",
            "results_present",
            "reproduction_command_present",
        }
        missing = sorted(required_state - set(state))
        if missing:
            errors.append(f"project {pid}: repository-state fields missing: {missing}")
        if status == "proposal_stub":
            if maturity != "M0" or evidence != "E0":
                errors.append(f"project {pid}: proposal_stub must remain M0/E0")
            if state.get("implementation_present") or state.get("results_present"):
                errors.append(f"project {pid}: proposal_stub conflicts with implementation/results flags")
        if maturity != "M0" and not state.get("implementation_present"):
            errors.append(f"project {pid}: maturity above M0 requires implementation_present=true")
        if maturity in {"M2", "M3", "M4", "M5"}:
            for field in ("tests_present", "results_present", "reproduction_command_present"):
                if not state.get(field):
                    errors.append(f"project {pid}: {maturity} requires {field}=true")
        path = p.get("path")
        if not isinstance(path, str) or not path.strip():
            errors.append(f"project {pid}: missing repository path")
        elif not (ROOT / path).exists():
            errors.append(f"project {pid}: registered path does not exist: {path}")

    if errors:
        raise SystemExit("registry validation failed:\n- " + "\n- ".join(errors))

    print(f"REGISTRY VALIDATION: PASS ({len(programs)} programs, {len(projects)} projects)")


if __name__ == "__main__":
    main()
