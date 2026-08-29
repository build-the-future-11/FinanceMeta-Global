#!/usr/bin/env python3
"""Validate FinanceMeta's machine-readable evidence registries.

This validator intentionally checks structure and conservative state semantics;
it does not promote any project or program based on unverified claims.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = ROOT / "registry" / "programs.json"
PROJECTS = ROOT / "registry" / "projects.json"

EVIDENCE = {"E0", "E1", "E2", "E3", "E4", "E5", "UNKNOWN"}
PROGRAM_STATUS = {
    "PLANNED_UNVERIFIED",
    "READY",
    "ACTIVE",
    "AT_RISK",
    "COMPLETE",
    "ARCHIVED",
}
PROJECT_STATUS = {
    "UNAUDITED",
    "PROPOSAL",
    "EXECUTABLE",
    "INTERNAL_EVIDENCE",
    "EXTERNAL_DATA",
    "EXTERNAL_REVIEW",
    "EXTERNAL_OUTCOME",
    "ARCHIVED",
}
RESULT_STATUS = {"UNVERIFIED", "UNTESTED", "POSITIVE", "NEGATIVE", "INCONCLUSIVE"}


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing registry: {path.relative_to(ROOT)}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path.relative_to(ROOT)}: {exc}") from exc


def require_unique(records: list[dict], label: str) -> None:
    ids = [r.get("id") for r in records]
    missing = [i for i, value in enumerate(ids) if not isinstance(value, str) or not value.strip()]
    if missing:
        raise SystemExit(f"{label}: missing/invalid id at indices {missing}")
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise SystemExit(f"{label}: duplicate ids: {duplicates}")


def main() -> None:
    programs_doc = load(PROGRAMS)
    projects_doc = load(PROJECTS)
    programs = programs_doc.get("programs")
    projects = projects_doc.get("projects")
    if not isinstance(programs, list) or not isinstance(projects, list):
        raise SystemExit("registries must contain list-valued 'programs' and 'projects'")

    require_unique(programs, "programs")
    require_unique(projects, "projects")
    program_ids = {p["id"] for p in programs}

    errors: list[str] = []
    for p in programs:
        if p.get("status") not in PROGRAM_STATUS:
            errors.append(f"program {p['id']}: invalid status {p.get('status')!r}")
        if p.get("evidence_level") not in EVIDENCE - {"UNKNOWN"}:
            errors.append(f"program {p['id']}: invalid evidence_level {p.get('evidence_level')!r}")
        if not str(p.get("claim_boundary", "")).strip():
            errors.append(f"program {p['id']}: missing claim_boundary")
        records = p.get("evidence_records")
        if not isinstance(records, list):
            errors.append(f"program {p['id']}: evidence_records must be a list")
        if p.get("status") == "PLANNED_UNVERIFIED" and p.get("evidence_level") != "E0":
            errors.append(f"program {p['id']}: planned/unverified programs must remain E0")
        if p.get("status") in {"ACTIVE", "COMPLETE"} and not records:
            errors.append(f"program {p['id']}: {p.get('status')} requires at least one linked evidence record")

    for p in projects:
        if p.get("status") not in PROJECT_STATUS:
            errors.append(f"project {p['id']}: invalid status {p.get('status')!r}")
        if p.get("evidence_level") not in EVIDENCE:
            errors.append(f"project {p['id']}: invalid evidence_level {p.get('evidence_level')!r}")
        if p.get("result_status") not in RESULT_STATUS:
            errors.append(f"project {p['id']}: invalid result_status {p.get('result_status')!r}")
        if p.get("program") not in program_ids:
            errors.append(f"project {p['id']}: unknown program {p.get('program')!r}")
        if not str(p.get("claim_boundary", "")).strip():
            errors.append(f"project {p['id']}: missing claim_boundary")
        if p.get("status") == "UNAUDITED":
            if p.get("evidence_level") != "UNKNOWN" or p.get("result_status") != "UNVERIFIED":
                errors.append(f"project {p['id']}: UNAUDITED must remain UNKNOWN / UNVERIFIED")
        if p.get("evidence_level") in {"E4", "E5"} and not p.get("evidence_record"):
            errors.append(f"project {p['id']}: external evidence level requires an evidence_record")

    if errors:
        raise SystemExit("registry validation failed:\n- " + "\n- ".join(errors))

    print(f"REGISTRY VALIDATION: PASS ({len(programs)} programs, {len(projects)} projects)")


if __name__ == "__main__":
    main()
