#!/usr/bin/env python3
"""Validate FinanceMeta evidence registries without promoting any claim."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

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
STATUS_MATURITY = {
    "proposal_stub": "M0",
    "executable": "M1",
    "internal_evidence": "M2",
    "external_data": "M3",
    "external_review": "M4",
    "external_outcome": "M5",
}
PROMOTED_PROGRAM_STATUS = {"ready", "active", "at_risk", "complete"}


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict[str, object] = {}
    duplicates: set[str] = set()
    for key, value in pairs:
        if key in result:
            duplicates.add(key)
        result[key] = value
    if duplicates:
        raise ValueError(f"duplicate JSON object keys: {sorted(duplicates)}")
    return result


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing registry: {path.relative_to(ROOT)}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise SystemExit(f"invalid JSON in {path.relative_to(ROOT)}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"registry root must be an object: {path.relative_to(ROOT)}")
    return data


def unique_ids(records: list[dict], label: str) -> None:
    if any(not isinstance(record, dict) for record in records):
        raise SystemExit(f"{label}: every record must be an object")
    ids = [r.get("id") for r in records]
    if any(not isinstance(v, str) or not v.strip() for v in ids):
        raise SystemExit(f"{label}: every record needs a non-empty string id")
    whitespace_ids = sorted(v for v in ids if v != v.strip())
    if whitespace_ids:
        raise SystemExit(f"{label}: ids must not contain leading/trailing whitespace: {whitespace_ids}")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in ids:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise SystemExit(f"{label}: duplicate ids: {sorted(duplicates)}")


def repository_path(path: str) -> Path:
    if path != path.strip():
        raise SystemExit(f"registry path must not contain leading/trailing whitespace: {path!r}")
    if "\\" in path:
        raise SystemExit(f"registry path must use POSIX separators: {path}")
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute() or path.startswith("/"):
        raise SystemExit(f"registry path must be repository-relative: {path}")
    if path in {"", "."} or any(part in {"", ".", ".."} for part in pure_path.parts):
        raise SystemExit(f"registry path must be normalized and repository-relative: {path}")
    if str(pure_path) != path:
        raise SystemExit(f"registry path must be normalized and repository-relative: {path}")

    candidate = (ROOT / path).resolve()
    root = ROOT.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"registry path escapes repository root: {path}") from exc
    return candidate


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
        status = p.get("status")
        record = p.get("evidence_record")
        if status not in PROGRAM_STATUS:
            errors.append(f"program {pid}: invalid status {status!r}")
        if p.get("minimum_evidence_level") not in EVIDENCE - {"E0"}:
            errors.append(f"program {pid}: minimum_evidence_level must be E1–E5")
        gate = p.get("launch_gate")
        if not isinstance(gate, str) or not gate.strip():
            errors.append(f"program {pid}: launch_gate must be a non-empty string")
        if status == "planned_until_evidence_record" and record:
            errors.append(f"program {pid}: planned status conflicts with linked evidence_record; review promotion explicitly")
        if status in PROMOTED_PROGRAM_STATUS and (not isinstance(record, str) or not record.strip()):
            errors.append(f"program {pid}: {status} requires an evidence_record path")
        if record is not None:
            if not isinstance(record, str) or not record.strip():
                errors.append(f"program {pid}: evidence_record must be a non-empty repository-relative path")
            else:
                try:
                    evidence_path = repository_path(record)
                except SystemExit as exc:
                    errors.append(f"program {pid}: invalid evidence_record: {exc}")
                else:
                    if not evidence_path.is_file():
                        errors.append(f"program {pid}: evidence_record file does not exist: {record}")

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
        if status in STATUS_MATURITY and maturity != STATUS_MATURITY[status]:
            errors.append(f"project {pid}: status {status} requires maturity {STATUS_MATURITY[status]}")
        if not isinstance(p.get("claim_boundary"), str) or not p["claim_boundary"].strip():
            errors.append(f"project {pid}: claim_boundary must be a non-empty string")
        if not isinstance(p.get("next_gate"), str) or not p["next_gate"].strip():
            errors.append(f"project {pid}: next_gate must be a non-empty string")
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
        non_boolean = sorted(field for field in required_state if field in state and not isinstance(state[field], bool))
        if non_boolean:
            errors.append(f"project {pid}: repository-state fields must be booleans: {non_boolean}")
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
        else:
            try:
                resolved_path = repository_path(path)
            except SystemExit as exc:
                errors.append(f"project {pid}: {exc}")
            else:
                if not resolved_path.is_dir():
                    errors.append(f"project {pid}: registered path is not a directory: {path}")

    if errors:
        raise SystemExit("registry validation failed:\n- " + "\n- ".join(errors))

    print(f"REGISTRY VALIDATION: PASS ({len(programs)} programs, {len(projects)} projects)")


if __name__ == "__main__":
    main()
