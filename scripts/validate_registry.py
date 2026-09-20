#!/usr/bin/env python3
"""Validate FinanceMeta evidence registries without promoting any claim."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = ROOT / "registry" / "programs.json"
PROJECTS = ROOT / "registry" / "projects.json"
OPERATIONS_QUEUE = ROOT / "registry" / "research_operations_queue.json"

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
HEX_SHA = re.compile(r"^[0-9a-f]{40}$")


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


def validate_operations_queue(data: dict) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append("research operations queue: schema_version must equal 1")

    last_verified = data.get("last_verified")
    if not isinstance(last_verified, str) or not last_verified.strip() or last_verified != last_verified.strip():
        errors.append("research operations queue: last_verified must be a canonical ISO date")
    else:
        try:
            parsed_last_verified = date.fromisoformat(last_verified)
        except ValueError:
            errors.append("research operations queue: last_verified must be a canonical ISO date")
        else:
            if parsed_last_verified.isoformat() != last_verified:
                errors.append("research operations queue: last_verified must be a canonical ISO date")

    claim_boundary = data.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip() or claim_boundary != claim_boundary.strip():
        errors.append("research operations queue: claim_boundary must be a canonical non-empty string")

    items = data.get("items")
    if not isinstance(items, list):
        errors.append("research operations queue: items must be a list")
        return errors

    unique_ids(items, "research operations queue items")

    for item in items:
        item_id = item["id"]
        for field in ("program", "state", "next_artifact", "blocker", "claim_status"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                errors.append(f"operations item {item_id}: {field} must be a canonical non-empty string")

        for field in ("owner_state", "reviewer_state", "verification_state", "canonical_general_form"):
            if field in item:
                value = item[field]
                if not isinstance(value, str) or not value.strip() or value != value.strip():
                    errors.append(f"operations item {item_id}: {field} must be a canonical non-empty string")

        if "held_out_access_authorized" in item:
            held_out_access_authorized = item["held_out_access_authorized"]
            if not isinstance(held_out_access_authorized, bool):
                errors.append(f"operations item {item_id}: held_out_access_authorized must be boolean")
            elif held_out_access_authorized:
                errors.append(
                    f"operations item {item_id}: held_out_access_authorized must remain false in this pre-result operations registry"
                )

        for field in ("primary_issue", "pull_request", "current_preresult_contract_pr"):
            if field in item:
                value = item[field]
                if type(value) is not int or value <= 0:
                    errors.append(f"operations item {item_id}: {field} must be a positive integer")

        if "verified_completed_general_applications" in item:
            value = item["verified_completed_general_applications"]
            if type(value) is not int or value < 0:
                errors.append(
                    f"operations item {item_id}: verified_completed_general_applications must be a non-negative integer"
                )

        if "dependencies" in item:
            dependencies = item["dependencies"]
            if (
                not isinstance(dependencies, list)
                or any(type(value) is not int or value <= 0 for value in dependencies)
            ):
                errors.append(f"operations item {item_id}: dependencies must be a list of positive integers")
            elif len(dependencies) != len(set(dependencies)):
                errors.append(f"operations item {item_id}: dependencies must not contain duplicates")

        if "exact_head" in item:
            exact_head = item["exact_head"]
            if not isinstance(exact_head, str) or not HEX_SHA.fullmatch(exact_head):
                errors.append(f"operations item {item_id}: exact_head must be a lowercase 40-character Git SHA")

    declared_pull_requests = {
        item["pull_request"]
        for item in items
        if type(item.get("pull_request")) is int and item["pull_request"] > 0
    }
    for item in items:
        contract_pr = item.get("current_preresult_contract_pr")
        if type(contract_pr) is int and contract_pr > 0 and contract_pr not in declared_pull_requests:
            errors.append(
                f"operations item {item['id']}: current_preresult_contract_pr must reference a pull_request declared in this queue"
            )

    return errors


def main() -> None:
    programs = load(PROGRAMS).get("programs")
    projects = load(PROJECTS).get("projects")
    operations_data = load(OPERATIONS_QUEUE)
    if not isinstance(programs, list) or not isinstance(projects, list):
        raise SystemExit("registry documents must contain list-valued programs/projects")

    unique_ids(programs, "programs")
    unique_ids(projects, "projects")
    errors: list[str] = validate_operations_queue(operations_data)

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
                if not resolved_path.exists():
                    errors.append(f"project {pid}: registered path does not exist: {path}")

    if errors:
        raise SystemExit("registry validation failed:\n- " + "\n- ".join(errors))

    operations = operations_data["items"]
    print(
        "REGISTRY VALIDATION: PASS "
        f"({len(programs)} programs, {len(projects)} projects, {len(operations)} operations items)"
    )


if __name__ == "__main__":
    main()
