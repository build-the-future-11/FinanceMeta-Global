#!/usr/bin/env python3
"""Validate the governed Finance4All four-month project registry.

This validator is deliberately conservative. It checks portfolio structure and
activation controls; it does not certify research, legal compliance, partners,
participants, production systems, or financial outcomes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VALID_WAVES = {"Core", "Gated", "Partner / capacity gated"}
VALID_STATES = {
    "OPEN_COMPONENT",
    "CANONICAL_EVIDENCE_REQUIRED",
    "SPEC_READY_DATA_BLOCKED",
    "LOCAL_PROTOTYPE",
    "DRAFT_REVIEW",
    "HELD_GATE",
}
ACTIVE_STATES = {"OPEN_COMPONENT", "LOCAL_PROTOTYPE"}
REQUIRED_PROJECT_FIELDS = {
    "id", "project", "wave", "portfolio_state", "next_action",
    "activation_gate", "owner", "reviewer", "decision",
    "evidence_level", "gate_review",
}


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("registry root must be an object")
    return value


def validate(registry: dict) -> list[str]:
    errors: list[str] = []
    projects = registry.get("projects")
    if registry.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(projects, list):
        return errors + ["projects must be an array"]

    ids: set[str] = set()
    core_count = 0
    active_research_like = 0

    for index, project in enumerate(projects, 1):
        prefix = f"projects[{index}]"
        if not isinstance(project, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = REQUIRED_PROJECT_FIELDS - project.keys()
        if missing:
            errors.append(f"{prefix} missing fields: {sorted(missing)}")
            continue

        pid = project["id"]
        if not isinstance(pid, str) or not pid.startswith("F4A-"):
            errors.append(f"{prefix} has invalid id")
        elif pid in ids:
            errors.append(f"duplicate id: {pid}")
        ids.add(pid)

        if project["wave"] not in VALID_WAVES:
            errors.append(f"{pid}: invalid wave {project['wave']!r}")
        if project["wave"] == "Core":
            core_count += 1
        if project["portfolio_state"] not in VALID_STATES:
            errors.append(f"{pid}: invalid portfolio_state {project['portfolio_state']!r}")

        for field in ("project", "next_action", "activation_gate"):
            if not isinstance(project[field], str) or len(project[field].strip()) < 10:
                errors.append(f"{pid}: {field} must be substantive")

        if project["decision"] != "HOLD_UNTIL_OWNER_REVIEW":
            errors.append(f"{pid}: activation decision must remain HOLD_UNTIL_OWNER_REVIEW until a reviewed state transition")

        if project["owner"] == "UNASSIGNED" and project["portfolio_state"] == "OPEN_COMPONENT":
            # Existing components may remain open for engineering, but they are
            # not considered programmatically activated.
            pass

        if project["portfolio_state"] == "HELD_GATE" and project["wave"] == "Core":
            errors.append(f"{pid}: core project cannot silently be converted to HELD_GATE; use an explicit reviewed scope decision")

        if pid in {"F4A-05", "F4A-06", "F4A-07", "F4A-08", "F4A-13"} and project["portfolio_state"] in ACTIVE_STATES:
            active_research_like += 1

    expected = {f"F4A-{i:02d}" for i in range(1, 49)}
    if ids != expected:
        errors.append(f"project ids must be exactly F4A-01..F4A-48; missing={sorted(expected-ids)}, extra={sorted(ids-expected)}")
    if len(projects) != 48:
        errors.append(f"expected 48 projects, found {len(projects)}")
    if core_count != 18:
        errors.append(f"expected 18 core projects, found {core_count}")

    cap = registry.get("controls", {}).get("active_research_flagship_cap")
    if cap != 5:
        errors.append("active_research_flagship_cap must remain 5 unless the portfolio charter is explicitly revised")
    if active_research_like > 5:
        errors.append(f"active research-like projects exceed cap: {active_research_like} > 5")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "registry",
        nargs="?",
        type=Path,
        default=Path("registry/finance4all_4month_projects.json"),
    )
    args = parser.parse_args(argv)
    try:
        registry = load(args.registry)
        errors = validate(registry)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}))
        return 2

    result = {
        "status": "PASS" if not errors else "FAIL",
        "project_count": len(registry.get("projects", [])),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
