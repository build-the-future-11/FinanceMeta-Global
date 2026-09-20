from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import validate_registry as vr


def valid_operations_queue() -> dict:
    return {
        "schema_version": 1,
        "last_verified": "2026-09-20",
        "claim_boundary": "Operational status only.",
        "items": [
            {
                "id": "ops-1",
                "program": "Operations",
                "state": "OPEN",
                "next_artifact": "review",
                "blocker": "pending review",
                "claim_status": "NO_RESULT_CLAIM",
                "held_out_access_authorized": False,
                "primary_issue": 1,
                "pull_request": 59,
                "current_preresult_contract_pr": 59,
                "dependencies": [2, 3],
                "exact_head": "a" * 40,
            }
        ],
    }


class RegistryValidationTests(unittest.TestCase):
    def test_unique_ids_rejects_non_object_records(self) -> None:
        with self.assertRaisesRegex(SystemExit, "every record must be an object"):
            vr.unique_ids([{"id": "valid"}, "not-an-object"], "projects")  # type: ignore[list-item]

    def test_unique_ids_rejects_surrounding_whitespace(self) -> None:
        with self.assertRaisesRegex(SystemExit, "leading/trailing whitespace"):
            vr.unique_ids([{"id": " PROJECT-1 "}], "projects")

    def test_load_rejects_non_object_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "registry.json"
            path.write_text("[]", encoding="utf-8")
            with patch.object(vr, "ROOT", root):
                with self.assertRaisesRegex(SystemExit, "registry root must be an object"):
                    vr.load(path)

    def test_load_rejects_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "registry.json"
            path.write_text('{"projects": [], "projects": []}', encoding="utf-8")
            with patch.object(vr, "ROOT", root):
                with self.assertRaisesRegex(SystemExit, "duplicate JSON object keys"):
                    vr.load(path)

    def test_repository_path_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(vr, "ROOT", root):
                with self.assertRaisesRegex(SystemExit, "normalized and repository-relative"):
                    vr.repository_path("../outside")

    def test_repository_path_rejects_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(vr, "ROOT", root):
                with self.assertRaisesRegex(SystemExit, "repository-relative"):
                    vr.repository_path(str(root / "project"))

    def test_repository_path_rejects_non_posix_separator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(vr, "ROOT", root):
                with self.assertRaisesRegex(SystemExit, "POSIX separators"):
                    vr.repository_path("nested\\project")

    def test_operations_queue_rejects_non_boolean_held_out_flag(self) -> None:
        data = valid_operations_queue()
        data["items"][0]["held_out_access_authorized"] = "false"
        errors = vr.validate_operations_queue(data)
        self.assertIn("operations item ops-1: held_out_access_authorized must be boolean", errors)

    def test_operations_queue_rejects_true_held_out_flag(self) -> None:
        data = valid_operations_queue()
        data["items"][0]["held_out_access_authorized"] = True
        errors = vr.validate_operations_queue(data)
        self.assertIn(
            "operations item ops-1: held_out_access_authorized must remain false in this pre-result operations registry",
            errors,
        )

    def test_operations_queue_rejects_invalid_last_verified_date(self) -> None:
        data = valid_operations_queue()
        data["last_verified"] = "2026-02-30"
        errors = vr.validate_operations_queue(data)
        self.assertIn("research operations queue: last_verified must be a canonical ISO date", errors)

    def test_operations_queue_rejects_invalid_current_pr_reference(self) -> None:
        data = valid_operations_queue()
        data["items"][0]["current_preresult_contract_pr"] = "59"
        errors = vr.validate_operations_queue(data)
        self.assertIn(
            "operations item ops-1: current_preresult_contract_pr must be a positive integer",
            errors,
        )

    def test_operations_queue_rejects_dangling_current_pr_reference(self) -> None:
        data = valid_operations_queue()
        data["items"][0]["current_preresult_contract_pr"] = 60
        errors = vr.validate_operations_queue(data)
        self.assertIn(
            "operations item ops-1: current_preresult_contract_pr must reference a pull_request declared in this queue",
            errors,
        )

    def test_operations_queue_rejects_noncanonical_optional_state(self) -> None:
        data = valid_operations_queue()
        data["items"][0]["owner_state"] = "  ACCEPTED  "
        errors = vr.validate_operations_queue(data)
        self.assertIn("operations item ops-1: owner_state must be a canonical non-empty string", errors)

    def test_operations_queue_rejects_invalid_exact_head(self) -> None:
        data = valid_operations_queue()
        data["items"][0]["exact_head"] = "not-a-sha"
        errors = vr.validate_operations_queue(data)
        self.assertIn(
            "operations item ops-1: exact_head must be a lowercase 40-character Git SHA",
            errors,
        )

    def test_operations_queue_rejects_duplicate_dependencies(self) -> None:
        data = valid_operations_queue()
        data["items"][0]["dependencies"] = [2, 2]
        errors = vr.validate_operations_queue(data)
        self.assertIn("operations item ops-1: dependencies must not contain duplicates", errors)

    def test_main_rejects_non_boolean_repository_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            programs_path = root / "registry" / "programs.json"
            projects_path = root / "registry" / "projects.json"
            operations_path = root / "registry" / "research_operations_queue.json"
            programs_path.parent.mkdir(parents=True)
            (root / "project").mkdir()
            programs_path.write_text(
                json.dumps(
                    {
                        "programs": [
                            {
                                "id": "PROGRAM-1",
                                "status": "ready",
                                "minimum_evidence_level": "E1",
                                "launch_gate": "verified",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            projects_path.write_text(
                json.dumps(
                    {
                        "projects": [
                            {
                                "id": "PROJECT-1",
                                "path": "project",
                                "maturity": "M1",
                                "evidence_level": "E1",
                                "status": "executable",
                                "claim_boundary": "bounded",
                                "next_gate": "verify",
                                "verified_repository_state": {
                                    "readme_present": True,
                                    "license_present": True,
                                    "implementation_present": True,
                                    "tests_present": "yes",
                                    "results_present": False,
                                    "reproduction_command_present": True,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            operations_path.write_text(json.dumps(valid_operations_queue()), encoding="utf-8")
            with (
                patch.object(vr, "ROOT", root),
                patch.object(vr, "PROGRAMS", programs_path),
                patch.object(vr, "PROJECTS", projects_path),
                patch.object(vr, "OPERATIONS_QUEUE", operations_path),
            ):
                with self.assertRaisesRegex(SystemExit, "repository-state fields must be booleans"):
                    vr.main()

    def test_main_accepts_valid_minimal_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            programs_path = root / "registry" / "programs.json"
            projects_path = root / "registry" / "projects.json"
            operations_path = root / "registry" / "research_operations_queue.json"
            programs_path.parent.mkdir(parents=True)
            (root / "project").mkdir()
            programs_path.write_text(
                json.dumps(
                    {
                        "programs": [
                            {
                                "id": "PROGRAM-1",
                                "status": "ready",
                                "minimum_evidence_level": "E1",
                                "launch_gate": "verified",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            projects_path.write_text(
                json.dumps(
                    {
                        "projects": [
                            {
                                "id": "PROJECT-1",
                                "path": "project",
                                "maturity": "M1",
                                "evidence_level": "E1",
                                "status": "executable",
                                "claim_boundary": "bounded",
                                "next_gate": "verify",
                                "verified_repository_state": {
                                    "readme_present": True,
                                    "license_present": True,
                                    "implementation_present": True,
                                    "tests_present": True,
                                    "results_present": False,
                                    "reproduction_command_present": True,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            operations_path.write_text(json.dumps(valid_operations_queue()), encoding="utf-8")
            with (
                patch.object(vr, "ROOT", root),
                patch.object(vr, "PROGRAMS", programs_path),
                patch.object(vr, "PROJECTS", projects_path),
                patch.object(vr, "OPERATIONS_QUEUE", operations_path),
            ):
                vr.main()


if __name__ == "__main__":
    unittest.main()
