from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import validate_registry as vr


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

    def test_main_rejects_non_boolean_repository_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            programs_path = root / "registry" / "programs.json"
            projects_path = root / "registry" / "projects.json"
            programs_path.parent.mkdir(parents=True)
            (root / "project").mkdir()
            programs_path.write_text(json.dumps({"programs": [{"id": "PROGRAM-1", "status": "planned_until_evidence_record", "minimum_evidence_level": "E1", "launch_gate": "verified"}]}), encoding="utf-8")
            projects_path.write_text(json.dumps({"projects": [{"id": "PROJECT-1", "path": "project", "maturity": "M1", "evidence_level": "E1", "status": "executable", "claim_boundary": "bounded", "next_gate": "verify", "verified_repository_state": {"readme_present": True, "license_present": True, "implementation_present": True, "tests_present": "yes", "results_present": False, "reproduction_command_present": True}}]}), encoding="utf-8")
            with (
                patch.object(vr, "ROOT", root),
                patch.object(vr, "PROGRAMS", programs_path),
                patch.object(vr, "PROJECTS", projects_path),
            ):
                with self.assertRaisesRegex(SystemExit, "repository-state fields must be booleans"):
                    vr.main()

    def test_main_accepts_valid_minimal_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            programs_path = root / "registry" / "programs.json"
            projects_path = root / "registry" / "projects.json"
            programs_path.parent.mkdir(parents=True)
            (root / "project").mkdir()
            programs_path.write_text(json.dumps({"programs": [{"id": "PROGRAM-1", "status": "planned_until_evidence_record", "minimum_evidence_level": "E1", "launch_gate": "verified"}]}), encoding="utf-8")
            projects_path.write_text(json.dumps({"projects": [{"id": "PROJECT-1", "path": "project", "maturity": "M1", "evidence_level": "E1", "status": "executable", "claim_boundary": "bounded", "next_gate": "verify", "verified_repository_state": {"readme_present": True, "license_present": True, "implementation_present": True, "tests_present": True, "results_present": False, "reproduction_command_present": True}}]}), encoding="utf-8")
            with (
                patch.object(vr, "ROOT", root),
                patch.object(vr, "PROGRAMS", programs_path),
                patch.object(vr, "PROJECTS", projects_path),
            ):
                vr.main()


    def _run_program(self, program: dict, *, make_record: bool = False) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "registry").mkdir()
            if make_record:
                (root / "evidence").mkdir()
                (root / "evidence" / "program.md").write_text("# Evidence", encoding="utf-8")
            programs_path = root / "registry" / "programs.json"
            projects_path = root / "registry" / "projects.json"
            programs_path.write_text(json.dumps({"programs": [program]}), encoding="utf-8")
            projects_path.write_text('{"projects":[]}', encoding="utf-8")
            with (
                patch.object(vr, "ROOT", root),
                patch.object(vr, "PROGRAMS", programs_path),
                patch.object(vr, "PROJECTS", projects_path),
            ):
                vr.main()

    def test_ready_program_requires_record(self) -> None:
        p = {"id": "PROGRAM-1", "status": "ready", "minimum_evidence_level": "E1", "launch_gate": "reviewed"}
        with self.assertRaisesRegex(SystemExit, "requires an evidence_record"):
            self._run_program(p)

    def test_ready_program_rejects_missing_record_file(self) -> None:
        p = {"id": "PROGRAM-1", "status": "ready", "minimum_evidence_level": "E1", "launch_gate": "reviewed", "evidence_record": "evidence/program.md"}
        with self.assertRaisesRegex(SystemExit, "evidence_record file does not exist"):
            self._run_program(p)

    def test_ready_program_rejects_path_traversal(self) -> None:
        p = {"id": "PROGRAM-1", "status": "ready", "minimum_evidence_level": "E1", "launch_gate": "reviewed", "evidence_record": "../outside.md"}
        with self.assertRaisesRegex(SystemExit, "invalid evidence_record"):
            self._run_program(p)

    def test_ready_program_accepts_existing_record(self) -> None:
        p = {"id": "PROGRAM-1", "status": "ready", "minimum_evidence_level": "E1", "launch_gate": "reviewed", "evidence_record": "evidence/program.md"}
        self._run_program(p, make_record=True)

    def _run_project(self, overrides: dict, *, project_is_file: bool = False) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "registry").mkdir()
            if project_is_file:
                (root / "project").write_text("not a directory", encoding="utf-8")
            else:
                (root / "project").mkdir()
            project = {
                "id": "PROJECT-1", "path": "project", "maturity": "M1",
                "evidence_level": "E1", "status": "executable",
                "claim_boundary": "synthetic only", "next_gate": "external dataset",
                "verified_repository_state": {
                    "readme_present": True, "license_present": True,
                    "implementation_present": True, "tests_present": True,
                    "results_present": False, "reproduction_command_present": True,
                },
            }
            project.update(overrides)
            programs_path = root / "registry" / "programs.json"
            projects_path = root / "registry" / "projects.json"
            programs_path.write_text('{"programs":[]}', encoding="utf-8")
            projects_path.write_text(json.dumps({"projects": [project]}), encoding="utf-8")
            with (
                patch.object(vr, "ROOT", root),
                patch.object(vr, "PROGRAMS", programs_path),
                patch.object(vr, "PROJECTS", projects_path),
            ):
                vr.main()

    def test_project_status_cannot_claim_unreached_maturity(self) -> None:
        with self.assertRaisesRegex(SystemExit, "status external_outcome requires maturity M5"):
            self._run_project({"status": "external_outcome"})

    def test_project_registered_path_must_be_directory(self) -> None:
        with self.assertRaisesRegex(SystemExit, "registered path is not a directory"):
            self._run_project({}, project_is_file=True)

    def test_project_rejects_structured_claim_boundary(self) -> None:
        with self.assertRaisesRegex(SystemExit, "claim_boundary must be a non-empty string"):
            self._run_project({"claim_boundary": {"statement": "unsupported"}})


if __name__ == "__main__":
    unittest.main()
