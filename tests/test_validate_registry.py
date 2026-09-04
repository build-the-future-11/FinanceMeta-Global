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

    def test_load_rejects_non_object_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "registry.json"
            path.write_text("[]")
            with patch.object(vr, "ROOT", root):
                with self.assertRaisesRegex(SystemExit, "registry root must be an object"):
                    vr.load(path)

    def test_repository_path_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(vr, "ROOT", root):
                with self.assertRaisesRegex(SystemExit, "escapes repository root"):
                    vr.repository_path("../outside")

    def test_main_rejects_non_boolean_repository_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            programs_path = root / "registry" / "programs.json"
            projects_path = root / "registry" / "projects.json"
            programs_path.parent.mkdir(parents=True)
            (root / "project").mkdir()
            programs_path.write_text(json.dumps({"programs": [{"id": "PROGRAM-1", "status": "ready", "minimum_evidence_level": "E1", "launch_gate": "verified"}]}))
            projects_path.write_text(json.dumps({"projects": [{"id": "PROJECT-1", "path": "project", "maturity": "M1", "evidence_level": "E1", "status": "executable", "claim_boundary": "bounded", "next_gate": "verify", "verified_repository_state": {"readme_present": True, "license_present": True, "implementation_present": True, "tests_present": "yes", "results_present": False, "reproduction_command_present": True}}]}))
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
            programs_path.write_text(json.dumps({"programs": [{"id": "PROGRAM-1", "status": "ready", "minimum_evidence_level": "E1", "launch_gate": "verified"}]}))
            projects_path.write_text(json.dumps({"projects": [{"id": "PROJECT-1", "path": "project", "maturity": "M1", "evidence_level": "E1", "status": "executable", "claim_boundary": "bounded", "next_gate": "verify", "verified_repository_state": {"readme_present": True, "license_present": True, "implementation_present": True, "tests_present": True, "results_present": False, "reproduction_command_present": True}}]}))
            with (
                patch.object(vr, "ROOT", root),
                patch.object(vr, "PROGRAMS", programs_path),
                patch.object(vr, "PROJECTS", projects_path),
            ):
                vr.main()


if __name__ == "__main__":
    unittest.main()
