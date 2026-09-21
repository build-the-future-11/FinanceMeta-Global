from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "registry-validation.yml"


class RegistryWorkflowEvidenceTests(unittest.TestCase):
    def test_evidence_records_executing_workflow_sha_not_only_event_sha(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            "printf 'workflow_commit=%s\\n' \"$GITHUB_WORKFLOW_SHA\"",
            workflow,
        )
        self.assertIn(
            "printf 'event_commit=%s\\n' \"$GITHUB_SHA\"",
            workflow,
        )
        self.assertIn(
            "printf 'workflow_ref=%s\\n' \"$GITHUB_WORKFLOW_REF\"",
            workflow,
        )

    def test_evidence_hashes_the_workflow_and_this_regression(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        digest_step = workflow.split("- name: Record registry digests", 1)[1]
        self.assertIn(".github/workflows/registry-validation.yml", digest_step)
        self.assertIn("tests/test_registry_workflow_evidence.py", digest_step)


if __name__ == "__main__":
    unittest.main()
