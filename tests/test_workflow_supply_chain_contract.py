from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
ENGINEERING_WORKFLOWS = (
    "fi-jepa-ci.yml",
    "five-foundations-resource.yml",
    "fmp-buildathon-contract.yml",
    "nov1-stock-pitch-position.yml",
    "registry-validation.yml",
    "september-evidence-ledger.yml",
    "september-literacy-pilot-protocol.yml",
    "workflow-supply-chain-contract.yml",
)
REMOTE_USES = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)", re.MULTILINE)
IMMUTABLE_REMOTE_ACTION = re.compile(r"^[^./\s][^\s]*@[0-9a-f]{40}$")


class WorkflowSupplyChainContractTest(unittest.TestCase):
    def _read(self, name: str) -> str:
        path = WORKFLOW_DIR / name
        self.assertTrue(path.is_file(), f"missing workflow: {path}")
        return path.read_text(encoding="utf-8")

    def test_engineering_workflows_pin_ubuntu_2404(self) -> None:
        for name in ENGINEERING_WORKFLOWS:
            with self.subTest(workflow=name):
                text = self._read(name)
                self.assertNotIn("ubuntu-latest", text)
                self.assertIn("runs-on: ubuntu-24.04", text)

    def test_remote_actions_use_immutable_commit_shas(self) -> None:
        for name in ENGINEERING_WORKFLOWS:
            with self.subTest(workflow=name):
                text = self._read(name)
                refs = REMOTE_USES.findall(text)
                self.assertTrue(refs, f"no action references found in {name}")
                for ref in refs:
                    if ref.startswith("./"):
                        continue
                    self.assertRegex(
                        ref,
                        IMMUTABLE_REMOTE_ACTION,
                        f"mutable remote action reference in {name}: {ref}",
                    )

    def test_checkout_is_exact_source_and_drops_credentials(self) -> None:
        for name in ENGINEERING_WORKFLOWS:
            with self.subTest(workflow=name):
                text = self._read(name)
                self.assertIn("SOURCE_SHA:", text)
                self.assertIn("ref: ${{ env.SOURCE_SHA }}", text)
                self.assertIn("persist-credentials: false", text)
                self.assertIn("git rev-parse HEAD", text)
                self.assertIn('"${SOURCE_SHA}"', text)

    def test_fi_jepa_evidence_binds_workflow_and_source_provenance(self) -> None:
        text = self._read("fi-jepa-ci.yml")
        self.assertIn("GITHUB_WORKFLOW_SHA", text)
        self.assertIn("GITHUB_WORKFLOW_REF", text)
        self.assertIn("event_commit=%s", text)
        self.assertIn("workflow_commit=%s", text)
        self.assertIn("workflow_ref=%s", text)
        self.assertIn(".github/workflows/fi-jepa-ci.yml", text)
        self.assertIn("find FI-JEPA/src FI-JEPA/tests", text)
        self.assertIn("xargs -0 sha256sum", text)


if __name__ == "__main__":
    unittest.main()
