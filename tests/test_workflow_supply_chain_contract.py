from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
ENGINEERING_WORKFLOWS = (
    "five-foundations-resource.yml",
    "fmp-buildathon-contract.yml",
    "nov1-stock-pitch-position.yml",
    "registry-validation.yml",
    "september-evidence-ledger.yml",
    "september-literacy-pilot-protocol.yml",
    "fintech-studio-01-starter-kit.yml",
    "train-the-trainer-packet-validation.yml",
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

    def test_fintech_studio_cancels_superseded_runs(self) -> None:
        text = self._read("fintech-studio-01-starter-kit.yml")
        self.assertIn("concurrency:", text)
        self.assertIn("group: fintech-studio-01-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}", text)
        self.assertIn("cancel-in-progress: true", text)

    def test_fintech_studio_push_runs_only_on_main(self) -> None:
        text = self._read("fintech-studio-01-starter-kit.yml")
        push_block = text.split("  push:\n", 1)[1].split("  pull_request:\n", 1)[0]
        self.assertIn("    branches:\n      - main\n", push_block)


if __name__ == "__main__":
    unittest.main()
