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
        self.assertIn("FI-JEPA/requirements.lock.txt", text)
        self.assertIn("find FI-JEPA/src FI-JEPA/tests", text)
        self.assertIn("xargs -0 sha256sum", text)

    def test_fi_jepa_runtime_test_and_build_dependencies_are_version_locked(self) -> None:
        workflow = self._read("fi-jepa-ci.yml")
        lock_path = ROOT / "FI-JEPA" / "requirements.lock.txt"
        pyproject_path = ROOT / "FI-JEPA" / "pyproject.toml"
        self.assertTrue(lock_path.is_file(), f"missing dependency lock: {lock_path}")
        self.assertTrue(pyproject_path.is_file(), f"missing package metadata: {pyproject_path}")

        observed: dict[str, str] = {}
        for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            self.assertRegex(
                line,
                r"^[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+$",
                f"dependency lock entry must use an exact version: {line}",
            )
            name, version = line.split("==", 1)
            observed[name.lower()] = version

        self.assertEqual(
            observed,
            {
                "iniconfig": "2.3.0",
                "numpy": "2.5.3",
                "packaging": "26.3",
                "pluggy": "1.6.0",
                "pygments": "2.21.0",
                "pytest": "8.4.2",
                "setuptools": "80.9.0",
            },
        )
        pyproject = pyproject_path.read_text(encoding="utf-8")
        self.assertIn('requires = ["setuptools==80.9.0"]', pyproject)
        self.assertNotIn('requires = ["setuptools>=', pyproject)
        self.assertIn(
            "python -m pip install --no-deps -r FI-JEPA/requirements.lock.txt",
            workflow,
        )
        self.assertIn(
            "python -m pip install -e FI-JEPA --no-deps --no-build-isolation",
            workflow,
        )
        self.assertNotIn("pip install -e 'FI-JEPA[dev]'", workflow)

    def test_fi_jepa_python_pip_and_build_backend_are_exact_locked(self) -> None:
        workflow = self._read("fi-jepa-ci.yml")
        self.assertIn("PYTHON_VERSION: '3.12.14'", workflow)
        self.assertIn("PIP_VERSION: '26.2.1'", workflow)
        self.assertIn("SETUPTOOLS_VERSION: '80.9.0'", workflow)
        self.assertIn("python-version: ${{ env.PYTHON_VERSION }}", workflow)
        self.assertIn("Verify exact Python and pip toolchain", workflow)
        self.assertIn("Verify exact build backend", workflow)
        self.assertIn("platform.python_version()", workflow)
        self.assertIn("importlib.metadata.version('setuptools')", workflow)
        self.assertIn("expected_python_version=%s", workflow)
        self.assertIn("expected_pip_version=%s", workflow)
        self.assertIn("expected_setuptools_version=%s", workflow)
        self.assertNotIn("python-version: '3.12'", workflow)


if __name__ == "__main__":
    unittest.main()
