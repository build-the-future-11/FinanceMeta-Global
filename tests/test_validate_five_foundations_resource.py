import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_five_foundations_resource.py"
RESOURCE_DIR = ROOT / "resources/five-foundations"
MANIFEST = RESOURCE_DIR / "resource_manifest.json"

spec = importlib.util.spec_from_file_location("validator", SCRIPT)
validator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(validator)


class FiveFoundationsValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads(MANIFEST.read_text())

    def test_current_release_passes(self) -> None:
        validator.validate(self.data, RESOURCE_DIR)

    def test_provider_eligibility_cannot_be_self_approved(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["jumpstart"]["provider_eligibility"] = "APPROVED"
        with self.assertRaises(AssertionError):
            validator.validate(mutated, RESOURCE_DIR)

    def test_submission_cannot_be_preclaimed(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["jumpstart"]["submission_completed"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, RESOURCE_DIR)

    def test_acceptance_cannot_be_preclaimed(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["jumpstart"]["accepted_or_listed"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, RESOURCE_DIR)

    def test_financial_advice_boundary_cannot_be_weakened(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["content_boundaries"]["financial_advice"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, RESOURCE_DIR)

    def test_free_access_boundary_cannot_be_weakened(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["access"]["price_usd"] = 10
        with self.assertRaises(AssertionError):
            validator.validate(mutated, RESOURCE_DIR)

    def test_public_link_check_cannot_be_silently_completed(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["jumpstart"]["resource_criteria"]["8_nationwide_access"] = "PASS"
        with self.assertRaises(AssertionError):
            validator.validate(mutated, RESOURCE_DIR)

    def test_missing_packaged_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            for filename in validator.EXPECTED_FILES - {"ANSWER_KEY.md"}:
                (temp_dir / filename).write_text("placeholder")
            with self.assertRaises(AssertionError):
                validator.validate(self.data, temp_dir)

    def test_frozen_pilot_boundary_cannot_be_weakened(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["pilot_relationship"]["this_resource_changes_frozen_intervention"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, RESOURCE_DIR)


if __name__ == "__main__":
    unittest.main()
