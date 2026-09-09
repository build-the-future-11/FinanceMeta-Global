import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_september_evidence_snapshot.py"
SNAPSHOT = ROOT / "operations/september-2026/evidence_snapshot.json"
LEDGER = ROOT / "operations/september-2026/EVIDENCE_LEDGER.md"

spec = importlib.util.spec_from_file_location("validator", SCRIPT)
validator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(validator)


class SeptemberEvidenceSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads(SNAPSHOT.read_text())

    def test_current_snapshot_passes(self) -> None:
        validator.validate(self.data, LEDGER)

    def test_unknown_cannot_be_treated_as_zero(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["metric_policy"]["unknown_is_zero"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, LEDGER)

    def test_planning_target_cannot_be_registration(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["metric_policy"]["planning_target_is_registration"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, LEDGER)

    def test_provisioned_key_cannot_count_as_usage(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["metric_policy"]["provisioned_key_is_usage"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, LEDGER)

    def test_scheduled_reviewer_cannot_count_as_participated(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["metric_policy"]["scheduled_reviewer_is_participated_reviewer"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, LEDGER)

    def test_nonzero_metric_requires_evidence(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["metrics"]["registrations"]["value"] = 25
        mutated["metrics"]["registrations"]["counting_rule"] = "UNIQUE_VALID_REGISTRATIONS"
        with self.assertRaises(AssertionError):
            validator.validate(mutated, LEDGER)

    def test_metric_with_evidence_requires_resolved_counting_rule(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["metrics"]["registrations"]["value"] = 25
        mutated["metrics"]["registrations"]["evidence_refs"] = ["some-export.csv"]
        with self.assertRaises(AssertionError):
            validator.validate(mutated, LEDGER)

    def test_haven_future_call_cannot_be_counted_as_delivered(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["workstreams"]["haven"]["external_outcome_counted"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, LEDGER)

    def test_jumpstart_acceptance_cannot_be_preclaimed(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["workstreams"]["five_foundations"]["accepted_or_listed"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, LEDGER)

    def test_internal_nov1_work_cannot_be_external_outcome(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["workstreams"]["nov1_stock_pitch"]["external_outcome_counted"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, LEDGER)


if __name__ == "__main__":
    unittest.main()
