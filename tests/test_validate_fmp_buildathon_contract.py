import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_fmp_buildathon_contract.py"
CONTRACT = ROOT / "operations/fintech-studio-buildathon/fmp_contract.json"
BRIEF = ROOT / "operations/fintech-studio-buildathon/FMP_TECHNICAL_BRIEF.md"

spec = importlib.util.spec_from_file_location("validator", SCRIPT)
validator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(validator)


class FMPBuildathonContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads(CONTRACT.read_text())

    def test_current_contract_passes(self) -> None:
        validator.validate(self.data, BRIEF)

    def test_fmp_cannot_be_publicly_confirmed_before_management_approval(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["fmp"]["public_partner_claim_allowed"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, BRIEF)

    def test_real_time_quotes_cannot_be_enabled(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["fmp"]["real_time_quotes_allowed"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, BRIEF)

    def test_rate_limit_cannot_drift(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["fmp"]["rate_limit_calls_per_minute"] = 1000
        with self.assertRaises(AssertionError):
            validator.validate(mutated, BRIEF)

    def test_attribution_cannot_drift(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["fmp"]["required_attribution"] = "Data by FMP"
        with self.assertRaises(AssertionError):
            validator.validate(mutated, BRIEF)

    def test_registration_cannot_be_preclaimed(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["event"]["registered_team_count"] = 25
        with self.assertRaises(AssertionError):
            validator.validate(mutated, BRIEF)

    def test_exact_dates_cannot_be_invented(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["event"]["exact_start_date"] = "2026-11-01"
        with self.assertRaises(AssertionError):
            validator.validate(mutated, BRIEF)

    def test_keys_cannot_count_as_actual_usage(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["outcome_accounting"]["provisioned_key_is_usage"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, BRIEF)

    def test_judging_weights_cannot_be_preclaimed(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["judging"]["weights_frozen"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, BRIEF)

    def test_public_key_exposure_cannot_be_enabled(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["participant_rules"]["public_key_exposure_allowed"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, BRIEF)


if __name__ == "__main__":
    unittest.main()
