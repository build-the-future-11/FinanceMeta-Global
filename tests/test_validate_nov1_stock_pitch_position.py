import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_nov1_stock_pitch_position.py"
POSITION = ROOT / "operations/nov1-stock-pitch/financemeta_position.json"

spec = importlib.util.spec_from_file_location("validator", SCRIPT)
validator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(validator)


class NovemberPositionValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads(POSITION.read_text())

    def test_current_position_passes(self) -> None:
        validator.validate(self.data)

    def test_partner_approval_cannot_be_faked(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["status"] = "PARTNER_APPROVED"
        with self.assertRaises(AssertionError):
            validator.validate(mutated)

    def test_scoring_must_sum_to_100(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["recommendations"]["scoring"]["structure_and_writing"] = 9
        with self.assertRaises(AssertionError):
            validator.validate(mutated)

    def test_future_market_performance_cannot_enter_scoring(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["integrity"]["future_market_performance_is_scoring_component"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated)

    def test_open_partner_decisions_cannot_be_silently_removed(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["open_partner_decisions"].remove("cash_prize_amount_and_funding_source")
        with self.assertRaises(AssertionError):
            validator.validate(mutated)


if __name__ == "__main__":
    unittest.main()
