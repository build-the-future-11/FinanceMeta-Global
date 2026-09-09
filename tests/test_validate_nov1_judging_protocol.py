import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_nov1_judging_protocol.py"
PROTOCOL = ROOT / "operations/nov1-stock-pitch/judging_protocol.json"
SPEC = ROOT / "operations/nov1-stock-pitch/JUDGING_RECORD_SPEC.md"

spec = importlib.util.spec_from_file_location("validator", SCRIPT)
validator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(validator)


class Nov1JudgingProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads(PROTOCOL.read_text())

    def test_current_protocol_passes(self) -> None:
        validator.validate(self.data, SPEC)

    def test_partner_approval_cannot_be_preclaimed(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["activation"]["partner_approved"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, SPEC)

    def test_future_market_performance_cannot_enter_scoring(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["anti_lookahead"]["future_market_performance_scoring_component"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, SPEC)

    def test_post_lock_news_cannot_enter_scoring(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["anti_lookahead"]["post_lock_news_or_filings_allowed_in_scoring"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, SPEC)

    def test_judge_count_requirement_cannot_be_invented(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["scoring_model"]["required_independent_scores"] = 2
        with self.assertRaises(AssertionError):
            validator.validate(mutated, SPEC)

    def test_aggregation_rule_cannot_be_selected_before_partner_decision(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["scoring_model"]["aggregation_rule"] = "MEAN_TOTAL_SCORE"
        with self.assertRaises(AssertionError):
            validator.validate(mutated, SPEC)

    def test_candidate_weights_cannot_drift(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["candidate_rubric"]["weights"]["thesis_and_reasoning"] = 35
        with self.assertRaises(AssertionError):
            validator.validate(mutated, SPEC)

    def test_standalone_ai_detector_cannot_auto_disqualify(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["integrity_gate"]["standalone_ai_detector_sufficient_for_auto_disqualification"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, SPEC)

    def test_score_records_cannot_be_silently_overwritten(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["per_score_record"]["silent_deletion_or_overwrite_allowed"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, SPEC)

    def test_post_event_returns_cannot_be_backfilled_into_judging(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["post_event_outcomes"]["subsequent_returns_may_be_backfilled_into_judging"] = True
        with self.assertRaises(AssertionError):
            validator.validate(mutated, SPEC)

    def test_blind_school_field_cannot_be_removed(self) -> None:
        mutated = copy.deepcopy(self.data)
        mutated["blindness_and_conflicts"]["hide_school_from_scorer"] = False
        with self.assertRaises(AssertionError):
            validator.validate(mutated, SPEC)


if __name__ == "__main__":
    unittest.main()
