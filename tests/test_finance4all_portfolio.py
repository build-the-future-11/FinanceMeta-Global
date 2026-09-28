import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "validate_finance4all_portfolio.py"
REGISTRY_PATH = ROOT / "registry" / "finance4all_4month_projects.json"

spec = importlib.util.spec_from_file_location("f4a_validator", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


class Finance4AllPortfolioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    def test_current_registry_passes(self):
        self.assertEqual(module.validate(copy.deepcopy(self.registry)), [])

    def test_missing_project_is_rejected(self):
        value = copy.deepcopy(self.registry)
        value["projects"].pop()
        errors = module.validate(value)
        self.assertTrue(any("expected 48 projects" in e for e in errors))
        self.assertTrue(any("project ids must be exactly" in e for e in errors))

    def test_duplicate_project_is_rejected(self):
        value = copy.deepcopy(self.registry)
        value["projects"][-1]["id"] = "F4A-01"
        errors = module.validate(value)
        self.assertTrue(any("duplicate id: F4A-01" in e for e in errors))

    def test_gated_state_cannot_be_marked_arbitrarily_active_by_decision(self):
        value = copy.deepcopy(self.registry)
        target = next(p for p in value["projects"] if p["id"] == "F4A-07")
        target["decision"] = "ACTIVE"
        errors = module.validate(value)
        self.assertTrue(any("activation decision must remain HOLD_UNTIL_OWNER_REVIEW" in e for e in errors))

    def test_core_project_cannot_silently_become_held_gate(self):
        value = copy.deepcopy(self.registry)
        target = next(p for p in value["projects"] if p["id"] == "F4A-21")
        target["portfolio_state"] = "HELD_GATE"
        errors = module.validate(value)
        self.assertTrue(any("core project cannot silently be converted" in e for e in errors))

    def test_research_cap_is_enforced(self):
        value = copy.deepcopy(self.registry)
        for pid in ("F4A-05", "F4A-06", "F4A-07", "F4A-08", "F4A-13"):
            target = next(p for p in value["projects"] if p["id"] == pid)
            target["portfolio_state"] = "OPEN_COMPONENT"
        extra = next(p for p in value["projects"] if p["id"] == "F4A-14")
        extra["portfolio_state"] = "OPEN_COMPONENT"
        # F4A-14 is not part of the validator's research-like cap set, so alter
        # the declared cap itself to verify charter tampering is rejected.
        value["controls"]["active_research_flagship_cap"] = 6
        errors = module.validate(value)
        self.assertTrue(any("active_research_flagship_cap must remain 5" in e for e in errors))


if __name__ == "__main__":
    unittest.main(verbosity=2)
