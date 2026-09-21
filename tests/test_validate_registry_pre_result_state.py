from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPERATIONS_QUEUE = ROOT / "registry" / "research_operations_queue.json"

# These are outcome-bearing states from docs/RESEARCH_PROGRAM_OPERATIONS_V1.md.
# The current operations registry is explicitly pre-result and requires
# held_out_access_authorized=false on every row, so claiming any of these states
# would contradict the repository's machine-readable authorization boundary.
OUTCOME_BEARING_STATES = {
    "HELD_OUT_UNLOCKED",
    "EXECUTED",
    "REVIEWED",
    "RELEASED",
    "RELEASE_WITH_LIMITATIONS",
}


class PreResultOperationsStateBoundaryTests(unittest.TestCase):
    def test_pre_result_queue_cannot_claim_outcome_bearing_state(self) -> None:
        data = json.loads(OPERATIONS_QUEUE.read_text(encoding="utf-8"))
        items = data.get("items")
        self.assertIsInstance(items, list)

        for item in items:
            item_id = item.get("id", "<missing-id>")
            self.assertIs(
                item.get("held_out_access_authorized"),
                False,
                f"{item_id}: the pre-result queue must explicitly deny held-out access",
            )
            self.assertNotIn(
                item.get("state"),
                OUTCOME_BEARING_STATES,
                f"{item_id}: outcome-bearing state contradicts the pre-result authorization boundary",
            )


if __name__ == "__main__":
    unittest.main()
