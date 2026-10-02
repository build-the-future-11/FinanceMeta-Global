from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts import validate_registry as vr


ROOT = Path(__file__).resolve().parents[1]
OPERATIONS_QUEUE = ROOT / "registry" / "research_operations_queue.json"

# `HELD_OUT_UNLOCKED` has one unambiguous meaning in the canonical research
# state machine: held-out access has been explicitly unlocked. The operations
# registry mixes research and ordinary operational rows, so broader words such
# as REVIEWED/EXECUTED/RELEASED are intentionally not banned globally here.
# They may be valid non-scientific operational states in other lanes.


class PreResultOperationsStateBoundaryTests(unittest.TestCase):
    def test_held_out_disabled_queue_cannot_claim_held_out_unlocked(self) -> None:
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
            self.assertNotEqual(
                item.get("state"),
                "HELD_OUT_UNLOCKED",
                f"{item_id}: HELD_OUT_UNLOCKED contradicts held_out_access_authorized=false",
            )

    def test_validator_rejects_held_out_unlocked_while_access_is_denied(self) -> None:
        data = {
            "schema_version": 1,
            "last_verified": "2026-09-21",
            "claim_boundary": "Pre-result operations only.",
            "items": [
                {
                    "id": "contradictory-pre-result-row",
                    "program": "Research",
                    "state": "HELD_OUT_UNLOCKED",
                    "next_artifact": "independent review",
                    "blocker": "held-out access remains denied",
                    "claim_status": "PRE_RESULT_ONLY",
                    "held_out_access_authorized": False,
                }
            ],
        }

        errors = vr.validate_operations_queue(data)
        self.assertIn(
            "operations item contradictory-pre-result-row: HELD_OUT_UNLOCKED contradicts held_out_access_authorized=false",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
