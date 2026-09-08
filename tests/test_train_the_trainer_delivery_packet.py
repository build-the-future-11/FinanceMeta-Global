from __future__ import annotations

import csv
import io
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "templates" / "train_the_trainer_delivery_packet.md"

EXPECTED_LEDGER_HEADER = [
    "cycle_id",
    "session_id",
    "facilitator_id",
    "site_id",
    "lesson_version",
    "protocol_id",
    "status",
    "session_date_iso",
    "attendance_instances",
    "unique_learners_at_session",
    "first_time_learners_cycle",
    "completion_count",
    "evidence_ref",
    "deviations",
    "permission_scope",
    "reviewer_id",
    "reviewed_at",
]


class TrainTheTrainerDeliveryPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = PACKET.read_text(encoding="utf-8")

    def test_packet_exists_and_is_explicitly_non_authorizing(self) -> None:
        self.assertTrue(PACKET.is_file())
        self.assertIn(
            "Status: operational template, not a launched program or a partner agreement.",
            self.text,
        )
        self.assertIn(
            "this packet authorizes neither a new experiment nor a causal effectiveness claim.",
            self.text.lower(),
        )

    def test_embedded_delivery_ledger_is_header_only_and_exactly_17_columns(self) -> None:
        match = re.search(r"```csv\n(?P<body>.*?)\n```", self.text, flags=re.DOTALL)
        self.assertIsNotNone(match, "packet must contain one fenced CSV ledger template")

        body = match.group("body").strip()
        rows = list(csv.reader(io.StringIO(body)))
        self.assertEqual(len(rows), 1, "committed ledger template must contain no activity rows")
        self.assertEqual(rows[0], EXPECTED_LEDGER_HEADER)
        self.assertEqual(len(rows[0]), 17)
        self.assertEqual(len(set(rows[0])), 17, "ledger column names must be unique")

    def test_facilitator_agenda_is_contiguous_and_totals_60_minutes(self) -> None:
        section = self.text.split("## 3. Proposed 60-minute facilitator session", 1)[1]
        section = section.split("## 4. Local delivery handoff", 1)[0]
        intervals = [
            (int(start), int(end))
            for start, end in re.findall(r"\|\s*(\d+)\s*[–-]\s*(\d+)\s*\|", section)
        ]
        self.assertEqual(intervals, [(0, 5), (5, 20), (20, 35), (35, 45), (45, 55), (55, 60)])
        self.assertEqual(sum(end - start for start, end in intervals), 60)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(intervals, intervals[1:])))

    def test_counting_contract_and_status_boundary_are_present(self) -> None:
        for status in ("PLANNED", "DELIVERED", "PARTIAL", "CANCELLED"):
            self.assertIn(status, self.text)
        self.assertIn("Blank means unknown/not collected; zero means a verified zero.", self.text)
        self.assertIn(
            "first_time_learners_cycle <= unique_learners_at_session\n  <= attendance_instances",
            self.text,
        )
        self.assertIn(
            "A `CANCELLED` row cannot contribute completed-session or learner counts.",
            self.text,
        )

    def test_referenced_frozen_protocol_assets_exist(self) -> None:
        required = [
            ROOT / "templates" / "program_evidence_record.md",
            ROOT / "evaluation" / "september-2026-financial-literacy-pilot" / "protocol.json",
            ROOT / "evaluation" / "september-2026-financial-literacy-pilot" / "INTERVENTION.md",
        ]
        for path in required:
            self.assertTrue(path.is_file(), f"referenced packet dependency is missing: {path.relative_to(ROOT)}")

    def test_private_financial_data_categories_are_explicitly_prohibited(self) -> None:
        prohibited = (
            "household income",
            "bank balances",
            "investment holdings",
            "account numbers",
        )
        for phrase in prohibited:
            self.assertIn(phrase, self.text)
        self.assertIn("Do not commit participant records to this public repository.", self.text)


if __name__ == "__main__":
    unittest.main()
