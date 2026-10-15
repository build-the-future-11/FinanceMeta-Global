from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from datetime import timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financemeta_data_audit import audit  # noqa: E402

HEADER = "timestamp,open,high,low,close,volume\n"
ROW = "2026-01-01T00:00:00Z,10,12,9,11,100\n"
CONFIG = {"provenance": {"source": "synthetic", "acquired_at": "2026-10-10", "license": "CC0"}}


class CsvAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "case.csv"
        self.source.write_text(HEADER + ROW, encoding="utf-8")

    def test_header_only_has_no_evidence_to_pass(self):
        self.source.write_text(HEADER, encoding="utf-8")
        report = audit.audit_csv(self.source, CONFIG)
        self.assertEqual(report["overall_status"], "FAIL")
        self.assertEqual(report["row_count"], 0)

    def test_duplicate_header_cannot_hide_an_impossible_price(self):
        self.source.write_text(
            "timestamp,open,high,low,close,close,volume\n"
            "2026-01-01,10,12,9,9999,11,100\n", encoding="utf-8"
        )
        report = audit.audit_csv(self.source, CONFIG)
        self.assertEqual(report["overall_status"], "FAIL")
        self.assertIn("unambiguous_header", [x["id"] for x in report["checks"] if x["status"] == "FAIL"])

    def test_unlabelled_header_is_rejected(self):
        self.source.write_text(HEADER.rstrip() + ",\n" + ROW.rstrip() + ",extra\n", encoding="utf-8")
        self.assertEqual(audit.audit_csv(self.source, CONFIG)["overall_status"], "FAIL")

    def test_extra_cell_is_not_silently_discarded(self):
        self.source.write_text(HEADER + ROW.rstrip() + ",unexpected\n", encoding="utf-8")
        self.assertEqual(audit.audit_csv(self.source, CONFIG)["overall_status"], "FAIL")

    def test_short_row_cannot_drop_an_optional_column(self):
        self.source.write_text(HEADER.rstrip() + ",note\n" + ROW, encoding="utf-8")
        self.assertEqual(audit.audit_csv(self.source, CONFIG)["overall_status"], "FAIL")

    def test_well_formed_optional_column_remains_supported(self):
        self.source.write_text(HEADER.rstrip() + ",note\n" + ROW.rstrip() + ",\n", encoding="utf-8")
        self.assertEqual(audit.audit_csv(self.source, CONFIG)["overall_status"], "PASS")

    def test_provenance_values_must_be_actual_nonblank_text(self):
        for key in ("source", "acquired_at", "license"):
            for value in (None, False, 0, [], {}, " "):
                with self.subTest(key=key, value=value):
                    config = {"provenance": {**CONFIG["provenance"], key: value}}
                    self.assertEqual(audit.audit_csv(self.source, config)["overall_status"], "FAIL")

    def test_invalid_intervals_are_rejected_even_for_one_row(self):
        for value in (float("nan"), float("inf"), float("-inf"), "nan", True, False, 0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.audit_csv(self.source, {**CONFIG, "expected_interval_seconds": value})

    def test_required_columns_cannot_disable_or_ambiguate_admission(self):
        for value in ([], ["close", "close"], [""], [None], "timestamp"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.audit_csv(self.source, {**CONFIG, "required_columns": value})

    def test_invalid_timestamp_column_is_rejected(self):
        for value in (None, "", " ", 42, False):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.audit_csv(self.source, {**CONFIG, "timestamp_column": value})

    def test_explicit_timestamp_format_uses_utc_for_naive_input(self):
        parsed = audit._parse_timestamp("2026-01-01", "%Y-%m-%d")
        self.assertIs(parsed.tzinfo, timezone.utc)
        self.assertEqual(parsed, audit._parse_timestamp("2025-12-31T19:00:00-05:00", None))

    def test_report_hash_identifies_the_exact_bytes_that_were_parsed(self):
        original = self.source.read_bytes()
        parse = audit._parse_timestamp

        def replace_after_read(raw, fmt):
            self.source.write_text("changed after parsing began", encoding="utf-8")
            return parse(raw, fmt)

        with patch.object(audit, "_parse_timestamp", side_effect=replace_after_read):
            report = audit.audit_csv(self.source, CONFIG)
        self.assertEqual(report["overall_status"], "PASS")
        self.assertEqual(report["source_sha256"], hashlib.sha256(original).hexdigest())
        self.assertNotEqual(report["source_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
