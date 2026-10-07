from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmark.run import CONFIG, base_rows, inject, run, write_csv
from financemeta_data_audit.audit import audit_csv, load_config


class AuditTests(unittest.TestCase):
    def test_empty_or_ambiguous_csv_does_not_pass(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.csv"
            for content in ("timestamp,open,high,low,close,volume\n", "", "timestamp,open,high,low,close,close,volume\n2026-01-01,1,2,0,1,1,0\n"):
                with self.subTest(content=content):
                    path.write_text(content)
                    self.assertEqual(audit_csv(path, CONFIG)["overall_status"], "FAIL")

    def test_null_provenance_does_not_pass(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "clean.csv"
            write_csv(path, base_rows(0))
            config = dict(CONFIG, provenance={"source": None, "license": None, "acquired_at": None})
            self.assertEqual(audit_csv(path, config)["overall_status"], "FAIL")

    def test_extra_or_missing_csv_fields_fail(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.csv"
            write_csv(path, base_rows(0))
            clean = path.read_text()
            path.write_text(clean.replace("1000\n", "1000,untracked\n"))
            self.assertEqual(audit_csv(path, CONFIG)["overall_status"], "FAIL")

    def test_clean_fixture_passes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "clean.csv"
            write_csv(path, base_rows(0))
            report = audit_csv(path, CONFIG)
            self.assertEqual(report["overall_status"], "PASS")

    def test_duplicate_fails(self):
        with tempfile.TemporaryDirectory() as td:
            rows = base_rows(1)
            inject(rows, "duplicate", 0)
            path = Path(td) / "duplicate.csv"
            write_csv(path, rows)
            report = audit_csv(path, CONFIG)
            statuses = {c["id"]: c["status"] for c in report["checks"]}
            self.assertEqual(statuses["duplicate_timestamps"], "FAIL")
            self.assertEqual(report["overall_status"], "FAIL")

    def test_yaml_subset_loads(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "config.yaml"
            cfg.write_text(
                "timestamp_column: timestamp\n"
                "expected_interval_seconds: 60\n"
                "provenance:\n"
                "  source: synthetic\n"
                "  acquired_at: 2026-09-25\n"
                "  license: CC0\n",
                encoding="utf-8",
            )
            data = load_config(cfg)
            self.assertEqual(data["expected_interval_seconds"], 60)
            self.assertEqual(data["provenance"]["source"], "synthetic")

    def test_frozen_benchmark_has_70_fixtures_and_passes(self):
        with tempfile.TemporaryDirectory() as td:
            summary = run(Path(td))
            self.assertEqual(summary["total_fixtures"], 70)
            self.assertTrue(summary["release_thresholds_pass"])
            self.assertEqual(summary["deterministic_repeat_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
