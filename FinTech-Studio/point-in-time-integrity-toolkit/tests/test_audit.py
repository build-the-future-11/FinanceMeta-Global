from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from financemeta_data_audit.audit import audit_csv, load_config  # noqa: E402
from benchmark.run import CONFIG, base_rows, inject, run, write_csv  # noqa: E402


class AuditTests(unittest.TestCase):
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
