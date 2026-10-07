from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financemeta_data_audit.backtest import COLUMNS, audit_backtest
from financemeta_data_audit.cli import main

CONFIG = {
    "asset": "synthetic-asset", "source": "hand-calculated engineering fixture", "license": "CC0 synthetic",
    "protocol_id": "backtest-ledger-unit-v1", "data_kind": "synthetic", "fee_bps": 5, "slippage_bps": 5,
    "invalidation_condition": "Any invalid timing or an arithmetic disagreement invalidates the fixture check.",
}


def rows():
    return [
        {"timestamp": "2026-01-01T00:00:00Z", "available_at": "2025-12-31T00:00:00Z",
         "trained_through": "2025-12-30T00:00:00Z", "target_end": "2026-01-02T00:00:00Z", "position": "1", "asset_return": "0.1"},
        {"timestamp": "2026-01-02T00:00:00Z", "available_at": "2026-01-01T00:00:00Z",
         "trained_through": "2025-12-30T00:00:00Z", "target_end": "2026-01-03T00:00:00Z", "position": "-1", "asset_return": "0.05"},
    ]


def write(path, data):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(data)


class BacktestTests(unittest.TestCase):
    def audit(self, data=None, config=None):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ledger.csv"
            write(path, rows() if data is None else data)
            before = path.read_bytes()
            report = audit_backtest(path, CONFIG if config is None else config)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(report["source_sha256"], hashlib.sha256(before).hexdigest())
            return report

    def test_hand_calculated_costs_reversal_and_terminal_liquidation(self):
        report = self.audit()
        self.assertEqual(report["overall_status"], "PASS")
        actual = report["performance"]["strategy"]
        # Open +1: 1 unit. Reverse to -1: 2 units; close short: 1 unit.
        self.assertEqual(actual["period_turnover"], [1, 3])
        self.assertAlmostEqual(actual["net_total_return"], 1.099 * 0.947 - 1)
        self.assertAlmostEqual(actual["max_drawdown"], 0.053)
        self.assertAlmostEqual(actual["sum_period_cost_fractions"], 0.004)
        self.assertAlmostEqual(report["performance"]["buy_and_hold"]["net_total_return"], 1.099 * 1.049 - 1)
        self.assertEqual(report["performance"]["cash"]["net_total_return"], 0)

    def test_single_interval_charges_both_ends(self):
        report = self.audit(rows()[:1])
        self.assertEqual(report["performance"]["strategy"]["period_turnover"], [2])
        self.assertAlmostEqual(report["performance"]["strategy"]["net_total_return"], 0.098)

    def test_future_features_and_training_withhold_metrics(self):
        for column, check in (("available_at", "feature_availability"), ("trained_through", "training_cutoff")):
            with self.subTest(column=column):
                data = rows()
                data[0][column] = "2026-01-01T00:00:01Z"
                report = self.audit(data)
                self.assertEqual(report["overall_status"], "FAIL")
                self.assertIsNone(report["performance"])
                self.assertEqual(next(c for c in report["checks"] if c["id"] == check)["examples"][0]["row"], 2)

    def test_offsets_are_compared_as_instants(self):
        data = rows()
        data[0]["available_at"] = "2026-01-01T05:30:00+05:30"
        self.assertEqual(self.audit(data)["overall_status"], "PASS")
        data[0]["available_at"] = "2026-01-01T05:30:01+05:30"
        self.assertEqual(self.audit(data)["overall_status"], "FAIL")

    def test_naive_timestamps_fail(self):
        data = rows()
        data[0]["timestamp"] = "2026-01-01"
        self.assertEqual(self.audit(data)["overall_status"], "FAIL")

    def test_empty_overlapping_duplicate_and_out_of_order_fail(self):
        overlap = rows()
        overlap[0]["target_end"] = "2026-01-02T12:00:00Z"
        for data in ([], overlap, [rows()[0], rows()[0]], list(reversed(rows()))):
            with self.subTest(data=data):
                report = self.audit(data)
                self.assertEqual(report["overall_status"], "FAIL")
                self.assertIsNone(report["performance"])

    def test_nonfinite_out_of_contract_and_bankrupt_paths_fail(self):
        for field, value in (("asset_return", "nan"), ("position", "inf"), ("position", "2"), ("asset_return", "-1")):
            data = rows()
            data[0][field] = value
            self.assertEqual(self.audit(data)["overall_status"], "FAIL")
        data = rows()
        data[1]["asset_return"] = "2"
        self.assertIsNone(self.audit(data)["performance"])

    def test_missing_or_invalid_cost_assumptions_fail(self):
        for value in (None, True, -1, float("inf"), "5"):
            config = copy.deepcopy(CONFIG)
            config["fee_bps"] = value
            with self.assertRaises(ValueError):
                self.audit(config=config)

    def test_duplicate_headers_and_bad_width_fail(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ledger.csv"
            write(path, rows())
            clean = path.read_text()
            for corrupted in (clean.replace("position,asset_return", "position,position"), clean.replace("1,0.1", "1,0.1,extra")):
                path.write_text(corrupted)
                report = audit_backtest(path, CONFIG)
                self.assertEqual(report["overall_status"], "FAIL")
                self.assertIsNone(report["performance"])

    def test_cli_returns_failure_and_does_not_overwrite_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = rows()
            data[0]["available_at"] = data[0]["target_end"]
            write(root / "ledger.csv", data)
            (root / "config.json").write_text(json.dumps(CONFIG))
            command = ["backtest", str(root / "ledger.csv"), "--config", str(root / "config.json"), "--out", str(root / "output")]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(command), 2)
            report_bytes = (root / "output/report.json").read_bytes()
            with self.assertRaises(FileExistsError):
                main(command)
            self.assertEqual((root / "output/report.json").read_bytes(), report_bytes)


if __name__ == "__main__":
    unittest.main()
