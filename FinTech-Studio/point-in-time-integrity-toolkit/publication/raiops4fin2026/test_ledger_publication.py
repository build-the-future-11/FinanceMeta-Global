import unittest

from analyze_ledger_controls import decimal_path


class DecimalOracleChecks(unittest.TestCase):
    def test_single_interval_charges_open_and_close(self):
        result = decimal_path(
            [{"position": "1", "asset_return": "0.01"}], "strategy", 10
        )
        self.assertEqual(result["net_total_return"], "0.008")
        self.assertEqual(result["turnover_units"], "2")

    def test_reversal_costs_two_units(self):
        result = decimal_path(
            [
                {"position": "1", "asset_return": "0"},
                {"position": "-1", "asset_return": "0"},
            ],
            "strategy",
            10,
        )
        self.assertEqual(result["turnover_units"], "4")
        self.assertEqual(result["net_total_return"], "-0.003997")

    def test_cash_has_no_exposure_cost(self):
        result = decimal_path([{"position": "1", "asset_return": "0.1"}], "cash", 25)
        self.assertEqual(result["net_total_return"], "0.0000")
        self.assertEqual(result["turnover_units"], "0")

    def test_fractional_exposure(self):
        result = decimal_path(
            [{"position": "0.5", "asset_return": "0.02"}], "strategy", 10
        )
        self.assertEqual(result["net_total_return"], "0.0090")
        self.assertEqual(result["turnover_units"], "1.0")


if __name__ == "__main__":
    unittest.main()
