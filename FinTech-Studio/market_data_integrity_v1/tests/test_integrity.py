import csv
import json
from pathlib import Path

from market_data_integrity.benchmark import run_benchmark
from market_data_integrity.core import sha256_file, validate_csv


def write_case(tmp_path: Path, rows):
    csv_path = tmp_path / "data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(rows)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "expected_interval_seconds": 86400,
        "timezone": "UTC",
        "provenance": {"source": "test"},
    }), encoding="utf-8")
    return csv_path, config_path


def clean_rows():
    return [
        {"timestamp": "2024-01-01T00:00:00Z", "open": "10", "high": "12", "low": "9", "close": "11", "volume": "100"},
        {"timestamp": "2024-01-02T00:00:00Z", "open": "11", "high": "13", "low": "10", "close": "12", "volume": "101"},
        {"timestamp": "2024-01-03T00:00:00Z", "open": "12", "high": "14", "low": "11", "close": "13", "volume": "102"},
    ]


def failed(report, name):
    return any(f["check"] == name and not f["passed"] for f in report["findings"])


def test_clean_fixture_passes_and_source_is_unchanged(tmp_path):
    csv_path, config = write_case(tmp_path, clean_rows())
    before = sha256_file(csv_path)
    report = validate_csv(csv_path, config)
    assert report["overall"] == "PASS"
    assert report["source"]["modified"] is False
    assert sha256_file(csv_path) == before


def test_duplicate_timestamp_fails_closed(tmp_path):
    rows = clean_rows()
    rows[2]["timestamp"] = rows[1]["timestamp"]
    csv_path, config = write_case(tmp_path, rows)
    assert failed(validate_csv(csv_path, config), "duplicate_timestamps")


def test_out_of_order_timestamp_fails_closed(tmp_path):
    rows = clean_rows()
    rows[1], rows[2] = rows[2], rows[1]
    csv_path, config = write_case(tmp_path, rows)
    assert failed(validate_csv(csv_path, config), "strictly_increasing")


def test_missing_interval_is_reported(tmp_path):
    rows = clean_rows()
    rows.pop(1)
    csv_path, config = write_case(tmp_path, rows)
    assert failed(validate_csv(csv_path, config), "expected_interval_gaps")


def test_nonfinite_impossible_ohlc_and_negative_volume_are_rejected(tmp_path):
    rows = clean_rows()
    rows[0]["close"] = "NaN"
    rows[1]["high"] = "1"
    rows[2]["volume"] = "-1"
    csv_path, config = write_case(tmp_path, rows)
    report = validate_csv(csv_path, config)
    assert failed(report, "finite_numeric_values")
    assert failed(report, "ohlc_constraints")
    assert failed(report, "non_negative_volume")


def test_missing_provenance_fails(tmp_path):
    csv_path, config = write_case(tmp_path, clean_rows())
    config.write_text(json.dumps({"expected_interval_seconds": 86400}), encoding="utf-8")
    assert failed(validate_csv(csv_path, config), "provenance_metadata")


def test_frozen_70_fixture_benchmark_passes(tmp_path):
    summary = run_benchmark(tmp_path / "benchmark")
    assert summary["fixtures"] == 70
    assert summary["overall"] == "PASS"
    assert summary["deterministic_repeat"] is True
    assert summary["source_modified"] is False
