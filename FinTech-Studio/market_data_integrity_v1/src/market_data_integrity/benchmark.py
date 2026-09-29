from __future__ import annotations

import argparse
import csv
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .core import validate_csv

CATEGORIES = ("clean", "duplicate", "out_of_order", "missing_interval", "nonfinite", "impossible_ohlc", "negative_volume")
CHECK_BY_CATEGORY = {
    "duplicate": "duplicate_timestamps",
    "out_of_order": "strictly_increasing",
    "missing_interval": "expected_interval_gaps",
    "nonfinite": "finite_numeric_values",
    "impossible_ohlc": "ohlc_constraints",
    "negative_volume": "non_negative_volume",
}


def base_rows(seed: int) -> list[dict[str, str]]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=seed * 40)
    rows = []
    for i in range(20):
        close = 100.0 + seed + i * 0.25
        rows.append({
            "timestamp": (start + timedelta(days=i)).isoformat().replace("+00:00", "Z"),
            "open": f"{close - 0.1:.4f}",
            "high": f"{close + 0.4:.4f}",
            "low": f"{close - 0.4:.4f}",
            "close": f"{close:.4f}",
            "volume": f"{1000 + seed * 10 + i:.2f}",
        })
    return rows


def apply_fault(rows: list[dict[str, str]], category: str) -> list[dict[str, str]]:
    rows = [dict(r) for r in rows]
    if category == "clean":
        return rows
    if category == "duplicate":
        rows[10]["timestamp"] = rows[9]["timestamp"]
    elif category == "out_of_order":
        rows[9], rows[10] = rows[10], rows[9]
    elif category == "missing_interval":
        rows.pop(10)
    elif category == "nonfinite":
        rows[10]["close"] = "NaN"
    elif category == "impossible_ohlc":
        rows[10]["high"] = "1"
    elif category == "negative_volume":
        rows[10]["volume"] = "-5"
    else:
        raise ValueError(category)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(rows)


def finding_failed(report: dict, check: str) -> bool:
    return any(f["check"] == check and not f["passed"] for f in report["findings"])


def run_benchmark(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_reports"
    raw_dir.mkdir(exist_ok=True)
    config = {
        "expected_interval_seconds": 86400,
        "timezone": "UTC",
        "provenance": {"source": "FinanceMeta deterministic synthetic v1 benchmark"},
    }
    config_path = output_dir / "benchmark_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    counts = {category: {"total": 0, "detected": 0} for category in CATEGORIES}
    deterministic = True
    modified = False

    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        for category in CATEGORIES:
            for seed in range(10):
                fixture = temp_dir / f"{category}-{seed}.csv"
                write_csv(fixture, apply_fault(base_rows(seed), category))
                first = validate_csv(fixture, config_path)
                second = validate_csv(fixture, config_path)
                deterministic = deterministic and first == second
                modified = modified or first["source"]["modified"]

                counts[category]["total"] += 1
                if category == "clean":
                    detected = first["overall"] == "PASS"
                else:
                    detected = finding_failed(first, CHECK_BY_CATEGORY[category])
                if detected:
                    counts[category]["detected"] += 1

                (raw_dir / f"{category}-{seed}.json").write_text(
                    json.dumps(first, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )

    rates = {category: counts[category]["detected"] / counts[category]["total"] for category in CATEGORIES}
    clean_false_positive_rate = 1.0 - rates["clean"]
    passed = (
        rates["duplicate"] == 1.0
        and rates["nonfinite"] == 1.0
        and rates["impossible_ohlc"] == 1.0
        and rates["negative_volume"] == 1.0
        and rates["out_of_order"] >= 0.95
        and rates["missing_interval"] >= 0.95
        and clean_false_positive_rate <= 0.05
        and deterministic
        and not modified
    )
    summary = {
        "benchmark": "market-data-integrity-v1",
        "fixtures": 70,
        "counts": counts,
        "rates": rates,
        "clean_false_positive_rate": clean_false_positive_rate,
        "deterministic_repeat": deterministic,
        "source_modified": modified,
        "overall": "PASS" if passed else "FAIL",
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen 70-fixture v1 benchmark")
    parser.add_argument("--output", default="benchmark-output")
    args = parser.parse_args()
    summary = run_benchmark(Path(args.output))
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["overall"] == "PASS" else 2)


if __name__ == "__main__":
    main()
