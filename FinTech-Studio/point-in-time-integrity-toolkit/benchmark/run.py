from __future__ import annotations

import csv
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financemeta_data_audit.audit import audit_csv  # noqa: E402

CONFIG = {
    "timestamp_column": "timestamp",
    "timestamp_format": "%Y-%m-%dT%H:%M:%SZ",
    "expected_interval_seconds": 60,
    "provenance": {"source": "synthetic-v1", "acquired_at": "2026-09-25", "license": "CC0-synthetic"},
}

FAULTS = [
    ("duplicate", "duplicate_timestamps"),
    ("out_of_order", "strictly_increasing_timestamps"),
    ("gap", "expected_interval_gaps"),
    ("non_finite", "non_finite_values"),
    ("ohlc", "ohlc_constraints"),
    ("negative_volume", "non_negative_volume"),
]


def base_rows(offset: int) -> list[dict[str, str]]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=offset)
    rows = []
    for i in range(20):
        price = 100 + offset * 0.1 + i * 0.05
        rows.append({
            "timestamp": (start + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": f"{price:.4f}",
            "high": f"{price + 0.5:.4f}",
            "low": f"{price - 0.5:.4f}",
            "close": f"{price + 0.1:.4f}",
            "volume": str(1000 + i),
        })
    return rows


def inject(rows: list[dict[str, str]], kind: str, variant: int) -> None:
    pos = 4 + (variant % 10)
    if kind == "duplicate":
        rows[pos]["timestamp"] = rows[pos - 1]["timestamp"]
    elif kind == "out_of_order":
        rows[pos]["timestamp"], rows[pos + 1]["timestamp"] = rows[pos + 1]["timestamp"], rows[pos]["timestamp"]
    elif kind == "gap":
        for i in range(pos, len(rows)):
            dt = datetime.strptime(rows[i]["timestamp"], "%Y-%m-%dT%H:%M:%SZ") + timedelta(minutes=1)
            rows[i]["timestamp"] = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    elif kind == "non_finite":
        rows[pos]["close"] = "nan"
    elif kind == "ohlc":
        rows[pos]["high"] = str(float(rows[pos]["low"]) - 1.0)
    elif kind == "negative_volume":
        rows[pos]["volume"] = "-1"
    else:
        raise ValueError(kind)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(rows)


def status_map(report: dict) -> dict[str, str]:
    return {item["id"]: item["status"] for item in report["checks"]}


def run(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    records = []
    deterministic = True

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fixture_index = 0
        for variant in range(10):
            rows = base_rows(variant)
            path = tmp / f"clean_{variant}.csv"
            write_csv(path, rows)
            a = audit_csv(path, CONFIG)
            b = audit_csv(path, CONFIG)
            deterministic = deterministic and (a == b)
            (reports_dir / f"{fixture_index:02d}_clean_{variant}.json").write_text(json.dumps(a, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            records.append({"kind": "clean", "expected": "PASS", "detected": a["overall_status"] == "PASS"})
            fixture_index += 1

        for kind, check_id in FAULTS:
            for variant in range(10):
                rows = base_rows(20 + fixture_index)
                inject(rows, kind, variant)
                path = tmp / f"{kind}_{variant}.csv"
                write_csv(path, rows)
                a = audit_csv(path, CONFIG)
                b = audit_csv(path, CONFIG)
                deterministic = deterministic and (a == b)
                detected = status_map(a).get(check_id) == "FAIL"
                (reports_dir / f"{fixture_index:02d}_{kind}_{variant}.json").write_text(json.dumps(a, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                records.append({"kind": kind, "expected_check": check_id, "detected": detected})
                fixture_index += 1

    summary = {"total_fixtures": len(records), "deterministic_repeat_rate": 1.0 if deterministic else 0.0, "categories": {}}
    clean = [r for r in records if r["kind"] == "clean"]
    summary["clean_false_positive_rate"] = sum(not r["detected"] for r in clean) / len(clean)
    for kind, _ in FAULTS:
        subset = [r for r in records if r["kind"] == kind]
        summary["categories"][kind] = {"n": len(subset), "detection_rate": sum(r["detected"] for r in subset) / len(subset)}

    thresholds = {
        "duplicate": 1.0,
        "out_of_order": 0.95,
        "gap": 0.95,
        "non_finite": 1.0,
        "ohlc": 1.0,
        "negative_volume": 1.0,
    }
    pass_thresholds = summary["clean_false_positive_rate"] <= 0.05 and summary["deterministic_repeat_rate"] == 1.0
    for kind, threshold in thresholds.items():
        pass_thresholds = pass_thresholds and summary["categories"][kind]["detection_rate"] >= threshold
    summary["release_thresholds_pass"] = pass_thresholds
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "benchmark" / "evidence"
    result = run(target)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["release_thresholds_pass"] else 1)
