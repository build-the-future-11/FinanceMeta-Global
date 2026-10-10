from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import __version__

REQUIRED_FIELDS = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class Finding:
    check: str
    passed: bool
    message: str
    rows: list[int]
    timestamps: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    value = yaml.safe_load(raw) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("config must be an object")
    return value


def _parse_timestamp(value: str, fmt: str | None) -> datetime:
    parsed = datetime.strptime(value, fmt) if fmt else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def validate_csv(csv_path: Path, config_path: Path) -> dict[str, Any]:
    source_hash_before = sha256_file(csv_path)
    config_hash = sha256_file(config_path)
    config = load_config(config_path)
    interval_value = config.get("expected_interval_seconds", 86400)
    if isinstance(interval_value, bool):
        raise ValueError("expected_interval_seconds must be finite and positive")
    interval_seconds = float(interval_value)
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("expected_interval_seconds must be finite and positive")

    provenance = config.get("provenance")
    provenance_ok = (
        isinstance(provenance, dict)
        and isinstance(provenance.get("source"), str)
        and bool(provenance["source"].strip())
    )
    timestamp_format = config.get("timestamp_format")
    mapping = config.get("columns", {})
    if not isinstance(mapping, dict):
        raise ValueError("columns must be a mapping")
    columns = {name: mapping.get(name, name) for name in REQUIRED_FIELDS}
    if (
        not all(isinstance(name, str) and name.strip() for name in columns.values())
        or len(set(columns.values())) != len(REQUIRED_FIELDS)
    ):
        raise ValueError("columns must map each required field to a distinct nonempty name")
    findings: list[Finding] = []
    rows: list[dict[str, Any]] = []
    parse_error_rows: list[int] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [columns[name] for name in REQUIRED_FIELDS if columns[name] not in header]
        if missing:
            findings.append(Finding("required_columns", False, f"missing columns: {missing}", [], []))
            source_hash_after = sha256_file(csv_path)
            return _report(csv_path, config_path, source_hash_before, source_hash_after, config_hash, config, findings, [], provenance_ok)
        findings.append(Finding("required_columns", True, "all required columns present", [], []))
        ambiguous = sorted({name for name in header if not name.strip() or header.count(name) > 1})
        findings.append(Finding("unambiguous_header", not ambiguous, f"ambiguous columns: {ambiguous}" if ambiguous else "column names are unique and nonempty", [], []))

        for idx, raw in enumerate(reader, start=2):
            if None in raw or any(value is None for value in raw.values()):
                parse_error_rows.append(idx)
                continue
            try:
                ts_text = raw[columns["timestamp"]]
                ts = _parse_timestamp(ts_text, timestamp_format)
                nums = {key: float(raw[columns[key]]) for key in ("open", "high", "low", "close", "volume")}
                rows.append({"row": idx, "timestamp_text": ts_text, "timestamp": ts, **nums})
            except Exception:
                parse_error_rows.append(idx)

    findings.append(Finding(
        "parseability",
        not parse_error_rows,
        "all rows parsed" if not parse_error_rows else "one or more rows failed timestamp/numeric parsing",
        parse_error_rows,
        [],
    ))

    if rows:
        duplicate_rows: list[int] = []
        duplicate_timestamps: list[str] = []
        seen: set[datetime] = set()
        for r in rows:
            if r["timestamp"] in seen:
                duplicate_rows.append(r["row"])
                duplicate_timestamps.append(r["timestamp"].isoformat())
            seen.add(r["timestamp"])
        findings.append(Finding("duplicate_timestamps", not duplicate_rows, "timestamps unique" if not duplicate_rows else "duplicate timestamps detected", duplicate_rows, duplicate_timestamps))

        disorder_rows: list[int] = []
        disorder_ts: list[str] = []
        for prev, cur in zip(rows, rows[1:]):
            if cur["timestamp"] <= prev["timestamp"]:
                disorder_rows.append(cur["row"])
                disorder_ts.append(cur["timestamp"].isoformat())
        findings.append(Finding("strictly_increasing", not disorder_rows, "timestamps strictly increasing" if not disorder_rows else "timestamps out of order", disorder_rows, disorder_ts))

        gap_rows: list[int] = []
        gap_ts: list[str] = []
        for prev, cur in zip(rows, rows[1:]):
            delta = (cur["timestamp"] - prev["timestamp"]).total_seconds()
            if delta > interval_seconds:
                gap_rows.append(cur["row"])
                gap_ts.append(f"{prev['timestamp'].isoformat()} -> {cur['timestamp'].isoformat()}")
        findings.append(Finding("expected_interval_gaps", not gap_rows, "no missing expected intervals" if not gap_rows else "missing expected interval(s) detected", gap_rows, gap_ts))

        nonfinite_rows: list[int] = []
        ohlc_rows: list[int] = []
        negative_volume_rows: list[int] = []
        for r in rows:
            values = [r[k] for k in ("open", "high", "low", "close", "volume")]
            if not all(math.isfinite(v) for v in values):
                nonfinite_rows.append(r["row"])
                continue
            if r["high"] < max(r["open"], r["close"], r["low"]) or r["low"] > min(r["open"], r["close"], r["high"]):
                ohlc_rows.append(r["row"])
            if r["volume"] < 0:
                negative_volume_rows.append(r["row"])

        findings.append(Finding("finite_numeric_values", not nonfinite_rows, "all required numeric values finite" if not nonfinite_rows else "non-finite values detected", nonfinite_rows, []))
        findings.append(Finding("ohlc_constraints", not ohlc_rows, "OHLC constraints satisfied" if not ohlc_rows else "impossible OHLC relationship detected", ohlc_rows, []))
        findings.append(Finding("non_negative_volume", not negative_volume_rows, "volume non-negative" if not negative_volume_rows else "negative volume detected", negative_volume_rows, []))
    else:
        for name in ("duplicate_timestamps", "strictly_increasing", "expected_interval_gaps", "finite_numeric_values", "ohlc_constraints", "non_negative_volume"):
            findings.append(Finding(name, False, "no valid parsed rows", [], []))

    findings.append(Finding("provenance_metadata", provenance_ok, "provenance metadata present" if provenance_ok else "config provenance.source is required", [], []))

    source_hash_after = sha256_file(csv_path)
    findings.append(Finding("read_only_source", source_hash_after == source_hash_before, "source file unchanged" if source_hash_after == source_hash_before else "source file changed during validation", [], []))

    return _report(csv_path, config_path, source_hash_before, source_hash_after, config_hash, config, findings, rows, provenance_ok)


def _report(csv_path: Path, config_path: Path, source_hash_before: str, source_hash_after: str, config_hash: str, config: dict[str, Any], findings: list[Finding], rows: list[dict[str, Any]], provenance_ok: bool) -> dict[str, Any]:
    timestamps = [r["timestamp"] for r in rows]
    return {
        "tool": "financemeta-market-data-integrity",
        "version": __version__,
        "overall": "PASS" if all(f.passed for f in findings) else "FAIL",
        "source": {"path": str(csv_path), "sha256_before": source_hash_before, "sha256_after": source_hash_after, "modified": source_hash_before != source_hash_after},
        "config": {"path": str(config_path), "sha256": config_hash, "expected_interval_seconds": config.get("expected_interval_seconds", 86400), "timezone": config.get("timezone", "UTC"), "provenance": config.get("provenance")},
        "rows": {"parsed": len(rows), "first_timestamp": min(timestamps).isoformat() if timestamps else None, "last_timestamp": max(timestamps).isoformat() if timestamps else None},
        "findings": [asdict(f) for f in findings],
    }


def markdown_summary(report: dict[str, Any]) -> str:
    lines = ["# Market Data Integrity Report", "", f"**Overall:** {report['overall']}", f"**Source SHA-256:** {report['source']['sha256_before']}", f"**Rows parsed:** {report['rows']['parsed']}", "", "| Check | Result | Detail |", "|---|---|---|"]
    for finding in report["findings"]:
        lines.append(f"| {finding['check']} | {'PASS' if finding['passed'] else 'FAIL'} | {finding['message']} |")
    lines.extend(["", "Passing this checker does not prove a dataset is unbiased, fully point-in-time safe, or suitable for investment decisions."])
    return "\n".join(lines) + "\n"

