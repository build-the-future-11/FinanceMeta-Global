from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

REQUIRED_DEFAULT = ["timestamp", "open", "high", "low", "close", "volume"]


def _config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_scalar(value: str) -> Any:
    raw = value.strip()
    if not raw:
        return ""
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none", "~"}:
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw.strip("\"'")


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse a deliberately small mapping-only YAML subset with one nesting level."""
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" in line:
            raise ValueError(f"tabs are not supported in YAML config (line {lineno})")
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"expected key: value in YAML config (line {lineno})")
        key, value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty key in YAML config (line {lineno})")
        if indent == 0:
            if value.strip() == "":
                child: dict[str, Any] = {}
                root[key] = child
                current = child
            else:
                root[key] = _parse_scalar(value)
                current = None
        elif indent == 2 and current is not None:
            if value.strip() == "":
                raise ValueError(f"nested maps deeper than one level are not supported (line {lineno})")
            current[key] = _parse_scalar(value)
        else:
            raise ValueError(f"unsupported YAML indentation at line {lineno}")
    return root


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        data = _load_simple_yaml(text)
    else:
        raise ValueError("config must be .json, .yaml, or .yml")
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")  # noqa: TRY004 - malformed serialized configuration
    return data


def _check(check_id: str, status: str, count: int = 0, examples: list[Any] | None = None, detail: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": check_id,
        "status": status,
        "count": int(count),
        "examples": examples or [],
    }
    if detail:
        item["detail"] = detail
    return item


def _parse_timestamp(raw: str, fmt: str | None) -> datetime:
    if fmt:
        dt = datetime.strptime(raw, fmt)  # noqa: DTZ007 - offsets preserved; naive input is explicitly normalized below
    else:
        value = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def audit_csv(csv_path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")  # noqa: TRY004 - configuration-validation error
    source = Path(csv_path)
    if not source.is_file():
        raise FileNotFoundError(source)

    # Snapshot config and input once: report hashes describe exactly the bytes
    # and configuration inspected, even if the caller later edits either.
    config = json.loads(json.dumps(config, allow_nan=False))
    required = config.get("required_columns", REQUIRED_DEFAULT)
    if (not isinstance(required, list) or not required
            or not all(isinstance(x, str) and x.strip() for x in required)
            or len(set(required)) != len(required)):
        raise ValueError("required_columns must be a non-empty list of unique column names")

    ts_col = str(config.get("timestamp_column", "timestamp"))
    expected_interval = config.get("expected_interval_seconds")
    if expected_interval is not None:
        if isinstance(expected_interval, bool):
            raise ValueError("expected_interval_seconds must be finite and > 0")
        expected_interval = float(expected_interval)
        if not math.isfinite(expected_interval) or expected_interval <= 0:
            raise ValueError("expected_interval_seconds must be finite and > 0")
    ts_format = config.get("timestamp_format")
    provenance = config.get("provenance")

    checks: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []

    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    with io.StringIO(source_bytes.decode("utf-8-sig"), newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        invalid_header = not fieldnames or len(set(fieldnames)) != len(fieldnames) or any(not x.strip() for x in fieldnames)
        checks.append(_check("unambiguous_header", "FAIL" if invalid_header else "PASS", int(invalid_header)))
        missing_cols = [c for c in required if c not in fieldnames]
        checks.append(_check("required_columns", "FAIL" if missing_cols else "PASS", len(missing_cols), missing_cols))
        rows = list(reader)
    checks.append(_check("nonempty_data", "PASS" if rows else "FAIL", 0 if rows else 1))
    malformed = [idx for idx, row in enumerate(rows, start=2) if None in row or any(value is None for value in row.values())]
    checks.append(_check("rectangular_rows", "FAIL" if malformed else "PASS", len(malformed), malformed[:10]))

    provenance_ok = isinstance(provenance, dict) and all(
        isinstance(provenance.get(key), str) and bool(provenance[key].strip())
        for key in ("source", "acquired_at", "license")
    )
    checks.append(_check("provenance_metadata", "PASS" if provenance_ok else "FAIL", 0 if provenance_ok else 1, [] if provenance_ok else ["source/acquired_at/license required"]))

    numeric_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in required]
    numeric_bad: list[dict[str, Any]] = []
    nonfinite: list[dict[str, Any]] = []
    parsed_numeric: list[dict[str, float]] = []
    for idx, row in enumerate(rows, start=2):
        parsed: dict[str, float] = {}
        for col in numeric_cols:
            raw = row.get(col, "")
            try:
                value = float(raw)
                parsed[col] = value
                if not math.isfinite(value):
                    nonfinite.append({"row": idx, "column": col, "value": raw})
            except (TypeError, ValueError):
                numeric_bad.append({"row": idx, "column": col, "value": raw})
        parsed_numeric.append(parsed)
    checks.append(_check("numeric_parsing", "FAIL" if numeric_bad else "PASS", len(numeric_bad), numeric_bad[:10]))
    checks.append(_check("non_finite_values", "FAIL" if nonfinite else "PASS", len(nonfinite), nonfinite[:10]))

    parsed_ts: list[tuple[int, datetime, str]] = []
    ts_bad: list[dict[str, Any]] = []
    if ts_col in (rows[0].keys() if rows else required):
        for idx, row in enumerate(rows, start=2):
            raw = row.get(ts_col, "")
            try:
                parsed_ts.append((idx, _parse_timestamp(raw, ts_format), raw))
            except (ValueError, TypeError, AttributeError, OverflowError):
                ts_bad.append({"row": idx, "value": raw})
    else:
        ts_bad.append({"row": 1, "value": f"missing timestamp column {ts_col}"})
    checks.append(_check("timestamp_parsing", "FAIL" if ts_bad else "PASS", len(ts_bad), ts_bad[:10]))

    duplicates: list[dict[str, Any]] = []
    seen: dict[datetime, int] = {}
    for idx, dt, raw in parsed_ts:
        if dt in seen:
            duplicates.append({"row": idx, "timestamp": raw, "first_row": seen[dt]})
        else:
            seen[dt] = idx
    checks.append(_check("duplicate_timestamps", "FAIL" if duplicates else "PASS", len(duplicates), duplicates[:10]))

    disorder: list[dict[str, Any]] = []
    for previous, current in pairwise(parsed_ts):
        if current[1] <= previous[1]:
            disorder.append({"previous_row": previous[0], "row": current[0], "previous": previous[2], "current": current[2]})
    checks.append(_check("strictly_increasing_timestamps", "FAIL" if disorder else "PASS", len(disorder), disorder[:10]))

    gaps: list[dict[str, Any]] = []
    if expected_interval is not None and not ts_bad:
        for previous, current in pairwise(parsed_ts):
            delta = (current[1] - previous[1]).total_seconds()
            if delta != expected_interval:
                gaps.append({"previous_row": previous[0], "row": current[0], "delta_seconds": delta, "expected_seconds": expected_interval})
    checks.append(_check("expected_interval_gaps", "FAIL" if gaps else "PASS", len(gaps), gaps[:10]))

    ohlc_bad: list[dict[str, Any]] = []
    volume_bad: list[dict[str, Any]] = []
    if not numeric_bad and not nonfinite:
        for idx, vals in enumerate(parsed_numeric, start=2):
            if (all(k in vals for k in ["open", "high", "low", "close"])
                    and (vals["high"] < max(vals["open"], vals["close"], vals["low"])
                         or vals["low"] > min(vals["open"], vals["close"], vals["high"]))):
                ohlc_bad.append({"row": idx, "open": vals["open"], "high": vals["high"], "low": vals["low"], "close": vals["close"]})
            if "volume" in vals and vals["volume"] < 0:
                volume_bad.append({"row": idx, "volume": vals["volume"]})
    else:
        for idx, vals in enumerate(parsed_numeric, start=2):
            if (all(k in vals and math.isfinite(vals[k]) for k in ["open", "high", "low", "close"])
                    and (vals["high"] < max(vals["open"], vals["close"], vals["low"])
                         or vals["low"] > min(vals["open"], vals["close"], vals["high"]))):
                ohlc_bad.append({"row": idx})
            if "volume" in vals and math.isfinite(vals["volume"]) and vals["volume"] < 0:
                volume_bad.append({"row": idx, "volume": vals["volume"]})
    checks.append(_check("ohlc_constraints", "FAIL" if ohlc_bad else "PASS", len(ohlc_bad), ohlc_bad[:10]))
    checks.append(_check("non_negative_volume", "FAIL" if volume_bad else "PASS", len(volume_bad), volume_bad[:10]))

    overall = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    return {
        "schema_version": "1.0",
        "package_version": "0.1.0",
        "source_file": source.name,
        "source_sha256": source_hash,
        "config_sha256": _config_sha256(config),
        "row_count": len(rows),
        "time_range": {
            "start": parsed_ts[0][2] if parsed_ts else None,
            "end": parsed_ts[-1][2] if parsed_ts else None,
        },
        "checks": checks,
        "overall_status": overall,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# FinanceMeta Market Data Audit",
        "",
        f"- Overall: **{report['overall_status']}**",
        f"- Rows: {report['row_count']}",
        f"- Source SHA-256: `{report['source_sha256']}`",
        f"- Config SHA-256: `{report['config_sha256']}`",
        "",
        "| Check | Status | Count |",
        "|---|---:|---:|",
    ]
    for item in report["checks"]:
        lines.append(f"| {item['id']} | {item['status']} | {item['count']} |")
    lines.extend(["", "This tool is read-only and does not certify point-in-time suitability for every research question."])
    return "\n".join(lines) + "\n"
