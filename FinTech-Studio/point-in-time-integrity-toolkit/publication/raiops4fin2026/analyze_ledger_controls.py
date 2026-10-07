#!/usr/bin/env python3
"""Enumerate fixed synthetic ledger faults; independently check Decimal arithmetic.

This is a software-control study, never an observed-market performance study.
No models, asset prices, thresholds or protected FI-JEPA outcomes are selected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(rows: list[dict[str, str]], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def fixed_cases(rows: list[dict[str, str]], fields: list[str]):
    yield "clean", "valid", "PASS", None, encode(rows, fields)
    for offset in (-420, 330):
        twin = deepcopy(rows)
        zone = timezone(timedelta(minutes=offset))
        for row in twin:
            for name in fields[:4]:
                row[name] = datetime.fromisoformat(row[name].replace("Z", "+00:00")).astimezone(zone).isoformat()
        yield f"same_instants_offset_{offset}", "equivalent_offsets", "PASS", None, encode(twin, fields)
    for i in range(len(rows)):
        for family, field, value, check in (
            ("future_feature", "available_at", (datetime.fromisoformat(rows[i]["timestamp"].replace("Z", "+00:00")) + timedelta(seconds=1)).isoformat(), "feature_availability"),
            ("future_training", "trained_through", (datetime.fromisoformat(rows[i]["timestamp"].replace("Z", "+00:00")) + timedelta(seconds=1)).isoformat(), "training_cutoff"),
            ("zero_target_interval", "target_end", rows[i]["timestamp"], "future_target"),
            ("unsupported_exposure", "position", "1.01", "numerical_contract"),
            ("nonfinite_return", "asset_return", "nan", "numerical_contract"),
            ("naive_decision", "timestamp", rows[i]["timestamp"].removesuffix("Z"), "parseability"),
        ):
            twin = deepcopy(rows)
            twin[i][field] = value
            yield f"{family}_{i}", family, "FAIL", check, encode(twin, fields)
    for i in range(1, len(rows)):
        for family, seconds in (("interval_gap", 1), ("interval_overlap", -1)):
            twin = deepcopy(rows)
            twin[i]["timestamp"] = (datetime.fromisoformat(rows[i]["timestamp"].replace("Z", "+00:00")) + timedelta(seconds=seconds)).isoformat()
            yield f"{family}_{i}", family, "FAIL", "contiguous_intervals", encode(twin, fields)
        twin = deepcopy(rows)
        twin[i]["timestamp"] = rows[i-1]["timestamp"]
        yield f"duplicate_decision_{i}", "duplicate_decision", "FAIL", "strict_decision_order", encode(twin, fields)
    clean = encode(rows, fields)
    yield "empty_rows", "schema_or_empty", "FAIL", "nonempty_data", clean.splitlines()[0] + "\n"
    yield "missing_column", "schema_or_empty", "FAIL", "schema", encode([{k:v for k,v in row.items() if k != fields[-1]} for row in rows], fields[:-1])
    yield "extra_column", "schema_or_empty", "FAIL", "schema", encode([{**row, "extra":"0"} for row in rows], fields + ["extra"])
    yield "duplicate_header", "schema_or_empty", "FAIL", "schema", clean.replace(",asset_return\n", ",position\n", 1)


def decimal_path(rows: list[dict[str, str]], method: str, cost_bps: int) -> dict:
    """Independent exact-decimal oracle, constructed from CSV numeric strings."""
    with localcontext() as context:
        context.prec = 70
        wealth, previous = Decimal(1), Decimal(0)
        turnover = Decimal(0)
        periods = []
        for i, row in enumerate(rows):
            position = Decimal(row["position"]) if method == "strategy" else Decimal(1 if method == "buy_and_hold" else 0)
            traded = abs(position - previous)
            if i == len(rows)-1:
                traded += abs(position)
            net = position * Decimal(row["asset_return"]) - Decimal(cost_bps) / Decimal(10000) * traded
            periods.append(str(net))
            wealth *= 1 + net
            turnover += traded
            previous = position
        return {"net_total_return": str(wealth-1), "turnover_units": str(turnover), "net_period_returns": periods}


def run(package: Path, output: Path) -> dict:
    sys.path.insert(0, str(package / "src"))
    from financemeta_data_audit.backtest import audit_backtest

    source = package / "examples/ledger_clean.csv"
    config_source = package / "examples/ledger_config.json"
    config = json.loads(config_source.read_text())
    rows = list(csv.DictReader(io.StringIO(source.read_text())))
    fields = list(rows[0])
    if len(rows) != 6 or digest(source) != "309490bd9893c4d49265887eb63a53bc2d19659505e102494dab762c428310ab":
        raise ValueError("expected the retained six-interval control, not a replacement ledger")
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    (output / "reports").mkdir()
    records = []
    reference = audit_backtest(source, config)["performance"]
    for name, family, expected, check, content in fixed_cases(rows, fields):
        path = output / "inputs" / f"{name}.csv"
        path.write_text(content)
        report = audit_backtest(path, config)
        failed = [item["id"] for item in report["checks"] if item["status"] == "FAIL"]
        valid = report["overall_status"] == expected
        valid &= (report["performance"] is None) if expected == "FAIL" else (report["performance"] == reference)
        valid &= check in failed if check is not None else not failed
        if not valid:
            raise RuntimeError(f"control {name} disagrees with the frozen expected behavior")
        (output / "reports" / f"{name}.json").write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
        records.append({"id":name, "family":family, "expected":expected, "actual":report["overall_status"], "failed_checks":failed, "performance_withheld":report["performance"] is None, "input_sha256":digest(path)})
    arithmetic = []
    for cost in (0, 1, 5, 10, 25):
        result = audit_backtest(source, {**config, "fee_bps":cost, "slippage_bps":0})
        for method in ("strategy", "buy_and_hold", "cash"):
            oracle = decimal_path(rows, method, cost)
            measured = result["performance"][method]
            error = abs(Decimal(str(measured["net_total_return"])) - Decimal(oracle["net_total_return"]))
            if error > Decimal("1e-14") or Decimal(str(measured["turnover_units"])) != Decimal(oracle["turnover_units"]):
                raise RuntimeError("independent Decimal cost oracle disagrees")
            for actual, expected in zip(measured["net_period_returns"], oracle["net_period_returns"], strict=True):
                if not math.isclose(actual, float(expected), abs_tol=1e-15, rel_tol=0):
                    raise RuntimeError("period arithmetic mismatch")
            arithmetic.append({"one_way_cost_bps":cost,"method":method,**oracle,"float_absolute_error":str(error)})
    families = {}
    for row in records:
        entry = families.setdefault(row["family"], {"cases":0,"expected_pass":0,"expected_fail":0,"observed_match":0})
        entry["cases"] += 1
        entry["expected_pass" if row["expected"] == "PASS" else "expected_fail"] += 1
        entry["observed_match"] += row["expected"] == row["actual"]
    summary = {"protocol":"FINANCEMETA_LEDGER_FAULT_ENUMERATION_V1", "study_kind":"synthetic exhaustive fault locations on one retained six-interval fixture", "status":"PASS", "source_sha256":{str(p.relative_to(package)):digest(p) for p in (source,config_source,package/'src/financemeta_data_audit/backtest.py')}, "case_count":len(records), "valid_cases":sum(x["expected"]=="PASS" for x in records), "invalid_cases":sum(x["expected"]=="FAIL" for x in records), "families":families, "cases":records, "arithmetic_checks":arithmetic, "limits":["No observed financial data, model fitting, hyperparameter search, new holdout, or FI-JEPA rerun.","Enumerated fixtures are coverage checks, not population sensitivity or specificity estimates.","Equivalent timestamp spellings preserve the same instants; metadata truth is not authenticated.","Cost sweep checks arithmetic only; no selected profitable configuration or trading recommendation."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    manifest = {str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.package.resolve(), args.output)
    print(json.dumps({k:result[k] for k in ("status","case_count","valid_cases","invalid_cases")}))
