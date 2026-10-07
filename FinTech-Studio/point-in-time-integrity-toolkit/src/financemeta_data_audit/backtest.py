"""Audit a single-asset prediction ledger without fitting or selecting a model.

Timestamps are assertions supplied by the producer, not independently verified
market publication times. The retained ledger must use non-overlapping,
contiguous return intervals and weights chosen before each interval.
"""
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

COLUMNS = ("timestamp", "available_at", "trained_through", "target_end", "position", "asset_return")


def _time(raw: str) -> datetime:
    value = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("backtest timestamps must include an explicit UTC offset")
    return value.astimezone(UTC)


def _cost(config: dict[str, Any], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be an explicit non-negative finite number")  # noqa: TRY004 - configuration-validation error
    if not math.isfinite(value) or not 0 <= value <= 10_000:
        raise ValueError(f"{key} must be between 0 and 10000 basis points")
    return float(value)


def _series(positions: list[float], returns: list[float], one_way_cost: float) -> dict[str, Any]:
    previous = 0.0
    turnover, costs, gross, net = [], [], [], []
    for index, (position, asset_return) in enumerate(zip(positions, returns)):
        # Charge opening/rebalancing and terminal liquidation. A -1 -> +1
        # reversal trades two units; a round trip is not one fee.
        traded = abs(position - previous)
        if index == len(positions) - 1:
            traded += abs(position)
        cost = traded * one_way_cost
        gross_return = position * asset_return
        net_return = gross_return - cost
        if not math.isfinite(net_return) or net_return <= -1:
            raise ValueError("non-positive wealth factor; leveraged/bankrupt paths are unsupported")
        turnover.append(traded)
        costs.append(cost)
        gross.append(gross_return)
        net.append(net_return)
        previous = position
    wealth, peak, max_drawdown = 1.0, 1.0, 0.0
    curve = [wealth]
    for value in net:
        wealth *= 1.0 + value
        if not math.isfinite(wealth) or wealth <= 0:
            raise ValueError("wealth accumulation is outside finite positive arithmetic")
        peak = max(peak, wealth)
        max_drawdown = max(max_drawdown, 1.0 - wealth / peak)
        curve.append(wealth)
    return {
        "net_total_return": wealth - 1.0,
        "max_drawdown": max_drawdown,
        "turnover_units": math.fsum(turnover),
        "sum_period_cost_fractions": math.fsum(costs),
        "gross_period_returns": gross,
        "net_period_returns": net,
        "period_turnover": turnover,
        "period_cost_fractions": costs,
        "net_wealth": curve,
    }


def audit_backtest(csv_path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Return timing/schema failures, or cost-aware metrics after all checks pass.

    No outcomes are inspected to tune a model or choose a configuration. This
    computes a deterministic audit of a caller-supplied, already frozen ledger.
    """
    if not isinstance(config, dict):
        raise ValueError("backtest config must be an object")  # noqa: TRY004 - configuration-validation error
    config = json.loads(json.dumps(config, allow_nan=False))
    fee = _cost(config, "fee_bps")
    slippage = _cost(config, "slippage_bps")
    for key in ("asset", "source", "license", "protocol_id", "invalidation_condition"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if config.get("data_kind") not in {"synthetic", "observed"}:
        raise ValueError("data_kind must explicitly be synthetic or observed")

    source = Path(csv_path)
    raw_bytes = source.read_bytes()
    findings: dict[str, list[dict[str, Any]]] = {
        key: [] for key in (
            "schema", "nonempty_data", "parseability", "strict_decision_order",
            "feature_availability", "training_cutoff", "future_target",
            "contiguous_intervals", "numerical_contract",
        )
    }

    def fail(check: str, row: int, detail: str) -> None:
        findings[check].append({"row": row, "detail": detail})

    reader = csv.DictReader(io.StringIO(raw_bytes.decode("utf-8-sig"), newline=""))
    header = reader.fieldnames or []
    if len(header) != len(set(header)) or set(header) != set(COLUMNS):
        fail("schema", 1, f"require exactly these unique columns: {', '.join(COLUMNS)}")
    raw_rows = list(reader)
    if not raw_rows:
        fail("nonempty_data", 1, "at least one prediction interval is required")
    parsed: list[dict[str, Any]] = []
    for row_number, row in enumerate(raw_rows, 2):
        if None in row or any(value is None for value in row.values()):
            fail("schema", row_number, "row width differs from header")
            continue
        try:
            values = {key: _time(row[key]) for key in COLUMNS[:4]}
            values["position"] = float(row["position"])
            values["asset_return"] = float(row["asset_return"])
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            fail("parseability", row_number, str(exc))
            continue
        values["row"] = row_number
        parsed.append(values)
        if values["available_at"] > values["timestamp"]:
            fail("feature_availability", row_number, "feature became available after the decision")
        if values["trained_through"] > values["timestamp"]:
            fail("training_cutoff", row_number, "training uses an outcome unavailable at the decision")
        if values["target_end"] <= values["timestamp"]:
            fail("future_target", row_number, "target interval must end after its decision")
        position, asset_return = values["position"], values["asset_return"]
        if not math.isfinite(position) or not -1 <= position <= 1:
            fail("numerical_contract", row_number, "position must be a finite weight in [-1, 1]")
        if not math.isfinite(asset_return) or asset_return <= -1:
            fail("numerical_contract", row_number, "asset_return must be finite and greater than -1")
    for previous, current in pairwise(parsed):
        if current["timestamp"] <= previous["timestamp"]:
            fail("strict_decision_order", current["row"], "decision timestamps must be unique and increasing")
        if current["timestamp"] != previous["target_end"]:
            fail("contiguous_intervals", current["row"], "return intervals overlap or leave an unaudited gap")

    performance = None
    if not any(findings.values()):
        positions = [row["position"] for row in parsed]
        returns = [row["asset_return"] for row in parsed]
        one_way_cost = (fee + slippage) / 10_000
        try:
            performance = {
                "strategy": _series(positions, returns, one_way_cost),
                "buy_and_hold": _series([1.0] * len(parsed), returns, one_way_cost),
                "cash": _series([0.0] * len(parsed), returns, one_way_cost),
            }
        except ValueError as exc:
            fail("numerical_contract", 0, str(exc))
    passed = not any(findings.values())
    return {
        "schema_version": "financemeta-backtest-ledger-v1",
        "source_file": source.name,
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "config": config,
        "row_count": len(raw_rows),
        "overall_status": "PASS" if passed else "FAIL",
        "checks": [
            {"id": key, "status": "FAIL" if failures else "PASS", "count": len(failures), "examples": failures[:10]}
            for key, failures in findings.items()
        ],
        "performance": performance if passed else None,
        "cost_model": "one-way fee+slippage per absolute weight change; initially flat; terminal liquidation charged",
        "claim_boundary": [
            "These are hypothetical ledger returns, not realized investment returns.",
            "Producer-supplied availability/training timestamps are checked for internal consistency only.",
            "No survivorship, corporate-action, publication-time, borrow-cost, market-impact, or execution-price certification.",
            "Costs use changes in declared target exposure; drift-induced rebalancing and financing are not modeled.",
            "Target exposures outside [-1, 1] and zero/negative wealth paths are unsupported.",
            "No model is trained or selected and no new protected study is authorized by this audit.",
        ],
    }


def render_backtest_markdown(report: dict[str, Any]) -> str:
    lines = ["# FinanceMeta backtest ledger audit", "", f"**Status: {report['overall_status']}**", "",
             f"Data kind: **{report['config']['data_kind']}**. Asset: `{report['config']['asset']}`.",
             f"Input SHA-256: `{report['source_sha256']}`", "",
             "| Check | Status | Failures |", "|---|---|---:|"]
    for check in report["checks"]:
        lines.append(f"| {check['id']} | {check['status']} | {check['count']} |")
    if report["performance"] is not None:
        lines.extend(["", "## Hypothetical cost-aware ledger results", "",
                      "| Method | Net total return | Max drawdown | Turnover units |", "|---|---:|---:|---:|"])
        for name, metrics in report["performance"].items():
            lines.append(f"| {name} | {metrics['net_total_return']:.6%} | {metrics['max_drawdown']:.6%} | {metrics['turnover_units']:.4f} |")
    else:
        lines.extend(["", "Performance metrics withheld because the ledger failed validation."])
    lines.extend(["", "## Declared invalidation condition", "", report["config"]["invalidation_condition"], "", "## Limits", ""])
    lines.extend(f"- {boundary}" for boundary in report["claim_boundary"])
    return "\n".join(lines) + "\n"
