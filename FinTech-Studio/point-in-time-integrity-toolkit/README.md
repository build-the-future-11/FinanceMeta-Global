# FinanceMeta Point-in-Time Market Data Integrity Toolkit v1

Read-only educational/research checker for local OHLCV CSV files and frozen single-asset prediction ledgers. It detects structural and declared timing failures without repairing the source or fitting a model. Software version 0.2 adds the ledger audit and closes empty/ambiguous CSV acceptance; the original 70-fixture v1 benchmark is unchanged.

## Run locally

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python benchmark/run.py benchmark/evidence
financemeta-data-audit check sample.csv --config dataset.json --out evidence/
financemeta-data-audit backtest examples/ledger_clean.csv --config examples/ledger_config.json --out /tmp/ledger-audit-new
python examples/reproduce.py /tmp/ledger-reproduction-new
```

## Config

JSON or a small dependency-free mapping-only YAML subset is supported.

```json
{
  "timestamp_column": "timestamp",
  "timestamp_format": "%Y-%m-%dT%H:%M:%SZ",
  "expected_interval_seconds": 60,
  "provenance": {
    "source": "provider-or-file-origin",
    "acquired_at": "2026-09-25",
    "license": "license-or-use-basis"
  }
}
```

## Checks

- nonempty data, unique nonblank headers and rectangular rows;
- required columns;
- numeric parsing;
- timestamp parsing;
- strict timestamp ordering;
- duplicate timestamps;
- expected-interval gaps;
- NaN/+Inf/-Inf;
- OHLC constraints;
- non-negative volume;
- provenance metadata.

## Evidence

The CLI writes `report.json` and `summary.md`, binding the report to source/config SHA-256 hashes.

The benchmark generates exactly 70 deterministic synthetic fixtures and retains one JSON report per fixture plus `benchmark_summary.json`.

The [completed ledger examples](examples/evidence/ledger-v1/summary.json) retain a six-interval arithmetic control and a future-feature control with performance withheld. [The research note](RESEARCH_NOTE.md) explains the exact ledger semantics, observed defects, example outcomes and remaining scientific gates. A separate [retained FI-JEPA audit](examples/evidence/fijepa-v1-retained-audit.json) recomputes the closed study's seed-level MSE comparison from a hash-pinned source, without training or reopening that study.

## Ledger contract

The `backtest` command requires exactly these six columns, with timestamps carrying explicit UTC offsets:

| Column | Meaning |
|---|---|
| `timestamp` | Decision instant and beginning of the next return interval |
| `available_at` | Latest availability instant across every feature used by that decision |
| `trained_through` | Latest availability instant of every training label used by the model |
| `target_end` | End of the future realized-return interval |
| `position` | Already-frozen declared target exposure in [-1, 1] |
| `asset_return` | Simple asset return over that interval; finite and greater than -1 |

Availability and training times must be no later than the decision. Decisions must strictly increase; intervals must be contiguous and nonoverlapping. Every failure withholds all performance metrics. A passing report includes hypothetical strategy, buy-and-hold and cash paths for the same intervals. Its cost approximation charges the explicitly configured one-way fee plus slippage per absolute change in declared exposure, including entry and terminal liquidation; it does not simulate drift-induced rebalancing, financing or actual fills. The producer must supply both cost assumptions, provenance, data kind, protocol ID and a failure condition. Producer-supplied timing metadata is checked for internal consistency; its truth requires external data-provenance review.

Commands return 0 for PASS and 2 for a failed audit. The output directory must be new; existing results are refused. Reports hash the exact bytes read and a snapshot of the configuration.

## Boundary

A PASS means the file passed its declared structural/timing checklist. It does not prove the dataset is survivorship-bias-free, correctly adjusted for corporate actions, fully point-in-time, suitable for investment decisions, or capable of producing profitable results. All shipped trading-ledger values are synthetic arithmetic fixtures. No observed market ledger was present in this repository for validation.
