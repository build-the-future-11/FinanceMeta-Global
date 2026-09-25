# FinanceMeta Point-in-Time Market Data Integrity Toolkit v1

Read-only educational/research checker for local OHLCV CSV files. It detects structural failures before modeling and never repairs the source file.

## Run locally

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python benchmark/run.py benchmark/evidence
financemeta-data-audit check sample.csv --config dataset.json --out evidence/
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

## Boundary

A PASS means the file passed this frozen v1 structural checklist. It does not prove the dataset is survivorship-bias-free, correctly adjusted for corporate actions, fully point-in-time, suitable for investment decisions, or capable of producing profitable results.
