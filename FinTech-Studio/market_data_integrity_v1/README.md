# Market Data Integrity Toolkit v1

Read-only OHLCV integrity checker for FinanceMeta FinTech Studio 01.

## Install

Run: python -m pip install -e '.[dev]'

## Configuration

Provide JSON or YAML with expected_interval_seconds, timezone, provenance.source, and optional column mappings / timestamp_format.

## Check one file

Run: financemeta-data-check data.csv config.json --json-out report.json --md-out report.md

Exit code is 0 only when every required check passes. Invalid input is reported, never repaired.

## Frozen benchmark

Run: financemeta-data-benchmark --output benchmark-output

This generates exactly 70 deterministic fixtures in seven frozen categories and retains a raw JSON report for every fixture plus benchmark_summary.json.

## Tests

Run: pytest

## Limitations

Passing v1 does not prove:
- point-in-time safety for every research target;
- freedom from survivorship or selection bias;
- correct corporate-action treatment;
- investment suitability;
- predictive value or profitability.

The checker validates structural integrity under the declared configuration. It does not silently clean or modify source data.
