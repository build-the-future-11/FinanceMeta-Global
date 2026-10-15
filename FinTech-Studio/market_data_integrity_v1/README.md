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

## Admission correction, 10 October 2026

CSV headers must be unique and nonblank, row widths must match the header, and
column mappings must name distinct required fields. Provenance requires actual
nonblank text. Expected intervals must be finite and positive; fractional
seconds are retained when checking for missing intervals, so a subsecond gap
cannot disappear through integer truncation.

The expanded suite passes 30 tests including the existing frozen 70-fixture
benchmark. Retained evidence and benchmark definitions are unchanged. These
corrections reject malformed input under the existing structural contract and
do not establish predictive or investment performance.

## Limitations

Passing v1 does not prove:
- point-in-time safety for every research target;
- freedom from survivorship or selection bias;
- correct corporate-action treatment;
- investment suitability;
- predictive value or profitability.

The checker validates structural integrity under the declared configuration. It does not silently clean or modify source data.

