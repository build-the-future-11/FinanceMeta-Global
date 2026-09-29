# FinTech Studio 01 — Market Data Integrity Toolkit v1

## Frozen brief

User: student research teams validating local historical OHLCV files before modeling.

Problem: malformed timestamps, duplicate or missing intervals, non-finite values, impossible OHLC relationships, negative volume, and missing provenance can silently invalidate downstream analysis.

v1 input: one local CSV containing timestamp/open/high/low/close/volume plus JSON or YAML configuration declaring timestamp format, timezone, expected interval, and provenance.

v1 output: JSON report and optional Markdown summary containing source/config hashes, row/time bounds, check results, affected rows/timestamps, package version, and overall PASS/FAIL.

## Required checks

1. required columns and numeric parsing;
2. timestamp parsing;
3. strictly increasing timestamps;
4. duplicate timestamps;
5. expected-interval gaps;
6. NaN / +Inf / -Inf;
7. OHLC constraints;
8. non-negative volume;
9. provenance metadata and source checksum.

## Guardrail

The tool is read-only. It never sorts, fills, deduplicates, interpolates, rewrites, or repairs source data.

## Frozen synthetic benchmark

Exactly 70 deterministic fixtures:
- 10 clean;
- 10 duplicate timestamp;
- 10 out of order;
- 10 missing interval;
- 10 non-finite;
- 10 impossible OHLC;
- 10 negative volume.

Release thresholds:
- 100% detection for duplicate, non-finite, impossible OHLC, negative volume;
- at least 95% for out-of-order and missing interval;
- clean false-positive rate at most 5%;
- deterministic repeat 100%;
- any source modification is automatic failure.

No fixtures may be changed in response to benchmark outcomes without incrementing the benchmark version.
