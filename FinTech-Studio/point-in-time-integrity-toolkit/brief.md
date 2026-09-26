# FinTech Studio 01 — Point-in-Time Market Data Integrity Toolkit v1

Status: frozen implementation brief. This artifact is a read-only structural checker for local OHLCV files. It is not a predictor, backtest, data repair tool, or investment-grade certification.

## User and problem
Student finance researchers need a reproducible way to detect timestamp disorder, duplicates, interval gaps, non-finite values, impossible OHLC relationships, negative volume, and missing provenance before downstream modeling.

## Frozen interface

```bash
financemeta-data-audit check data.csv --config dataset.json --out evidence/
```

Input: CSV plus JSON/YAML configuration. Output: `report.json` and `summary.md` with source/config SHA-256 hashes, per-check findings, affected examples, and overall PASS/FAIL.

## v1 checks
1. required columns;
2. numeric parsing;
3. timestamp parsing;
4. strictly increasing timestamps;
5. duplicate timestamps;
6. expected-interval gaps;
7. NaN/+Inf/-Inf;
8. OHLC constraints;
9. non-negative volume;
10. provenance metadata.

## Fail-closed boundary
The tool never sorts, fills, deduplicates, interpolates, rewrites, or silently repairs the source. A report-generation/config failure cannot produce a PASS.

## Frozen benchmark
Exactly 70 deterministic synthetic fixtures: 10 clean controls plus 10 fixtures for each of duplicate timestamps, out-of-order timestamps, missing intervals, non-finite values, impossible OHLC relationships, and negative volume.

Release thresholds:
- duplicate/non-finite/OHLC/negative-volume detection: 100%;
- out-of-order/gap detection: >=95%;
- clean false-positive rate: <=5%;
- deterministic repeat rate: 100%;
- any source modification: automatic failure.

## Claim boundary
A PASS means only that the file passed this frozen v1 structural checklist. It does not prove full point-in-time correctness, survivorship-bias freedom, corporate-action correctness, investment suitability, or profitable predictability.
