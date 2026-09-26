# FinTech Studio 01 — Point-in-Time Market Data Integrity Toolkit v1

**Status:** frozen build brief / no benchmark result yet.

## User

Student quantitative-finance researchers who receive historical OHLCV CSV files and need to know whether the file is structurally safe enough to enter a modeling pipeline.

## Problem

A polished notebook can hide basic data failures: duplicate or disordered timestamps, gaps, non-finite values, impossible OHLC relationships, negative volume, ambiguous timezone assumptions, and missing provenance. The first Studio artifact should catch these failures without silently repairing source data.

## Smallest artifact

A read-only Python package and CLI:

```bash
financemeta-data-audit check data.csv --config dataset.json --out evidence/
```

Inputs:
- local CSV;
- JSON or YAML config;
- required columns: timestamp/open/high/low/close/volume.

Outputs:
- `report.json`;
- `summary.md`;
- source SHA-256;
- config SHA-256;
- per-check findings;
- affected row/timestamp references;
- package version;
- PASS/FAIL.

## v1 checks

- required columns;
- numeric parsing;
- timestamp parsing;
- strictly increasing timestamps;
- duplicate timestamps;
- expected interval gaps;
- NaN/+Inf/-Inf;
- OHLC consistency;
- non-negative volume;
- provenance metadata;
- source checksum.

## Fail-closed rules

- never modify the source file;
- never silently sort, deduplicate, interpolate, fill, or coerce an invalid field;
- unknown timezone/config ambiguity is an explicit failure or warning according to the frozen rule;
- malformed config prevents a PASS;
- report generation failure prevents a PASS.

## Evidence schema

```json
{
  "schema_version": "1.0",
  "source_sha256": "<hash>",
  "config_sha256": "<hash>",
  "row_count": 0,
  "time_range": {"start": null, "end": null},
  "checks": [
    {
      "id": "duplicate_timestamps",
      "status": "PASS",
      "count": 0,
      "examples": []
    }
  ],
  "overall_status": "PASS"
}
```

## Frozen synthetic benchmark

Exactly 70 deterministic fixtures:
- 10 clean;
- 10 duplicate timestamp;
- 10 out-of-order;
- 10 missing interval;
- 10 non-finite;
- 10 impossible OHLC;
- 10 negative volume.

Primary release thresholds:
- duplicate/non-finite/OHLC/negative-volume detection: 100%;
- out-of-order/gap detection: >=95%;
- clean false-positive rate: <=5%;
- repeat determinism: 100%;
- any source modification: automatic failure.

## Suggested module split

- `config.py` — parse/validate config;
- `reader.py` — read-only CSV parsing;
- `checks/schema.py`;
- `checks/timestamps.py`;
- `checks/numeric.py`;
- `checks/ohlcv.py`;
- `evidence.py` — hashes + JSON receipt;
- `cli.py`;
- `benchmark/fixtures.py`;
- `benchmark/run.py`;
- `tests/`.

## Builder split

Builder A:
- config;
- reader;
- schema/timestamp checks;
- unit tests.

Builder B:
- numeric/OHLCV checks;
- evidence writer;
- benchmark fixtures/runner;
- unit tests.

Independent reviewer:
- verifies frozen fixture counts;
- validates failure cases;
- checks no source mutation;
- reproduces benchmark receipt from a clean checkout.

## Definition of done

The artifact is complete only when the frozen 70-fixture benchmark runs deterministically, emits machine-readable evidence, meets the frozen thresholds, and the reviewer can reproduce it. Passing v1 does not prove full point-in-time safety for arbitrary financial datasets.
