# F4A-07 — FI-JEPA temporal-integrity fix receipt

**Date:** 2026-09-28  
**Repository:** `Finance-Meta-Research/FI-JEPA`  
**Pull request:** #7 — `F4A-07: make FI-JEPA temporal preprocessing fail closed`  
**Head:** `89b4f2e767f2adeb62dcb2b1cd594185c07b323c`

## Defect found during canonical audit

The generic FI-JEPA data preparation path performed backward fill before the chronological train/validation/test split and used two-sided interpolation during optional resampling. That behavior can cause a value observed later in time to populate an earlier missing row.

The built-in macrodata benchmark derives and fills its initial lagged values separately, so discovering this generic defect is **not** by itself evidence that the saved macro result table is invalid or valid. Saved results still require an exact replay/evidence audit.

## Fix opened

PR #7:

- rejects missing/unparsable timestamps instead of synthesising them;
- rejects duplicate asset/timestamp rows;
- uses per-asset past-only forward fill;
- fails closed when leading missing values remain instead of copying future values backward;
- resamples with right-edge labels and past-only carry-forward;
- preserves train-only normaliser fitting;
- adds planted regression tests for future backfill, cross-asset filling, duplicate timestamps and train-only normalisation;
- adds a scoped GitHub Actions workflow.

## CI receipt

GitHub Actions run `36454963553`, workflow **Temporal Integrity**, completed successfully. The dedicated test job and the `Temporal integrity tests` step both concluded `success`.

## What this does not prove

The CI receipt verifies the new code/tests executed successfully. It does not certify:

- point-in-time correctness of every external dataset;
- the historical availability of the revised `statsmodels` macrodata values;
- saved benchmark/paper result provenance;
- fair comparator selection;
- paper figures/tables against raw retained outputs;
- tradable or economically significant performance.

F4A-07 therefore remains held. The next gate is reviewer acceptance of PR #7 followed by a pinned canonical commit and replay/audit of the saved benchmark artifacts.
