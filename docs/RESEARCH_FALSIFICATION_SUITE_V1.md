# FinanceMeta Research Falsification Suite v1

## Purpose

The suite is a common set of adversarial checks intended to break fragile finance-research claims before release. Projects select applicable checks in the pre-result contract and retain failures.

## Required result schema

Each check should emit a row with:

`project_id, check_id, config_hash, source_sha, data_hash, seed, status, primary_metric, control_metric, threshold, interpretation, artifact_path`

Allowed `status` values:
- PASS;
- FAIL;
- INCONCLUSIVE;
- NOT_APPLICABLE;
- NOT_RUN.

## F01 — Label permutation

Randomly permute labels/targets within the permitted design.

Expected behavior: predictive signal should collapse toward the task-appropriate null.

Failure signal: comparable performance after permutation.

## F02 — Timing shift

Shift features forward/backward by a prespecified interval.

Expected behavior: future-dependent alignment should not improve legitimate performance; a deliberately broken alignment should normally degrade.

Failure signal: implausible improvement after a shift that introduces or exposes future information.

## F03 — Split-boundary audit

Enumerate feature and label windows at train/validation/test boundaries.

Expected behavior: no sample consumes observations outside its assigned split under the frozen convention.

Output boundary-exclusion counts.

## F04 — Seed sensitivity

Run the fixed seed list.

Expected behavior: the claim should not depend on one favorable seed unless the claim explicitly concerns variance.

Report every seed.

## F05 — Train-window perturbation

Repeat with prespecified nearby historical windows.

Expected behavior: a robust structural claim should not require one exact start date without explanation.

## F06 — Feature ablation

Remove prespecified feature families one at a time.

Expected behavior: interpretation should reflect which information actually carries performance.

Ablation is not feature fishing; the set is frozen before held-out inspection.

## F07 — Placebo features

Add or substitute noise/placebo features.

Expected behavior: a valid pipeline should not consistently extract large "signal" from placebos.

## F08 — Regime exclusion

Train/re-evaluate under prespecified regime exclusions where scientifically meaningful.

Expected behavior: reveal whether a headline average is entirely driven by one regime.

Never delete the adverse regime from the main report.

## F09 — Parameter sensitivity

Evaluate a bounded grid around prespecified key assumptions.

Expected behavior: conclusions should not flip under trivial perturbations without that fragility being reported.

## F10 — Cost sensitivity

For simulated strategy layers only, rerun with at least two prespecified fee/slippage levels plus the base case.

Report turnover alongside net performance.

## F11 — Baseline dominance check

Re-run the transparent baselines under exactly the candidate pipeline's split/preprocessing conventions.

Failure signal: candidate advantage disappears when baselines receive fair treatment.

## F12 — Duplicate / near-duplicate audit

Check whether duplicate records or overlapping windows leak near-identical examples across splits.

## F13 — Universe perturbation

Where applicable, evaluate a prespecified alternate universe or membership rule.

Purpose: detect survivor/universe dependence.

## F14 — Multiple-testing ledger

Record every materially distinct model/configuration comparison inspected during the study.

Use the ledger to bound claims and apply a correction or exploratory label where necessary.

## Release rule

A FAIL does not automatically invalidate all research. It changes the permitted claim.

Examples:
- failed timing shift/leakage check: block predictive interpretation until repaired;
- high seed variance: permit an instability finding, not a robust-performance claim;
- cost sensitivity failure: permit a gross-simulation result, not a cost-robust economic claim;
- baseline dominance failure: permit an implementation result, not a superiority claim.

## Machine-readable contract example

```json
{
  "suite_version": "1.0",
  "project_id": "FI-JEPA",
  "frozen_checks": ["F01","F02","F03","F04","F05","F06","F11","F12","F14"],
  "source_sha": "<commit>",
  "data_hash": "<hash>",
  "config_hash": "<hash>",
  "result_state": "UNTESTED"
}
```

Do not fill outcome fields before execution.
