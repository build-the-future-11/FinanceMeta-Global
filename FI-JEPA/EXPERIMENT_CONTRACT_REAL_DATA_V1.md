# FI-JEPA Real-Data Experiment Contract v1

**Status:** PRE-RESULT / protocol only.  
**Current project maturity remains:** M1 / E1 until evidence justifies promotion.

## Question

Does a JEPA-style latent predictive objective learn financial time-series representations that transfer more robustly across chronological market regimes than transparent non-JEPA representation baselines under an identical leakage-safe protocol?

## Null

After fair preprocessing, identical chronological splits, and a fixed probe budget, FI-JEPA representations do not materially improve the prespecified downstream representation metrics over transparent baselines, or any apparent gain is unstable across seeds/regimes.

## Confirmatory boundary

This contract must be frozen before the confirmatory held-out test is accessed. Validation may be used only within the fixed search budget. Held-out outcomes must not change:
- feature set;
- architecture family;
- split;
- regime rule;
- probe;
- baseline set;
- seed list;
- headline metric.

## Dataset gate

No real-data execution until the dataset manifest records:
- provider and license;
- acquisition date;
- immutable raw-data hash;
- point-in-time timestamp semantics;
- timezone;
- universe construction;
- corporate-action handling if applicable;
- survivorship/delisting policy;
- missing-data policy;
- allowed redistribution.

The first real-data implementation should favor a small, auditable universe over a broad dataset with uncertain provenance.

## Chronology

Use non-overlapping chronological train, validation, and held-out test periods.

All context windows must end before their targets begin. Feature/target windows crossing a split boundary are excluded and counted. Fitting statistics are learned from the permitted historical segment only.

## Representation task

For each sample:
- context: past multivariate block;
- target: strictly later block;
- online/context encoder produces context representation;
- EMA target encoder produces target representation;
- predictor estimates target representation from context representation.

No raw target reconstruction objective is required for the headline FI-JEPA condition.

## Baselines

Minimum:
1. raw standardized-window features + frozen linear probe;
2. PCA representation + frozen linear probe;
3. autoregressive/statistical features + frozen linear probe;
4. simple non-JEPA encoder trained with a transparent objective;
5. FI-JEPA.

The same downstream probe family and budget must be used wherever the representation dimensions permit.

## Primary metric

Primary metric should be a representation-transfer metric chosen before test access, such as held-out probe error on a bounded forecasting/classification target.

Secondary metrics:
- directional accuracy where meaningful;
- calibration where probabilistic;
- per-regime probe performance;
- seed dispersion;
- representation collapse diagnostics;
- nearest-neighbor or covariance diagnostics if prespecified.

Any simulated trading layer is secondary and separate.

## Seeds

Use a fixed prespecified seed list. Report every seed. No "best seed" result.

## Regimes

Regimes must be defined from information available at or before each observation and frozen before held-out inspection. Report aggregate and per-regime results; do not delete losing regimes.

## Ablations

Prespecify at least:
- context length;
- target horizon;
- EMA momentum;
- predictor depth;
- latent dimension;
- masking/context-target block rule.

Ablation budgets must be fixed before confirmatory test access.

## Falsification checks

At minimum:
- label/target permutation control for downstream probe;
- feature timing shift;
- seed sensitivity;
- train-window perturbation;
- representation collapse checks;
- context/target chronology assertion;
- split-boundary exclusion audit.

## Evidence package

Required files for a real-data run:
- `experiment_contract.json`;
- `data_manifest.json`;
- `split_manifest.json`;
- `config.json`;
- `results/per_seed.csv`;
- `results/per_regime.csv`;
- `results/ablations.csv`;
- `results/falsification.csv`;
- `findings.md`;
- exact reproduction command;
- source SHA.

## Interpretation rules

A result is not evidence of "alpha" merely because a representation probe improves.

If FI-JEPA:
- loses to simple baselines: retain as NEGATIVE;
- wins only on one seed/regime: report instability;
- wins validation but not held-out: report NEGATIVE/INCONCLUSIVE according to the frozen rule;
- wins representation metrics but not a separate simulated strategy layer: do not merge those claims;
- fails falsification: block the stronger interpretation.

## Promotion gate

Promotion beyond M1/E1 requires an executed, reproducible, provenance-complete real-data package. External review or reproduction is a separate later gate.
