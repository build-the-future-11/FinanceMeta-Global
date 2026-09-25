# FinanceMetaBench v0.1 — Pre-Result Specification

**Status:** specification only. No leaderboard or performance claim exists yet.

## Goal

Create a small, reproducible finance-ML benchmark that rewards chronology, provenance, calibration, robustness, and transparent baselines rather than one headline backtest number.

## v0.1 task families

### T1 — Next-horizon directional classification
Predict a prespecified future direction label using only information available at decision time.

### T2 — Volatility forecasting
Forecast a prespecified realized-volatility target.

### T3 — Cross-sectional ranking
Rank eligible assets/entities on a frozen future target using a point-in-time universe.

### T4 — Regime classification
Classify a prespecified regime definition built without future leakage.

### T5 — Representation transfer
Freeze an encoder/representation and evaluate with a fixed low-capacity probe on one or more downstream targets.

The first public release may include fewer tasks if provenance quality is stronger that way.

## Dataset acceptance gate

A dataset enters the benchmark only if:
- license/use permits the intended benchmark release;
- source and acquisition date are recorded;
- immutable hashes are stored;
- timestamp semantics are documented;
- universe construction is reproducible;
- survivorship/restatement risks are documented;
- split construction is reproducible.

## Common chronology

Each task must use chronological partitions. Random split results may exist only as explicitly labeled negative controls.

Any rolling normalization/fitting must use information available at that point in the timeline.

## Baseline set

Every task includes:
- naive baseline;
- transparent linear/statistical baseline;
- one task-appropriate classical ML baseline;
- optional advanced model families.

No submission is compared only against other complex models.

## Metrics

### Predictive metrics
Use task-appropriate metrics such as:
- MAE/RMSE;
- balanced accuracy/F1 where class imbalance warrants;
- rank correlation;
- Brier score/calibration error.

### Robustness metrics
- seed dispersion;
- per-regime performance;
- temporal slice performance;
- falsification outcomes.

### Economic simulation metrics
Only for tasks with an explicitly separate simulated decision layer:
- turnover;
- gross and net simulated return;
- drawdown;
- cost sensitivity.

These are secondary to the core predictive benchmark unless a future benchmark version explicitly defines otherwise.

## Submission artifact

Each submission must provide:
- model identifier/version;
- source SHA;
- environment lock;
- dataset manifest hash;
- configuration;
- seeds;
- raw predictions where redistribution rules permit;
- metrics JSON;
- falsification JSON;
- runtime/hardware metadata;
- claim/limitation statement.

## Anti-leaderboard gaming rules

- fixed public metric definitions;
- hidden/held-out labels remain inaccessible until authorized evaluation;
- bounded submission frequency if an external evaluation service is later introduced;
- no manual deletion of failed seeds;
- all material configurations logged;
- post-hoc ensembles require explicit versioning;
- benchmark organizers may invalidate runs that violate provenance or chronology even if metrics are high.

## v0.1 promotion gate

FinanceMetaBench becomes an executed internal benchmark only when:
1. at least one dataset clears the acceptance gate;
2. task splits are frozen and hashed;
3. baseline implementations reproduce;
4. an independent reviewer confirms chronology/leakage rules;
5. result storage is machine-readable and source-bound.

A public leaderboard is a later decision, not implied by the v0.1 specification.
