# FinanceBench v0

FinanceBench is FinanceMeta's evaluation contract for financial machine-learning experiments. It is designed to make temporal integrity, leakage control, baseline fairness, uncertainty, trading assumptions, reproducibility, and claim calibration explicit before a result is promoted.

## v0 contract

Every benchmark package must contain:

- `benchmark.json` matching `schema/financebench-v0.schema.json`;
- a frozen task/target/horizon definition;
- dataset provenance and availability-time assumptions;
- chronological or walk-forward split policy;
- baseline registry;
- metric registry;
- leakage audit;
- transaction-cost assumptions where applicable;
- reproducibility receipt with commit/config/environment;
- machine-readable result card;
- claim ledger;
- final release decision.

## Evaluation sequence

```text
DATA PROVENANCE
  -> TEMPORAL INTEGRITY
  -> TRAIN/VALIDATION SELECTION
  -> CONFIG FREEZE
  -> PROTECTED EVALUATION
  -> UNCERTAINTY + COST SENSITIVITY
  -> CLAIM AUDIT
  -> RELEASE DECISION
```

A leaderboard is explicitly out of scope until the integrity contract is stable.

## Pilot

The first pilot should wrap an already-frozen FinanceMeta experiment without changing its scientific conclusion. The pilot validates the benchmark contract, not the hypothesis.
