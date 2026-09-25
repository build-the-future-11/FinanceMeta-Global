# FinanceBench v0

FinanceBench is a reusable evidence contract for FinanceMeta financial-ML work. It is designed to make chronology, leakage, baselines, regimes, costs, and claim boundaries reviewable before a result is promoted.

## Required manifest fields

Each evaluation must provide a JSON manifest with:

- `version`: exactly `financebench-v0`
- `project_id`
- `dataset.name`, `dataset.version`, `dataset.as_of`
- `chronology.train_end`, `chronology.validation_end`, `chronology.test_end`
- `chronology.point_in_time_verified`
- `evaluation.walk_forward`
- `evaluation.regimes`
- `evaluation.transaction_cost_bps`
- `baselines`
- `metrics`
- `claim_boundary`
- `reproduction.command`
- `reproduction.source_revision`

## Hard gates

1. **Point-in-time integrity**
   Every feature used at timestamp t must be available at or before t under the declared data source/version.

2. **Chronology**
   `train_end < validation_end < test_end`. Random shuffled splits are not a substitute for temporal evaluation.

3. **Walk-forward evidence**
   The manifest must state whether a walk-forward protocol is used. If false, the claim boundary must explain why.

4. **Regime analysis**
   Regimes must be declared by rule rather than chosen after viewing model outcomes.

5. **Transaction costs**
   Portfolio/trading claims require an explicit cost-sensitivity grid. A zero-cost result may still be reported, but cannot support a net-performance claim by itself.

6. **Baselines**
   Include at least one simple baseline appropriate to the task. A complex model beating only another complex model is insufficient evidence of practical value.

7. **Claim separation**
   Forecast accuracy, simulated portfolio performance, paper-trading/live performance, and deployability are separate claim levels.

## Recommended claim levels

- `FORECAST_ONLY`: predictive metric only.
- `SIMULATION_GROSS`: simulated portfolio result before costs.
- `SIMULATION_NET`: simulated result after declared costs.
- `PAPER_TRADING`: prospective paper-trading evidence.
- `LIVE`: live capital evidence.

A manifest may declare a lower claim level than the evidence could potentially support. It must not declare a higher one.

## Example validation

```bash
python scripts/validate_financebench_manifest.py evaluation/financebench-v0/example-manifest.json
```

The validator checks structure and a small number of fail-closed integrity rules. It does not certify that the underlying data or strategy is scientifically valid; reviewers must verify the source artifacts.
