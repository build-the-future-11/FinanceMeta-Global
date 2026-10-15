# FinanceBench v0

FinanceBench is a reusable evidence contract for FinanceMeta financial-ML work. It makes chronology, leakage, baselines, regimes, costs, and claim boundaries reviewable without forcing older frozen experiments to add analyses after outcomes are known.

## Required manifest fields

Each evaluation must provide a JSON manifest with:

- `version`: exactly `financebench-v0`
- `project_id`
- `dataset.name`, `dataset.version`, `dataset.as_of`
- `chronology.train_end`, `chronology.validation_end`, `chronology.test_end`
- `chronology.point_in_time_verified`
- `evaluation.walk_forward`
- `evaluation.regime_analysis_status`
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
   The manifest must state whether a walk-forward protocol is used.

4. **Regime analysis without hindsight**
   Set `regime_analysis_status` to:
   - `PREDECLARED` when regime definitions were fixed before outcome inspection; or
   - `NOT_EVALUATED` when the frozen study did not include regime analysis.

   If `NOT_EVALUATED`, `regimes` must be empty. Do not retroactively invent regimes merely to satisfy FinanceBench. Simulation/trading claim levels require predeclared regime analysis.

5. **Transaction costs**
   Portfolio/trading claims require an explicit cost-sensitivity grid. A zero-cost result may still be reported, but cannot support a net-performance claim by itself.

6. **Baselines**
   Include at least one simple baseline appropriate to the task. A complex model beating only another complex model is insufficient evidence of practical value.

7. **Claim separation**
   Forecast accuracy, simulated portfolio performance, paper-trading/live performance, and deployability are separate claim levels.

## Claim levels

- `FORECAST_ONLY`: predictive metric only.
- `SIMULATION_GROSS`: simulated portfolio result before costs.
- `SIMULATION_NET`: simulated result after declared costs.
- `PAPER_TRADING`: prospective paper-trading evidence.
- `LIVE`: live capital evidence.

A manifest may declare a lower claim level than the evidence could potentially support. It must not declare a higher one.

## Frozen-study rule

FinanceBench can be applied retrospectively for **packaging/audit only** when a study is already frozen. Missing nonessential analyses must be recorded as not evaluated, not added after outcome inspection. Applying the standard does not authorize new data access, tuning, robustness checks, or result-bearing runs.

## Example validation

```bash
python scripts/validate_financebench_manifest.py evaluation/financebench-v0/example-manifest.json
```

The validator checks structure and fail-closed integrity rules. It does not certify the underlying data or scientific result; reviewers must verify the retained source artifacts.
