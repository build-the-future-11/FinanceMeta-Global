# FinanceMeta backtest ledger audit

**Status: PASS**

Data kind: **synthetic**. Asset: `SYNTHETIC_ARITHMETIC_FIXTURE`.
Input SHA-256: `309490bd9893c4d49265887eb63a53bc2d19659505e102494dab762c428310ab`

| Check | Status | Failures |
|---|---|---:|
| schema | PASS | 0 |
| nonempty_data | PASS | 0 |
| parseability | PASS | 0 |
| strict_decision_order | PASS | 0 |
| feature_availability | PASS | 0 |
| training_cutoff | PASS | 0 |
| future_target | PASS | 0 |
| contiguous_intervals | PASS | 0 |
| numerical_contract | PASS | 0 |

## Hypothetical cost-aware ledger results

| Method | Net total return | Max drawdown | Turnover units |
|---|---:|---:|---:|
| strategy | -2.446481% | 3.316631% | 8.0000 |
| buy_and_hold | 1.158764% | 2.000000% | 2.0000 |
| cash | 0.000000% | 0.000000% | 0.0000 |

## Declared invalidation condition

Withhold every performance metric if any input column, timestamp ordering, latest feature availability, latest training-label availability, target interval, or numerical-contract check fails. This fixture is not evidence of economic performance.

## Limits

- These are hypothetical ledger returns, not realized investment returns.
- Producer-supplied availability/training timestamps are checked for internal consistency only.
- No survivorship, corporate-action, publication-time, borrow-cost, market-impact, or execution-price certification.
- Costs use changes in declared target exposure; drift-induced rebalancing and financing are not modeled.
- Target exposures outside [-1, 1] and zero/negative wealth paths are unsupported.
- No model is trained or selected and no new protected study is authorized by this audit.
