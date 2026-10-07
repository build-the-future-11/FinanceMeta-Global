# FinanceMeta backtest ledger audit

**Status: FAIL**

Data kind: **synthetic**. Asset: `SYNTHETIC_ARITHMETIC_FIXTURE`.
Input SHA-256: `182d5b5f28d15a3add08dbbdc08a5913d230ab372f50163bf570f0300a3fe96c`

| Check | Status | Failures |
|---|---|---:|
| schema | PASS | 0 |
| nonempty_data | PASS | 0 |
| parseability | PASS | 0 |
| strict_decision_order | PASS | 0 |
| feature_availability | FAIL | 1 |
| training_cutoff | PASS | 0 |
| future_target | PASS | 0 |
| contiguous_intervals | PASS | 0 |
| numerical_contract | PASS | 0 |

Performance metrics withheld because the ledger failed validation.

## Declared invalidation condition

Withhold every performance metric if any input column, timestamp ordering, latest feature availability, latest training-label availability, target interval, or numerical-contract check fails. This fixture is not evidence of economic performance.

## Limits

- These are hypothetical ledger returns, not realized investment returns.
- Producer-supplied availability/training timestamps are checked for internal consistency only.
- No survivorship, corporate-action, publication-time, borrow-cost, market-impact, or execution-price certification.
- Costs use changes in declared target exposure; drift-induced rebalancing and financing are not modeled.
- Target exposures outside [-1, 1] and zero/negative wealth paths are unsupported.
- No model is trained or selected and no new protected study is authorized by this audit.
