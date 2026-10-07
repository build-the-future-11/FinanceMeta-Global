# Auditable prediction ledgers and retained negative evidence

## Status and question

**Completed:** an executable read-only audit tool, regression tests, retained synthetic arithmetic/future-information examples, and a provenance-bound replay of an existing negative forecasting result. **Research release status:** draft; an independent reviewer and observed trading-ledger study remain absent. The engineering artifact has internal evidence (M2/E2); no peer-reviewed, externally validated or profitable strategy result is claimed.

The bounded engineering question is whether a producer's frozen prediction ledger can be rejected before performance reporting when its declared schema, information availability, training cutoff, interval chronology or numerical inputs violate an explicit contract. A separate arithmetic check asks whether retained seed-level FI-JEPA outcomes reproduce the reported matched-baseline comparison. These are different evidence types; neither establishes new forecasting accuracy.

## Implementation and identified defects

The prior OHLCV checker could return PASS for a header-only file. It did not reject duplicate headers or inconsistent row width, and a null provenance value could be accepted after conversion to a string. The checker now rejects those inputs. It snapshots source bytes once so the report hash describes precisely the input it inspected. This preserves the original fixed 70-fixture benchmark and adds regression cases without changing its outcomes.

The new `backtest` command consumes a frozen single-asset CSV. `available_at` means the latest information-availability timestamp over all features used in that row; `trained_through` means the latest availability timestamp over all training labels used. Both must be no later than the decision. This requirement is stronger than merely sorting observation dates, but the tool cannot independently prove that producer-supplied metadata is truthful. The full contract is in [README.md](README.md).

The checker rejects naive timestamps, duplicate or disordered decisions, overlapping or missing return intervals, future features/training labels, nonfinite values and unsupported exposures. Exact timezone offsets are compared as instants. The invalidation rule is fixed: any failed check withholds every performance path, including baselines. Errors list affected CSV rows in machine-readable output.

For valid ledgers, deterministic arithmetic reports strategy, buy-and-hold and cash wealth for identical intervals. The explicit cost approximation is

`period_return = position * asset_return - (fee_bps + slippage_bps) / 10000 * traded_exposure`.

`traded_exposure` is the absolute change from the previous declared target, initially zero; terminal liquidation adds the final absolute target exposure. Reversing from -1 to +1 trades two exposure units. Cash has no market exposure or interest. The method charges a declared exposure-based approximation, not a broker execution or exact self-financing portfolio simulation. Drift-induced rebalancing, financing, borrow availability, corporate actions, impact and trade execution are outside the contract. Reports expose the period costs, turnover, returns and wealth, rather than only an aggregate headline.

## Executed controls

The manually specified six-interval fixture includes long, short, flat and fractional exposures. It has an intentionally losing strategy; no parameter was fitted or chosen from observed market prices. The future-feature control keeps all values identical except that the first feature becomes available one second after its decision.

| Evidence | Observed outcome | Interpretation |
|---|---|---|
| Clean six-interval ledger | PASS; hypothetical strategy -2.446481%, matched buy-and-hold +1.158764%, cash 0% | Arithmetic illustration only; the declared strategy loses even though its file passes validation |
| One-second future-feature twin | FAIL; all performance metrics withheld | Information-consistency failure is visible before publishing a P&L summary |
| Existing frozen OHLCV benchmark | All 70 expected fixture classifications retained | Existing synthetic coverage remains intact |
| Unit suite | 17 tests pass | Includes hand-calculated reversal/terminal costs, timezone equality, malformed input, missing assumptions, future information and no-overwrite behavior |

All example input CSVs and configs, detailed reports, summaries and output hashes are retained under [examples](examples/). The CLI report also binds input/config hashes. Deterministic fixture replay is compared byte-for-byte in CI. These controls demonstrate behavior on known errors; they are not estimates of detection sensitivity on naturally occurring market-data failures.

## Retained FI-JEPA evidence

The independent arithmetic replay reads canonical [FI-JEPA `paper_results.json` at c94e1616](https://github.com/Finance-Meta-Research/FI-JEPA/blob/c94e1616eba1b2a7415ead4695def1d3b93094db/experiments/paper_results.json). It accepts only SHA-256 `525fe5f141bba765b3346c1aca3b327303703e5f98142b0f25d664211e10f7c8`, protocol `FIJEPA_MACRODATA_PAPER_V1_20260909`, and all three predeclared seeds. The source is a retained GDP-growth proxy prediction study on statsmodels macrodata; it is not a trading ledger.

| Seed | Full latent probe MSE | Matched raw-context Ridge MSE | Full minus Ridge |
|---:|---:|---:|---:|
| 7 | 0.8945053220 | 0.4419237077 | +0.4525816143 |
| 17 | 0.8618583679 | 0.4419237077 | +0.4199346602 |
| 27 | 1.0040714741 | 0.4419237077 | +0.5621477664 |

The recomputed mean difference is **+0.4782213469 MSE**, agreeing with the retained aggregate. Full latent MSE is worse for every retained seed. The reported bootstrap interval [0.4199346602, 0.5621477664] is retained as an upstream value; this tool does not rerun bootstrap inference. Full and `no_operator_split` MSE are identical for all three seeds. The source's one-stage setup makes that operator-removal condition non-identifying, so it does not support a positive mechanism claim.

The canonical study remains **CLOSED NEGATIVE / BOUNDARY**, consistent with its [final status](https://github.com/Finance-Meta-Research/FI-JEPA/blob/c94e1616eba1b2a7415ead4695def1d3b93094db/FINAL_STATUS_2026-09-30.md). This replay did not fit a model, recompute dataset splits, inspect a new holdout, retry seeds, change endpoints or authorize a successor. The audit validates source binding and arithmetic; it does not independently revalidate original training execution or dataset vintage integrity.

## Reproduce

From this package directory, Python 3.11+ is sufficient and no runtime dependency is required:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/reproduce.py /tmp/financemeta-ledger-new
python benchmark/run.py /tmp/financemeta-benchmark-new
cmp /tmp/financemeta-benchmark-new/benchmark_summary.json benchmark/evidence/benchmark_summary.json
curl --fail --location 'https://raw.githubusercontent.com/Finance-Meta-Research/FI-JEPA/c94e1616eba1b2a7415ead4695def1d3b93094db/experiments/paper_results.json' --output /tmp/fijepa-v1-pinned.json
python examples/audit_retained_fijepa.py /tmp/fijepa-v1-pinned.json --output /tmp/fijepa-audit-new.json
cmp /tmp/fijepa-audit-new.json examples/evidence/fijepa-v1-retained-audit.json
```

Existing output paths are refused. The workflow pins Python and source checkout, repeats both fixture and retained-evidence replays and retains generated outputs. Source edits never alter the upstream closed result.

## Submission scope and remaining gates

The current contribution can support an honest financial-AI operations or tools demonstration: reproducible chronology checks, failure examples, explicit cost assumptions and preserved negative evidence. It is not yet an empirical trading or forecasting paper for ICAIF/KDD. No observed trading ledger with licensed prices and independently substantiated feature-publication/training-label times was present in this repository.

A new empirical study would require an identified permissible dataset and vintage/availability record, frozen question and evaluation protocol, matched method/search budget, held-out or chronological evaluation, realistic costs and universe accounting, retained uncertainty and failures, and independent review under the repository's release standard. The existing FI-JEPA v1 result must remain closed; a materially distinct successor would need its own authorization and protocol. No external review, acceptance, user adoption or finance-program scale is inferred from this code delivery.
