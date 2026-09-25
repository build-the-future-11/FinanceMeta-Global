# FinanceMeta Research Standard v1

## Scope

This standard applies to FinanceMeta empirical finance, quantitative research, forecasting, representation learning, market microstructure, portfolio simulation, and financial-ML projects. It extends the repository-wide operating system with a concrete experiment contract.

## 1. Required pre-result contract

Before outcome-bearing held-out evaluation, freeze:

- research question;
- primary hypothesis and null;
- primary metric;
- secondary metrics;
- asset/entity universe;
- date range;
- prediction/decision horizon;
- timestamp convention;
- raw data fields;
- feature availability rule;
- label definition;
- train/validation/test chronology;
- fitting/normalization scope;
- baseline set;
- model family;
- hyperparameter/search budget;
- seed policy;
- regime definition;
- cost/execution assumptions if a simulated strategy is studied;
- primary failure condition;
- allowed post-freeze changes.

Every change after freeze must be logged with reason and whether it invalidates confirmatory status.

## 2. Data provenance minimum

Each dataset gets a machine-readable manifest with:

- canonical name;
- provider/source;
- access date;
- license/use restriction;
- immutable file/object hash where possible;
- timezone;
- raw timestamp semantics;
- corporate-action policy where relevant;
- universe construction rule;
- delisting/survivorship treatment;
- missing-data policy;
- known revisions/restatements;
- point-in-time availability notes.

Do not infer point-in-time validity because a historical dataset contains timestamps.

## 3. Leakage firewall

Projects must test at least:

1. split boundaries do not share observations;
2. every feature window ends before the decision timestamp;
3. every label window is contained entirely inside its assigned split;
4. normalization/fitting statistics use training-available information only;
5. feature selection uses no held-out information;
6. regime definitions do not use future outcomes;
7. universe selection does not silently exclude failed/delisted entities where that would bias the task;
8. target-derived columns cannot enter feature construction;
9. duplicate or near-duplicate records cannot cross splits unnoticed.

A leakage failure blocks result interpretation until repaired and rerun.

## 4. Baseline ladder

The minimum comparison ladder is:

- naive/persistence or historical-mean baseline;
- transparent linear/statistical baseline;
- task-appropriate classical baseline;
- candidate model.

A candidate model is not "better" unless the same data, split, preprocessing, search budget, and evaluation rules are applied fairly.

## 5. Reproducibility contract

A reviewable result must bind:

- repository;
- exact commit SHA;
- data-manifest hash;
- configuration hash;
- environment/dependency lock;
- random seeds;
- command used;
- raw result file;
- generated table/figure derivation.

A screenshot or copied metric is not sufficient evidence.

## 6. Statistical discipline

Where repeated sampling/seeds are applicable:

- retain all prespecified seeds;
- report central tendency and dispersion;
- include confidence intervals or bootstrap intervals where meaningful;
- distinguish exploratory from confirmatory comparisons;
- correct or explicitly bound multiple testing where many hypotheses are evaluated;
- report effect size, not only significance;
- retain underperforming regimes and subgroups.

## 7. Simulated economic evaluation

If an experiment adds a strategy layer, report separately from predictive/representation metrics:

- gross simulated return;
- assumed fees;
- spread/slippage convention;
- turnover;
- net simulated return;
- drawdown;
- cost sensitivity;
- execution timing;
- rebalance frequency.

These are simulated research quantities, not realized performance or investment advice.

## 8. Falsification requirement

Every substantive result must include applicable checks from `RESEARCH_FALSIFICATION_SUITE_V1.md`. A result that disappears under an expected-invariance check must be treated as weakened or falsified, not hidden.

## 9. Claim states

Use exactly one result state:

- POSITIVE: prespecified claim survives the defined tests;
- NEGATIVE: prespecified claim fails;
- INCONCLUSIVE: evidence cannot distinguish;
- UNTESTED: outcome-bearing evaluation has not occurred.

The result state is distinct from repository maturity or publication status.

## 10. Review gate

A release-ready package requires:
- no unresolved leakage blocker;
- reproducible baseline;
- raw machine-readable outputs;
- falsification results;
- explicit limitations;
- claim text bounded by the actual evidence;
- reviewer note from someone other than the primary implementer where feasible.

## 11. Prohibited shortcuts

Do not:
- random-split time series merely to improve metrics;
- choose seeds after inspecting outcomes;
- change baselines because they outperform the candidate;
- alter regimes after seeing test performance;
- silently drop failed runs;
- backfill preregistration language after outcome inspection;
- describe synthetic or simulated outcomes as real-market results;
- equate green CI with scientific validation;
- equate a draft/preprint with peer-reviewed acceptance.
