# FinanceMeta Research Release Gate

A FinanceMeta research artifact is **not release-ready** until every mandatory gate below is satisfied or explicitly waived with a written reason and reviewer sign-off.

## Gate A — Question and claim lock

- [ ] Research question is falsifiable.
- [ ] Primary hypothesis and null hypothesis are written before protected evaluation.
- [ ] Primary metric is frozen.
- [ ] Failure / falsification condition is frozen.
- [ ] Public claim is written narrowly enough that a negative or inconclusive result remains publishable.

## Gate B — Data provenance and temporal integrity

- [ ] Dataset source, license, version and hash are recorded.
- [ ] Observation timestamps and availability timestamps are distinguished where relevant.
- [ ] All feature construction is performed using only information available at prediction time.
- [ ] Train/validation/test or walk-forward policy is frozen before held-out evaluation.
- [ ] Any normalization, imputation, feature selection and target construction are fit only on allowed historical data.
- [ ] Corporate actions, survivorship, delistings and universe membership are handled explicitly where relevant.
- [ ] A deliberately leaky control or other leakage diagnostic exists for pipelines where leakage is a material risk.

## Gate C — Baselines and controls

- [ ] At least one trivial baseline is included.
- [ ] At least one strong domain-relevant baseline is included.
- [ ] Hyperparameter/search budget is comparable across methods.
- [ ] Baselines are not removed because they outperform the proposed method.
- [ ] Ablations isolate the claimed mechanism where feasible.

## Gate D — Statistical discipline

- [ ] Seeds / resamples are predeclared where stochasticity matters.
- [ ] Uncertainty is reported, not only point estimates.
- [ ] Multiple-testing or model-selection effects are acknowledged where relevant.
- [ ] Transaction costs, slippage and turnover are included for trading-oriented claims.
- [ ] Economic significance is separated from statistical significance.
- [ ] Failed seeds and null runs are retained unless an exclusion rule was frozen in advance.

## Gate E — Reproducibility

- [ ] Commit SHA is recorded.
- [ ] Environment/dependency lock exists.
- [ ] Exact reproduction command exists.
- [ ] Config files are retained.
- [ ] Raw machine-readable outputs are retained or persistently linked.
- [ ] Tables and figures trace to retained outputs.
- [ ] Deterministic smoke test passes.
- [ ] A second person can execute the bounded reproduction path.

## Gate F — Evidence and claim audit

- [ ] Positive, negative and contradictory evidence are all recorded.
- [ ] Synthetic results are labeled synthetic.
- [ ] Simulated returns are not described as realized returns.
- [ ] No external-validation field is populated from outreach, plans or self-description.
- [ ] Reviewer confirms every headline claim is supported by an artifact.
- [ ] Limitations include the strongest plausible alternative explanation.

## Gate G — Release decision

Allowed decisions:

- **RELEASE** — all mandatory gates passed.
- **RELEASE_WITH_LIMITATIONS** — bounded release with explicit reviewer-approved limitations.
- **CONTINUE** — more decisive evidence required.
- **FREEZE** — protocol or integrity problem blocks further protected evaluation.
- **ARCHIVE** — project is no longer worth additional execution.

A negative or inconclusive result may still receive **RELEASE** if the protocol and evidence quality pass the gate.

## Required release packet

1. manuscript/report;
2. research evidence record;
3. code and environment lock;
4. frozen protocol/config;
5. raw or traceable outputs;
6. generated tables/figures;
7. limitations and negative findings;
8. reviewer decision;
9. release notes with commit SHA.

## Finance-specific hard stops

Do **not** unlock protected evaluation if any of the following remains unresolved:

- look-ahead leakage;
- global preprocessing fit before chronological split;
- survivorship bias with an unqualified historical-performance claim;
- unrecorded universe changes;
- unlabeled synthetic or simulated results;
- post-hoc metric replacement;
- hidden seed filtering;
- transaction-cost-free profitability claims;
- held-out test inspection before configuration freeze.

This checklist operationalizes the integrity requirements in `OPERATING_SYSTEM_2026.md`.