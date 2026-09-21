# FinanceMeta / Finance4All validation — 2026-09-22

This document is an evidence-bound validation snapshot. It does not promote any program, research result, deployment, partner relationship, market-performance claim, or production-security claim. Where repository state and public/product copy differ, the narrower evidence-supported statement wins.

## Research status

### FinanceMeta-Global FI-JEPA

Current registry status remains **M1 / E1 — executable synthetic baseline only**. The package is useful as an engineering/reproducibility baseline, not evidence of novelty, real-market predictive power, alpha, profitability, or trading suitability.

A research-integrity defect remains on `main@112154d54e26889b0a65f15b2feefa8121ffc4c8`: synthetic data are standardized using mean/std from the full generated timeline before the chronological train/validation boundary is created. This allows validation-period distribution information to influence training normalization.

Draft PR #63 is the canonical repair surface. It moves normalization after split construction, fits scaling statistics only on pre-split observations, applies the frozen transform to validation, and adds regressions showing post-split perturbations cannot alter normalized training tensors. Its exact head passed the FI-JEPA CI lane. Do not duplicate this fix or use post-repair metric changes as a tuning signal.

Even after #63, promotion beyond M1/E1 requires point-in-time data/provenance, survivorship controls, multiple frozen seeds, walk-forward evaluation, stronger linear/autoregressive baselines, regime testing, ablations, collapse diagnostics, confidence intervals, and machine-readable output/config/commit provenance. A trading study, if any, must be separate from representation-learning validation and must state transaction-cost/slippage assumptions explicitly.

### Finance-Meta-Research/FI-JEPA

Treat this as a separate research surface from the compact `FinanceMeta-Global/FI-JEPA` baseline until a canonical ownership/migration decision is made.

Its repository contains a paper-evidence gate whose current status is **FAIL**. The gate rejects selecting among overlapping historical result families after seeing outcomes and requires one canonical `experiments/paper_results.json` with at least three predeclared seeds, complete required variant-by-seed cells, and an explicit downstream baseline for every seed. That canonical artifact is currently absent.

Therefore phrases such as “publishable preprint” are repository-intent language, not evidence that the paper is currently submission-ready or scientifically validated.

### Eigen-JEPA

The retained final-rigor-v2 synthetic package passes its machine gate with five frozen seeds and four variants. The scientific result is mixed, not a full-model-dominance result. The retained evidence reports that `full` improves some metrics versus `no_memory`, while Tail F1 is identical across variants, `no_gate` has better mean Drift MSE, and `no_regime` has the best mean Eig NMSE among the four paper-facing variants.

No preregistered significance test or practical-effect threshold exists for that frozen comparison. Keep all component comparisons descriptive. Do not claim trading alpha, deployable trading performance, real-market validation, cross-market validation, statistical significance, or superiority to unevaluated classical covariance baselines. The separate real-market confirmation remains pre-outcome.

### EigenFinance

Status is **RESERVED_PLACEHOLDER**. There is no implementation, experiment, dataset, paper, product, or result committed. Do not present the name as an active validated research result. Activation requires a reviewed charter defining the question, owner, data provenance, split/leakage rules, baselines, transaction-cost/latency assumptions where relevant, metrics, seeds, stop rules, retained artifacts, and claim boundaries.

## Product / portal status

Canonical Finance4All member-portal source is `build-the-future-11/finance4all-global-reach`. Current `main` is `0604470b6275973038e46151d1103e128da83652`; Vercel reports a completed deployment for that SHA, and the latest checked Production Health run on the same SHA succeeded.

That is deployment/health evidence only. It does not by itself certify production migration history, production RLS behavior, credentialed email/Google/recovery flows, two-member isolation, retention/deletion operations, or all member-journey behavior.

The repository has strong source-side authorization and release controls, plus draft current-main hardening surfaces, but production certification must remain layer-separated. Do not describe repository-green RLS/migration/auth tests as equivalent to live production certification.

The September 20 product refresh also records unfinished areas that remain evidence boundaries: owner-only administration has not been newly enforced and live-certified; onboarding/saved-item behavior has source tests but not a fresh management-backed live certification; the broad external-opportunity catalog is not complete; and operational dates/partner/program records still require real source data.

## Public-claim audit

The current product refresh labels the following as founder-supplied: **100,000+ students reached**, **1M+ LinkedIn impressions**, and **six continents**. These metrics are rendered on the landing page.

Until retained source receipts are attached to a claim ledger, treat these as founder-reported marketing claims rather than independently verified organizational evidence. Do not use them as E-level evidence, research validation, participant completion counts, partner outcomes, or application/sponsor proof without the underlying source.

Likewise, “flagship” positioning does not establish that a cohort is currently open, completed, or externally validated. Proposed programs must remain visibly differentiated from launched/evidenced programs.

## Program operations

`registry/programs.json` still records Labs, Axiom Pathways, Debrief, FinTech Studio, Chapters, Fellowship, and the Investment / Quantitative Competition as `planned_until_evidence_record`.

Do not promote a program because a landing-page card, launch-control document, application route, or recruitment copy exists. Promotion requires the program-specific evidence gate: named accountable owner, frozen operating rules where relevant, participant/output records, review artifacts, reproducibility or judging records, and final evidence records.

For finance research cohorts specifically, minimum scientific controls should include:

- predeclared question and primary metric;
- point-in-time data provenance and temporal split policy;
- leakage and survivorship checks;
- simple and strong matched baselines;
- regime/walk-forward analysis where relevant;
- transaction-cost/slippage assumptions for any trading/backtest claim;
- multiple frozen seeds when stochastic models are compared;
- retained negative/null/adverse outcomes;
- machine-readable results bound to source/config/data identity;
- explicit separation of representation/prediction research from investment advice or live-trading claims.

## Current validation gates

1. **FI-JEPA leakage repair:** independently review draft PR #63; integrate only after the research-methods boundary is accepted; then regenerate synthetic evidence without outcome-driven retuning.
2. **Research canonicalization:** designate the canonical FI-JEPA research repository and map/deprecate duplicate evidence surfaces. Do not merge historical result families into one stronger narrative.
3. **Production + claim certification:** close exact-SHA production migration/RLS/auth gates, then update production-readiness evidence. In parallel, attach receipts for public reach/partner/program claims or narrow the copy to clearly attributed founder-reported statements.

## Non-claims

This validation pass did not run a new outcome-bearing financial experiment, inspect a sealed holdout, change a frozen seed/threshold/protocol, merge a PR, deploy a release, mutate production data, certify investment performance, or promote a program/evidence level.
