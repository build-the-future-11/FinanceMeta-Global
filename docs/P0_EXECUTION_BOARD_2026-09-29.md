# FinanceMeta P0 Execution Board — 2026-09-29

This board consolidates the current live GitHub work into one execution order. It records **what must close next**; it does not claim completion until linked evidence exists.

## P0.1 — Product and production closure

### Member portal — finance4all-global-reach

Source issues:
- #25 — P1 FinanceMeta product closure: auth, RLS, portal and production evidence
- #48 — reconcile production Supabase schema with migration ledger

Exit criteria:
- [ ] production schema equals migration ledger;
- [ ] signup/login/logout/password-reset paths verified;
- [ ] role/authorization matrix documented;
- [ ] RLS tests cover anonymous, member and privileged paths;
- [ ] one real end-to-end path is evidenced: signup -> onboarding -> learn -> save -> apply/register;
- [ ] no privileged service-role credential is exposed client-side;
- [ ] production URL and build SHA are recorded;
- [ ] rollback path is documented.

## P0.2 — Quant Cohort 01 integrity lock

Source issues:
- #35 — regime-robust, leakage-safe benchmark sprint
- #37 — raw-data integrity lock
- #38 — staffing launch gate and reviewer handoff
- #43 — leakage-audit harness
- #44 — point-in-time OHLCV pipeline
- #45 — Elliptic graph-leakage audit
- #46 — point-in-time financial-news alignment audit

Execution order:
1. raw-data provenance lock;
2. point-in-time feature pipeline;
3. labeled leakage controls;
4. train/validation-only reproduction;
5. config + source SHA freeze;
6. reviewer sign-off;
7. held-out unlock;
8. result + uncertainty report;
9. release decision.

Protected evaluation remains locked until steps 1–6 are evidenced.

## P0.3 — FinTech Studio 01

Source issues:
- #47 — evidence-to-prototype launch gate
- #58 — Point-in-Time Market Data Integrity Toolkit v1

Exit criteria:
- [ ] bounded user/problem statement;
- [ ] two builders explicitly accept scope;
- [ ] reviewer accepts rubric;
- [ ] brief frozen before implementation;
- [ ] runnable implementation;
- [ ] core automated tests;
- [ ] machine-readable evidence output;
- [ ] security/privacy review where applicable;
- [ ] findings and limitations;
- [ ] reviewer decision.

## P0.4 — FinanceBench v0

Source issue:
- #71 — unified financial ML evaluation standard

Minimum v0 scope:
- [ ] task schema;
- [ ] dataset/provenance schema;
- [ ] temporal-split schema;
- [ ] baseline registry;
- [ ] metric registry;
- [ ] leakage checks;
- [ ] reproducibility manifest;
- [ ] result-card format;
- [ ] one small reference benchmark implemented end-to-end.

FinanceBench must not become a leaderboard before the integrity schema is working.

## P0.5 — FI-JEPA next gate

Source issue:
- #62 — eliminate global normalization leakage before chronological split

Current registry boundary: M1/E1 executable synthetic baseline only.

Next gate:
- [ ] preprocessing fit exclusively on permitted training history;
- [ ] licensed point-in-time market data provenance;
- [ ] stronger baselines;
- [ ] multiple seeds;
- [ ] walk-forward evaluation;
- [ ] ablations;
- [ ] uncertainty intervals;
- [ ] raw outputs with commit/config provenance.

Do not make alpha/profitability/novelty claims before these gates pass.

## P0.6 — External proof and recruiting

Source issues:
- #11 — September external-proof sprint
- #30 — FMP buildathon sponsorship
- #42 — qualified-traffic experiment and routing gate
- #56 — P0 execution queue

Rules:
- route serious applicants into exactly one primary track;
- every trial gets owner, artifact, due date and evidence location;
- stop broad recruiting when reviewer capacity is the bottleneck;
- a warm reply/call is not a partnership;
- a partnership claim requires a concrete delivered or accepted contribution.

## Weekly scorecard

Record only evidence-backed counts:
- qualified visits;
- completed applications;
- trial-worthy applicants;
- explicit acceptances;
- active trials;
- reviewable artifacts completed;
- independent reviews/reproductions;
- released research artifacts;
- prototypes continued;
- externally delivered outcomes.

## Stop rules

- Pause or merge any program after two review cycles with no owner or no reviewable artifact.
- Do not unlock protected tests to manufacture a positive result.
- Do not add another FinanceMeta program until Studio 01 and Quant Cohort 01 each have a reviewed artifact.
- Negative and inconclusive results count as completed research outputs when protocol integrity passes.
