# FinanceMeta FinTech Studio 01 — 7-day evidence-to-prototype starter kit

**Status:** FORMING / BUILDER SHORTLIST OPEN  
**Authority:** issue #47  
**Purpose:** turn a narrow finance problem into one reproducible prototype and evidence packet without implying live-trading performance, personalized financial advice, or a launched standing cohort.

## Cycle contract

Each builder owns one bounded 7-day cycle:

1. freeze `brief.md` before implementation;
2. build the smallest artifact that can test the problem;
3. retain deterministic tests and source/provenance information;
4. evaluate against the predeclared primary metric;
5. write `findings.md` including failures and unresolved ambiguity;
6. request review only after the evidence packet exists.

A null, adverse, or stopped prototype is a valid completion when the process is reproducible and the evidence is retained.

## Allowed first-cycle lanes

### A. Financial-data utility
Examples: public-data ingestion, statement normalization, market/event alignment, validation, reproducible analytics.

Minimum evidence: deterministic input/output path, provenance, explicit missing/conflicting-data behavior, validation tests.

### B. Consumer-finance education / decision support
Examples: fee/interest explainer, budgeting simulator, savings trade-off tool, scenario calculator.

Minimum evidence: transparent assumptions, edge-case tests, no personalized investment advice, at least one user-comprehension check.

### C. Research workflow tooling
Examples: leakage checker, backtest-hygiene audit, experiment-manifest builder, claim/evidence tracker.

Minimum evidence: one reproducibility case, automated proof that at least one invalid condition is caught, limitations retained.

## Required participant packet

Copy the templates in this directory into the builder's project repository and replace placeholders before review:

- `brief.md` — frozen before implementation;
- runnable code or a clearly executable prototype;
- `README.md` — setup, assumptions, limitations, reproduce command;
- at least 5 automated tests for core logic where code is involved;
- `evidence.json` — source/input revision and primary evaluation result;
- `findings.md` — what worked, failed, remained ambiguous, and continue/stop recommendation.

The templates here are intentionally blank. Their presence is not evidence that a project ran.

## Review rubric

Score each dimension 0–2:

- specific, user-grounded problem;
- explicit data/provenance rules;
- evaluation frozen before outcomes;
- reproducible implementation;
- appropriate privacy/security/financial guardrails;
- negative or failed evidence retained;
- conclusions stay inside the evidence.

A polished interface cannot rescue an invalid data or evaluation path.

## Launch boundary

Do **not** call Studio 01 launched until issue #47's staffing and operating gates are actually satisfied. In particular, this starter kit does not establish two accepted builders, a reviewer, a review window, completed prototypes, external usage, or any financial outcome.

Public-safe description until then: **FinTech Studio 01 is forming and builder shortlisting is open.**
