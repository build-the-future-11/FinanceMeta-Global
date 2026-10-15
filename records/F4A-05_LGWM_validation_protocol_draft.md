# F4A-05 — LGWM systemic-risk validation protocol draft

**Status:** DRAFT / NOT FROZEN / NO OUTCOME RUN AUTHORISED  
**Purpose:** convert the portfolio line into one falsifiable study specification before touching result-bearing evaluation.

## Canonical-state prerequisite

Before adoption, the research lead must identify the canonical LGWM repository/commit, existing protocol, simulator version, datasets, prior outputs and any frozen seeds/endpoints. If this draft conflicts with a prior frozen protocol, preserve the prior protocol and open a successor study ID rather than silently changing it.

## Proposed sharp question

Under predeclared liquidation-network stress scenarios, does the LGWM representation/model improve **out-of-sample prediction of cascade severity** over matched simple baselines when all methods receive the same information available at the prediction cutoff and are evaluated on the same held-out scenarios?

This is a simulator/benchmark question. It is not a claim that the model predicts real financial crises, establishes causal mechanisms in real markets, or supports trading.

## Unit and endpoint

- Unit: one pre-generated market/network scenario at a predeclared prediction cutoff.
- Primary endpoint: error on total cascade severity measured by a single frozen scalar definition from the canonical simulator.
- Secondary endpoints: calibration of severity intervals/probabilities if the model emits them; node-level ranking quality only if defined before execution.
- Every endpoint must be computable from retained raw outputs.

## Minimum baselines

1. persistence/no-change or prior-state baseline;
2. simple aggregate-feature linear/generalized-linear baseline;
3. graph-statistic baseline using the same observable network snapshot;
4. matched-capacity neural baseline if the LGWM is neural.

The final baseline set must be frozen before held-out evaluation. A baseline may not be removed because it performs well.

## Split and information policy

- scenario generation/data version is pinned;
- train/validation/test membership is generated and retained before outcome inspection;
- feature availability is defined at the prediction cutoff;
- no tuning on held-out test scenarios;
- if walk-forward evaluation is used, it requires a separately reviewed per-fold availability contract rather than bypassing the fixed-holdout checker.

## Predeclared stress families

Use only stress families supported by the canonical simulator. Candidate families for review:

- exogenous asset-price shock magnitude;
- leverage/liquidity constraint severity;
- network concentration/degree perturbation;
- correlated shock intensity;
- transaction/liquidation friction variation;
- missing/noisy network observation as a robustness test.

The exact grid and number of seeds must be frozen before result-bearing execution.

## Falsification / failure criteria

The flagship claim is not supported if any of the following occurs:

- LGWM fails to improve the frozen primary endpoint over the strongest fair baseline under the predeclared comparison;
- apparent gains disappear under the predeclared seed/robustness analysis;
- a leakage, split, simulator or metric defect invalidates the comparison;
- results depend on a post-hoc scenario subset or threshold;
- independent replay cannot reproduce the reported direction within the defined tolerance.

A negative or inconclusive result is a valid closure.

## Required evidence before release

- canonical protocol ID and commit;
- source/simulator licence and version notes;
- frozen configs/seeds/splits;
- infrastructure-gate receipt;
- raw per-scenario predictions and labels;
- baseline outputs;
- table/figure generation code;
- uncertainty/paired comparison;
- ablations tied to a stated mechanism;
- failed runs and discrepancy log;
- claim ledger and limitations;
- independent reproduction attempt.

## Explicitly prohibited wording

Do not describe simulator performance as real-world crisis prediction, causal identification, systemic-risk prevention, investment performance or production readiness.

## Adoption gate

This draft becomes `FROZEN` only after a named research owner and reviewer reconcile it against the canonical LGWM state and sign the exact question, endpoint, baseline set, data/simulator version, split, seeds and stop rules.
