# FinanceMeta Regime Intelligence Benchmark — Specification

**Status:** pre-result specification.

## Research question

Do market-regime representations add reproducible downstream information beyond simple observable regime rules when evaluated chronologically and without post-hoc regime construction?

## Candidate methods

- volatility threshold rules;
- trend/drawdown rules;
- Gaussian mixture model;
- hidden Markov model;
- PCA/factor clustering;
- simple supervised classifier where labels are validly defined;
- neural representation clustering;
- FI-JEPA representation clustering.

Methods may be reduced for v0.1 if provenance/review capacity is limited.

## Regime definition policy

A regime must be:
- computable from information available at or before the classification timestamp;
- frozen before held-out outcome inspection;
- interpretable enough to reproduce;
- separated from downstream outcome labels.

Post-hoc labels such as "the crash regime" may be used descriptively after the fact but not as a confirmatory predictor input unless independently defined.

## Primary evaluations

1. temporal stability;
2. transition stability;
3. separation on prespecified observable quantities;
4. usefulness for a frozen downstream task;
5. cross-period transfer;
6. seed sensitivity for stochastic methods.

## Downstream test

Use one low-capacity frozen downstream model both:
- without regime features;
- with regime features.

The regime method earns evidence only if the delta is evaluated under identical chronology and preprocessing.

## Falsification

- permuted regime labels;
- time-shifted regime assignments;
- alternate nearby training windows;
- seed sensitivity;
- regime-count sensitivity;
- simple-rule baseline dominance.

## Reporting

Always include:
- regime occupancy;
- transition matrix;
- per-regime downstream performance;
- aggregate downstream performance;
- adverse/rare regime behavior;
- uncertainty/sensitivity;
- null/negative results.

## Interpretation boundary

A visually compelling clustering is not sufficient evidence that regimes are economically meaningful. A regime method that improves one downstream slice but not the prespecified aggregate should be reported as slice-specific, not globally superior.
