# FinanceMeta Private-Markets IRR Engine — Specification

**Status:** pre-implementation specification.

## Goal

Build an auditable educational/research engine for private-market cash-flow analysis. It should make every assumption inspectable and produce reproducible calculations rather than opaque spreadsheet outputs.

## Core functions

- periodic IRR;
- irregular-date XIRR;
- MOIC;
- DPI;
- RVPI;
- TVPI;
- gross-to-net fee/carry waterfall where explicitly configured;
- entry/exit multiple scenarios;
- leverage/debt schedule scenarios;
- downside/base/upside cases;
- Monte Carlo sensitivity where assumptions are frozen and clearly labeled;
- public-market-equivalent comparison as a separate module.

## Input contract

Each cash flow must record:
- date;
- amount;
- currency;
- direction/type;
- source;
- scenario tag;
- whether realized or assumed.

Model configuration records:
- valuation date;
- compounding convention;
- exit timing;
- exit multiple;
- debt terms;
- fee/carry assumptions;
- FX treatment if applicable.

## Auditability

Every computed metric must link to:
- exact input hash;
- config hash;
- formula/version;
- source SHA;
- generated timestamp.

The report must display cash-flow series and assumptions before headline outputs.

## Numerical rules

- XIRR solver must expose convergence state;
- multiple roots or non-convergence are not silently collapsed into one number;
- invalid sign patterns return an explicit diagnostic;
- date ordering is validated;
- scenario outputs are clearly separated from historical cash flows;
- Monte Carlo output reports distributional summaries rather than one preferred draw.

## Validation suite

At minimum:
- known textbook/simple cash-flow cases;
- single-investment/single-exit analytic case;
- irregular-date XIRR case;
- no-sign-change failure;
- multiple-sign-change warning case;
- zero/near-zero cash flow;
- leverage scenario reconciliation;
- deterministic Monte Carlo seed replay.

## Output

Machine-readable:
- `analysis.json`;
- `cashflows.csv`;
- `sensitivity.csv`.

Human-readable:
- assumptions;
- core metrics;
- sensitivity table;
- cash-flow timeline;
- limitations;
- audit receipt.

## Claim boundary

This is an analysis/education tool. It does not predict realized private-market performance, provide investment advice, or validate third-party fund claims. Scenario outputs are assumptions, not forecasts.
