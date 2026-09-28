# F4A-03 / 05 / 06 / 07 — canonical-state reconciliation

**Date:** 2026-09-28  
**Status:** evidence reconciliation; no outcome access/run authorised.

This record captures what can be verified from the managed GitHub repositories now. It is deliberately not a scientific interpretation.

## F4A-03 — Frozen-study closure

The four-month portfolio names **DistilBERT-SST2 Stage 3** and an **Elliptic topology/leakage audit** as the two closure targets.

Repository-name searches in the connected `Finance-Meta-Research` and `build-the-future-11` organisations did not locate a repository named for DistilBERT/SST2 or Elliptic. This does **not** prove that the evidence is absent: it may live under another repository/path, archived source, private workspace, local receipt, or a differently named project.

### Required next action

Do not create a replacement experiment. Locate the exact canonical repository/file/receipt IDs from the prior frozen work. The closure record must preserve the original endpoint, data version, split, seeds and stop rule. If those artifacts cannot be located, close the item as **CANONICAL_EVIDENCE_MISSING** with a gap log rather than reconstructing a favorable result.

## F4A-05 — LGWM

Closest managed repository found: `Finance-Meta-Research/LGWM-Hedge-Fund`.

Its current README explicitly marks it as a **reserved project placeholder** and states that it contains no hedge fund, trading system, strategy implementation, backtest, portfolio or performance record. It requires a reviewed charter before activation and forbids implying live capital/AUM/returns.

### Reconciliation decision

The F4A-05 validation protocol draft in this PR must **not** be treated as a successor to an existing implementation until another canonical LGWM repository/commit is identified or this placeholder is deliberately chartered into the bounded systemic-risk research project. No result-bearing run is authorised.

If this placeholder is the intended canonical home, the first state transition is G0/G1 charter + novelty work, not G3/G4 experimentation.

## F4A-06 — Eigen-Finance

Managed repository found: `Finance-Meta-Research/EigenFinance`.

Its current README marks it **RESERVED_PLACEHOLDER** and says no implementation, experiment, dataset, paper, product or result is committed. The scope is not frozen and the activation gate requires an objective, owner, evidence plan and relationship to existing FinanceMeta research.

### Reconciliation decision

The portfolio language "preserve current results" cannot be satisfied from this repository because this repository documents no results. Keep F4A-06 held. Either:

1. identify the prior Eigen-Finance evidence package elsewhere and bind it here without changing it; or
2. treat this repository as a new/successor study with a new protocol ID, never as a continuation of unverified results.

## F4A-07 — FI-JEPA

Managed repository found: `Finance-Meta-Research/FI-JEPA`.

Its README describes a materially developed repository with PyTorch implementation, configs, benchmark/ablation scripts, saved experiment outputs, checkpoints, figures and a LaTeX preprint. In contrast, the central `FinanceMeta-Global` registry currently describes FI-JEPA at M1/E1 as implementation + deterministic synthetic baseline only, with no market-performance/research result authorised.

### Reconciliation decision

This is a **state discrepancy**, not permission to choose the more advanced description. Keep F4A-07 held while the exact canonical commit and evidence are reviewed.

Required reconciliation:

- pin the intended canonical FI-JEPA repository and commit;
- inventory configs/data sources/splits/seeds and saved outputs;
- distinguish synthetic, macrodata and any market-data runs;
- verify point-in-time/availability semantics before accepting any financial benchmark;
- map paper tables/figures to retained raw outputs;
- preserve negative/failed ablations;
- update the central registry only after this audit.

No representation-quality result should be interpreted as tradable alpha or investment performance.

## Cross-project action created by this reconciliation

1. F4A-03: locate canonical frozen-study receipts; no reconstructed endpoints.
2. F4A-05: identify actual LGWM canonical implementation or explicitly charter the placeholder as a new bounded research study.
3. F4A-06: keep held until prior evidence is found or a new study ID is created.
4. F4A-07: reconcile the developed FI-JEPA repository against the conservative central registry before any promotion.
5. F4A-04: record each resolved canonical repository/commit in the claims/preregistration registry rather than relying on project names.

This record reduces uncertainty about repository state. It does not raise any project to an evidence or claims gate.
