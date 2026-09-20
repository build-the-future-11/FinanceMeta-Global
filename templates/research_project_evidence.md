# FinanceMeta Research Project Evidence Record

## Identity

- Project ID:
- Title:
- Repository path / URL:
- Program:
- Responsible researcher(s):
- Research owner:
- Data / integrity owner:
- Independent reproducibility reviewer:
- Report / claim reviewer:
- Maturity: `M0 | M1 | M2 | M3 | M4 | M5`
- Evidence level: `E0 | E1 | E2 | E3 | E4 | E5`
- Result status: `UNTESTED | POSITIVE | NEGATIVE | INCONCLUSIVE | MIXED`
- Release status: `DRAFT | HOLD | REVIEW | RELEASED | ARCHIVED`
- Held-out access authorized: `false`

## Scientific question

**Question:**

**Hypothesis:**

**Null hypothesis:**

**Primary metric:**

**Secondary metrics:**

**Predeclared falsification / failure condition:**

**What result would stop continuation:**

## Data provenance and point-in-time safety

- Dataset(s):
- Source / DOI / URL:
- License:
- Retrieval UTC timestamp:
- Raw version / hash:
- Train / validation / test chronology:
- Universe construction rule:
- Timestamp availability rule:
- Corporate-action / survivorship policy:
- Missing/conflicting observation policy:
- Leakage controls:
- Exclusions and reasons:

### Leakage checklist

- [ ] no random split for time-dependent evaluation unless explicitly justified as a negative control
- [ ] feature values use only information available at the decision timestamp
- [ ] normalization/statistics fit on training data only
- [ ] target horizon cannot bleed into feature windows
- [ ] market/news/event timestamps use availability time, not scrape time, where applicable
- [ ] survivor-only universe bias addressed or explicitly retained as a limitation
- [ ] duplicate/syndicated events cannot silently multiply evidence
- [ ] held-out outcomes were not used to define regimes, thresholds, baselines, or exclusions

## Baselines and controls

| Baseline / control | Why fair | Source / implementation | Frozen before held-out? | Status |
|---|---|---|---|---|
| | | | | |

Required where applicable:
- naive/persistence or historical-statistic baseline;
- transparent linear/statistical baseline;
- candidate method under a fixed search budget;
- placebo/permutation/timing-shift negative control expected to destroy genuine predictability.

## Regime contract

Regimes must be defined before held-out outcome inspection using information available without the held-out result.

| Regime | Definition | Data used to define it | Frozen source/config | Why scientifically relevant |
|---|---|---|---|---|
| | | | | |

- Regime assignment code/config:
- Losing regimes retained: `YES / NO / N/A`
- Post-hoc regime slicing prohibited: `YES`

## Transaction-cost and execution contract

If economic performance is reported, freeze before held-out access:

- fee model:
- spread model:
- slippage model:
- latency / execution timing:
- turnover definition:
- position-sizing rule:
- rebalance frequency:
- financing/borrow assumptions, if relevant:
- minimum two prespecified cost-sensitivity levels:
- capacity/liquidity limitations:

No gross-return result may be presented as net performance.

## Frozen execution specification

- Protocol freeze UTC:
- Commit:
- Config:
- Seeds:
- Hyperparameters / search budget:
- Environment lock:
- Hardware constraints:
- Selected configuration frozen before held-out access: `YES / NO`
- Reviewer held-out unlock receipt:

```bash
# exact reproduction command
```

## Executed evidence

| Run / artifact | Commit/config | Seed | Split/regime | Status | Raw output | Notes |
|---|---|---:|---|---|---|---|
| | | | | | | |

## Results

| Method | Metric | Gross / net | Estimate | Uncertainty / paired test | Source |
|---|---|---|---:|---|---|
| | | | | | |

Required reporting where applicable:
- per-seed rows;
- per-regime rows;
- cost-sensitivity rows;
- turnover;
- placebo/timing-shift result;
- baseline comparison under the same chronology and assumptions.

## Negative / contradictory evidence

Record baseline wins, failed seeds, losing regimes, null effects, instability, cost sensitivity, leakage discoveries, and evidence that weakens the proposed mechanism.

| Finding | Consequence for claim | Artifact |
|---|---|---|
| | | |

## Statistical discipline

- Unit of analysis:
- Number of independent seeds / folds / periods:
- Uncertainty method:
- Multiple-comparison handling, if relevant:
- Paired comparison definition:
- Effect-size threshold frozen before test access:
- Exploratory analyses clearly labeled: `YES / NO`

Do not convert repeated observations from one simulation/path into false independent sample size.

## Claim ledger

| Claim | Evidence | Confidence | Allowed? | Public wording |
|---|---|---|---|---|
| | | | `YES / NO` | |

Explicitly separate:
- predictive evidence;
- simulated economic evidence;
- live-market evidence (`NONE` unless independently established);
- exploratory observations;
- unsupported or prohibited claims.

## Reproducibility gate

- [ ] clean install instructions
- [ ] exact dependency/environment lock
- [ ] deterministic smoke test
- [ ] exact experiment command
- [ ] seeds/configs retained
- [ ] raw outputs retained or persistently linked
- [ ] result tables trace to raw outputs
- [ ] figures trace to retained results
- [ ] failed runs preserved
- [ ] negative/null outcomes preserved
- [ ] no placeholder or fabricated result values
- [ ] independent reviewer can reproduce the principal table
- [ ] source SHA and selected config were frozen before held-out access

## Reviewer handoff

- Reviewer name / handle:
- Scope reviewed:
- Exact source SHA reviewed:
- Exact artifact hashes / paths:
- Held-out access history checked: `YES / NO`
- Leakage review: `PASS | REQUEST_CHANGES | NOT_APPLICABLE`
- Reproduction: `PASS | REQUEST_CHANGES | NOT_RUN`
- Claim review: `PASS | REQUEST_CHANGES`
- Reviewer conflicts disclosed:
- Decision date:

## Review rubric

| Dimension | Score 0–4 | Reviewer note |
|---|---:|---|
| Question | | |
| Baselines | | |
| Data integrity / point-in-time safety | | |
| Leakage controls | | |
| Regime discipline | | |
| Transaction-cost / execution realism | | |
| Reproducibility | | |
| Statistical discipline | | |
| Negative-result retention | | |
| Claim calibration | | |

**Total / 40:**

## External validation

- External-data evidence:
- External reviewer:
- Independent replication:
- Publication / competition / adoption outcome:
- Evidence links:

Do not populate external-validation fields from plans, outreach, or self-description.

## Decision

`RELEASE | RELEASE_WITH_LIMITATIONS | CONTINUE | FREEZE | ARCHIVE`

Reason:

### Claims explicitly prohibited after review

-

### Next decisive experiment / action

-
