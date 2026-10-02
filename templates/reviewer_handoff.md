# FinanceMeta Independent Reviewer Handoff

## Project identity
- Project ID:
- Research owner:
- Data / integrity owner:
- Reviewer:
- Review date:
- Source repository:
- Exact source SHA:
- Protocol version / freeze UTC:
- Environment lock:
- Evidence record path:

## Authorization state
- Held-out access authorized: `NO`
- If `YES`, authorization receipt / issue comment / artifact:
- Outcome access already occurred before this review: `YES / NO / UNKNOWN`
- If yes/unknown, describe exactly what was inspected and by whom:

## Inputs to review
- Data manifest / hashes:
- Split manifest:
- Baseline implementations:
- Candidate implementation:
- Config / seeds:
- Regime definitions:
- Transaction-cost / execution config:
- Negative control / placebo:
- Raw-output paths:
- Result-table generation path:

## Independence disclosure
- Did reviewer choose candidate model/config? `YES / NO`
- Did reviewer inspect held-out outcomes before protocol freeze? `YES / NO`
- Financial/personal/project conflicts:
- Role overlap that may affect independence:

## Leakage review

| Check | PASS / FAIL / N/A | Evidence / note |
|---|---|---|
| decision-time feature availability | | |
| train-only fitting of normalization/statistics | | |
| chronological split integrity | | |
| target-window overlap | | |
| survivorship / universe construction | | |
| revised/edited data availability | | |
| duplicate/syndicated event handling | | |
| graph/topology future exposure | | |
| missing-data handling | | |
| regime definitions outcome-blind | | |

Any FAIL blocks confirmatory interpretation of the affected result.

## Baseline fairness
- Same eligible rows / chronology: `YES / NO`
- Same target definition: `YES / NO`
- Same cost/execution convention where applicable: `YES / NO`
- Search budget frozen and comparable: `YES / NO`
- Naive baseline present: `YES / NO / N/A`
- Transparent statistical baseline present: `YES / NO / N/A`
- Negative control behaves as expected: `YES / NO / N/A`

## Regime review
- Regimes frozen before held-out inspection: `YES / NO / N/A`
- All prespecified regimes reported: `YES / NO / N/A`
- Losing regimes retained: `YES / NO / N/A`
- Any post-hoc regimes clearly exploratory: `YES / NO / N/A`

## Transaction-cost / execution review
- Gross and net results separated: `YES / NO / N/A`
- Fee assumption frozen: `YES / NO / N/A`
- Spread/slippage frozen: `YES / NO / N/A`
- Turnover calculation reproducible: `YES / NO / N/A`
- At least two cost-sensitivity levels retained: `YES / NO / N/A`
- Result survives / fails frozen cost model:

## Reproduction
```bash
# exact commands run by reviewer
```

- Clean checkout used: `YES / NO`
- Principal table reproduced: `YES / NO / PARTIAL`
- Artifact hashes match: `YES / NO / PARTIAL`
- Failed/null runs retained: `YES / NO`
- Differences from owner output:

## Claim audit
| Proposed claim | Evidence supports? | Allowed wording / correction |
|---|---|---|
| | | |

## Decision
`PASS_FOR_HELD_OUT_UNLOCK | PASS_FOR_REVIEW | RELEASE_WITH_LIMITATIONS | REQUEST_CHANGES | INVALIDATED`

Reason:

Blocking changes:
1.

Non-blocking limitations:
1.

Exact next artifact:

This review is not investment advice and does not establish live-market profitability.