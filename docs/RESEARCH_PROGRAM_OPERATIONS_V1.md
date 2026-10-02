# FinanceMeta Research Program Operations v1

This document governs researcher intake, project activation, reviewer assignment, chapter handoff, experiment execution, and release status across FinanceMeta / Finance4All research programs.

## 1. State machine

Every applicant and project must occupy exactly one explicit state.

### Contributor state
`APPLIED -> SCREENED -> TRIAL_ASSIGNED -> TRIAL_SUBMITTED -> REVIEWED -> ACTIVE | HOLD | CLOSED`

Rules:
- Form completion is not activation.
- A contributor is `ACTIVE` only after a named owner accepts their scope and one bounded artifact passes review.
- Do not double-book a person into standing Quant + FinTech Studio roles before one route is reviewed.

### Research project state
`INTAKE -> CONTRACT_DRAFT -> CONTRACT_FROZEN -> ENGINEERING_READY -> REVIEWER_READY -> HELD_OUT_UNLOCKED -> EXECUTED -> REVIEWED -> RELEASED | RELEASE_WITH_LIMITATIONS | ARCHIVED`

Hard gates:
- no held-out access before `HELD_OUT_UNLOCKED`;
- no public result claim before `REVIEWED`;
- no release without exact source/config/artifact provenance;
- negative, mixed, or null evidence is a valid terminal result.

## 2. Project intake minimum

A project may enter `CONTRACT_DRAFT` only when it has:
1. one narrow falsifiable question;
2. one responsible research owner;
3. one planned independent reviewer who did not choose the candidate result after seeing outcomes;
4. dataset/source family and license;
5. train/validation/test chronology;
6. primary metric and failure condition;
7. fair baseline set;
8. explicit leakage risks;
9. regime rules where applicable;
10. transaction-cost/execution assumptions if economic performance will be reported.

Use `templates/research_project_evidence.md` as the canonical evidence record.

## 3. Pre-result freeze

Before any held-out outcome is inspected, freeze and retain:
- protocol UTC timestamp;
- exact source commit;
- exact environment/dependency lock;
- data snapshot/hash or deterministic retrieval manifest;
- feature/availability rules;
- train/validation/test boundaries;
- baselines and candidate model family;
- hyperparameter/search budget;
- seeds;
- primary/secondary metrics;
- falsification condition;
- regime definitions;
- fees/spread/slippage/turnover convention;
- placebo/negative control;
- exclusion/failure rules.

Any amendment after freeze but before held-out access must be versioned, justified, and reviewed. Any change after outcome access is exploratory unless a new independently frozen confirmatory study is created.

## 4. Leakage audit

Minimum reviewer questions:
- Could any feature use information published/known after the decision timestamp?
- Were scalers, PCA, encoders, statistics, or universe filters fit using validation/test data?
- Does the target horizon overlap feature construction?
- Are revised financial statements, edited articles, or delayed timestamps treated as originally available?
- Does graph/message-passing topology expose future nodes/edges?
- Was survivorship introduced by selecting only currently listed assets?
- Were regime definitions chosen after inspecting held-out performance?
- Were missing observations forward-filled in a way unavailable in real time?

A discovered leak invalidates the affected comparison. Preserve the original result as diagnostic evidence; do not delete it.

## 5. Baseline policy

A quantitative experiment must include, where meaningful:
- a naive/persistence/historical-statistic baseline;
- a transparent linear/statistical baseline;
- candidate model under a frozen search budget;
- at least one negative control/placebo expected to remove genuine signal.

All methods must share the same eligible samples, chronology, target, cost convention, and outcome access policy unless the difference is explicitly the object of study.

## 6. Regime testing

Regimes are allowed only if their definitions are frozen without held-out outcome inspection. Each regime record must include:
- deterministic definition;
- source variables used to assign it;
- whether those variables were available at the relevant time;
- minimum sample-size rule;
- expected scientific purpose.

Report every prespecified regime, including losing or unstable ones. Post-hoc slices may appear only under an `EXPLORATORY` heading and must not be promoted to confirmatory evidence.

## 7. Transaction costs and execution

Any strategy/economic-performance claim must report both gross and net results. Freeze:
- fees;
- spread;
- slippage;
- execution timestamp/latency;
- turnover calculation;
- rebalance frequency;
- financing/borrow assumptions if relevant;
- at least two prespecified cost-sensitivity levels.

If a claimed edge disappears under the frozen cost model, that is the result. Do not weaken the cost model after inspection.

## 8. Reviewer assignment

Every outcome-bearing project needs:
- **research owner** — owns scientific contract and implementation;
- **data/integrity owner** — owns provenance, timestamps, data-quality evidence;
- **independent reproducibility reviewer** — did not select the winning model/config;
- **claim reviewer** — checks manuscript/public wording against retained evidence.

One person may cover more than one role only when independence is not compromised and that overlap is disclosed. Held-out unlock requires an explicit reviewer receipt.

Use `templates/reviewer_handoff.md`.

## 9. Researcher onboarding

First two weeks for a new research contributor:

### Week 1
- read this operating gate;
- complete a bounded methods/data/reproduction screen;
- submit one auditable artifact with exact reproduce command;
- receive a written review decision.

### Week 2
- accept one named project scope;
- create/fill the evidence record;
- freeze a protocol or complete a pre-result infrastructure task;
- receive owner + reviewer assignment.

No standing research title is earned from application, attendance, or a biography form alone.

## 10. Chapter onboarding

A chapter is operational only when it has:
- named chapter lead;
- named central reviewer/liaison;
- one bounded 30-day output;
- evidence location;
- reporting date;
- chapter-safe claim boundary;
- no use of member count/event planning as proof of research delivery.

Suggested first outputs: reproducibility workshop, data-integrity audit, evidence-first research note, bounded FinTech Studio prototype, or independently reviewed educational resource.

Use `templates/chapter_activation_handoff.md`.

## 11. Status tracker fields

Every active research/program row must record:
- ID;
- program/lane;
- owner;
- reviewer;
- current state;
- exact next artifact;
- due date;
- evidence path;
- held-out authorization (`false` by default);
- blocker;
- last verified date;
- claim status.

Stale rule: if two review cycles pass without a responsible owner or reviewable artifact, merge/pause/archive the work instead of keeping a zombie program.

## 12. Handoff definition of done

A handoff is complete only when the receiver can answer, without asking the previous owner:
- what question is being tested;
- what is frozen;
- what data are allowed;
- what has/has not been inspected;
- what command reproduces current evidence;
- which claim is allowed;
- what next artifact closes the gate;
- who reviews it.

## 13. Current operating priorities

1. Quant Cohort 01 staffing + raw-data lock before any held-out evaluation (#35, #37, #38).
2. Convert current applicant routing into one explicit route + trial + owner + reviewer per candidate (#56).
3. Staff FinTech Studio 01 and freeze builder briefs before implementation (#47, #58).
4. Protect canonical `main` with stable human-review/CI gates (#50).
5. Close external-proof items with delivered evidence rather than planned-partner language (#11, #30).

This document does not authorize model training, held-out access, outreach, launch claims, or research-result changes.