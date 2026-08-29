# FinanceMeta Operating System 2026

## Purpose

FinanceMeta is intended to be a student finance, economics, markets, quantitative research, and fintech talent network. This document defines the operating standard required for that ambition to become measurable work rather than promotional activity.

This is an operating specification. It does **not** assert that every program below is currently active, that any membership target has been reached, or that any partnership, publication, internship, sponsor, or competition outcome exists unless a linked evidence record says so.

## 1. Evidence hierarchy

FinanceMeta uses the following evidence levels for public claims.

| Level | Meaning | Examples |
|---|---|---|
| E0 | Idea or proposal | planned chapter, proposed competition |
| E1 | Implemented artifact | repository, handbook, application form, curriculum |
| E2 | Executed internally | completed cohort session, internal benchmark, judged pilot |
| E3 | External participation | independently registered participant, external speaker, university reviewer |
| E4 | External outcome | published work, verified internship, deployed product with real users, sponsor agreement |
| E5 | Independent validation | external replication, citation, audited outcome, peer-reviewed acceptance |

Every public metric should identify its evidence level and source. Planned targets are never reported as current achievements.

## 2. Program portfolio

FinanceMeta may operate the following program families. Each family is considered **planned** until its minimum launch gate is met and linked from a program evidence record.

### Labs

Purpose: structured student research in quantitative finance, economics, market microstructure, causal inference, forecasting, and financial technology.

Minimum launch gate:

- named research lead;
- written research question;
- baseline or comparison method;
- data provenance statement;
- reproducible repository or analysis package;
- weekly review cadence;
- final artifact requirement;
- evidence ledger.

### Axiom Pathways

Purpose: guided technical learning pathways that culminate in a verifiable artifact.

Minimum launch gate:

- syllabus with prerequisites and outcomes;
- exercises or projects with answer/review criteria;
- completion definition stricter than attendance;
- participant progress log;
- final demonstrable output.

### Debrief

Purpose: concise analysis of markets, economics, policy, and financial systems.

Minimum launch gate:

- editorial standard;
- source and citation requirements;
- correction policy;
- author attribution;
- distinction between analysis, opinion, and investment advice.

### FinTech Studio

Purpose: build functional financial software rather than presentation-only projects.

Minimum launch gate:

- problem statement and user;
- working repository;
- security/privacy review where financial or personal data is involved;
- testable demo;
- usage evidence if a product is publicly described as used.

### Chapters

Purpose: local execution nodes for FinanceMeta programs.

A chapter is **active** only if, within the prior 45 days, it has produced at least one evidence-backed output such as a workshop with attendance evidence, a reviewed research milestone, a technical demo, a competition submission, or a published analysis. A dormant social-media group is not counted as active.

### Fellowship

Purpose: small, selective project teams producing research or technical work.

Minimum launch gate:

- application rubric;
- project matching process;
- milestone schedule;
- named reviewers;
- documented feedback;
- final release gate.

### Investment / quantitative competition

Purpose: evaluate analytical reasoning and research discipline rather than reward unsupported return claims.

Minimum launch gate:

- frozen rules before submissions open;
- scoring rubric;
- conflict-of-interest policy;
- data and time-window rules;
- anti-leakage / anti-lookahead requirements where relevant;
- reproducible judging record;
- publication of methodology and aggregate outcomes.

## 3. Research lifecycle

Every FinanceMeta research project should move through the same lifecycle:

```text
QUESTION
  -> PROTOCOL
  -> DATA PROVENANCE
  -> BASELINE
  -> EXECUTION
  -> EVIDENCE AUDIT
  -> REVIEW
  -> RELEASE / NEGATIVE RESULT / ARCHIVE
```

Projects do not remain indefinitely in an ambiguous “active” state.

### Required project record

Each project should include:

- project ID;
- research question;
- hypothesis and null hypothesis;
- primary metric;
- baseline(s);
- dataset identity and license;
- train/validation/test policy where applicable;
- seed policy where applicable;
- predeclared failure condition;
- repository and commit;
- exact reproduction command;
- raw result location;
- result status: positive, negative, inconclusive, or untested;
- reviewer notes;
- release decision.

### Integrity rules

FinanceMeta research must not:

- backfill a preregistration after results are known;
- discard failed seeds without a predeclared reason;
- replace a baseline because it wins;
- report synthetic results as market performance;
- use future information in historical prediction tests;
- report paper-trading output as realized financial return;
- present simulated users, capital, partners, or transactions as real;
- turn an inconclusive result into a positive conclusion by changing the claim after the fact.

Negative results remain first-class outputs.

## 4. Participant outcome model

Raw registrations and social audience size are secondary metrics. The primary outcome table should track completed work.

| Metric | Counting rule | Evidence required |
|---|---|---|
| Active builders | produced a qualifying output in trailing 45 days | artifact/event record |
| Active research teams | reviewed milestone in trailing 45 days | research ledger |
| Completed projects | passed a defined release gate | repository/demo/report |
| Research releases | public report/preprint/package with provenance | persistent link |
| Technical demos | executable or recorded working system | demo + repo |
| Competition submissions | actually submitted, not merely planned | submission receipt |
| Internships/opportunities | externally confirmed outcome | confirmation record |
| External collaborators | performed a documented contribution | review, commit, session, letter |
| Chapters active | meets chapter activity definition | chapter evidence record |

No metric is updated from self-report alone when independent evidence is reasonably available.

## 5. Chapter operating standard

Each chapter maintains a monthly evidence record with:

- chapter name and lead;
- reporting period;
- active participants under the 45-day rule;
- events actually completed;
- research/build milestones;
- links to artifacts;
- problems/blockers;
- next month commitments.

### Chapter status

- **ACTIVE**: evidence-backed qualifying output within 45 days.
- **AT RISK**: no qualifying output for 46–75 days.
- **DORMANT**: no qualifying output for more than 75 days.
- **CLOSED**: formally wound down.

## 6. Research review rubric

Score each dimension from 0 to 4.

| Dimension | 0 | 2 | 4 |
|---|---|---|---|
| Question | unclear | testable | precise and important |
| Baselines | absent | basic | strong and fair |
| Data integrity | unknown | documented | versioned, leakage-audited |
| Reproducibility | cannot run | partial | one-command + provenance |
| Statistical discipline | unsupported | basic | predeclared + uncertainty |
| Claim calibration | exaggerated | mostly aligned | exactly bounded by evidence |
| Communication | unclear | understandable | reviewer-ready |

Suggested release gate: no zero in any category and at least 20/28 overall. A high rubric score does not imply scientific novelty or external validation.

## 7. Project maturity rubric

### M0 — Proposal

Question exists; no implementation.

### M1 — Executable start

Code or analysis runs on a bounded fixture.

### M2 — Reproducible internal evidence

Multiple runs/controls where needed, raw outputs, provenance, and verification exist.

### M3 — External-data or real-user evidence

The project is tested beyond its author-created fixture.

### M4 — Independent review

An external domain reviewer has inspected method and evidence.

### M5 — External outcome

Publication, independent replication, externally used tool, verified product adoption, or another durable outside result.

## 8. Weekly operating cadence

### Monday: portfolio triage

- update project states;
- identify missing evidence;
- freeze the week’s top three outcomes;
- stop or defer low-leverage work.

### Midweek: review

- inspect raw outputs and blockers;
- challenge baselines and leakage assumptions;
- request corrective work before polish.

### Weekend: release review

- verify completed artifacts;
- update evidence ledger;
- publish only claims that passed their gates;
- record negative and inconclusive outcomes;
- select the next cycle.

## 9. Partnership standard

A partnership is reported publicly only when there is a concrete reciprocal commitment such as a signed agreement, scheduled and accepted program contribution, provided resource, verified mentorship role, or completed joint output.

A cold email, introductory call, logo permission request, or unanswered proposal is not a partnership.

Partner records should include:

- organization;
- contact/role internally (private if needed);
- purpose;
- concrete commitment;
- start/end or review date;
- evidence location;
- public wording approved for use.

## 10. Sponsor standard

Sponsors must not purchase favorable research conclusions, judging outcomes, admissions, investment recommendations, or editorial coverage disguised as independent analysis. Sponsorship benefits and editorial/research independence should be documented separately.

## 11. Publication workflow

```text
DRAFT
 -> INTERNAL METHODS REVIEW
 -> EVIDENCE LOCK
 -> DOMAIN REVIEW
 -> CLAIM AUDIT
 -> REPRODUCIBILITY CHECK
 -> RELEASE
```

A release package should contain:

- manuscript/report;
- code;
- environment or lockfile;
- raw/aggregated results;
- table/figure generation path;
- limitations;
- citation metadata;
- changelog or release notes.

## 12. Current repository boundary

As of this document’s creation, this repository should be treated as an **infrastructure and research container**, not as evidence that all FinanceMeta programs above are operating. Current contents must be inspected directly before making any activity claim. Program evidence should be linked here only after it exists.

## 13. Next repository gates

1. Create a machine-readable program/project registry.
2. Add contribution and research-intake templates.
3. Give every research directory a status + evidence file.
4. Add CI for any executable research package.
5. Publish an evidence-backed quarterly output report rather than a member-count report.
6. Add an external-review log with consent-aware attribution.

The objective is simple: a visitor should be able to distinguish **what FinanceMeta wants to build** from **what FinanceMeta has actually built and verified** within five minutes.
