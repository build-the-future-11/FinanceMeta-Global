# FinanceMeta Train-the-Trainer Delivery Packet

**Status: operational template, not a launched program or a partner agreement.**

Use one copy for a bounded facilitator-training cycle. This supplements
[Program Evidence Record](program_evidence_record.md), rather than replacing its
launch gates, evidence levels or public-claim audit. Keep completed copies in a
permission-controlled workspace. Do not commit participant records to this public repository.

## 1. Establish one source of truth before invitations

| Field | Required record |
|---|---|
| Cycle ID | Stable internal identifier |
| Accountable coordinator | Named owner, recorded privately |
| Curriculum owner / trainer | Confirmed person, not a prospect |
| Confirmed session | ISO date, start/end and named timezone; copy the exact agreed date, not “next Wednesday” |
| Curriculum | Lesson title, version, source and permitted use/adaptation |
| Intended learners | Bounded audience and eligibility rule |
| Facilitator cohort | Private confirmed roster, availability and delivery commitment |
| Local-delivery window | Agreed dates, not an assumed promise |
| Evidence owner | Person checking receipts and preparing the report |
| Partner attribution | Private permission record; consent to participate is not consent to public endorsement |

Until all applicable launch gates are supported, retain `PLANNED`, not `READY`.
Do not send multiple competing dates or scopes. The coordinator keeps one dated
change log and one current invitation. Cancellation or a revised date is a new
record, not an overwrite of the prior commitment.

## 2. Keep curriculum and evaluation lineages separate

Select a lesson only after permission and suitability are reviewed. Partner
training is not automatically part of the frozen September literacy evaluation.

If using `FINANCEMETA-LITERACY-SEP2026-v1`, reference its exact
[protocol](../evaluation/september-2026-financial-literacy-pilot/protocol.json)
and [intervention](../evaluation/september-2026-financial-literacy-pilot/INTERVENTION.md).
Do not replace the frozen 35-minute module, change its assessment, disclose answers
prematurely or alter its scoring to fit a partner lesson. Material changes require
a separately versioned, reviewed protocol before evaluation begins.

If using a different lesson, record it as a separate delivery cycle. Do not pool
its attendance or assessment results into the frozen pilot by default. This
packet authorizes neither a new experiment nor a causal effectiveness claim.

## 3. Proposed 60-minute facilitator session

This agenda is for **training facilitators**, not a replacement learner curriculum.
The trainer must agree to the final agenda and duration before invitations.

| Minutes | Activity | Receipt |
|---|---|---|
| 0–5 | Confirm learning objective, audience and boundaries | Agreed objective |
| 5–20 | Trainer demonstrates the approved lesson | Exact lesson/version reference |
| 20–35 | Facilitators rehearse one teaching segment | Teach-back checklist |
| 35–45 | Work through likely questions and permitted adaptations | Clarification/deviation log |
| 45–55 | Confirm local delivery, consent and evidence capture | One private plan per facilitator |
| 55–60 | Resolve blockers and name the report owner | Action list with owners |

Teach-back checks: explains the intended concept accurately; distinguishes an
educational example from individualized advice; follows the curriculum and
assessment boundary; knows how to retain uncertainty/failure; can complete the
minimal evidence record. An incomplete teach-back is a preparation task, not an
attendance-based qualification. Record the trainer's actual assessment.

## 4. Local delivery handoff

Before a local session, verify host permission, applicable participant/guardian
consent requirements, curriculum permission, accessibility/logistics and the
agreed evidence plan. Do not ask learners for household income, bank balances,
investment holdings, account numbers or other personal financial records.

The facilitator records the actual lesson version and session identity before
delivery. Retain the first observation and all deviations; do not silently redo
a weak session and present only the successful attempt. A cancellation, partial
session or missing evidence stays visible.

Record attendance separately from learning outcomes. Only administer an
assessment under its approved protocol. A session attendance receipt alone does
not establish learning, financial behavior change or external validation.

## 5. Minimal private delivery ledger

Create a CSV using this header. It intentionally contains **no populated rows**:

```csv
cycle_id,session_id,facilitator_id,site_id,lesson_version,protocol_id,status,session_date_iso,attendance_instances,unique_learners_at_session,first_time_learners_cycle,completion_count,evidence_ref,deviations,permission_scope,reviewer_id,reviewed_at
```

One row represents one planned/attempted local session, never one promotional
post. Use `PLANNED`, `DELIVERED`, `PARTIAL` or `CANCELLED`. Keep an append-only change
log outside the CSV if a row is corrected. Duplicate `session_id` values must be
resolved before aggregation; do not count both copies.

**Counting and validity rules**

- Blank means unknown/not collected; zero means a verified zero. Never replace
  unknowns with zero to make a report look complete. Counts must be nonnegative
  integers, not estimates inferred from group size or registrations.
- `attendance_instances` counts observed attendance in that session.
  `unique_learners_at_session` uses stable private learner IDs within the session.
  Do not add per-session unique counts and call the result unique people.
- `first_time_learners_cycle` requires a private cross-session deduplication
  registry. Its sum may be reported as cycle-level unique learners only when that
  registry is complete and checked. Otherwise report attendance instances and
  mark cycle-unique reach unknown.
- A row should satisfy `first_time_learners_cycle <= unique_learners_at_session
  <= attendance_instances`. `completion_count` cannot exceed the session's unique
  learners and must follow the predeclared completion rule.
- `DELIVERED` requires actual date, lesson version and an accessible permitted
  evidence reference. Missing evidence is reported as unverified, not “complete.”
  A `CANCELLED` row cannot contribute completed-session or learner counts.
- Trainer/facilitator attendance is not student attendance. Keep it in a separate
  roster. Meeting sign-ups, messages, impressions and interest are not delivery.

Evidence references point to an access-controlled record. Public reports use
aggregated counts and approved attribution only. No learner names, email
addresses, identifiable screenshots or private partner communications belong in
this repository.

## 6. Review and closeout

The evidence owner checks each delivered session against its receipt and reviews
all partial/cancelled rows before publishing totals. Resolve contradictory counts
or retain them as a limitation. Do not exclude failures from the denominator.

Use this report outline in the existing Program Evidence Record:

1. **Scope:** dates, curriculum/version and what the cycle actually attempted.
2. **Execution:** confirmed facilitators trained; planned, delivered, partial and
   cancelled local sessions, each with a counting rule and evidence reference.
3. **Reach:** attendance instances, independently deduplicated unique learners
   where available, and missingness. Keep targets separate from actuals.
4. **Observations:** concepts participants found useful/difficult, implementation
   problems and operational deviations; no invented testimonials.
5. **Limits:** missing evidence, selection effects, unmeasured learning and any
   reason results should not be generalized.
6. **Decision:** continue, revise or stop, with a named owner and bounded next step.

Send the private report through the agreed channel after review. A request for
feedback or a positive reply is not institutional endorsement. Public release
requires a separate claim/attribution check; this template is not that approval.
