# FinanceMeta September 2026 External-Proof Evidence Ledger

**Reporting window:** September 1–30, 2026  
**Current snapshot:** September 9, 2026  
**Purpose:** preserve what actually happened, what is only planned, and which quantitative claims are still unsupported.

This ledger exists to prevent a common failure mode: converting a warm reply, scheduled call, planning target, provisioned resource, or merged internal artifact into an external outcome.

## Evidence rules

A number may be reported as an outcome only when its source is preserved and the counting rule is explicit.

Do **not** treat any of the following as equivalent:

- planning target ↔ registration;
- registration ↔ active participant/team;
- active participant/team ↔ completion;
- provisioned API key ↔ actual API usage;
- sponsor offer ↔ sponsor resource actually used;
- scheduled reviewer/judge ↔ reviewer/judge who actually scored work;
- prepared submission ↔ external submission;
- external submission ↔ acceptance/listing;
- scheduled call ↔ delivered collaboration;
- merged internal documentation ↔ external program outcome.

Unknown is recorded as **unknown**, not `0`.

## Required September metrics

| Metric | Current verified value | Counting rule | Required evidence |
| --- | --- | --- | --- |
| Registrations | Unknown | Unique valid registrations in the frozen registration source | source export/snapshot + dedupe rule |
| Active builders/teams | Unknown | Registrant/team with qualifying participation activity | activity definition + event log |
| Completions | Unknown | Participant/team meeting the frozen completion definition | completion rule + artifact/activity evidence |
| Submitted artifacts | Unknown | Unique accepted submission records received by deadline | submission index/export |
| External reviewers who actually participated | Unknown | External reviewer with at least one preserved evaluation/action | reviewer record + evaluation evidence |
| Sponsor contributions actually used | Unknown | Sponsor-provided resource demonstrably consumed in delivery/project work | usage/provisioning evidence tied to activity |
| Failures/data issues/complaints | Unknown | Preserved incident/complaint records within the reporting window | incident log/source record |

No aggregate impact claim should be built from these rows until the unknown values have source-backed counts.

## Current workstream states

### FMP / FinTech Studio Buildathon

**Current state:** technical terms confirmed; final management/leadership approval pending.

Preserved internal evidence:

- technical/data boundary is frozen in `operations/fintech-studio-buildathon/FMP_TECHNICAL_BRIEF.md`;
- machine-readable launch gates are frozen in `operations/fintech-studio-buildathon/fmp_contract.json`;
- merged implementation commit: `f12cc9d101e117d4196a4ff3e76839a5f3d85ea9`.

Not yet countable as external outcome:

- confirmed public sponsorship/partnership;
- registered teams;
- keys issued;
- teams actually using FMP;
- completed prototypes;
- finalist/judge activity.

The 15–25-team figure is a planning estimate, not a registration number.

### HAVEN contribution

**Current state:** pre-call packet prepared; Sep. 15 planning call is future work.

Preserved internal evidence:

- `operations/haven-2026-09-15/PLANNING_PACKET.md`.

Not yet countable as delivered collaboration:

- the call itself;
- agreed contribution/rubric/template;
- post-Sep. 25 seminar/workshop;
- participant work reviewed under an external rubric.

### Five Foundations / standards path

**Current state:** resource package audited and canonical public-site code merged; Jump$tart submission/provider eligibility and live canonical-domain verification remain open.

Preserved internal evidence:

- resource/audit package under `resources/five-foundations/`;
- public-site implementation commit `9398aa74f65c9adc0bd6124735cfb27abc07fb69` in `build-the-future-11/finance4all-global-reach`;
- canonical URL gate commit `a4162ac90b56aff8c9f4a785ecca6b8b2ad4404e`;
- Vercel status on the public-site commit reports a deployment build-rate limit, so production equivalence is not currently claimed.

Not yet countable:

- successful live canonical publication on `finance-meta.org`;
- Jump$tart submission;
- Jump$tart provider eligibility confirmation;
- acceptance/listing/reviewer outcome.

### November 1 stock-pitch competition

**Current state:** FinanceMeta internal partner-position and reproducible judging proposal prepared; joint partner rules remain unresolved.

Preserved internal evidence:

- `operations/nov1-stock-pitch/FINANCEMETA_POSITION.md`;
- `operations/nov1-stock-pitch/financemeta_position.json`;
- `operations/nov1-stock-pitch/JUDGING_RECORD_SPEC.md`;
- `operations/nov1-stock-pitch/judging_protocol.json`;
- judging/reproducibility merge commit `8c139d45449b6dc1e3fcab8b61b6a67b963e67bf`.

Not yet countable:

- final partner-approved rulebook;
- judges committed under the final process;
- registrations/submissions;
- scoring activity/results.

## Evidence-entry format

For each later quantitative metric, preserve:

1. `metric_id`;
2. reporting window;
3. source timestamp;
4. source reference/location;
5. counting rule/version;
6. raw count;
7. exclusions/deduplication;
8. final reported count;
9. known limitations;
10. reviewer/approver of the count.

If the source cannot be preserved, the metric remains unsupported.

## Sep. 30 reporting gate

Before producing the final internal evidence sheet:

- replace unknowns only with source-backed counts;
- report the source window and snapshot timestamp;
- separate registrations, activity, completions, and submissions;
- identify external reviewers by actual participation, not intent;
- count sponsor resources only when actually used;
- include failures, data-quality incidents, credential issues, and complaints rather than reporting success only;
- preserve zeroes when verified, but never convert missing data into zero;
- explicitly list unresolved sources/limitations.

The final report should be reproducible from the preserved evidence, not reconstructed from memory.