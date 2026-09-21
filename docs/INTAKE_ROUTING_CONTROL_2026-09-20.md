# FinanceMeta Intake Routing Control — 2026-09-20

This is an aggregate operational snapshot. Do not copy applicant names, emails, school identities, or other PII into this repository.

## Current intake state

Verified from connected Tally metadata/submission counts on 2026-09-20:

| Form | Status | Completed submissions | Operating decision |
|---|---|---:|---|
| FinanceMeta — General Application (`5B7blP`) | open | 4 | **CANONICAL GENERAL INTAKE** |
| FinanceMeta — Fall 2026 Research Cohorts (`5B7YlP`) | open | 2 | existing cohort-specific responses; do not create another replacement form |
| Quantitative Markets & Microstructure Lab Cohort 01 (`0Q9qZy`) | open | 1 | existing specialized intake; route through central tracker after review |
| Financial Systems, Access & Policy Lab Cohort 01 (`lbG8Yk`) | open | 1 | existing specialized intake; route through central tracker after review |
| Financial Foundations (`q4rLG9`) | open | 1 | existing beginner/foundations route |
| FinanceMeta — Fall 2026 New Programs (`rjeKXL`) | open | 0 | redundant until a real program requires separate fields |
| FinanceMeta — Research, FinTech & Programs (`eqVq4O`) | open | 0 | redundant with General Application unless a field requirement is proven |
| Finance4All India — School & Community Pilot Interest (`q4rlKd`) | open | 0 | keep only as institutional/pilot interest, not researcher intake |
| FinanceMeta — Chapter Registration (`XxaB1j`) | open | 0 | chapter intake only; activation still requires reviewed 30-day output |
| FinTech Studio Application (`Me4d6p`) | closed | 0 | historical/closed; do not reopen by default |
| Quant Research Cohort — Applications (`xXqylk`) | closed | 0 | historical/closed; do not reopen by default |
| Financial Foundations duplicate (`lbGBbX`) | open | 0 | duplicate surface; should not become a second source of truth |
| Bu1LD + FinanceMeta Accepted Member Onboarding (`1AL4r1`) | open | 6 | onboarding surface only; onboarding completion is not research activation |
| Bu1LD + FinanceMeta Weekly Project Check-In (`aQRQVq`) | open | 0 | reporting surface; currently no evidence of adoption |

## Primary bottleneck

The General Application still has exactly **4 completed submissions**, matching the earlier operating issue rather than showing a surge in applicant volume. Meanwhile several newer/specialized FinanceMeta forms have zero submissions. The current problem is therefore **routing and activation discipline**, not creating more forms.

## Canonical routing rule

1. New individual contributors enter through `5B7blP` unless a specialized form requires genuinely different information.
2. Existing specialized-form submissions are reviewed once and mapped into the same internal route states; do not ask applicants to refill biographies.
3. Every serious applicant receives exactly one current primary route:
   - Quant Research;
   - FinTech Studio;
   - Financial Foundations;
   - Research / Writing;
   - Community / Programs / Chapters;
   - Hold / No current fit.
4. Every trial record requires:
   - named owner;
   - named reviewer;
   - one bounded first artifact;
   - exact due date;
   - evidence location;
   - review outcome.
5. `APPLIED`, `ONBOARDED`, and `ACTIVE` are distinct states.

## No-new-form gate

Do not create another FinanceMeta contributor/research application form until all of these are true:
- the missing field cannot be added to or collected after the General Application;
- the target audience is genuinely different;
- an owner exists to review the new queue;
- a routing path exists after submission;
- duplicate intake cannot be handled by URL parameters or a lightweight follow-up screen.

## Chapter-specific rule

Chapter registration may remain a separate institutional surface, but a chapter stays `FORMING` until:
- chapter lead is named;
- central reviewer is named;
- one 30-day output is frozen;
- output passes review;
- evidence is retained using `templates/chapter_activation_handoff.md`.

## Privacy rule

Public GitHub artifacts may record anonymous counts, route state, artifact state, and review state. Applicant PII remains in the private intake/email surface.

## Next operating moves

1. Route the four current General Application submissions using artifact evidence, not biography length.
2. Resolve the explicit three-role Quant Cohort staffing gate in #38 before broadening recruitment.
3. Stop creating/reopening zero-submission duplicate forms until the existing queues are reviewed.
4. Convert accepted/onboarded people into bounded reviewed work using the templates in this PR.
5. Re-measure completed applications, accepted trials, submitted artifacts, and passed reviews after the next distribution cycle; optimize for reviewed artifacts, not raw form count.
