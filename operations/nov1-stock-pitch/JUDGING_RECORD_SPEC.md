# November 1 Stock Pitch — Reproducible Judging Record Specification

**Status:** FINANCEMETA INTERNAL DRAFT — NOT PARTNER APPROVED  
**Applies to:** Empiric Pitch / November 1, 2026 stock-pitch competition planning  
**Purpose:** make scoring auditable and anti-lookahead before any final partner rulebook is adopted.

This document does not freeze a partner decision. It defines the evidence FinanceMeta believes should exist for every scored submission so that a result can later be reconstructed without relying on memory, future price moves, or unpublished judge reasoning.

## 1. Governance boundary

The August 17 operating brief makes the competition rules a joint partner decision. Therefore:

- this specification is a FinanceMeta internal position only;
- the candidate 30/25/20/15/10 scoring weights remain a recommendation until all partners sign off;
- judge count, scoring-window length, results date, prizes, eligibility, entry format, memo cap, and final co-branding remain open partner decisions;
- Empiric owns the platform, registration, blind-judging infrastructure, and scoring implementation under the current brief;
- no public competition copy may state that this judging specification is final unless the partner brief is updated first.

## 2. Core anti-lookahead rule

A stock pitch must be judged as the submission that existed when editing locked — **not** as a retrospective prediction contest.

Judges must not increase or decrease a score because of:

- price moves after the submission lock;
- earnings releases, guidance, filings, mergers, litigation, macro events, or news first published after the lock;
- later analyst commentary or consensus changes;
- later revisions to the issuer's historical data that were unavailable to the participant at lock time;
- whether the thesis ultimately became profitable after the deadline.

The score asks: *Was this a coherent, evidence-backed, well-calibrated investment case using information available at the time?*

Future market performance is never a scoring component.

## 3. Submission evidence frozen before judging

For each submission, preserve or reference at minimum:

1. stable submission ID;
2. submission-lock timestamp in UTC;
3. final memo artifact or platform snapshot identifier;
4. SHA-256 (or platform-equivalent immutable digest) of the final exported memo where available;
5. list of cited source URLs as captured at submission;
6. participant AI-assistance disclosure;
7. platform authenticity/integrity record identifier;
8. team/individual entry type from the final partner-approved rulebook;
9. competition/rubric version used for scoring.

Do not silently replace the frozen artifact because a URL later changes or a participant asks to clarify the thesis after the deadline.

## 4. Citation/time-window rule

The participant's cited evidence should be evaluated against what was represented in the locked submission.

Where source timestamps or archived evidence are available, preserve them. If a cited webpage changes after submission, judges should use the captured submission-time evidence or platform record when available rather than rewarding or penalizing the participant for later edits to an external page.

If a critical source cannot be reconstructed, record the limitation instead of substituting later information.

## 5. Blind assignment and conflicts

The judging interface should hide author identity, school, and country from scorers.

Conflict screening should happen **before** blind assignment through a separate coordinator or platform process. At minimum, disclose and recuse for a known material conflict such as:

- family/household relationship;
- direct mentorship, supervision, employment, or current close collaboration with the participant;
- same small organization/team where impartial scoring would reasonably be questioned;
- direct involvement in preparing the submission;
- material professional or financial conflict relating to the issuer or participant that could reasonably affect impartiality.

A judge who recognizes a participant from content or context during scoring must stop and flag the assignment rather than continuing anonymously.

The public record need not reveal private conflict details. It should preserve that a conflict check occurred and whether recusal/reassignment happened.

## 6. Candidate scoring rubric

FinanceMeta's current internal recommendation mirrors the August 17 proposed partner brief:

| Component | Candidate weight | What the record must capture |
| --- | ---: | --- |
| Thesis and reasoning | 30 | component score + concise rationale |
| Financial analysis and valuation | 25 | component score + concise rationale |
| Evidence and citations | 20 | component score + concise rationale |
| Risk assessment | 15 | component score + concise rationale |
| Structure and writing | 10 | component score + concise rationale |

These weights are not partner-approved merely because this specification encodes them. The active rubric version must be supplied by the final partner brief.

## 7. Integrity gate is separate from merit score

Authenticity/integrity is a gate, not a weighted bonus or penalty.

- Editing/proofreading assistance may be allowed only under the final disclosure rule.
- A standalone AI detector is insufficient evidence for automatic disqualification.
- Paste events, version history, disclosure, build-time evidence, and other platform signals may inform a human integrity review.
- A submission placed on integrity hold should not be silently assigned a reduced merit score as a substitute for an integrity decision.
- Integrity decisions must preserve a reason code and reviewer/action timestamp.

## 8. Independent scoring and disagreements

FinanceMeta's preferred operating model is two independent merit scores per submission when judge capacity permits, with a third review for finalists or material disagreement.

Because judge capacity is an open partner decision, this is not a guaranteed format yet.

If the final process uses multiple scores, preserve each score independently. Do not overwrite one judge's record with a consensus number.

Any aggregation rule must be versioned before results are calculated. Examples could include mean total score or a calibrated panel rescore for finalists, but the final rule must come from the partner-approved brief rather than being selected after seeing results.

## 9. Calibration

Before live scoring, judges should receive:

- the exact active rubric version;
- short anchor examples or a calibration memo;
- the anti-lookahead rule;
- conflict/recusal instructions;
- integrity escalation instructions;
- guidance that polish cannot substitute for unsupported reasoning.

Record the calibration-material version associated with each scoring round.

## 10. Required per-score record

Each judge score should retain:

- submission ID;
- blinded judge ID or stable reviewer pseudonym;
- assignment/scoring-round ID;
- active rubric version;
- calibration version;
- conflict check status;
- component scores;
- computed total score;
- concise rationale for each component or a structured overall rationale as required by the final rubric;
- integrity-gate status at scoring time;
- scoring timestamp;
- whether the judge requested recusal or escalation;
- whether the record was superseded, and why.

A correction should create a new version or amendment record rather than deleting the old score without trace.

## 11. Results reproducibility packet

Before announcing winners, preserve an internal packet containing:

1. final partner-approved brief/rubric version;
2. submission IDs and immutable artifact references;
3. all judge score records;
4. conflict/recusal/reassignment records;
5. calibration version;
6. integrity decisions and reason codes;
7. aggregation/finalist-selection rule version;
8. computation output used to form the finalist/winner slate;
9. manual overrides, if any, with named approver and reason;
10. final announced result set.

The packet may pseudonymize or restrict private participant/judge data, but it must be sufficient for authorized reviewers to reconstruct the ranking.

## 12. No outcome leakage into post-event writeups

Post-event reporting may later discuss how pitches performed in the market, but that analysis must be clearly separate from the competition score and cannot be backfilled into judging records.

If a later research analysis uses subsequent returns, freeze a separate post-outcome protocol, horizons, benchmark, corporate-action handling, delisting handling, and missing-data rules before examining those outcomes.

## 13. Current release gate

This specification becomes an active competition rule only after the joint partner brief explicitly adopts the relevant judging, conflict, time-window, and aggregation provisions.

Until then, the strongest defensible claim is:

> FinanceMeta has prepared a reproducible, anti-lookahead judging-record specification for partner review; it is not the final competition rulebook.
