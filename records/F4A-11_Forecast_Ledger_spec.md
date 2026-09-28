# F4A-11 — Forecast Ledger scoring and resolution contract

**Status:** specification draft; forecast round unopened.

## Objective

Collect timestamped probabilistic macro/market forecasts and later score calibration and accuracy under rules that cannot be changed after submissions are visible.

## Question contract

Every question must contain:

- stable question ID and immutable wording;
- opening and closing timestamps with timezone;
- exactly defined quantity/event;
- authoritative resolution source and series/table identifier;
- data vintage/revision rule;
- resolution window and fallback rule;
- permitted answer type;
- status: DRAFT, OPEN, CLOSED_UNRESOLVED, RESOLVED, VOID.

Questions are frozen before `OPEN`. Any substantive edit after opening creates a new question ID.

## Answer types

### Binary

Probability in [0,1]. Primary score: Brier score

`(p - y)^2`

where `y` is 0 or 1.

### Numeric

Require a point forecast and, where feasible, a central predictive interval. The question must freeze the unit, transform and rounding rule. Aggregate absolute/squared error may be reported, but comparisons must use the same resolved question set.

## Timestamp policy

The ledger stores the server/registry receipt time, not an editable self-reported time. Forecast text may include reasoning, but the scored probability/value is immutable after the question cutoff. A revision is a new timestamped submission and does not erase the prior one.

## Calibration reporting

For binary questions report:

- mean Brier score across resolved questions;
- reliability table/bins only when sample size makes the bin meaningful;
- number resolved / submitted;
- unanswered and void questions separately.

Do not publish a calibration curve that hides small denominators or unresolved questions.

## Resolution and revision policy

Use the predeclared source vintage. If the question asks for a first-release value, later revisions do not change the score. If it asks for a value as of a later vintage, the exact vintage date must be in the question. Ambiguous or source-broken questions are VOID rather than resolved opportunistically.

## First-round launch gate

Before the first forecast is accepted:

1. freeze 20 resolvable question records;
2. independently inspect the resolution source/identifier for every question;
3. freeze the scoring implementation and tie/rounding rules;
4. demonstrate append-only timestamps on synthetic submissions;
5. publish the unresolved/void policy;
6. name the resolution reviewer.

No current economic value or outcome is encoded in this file, and no retrospective forecast may be entered as if it were contemporaneous.
