# F4A-37 — AlphaForge launch gate and evaluation contract

**Status:** DRAFT / LAUNCH BLOCKED.

AlphaForge is an evidence-first quantitative research challenge. Recruitment does not open until the dataset, task, temporal split, evaluator and reviewer roster are frozen.

## Non-negotiable task properties

- legally usable, documented data with availability timestamps;
- one bounded empirical question;
- temporal split that prevents future information from entering earlier predictions;
- public train/development material and protected final test;
- no live trading, brokerage connection, managed capital or investment recommendation;
- scoring based on a research endpoint, not raw realised-return claims;
- reproducible submission package with code/config/environment and claim statement.

## Required submission

Each team submits:

1. machine-readable predictions for required row IDs;
2. source code and exact reproduction command;
3. model/config/seed record;
4. data/provenance statement;
5. baseline result using the official starter;
6. short technical report including negative/failure evidence;
7. compute/runtime disclosure;
8. conflict/assistance disclosure.

Missing required IDs, duplicate IDs, nonfinite predictions or use of prohibited data fails validation rather than being silently repaired.

## Evaluation design to freeze before launch

- primary metric and direction;
- baseline definition;
- protected test split;
- any regime/stress slices;
- uncertainty or paired-comparison method;
- resource limits;
- tie rule;
- disqualification conditions;
- reviewer rubric.

A public leaderboard, if used, must not expose the final protected test or allow repeated tuning against it.

## Suggested reviewer weighting

The exact weights remain proposed until reviewers approve them, but the rubric must reward:

- data/provenance and leakage discipline;
- baseline fairness;
- reproducibility;
- statistical/robustness discipline;
- claim calibration;
- clarity.

Raw return alone is not a judging criterion.

## Launch checklist

- [ ] dataset use/redistribution confirmed
- [ ] availability timestamps documented
- [ ] task wording frozen
- [ ] train/dev/final-test construction frozen
- [ ] starter baseline reproducible
- [ ] evaluator independently replayed
- [ ] hidden final-test access controlled
- [ ] reviewer roster confirmed
- [ ] conflicts policy accepted
- [ ] submission retention/privacy policy accepted
- [ ] F4A-01/02/04 integrity stack integrated
- [ ] F4A-47 release review passed

Until every checked item has retained evidence, the challenge remains `LAUNCH BLOCKED`.
