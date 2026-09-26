# FinanceMeta Execution Wave — 25 September 2026

## Purpose

This wave converts the current FinanceMeta portfolio from broad intent into a bounded closure-and-proof program. It does not authorize held-out evaluation, production credential use, live trading, public partnership claims, or publication claims without the existing evidence gates.

The wave has five P0 outputs:

1. close the Quant Cohort leakage/pre-result gate;
2. move FI-JEPA from synthetic M1/E1 toward a reviewable real-data protocol without looking at held-out outcomes prematurely;
3. establish one reusable falsification suite for FinanceMeta research;
4. freeze FinanceMetaBench v0.1 before collecting benchmark outcomes;
5. staff and run the first 12-week researcher program around reviewable artifacts rather than enrollment counts.

## P0 control board

| Workstream | Current boundary | Next accepted artifact | Stop condition |
|---|---|---|---|
| Quant Cohort 01 | Pre-result / leakage repair | split-contained feature+label window spec, boundary-exclusion counts, source/data hashes, reproducible train/validation run | no modeling if leakage gate is unresolved |
| FI-JEPA | M1/E1 synthetic executable baseline | frozen real-data experiment contract, provenance manifest, baseline matrix, walk-forward split definition | no held-out run before reviewer unlock |
| Research integrity | fragmented project-level checks | common falsification-suite contract + machine-readable result schema | do not accept a headline result without applicable checks |
| FinanceMetaBench | proposed | frozen task/metric/data-contract specification | no leaderboard until data licensing/provenance is valid |
| Researcher program | forming | named owner/reviewer per project, artifact-first 12-week cadence | pause intake when reviewer capacity is saturated |
| Portal/product | source-green but production evidence incomplete | preserve product repo closure gates in finance4all-global-reach | do not treat CI as production verification |

## 7-day sequence

### Day 1 — freeze contracts
- Adopt the research standard in `RESEARCH_STANDARD_V1.md`.
- Freeze FI-JEPA real-data protocol before any held-out access.
- Freeze the falsification-suite test catalog and output schema.
- Freeze FinanceMetaBench task definitions without publishing results.

### Day 2 — provenance and leakage
- Build/verify dataset manifests containing source, license, timestamp convention, asset universe, acquisition date, immutable hash, and allowed-use notes.
- Verify every feature and label window is split-contained.
- Record boundary exclusions explicitly.
- Verify fitting, normalization, imputation, feature selection, and regime construction use training-available information only.

### Day 3 — baselines
For each applicable project, require:
- naive/persistence baseline;
- transparent linear baseline;
- task-appropriate autoregressive or tree baseline;
- fixed search budget;
- identical split and preprocessing discipline.

No advanced model earns interpretive weight until the simple baselines reproduce.

### Day 4 — falsification
Run only the checks authorized by the frozen protocol:
- label permutation;
- feature timing shift;
- seed sensitivity;
- train-window perturbation;
- feature ablation;
- cost sensitivity for simulated strategy layers only;
- placebo feature/control;
- regime exclusion;
- parameter sensitivity.

Failed falsification checks block stronger claims; they are not deleted.

### Day 5 — independent reproduction
A reviewer who did not write the primary implementation should reproduce:
- data-manifest validation;
- one baseline table;
- one falsification table;
- exact source/config hash.

### Day 6 — evidence package
Each research artifact should include:
- `experiment_contract.json`;
- `data_manifest.json`;
- `environment.lock` or equivalent;
- raw machine-readable results;
- `findings.md` with positive/negative/inconclusive/untested status;
- reproduction command;
- claim boundary;
- reviewer note.

### Day 7 — release review
For every project choose exactly one:
- RELEASE;
- REVISE;
- NEGATIVE RESULT RELEASE;
- HOLD;
- ARCHIVE.

No project stays "active" without a dated next evidence gate.

## Portfolio rules

- Maximum five simultaneous research lanes requiring active review.
- A new lane opens only when one active lane reaches RELEASE, HOLD, or ARCHIVE.
- Negative results count as completed research when the protocol and evidence package are complete.
- Public metrics must distinguish proposals, executable artifacts, internally executed work, external participation, external outcomes, and independent validation.
- Simulated portfolio or backtest output is never represented as realized return or investment advice.
- Private participant records, partner negotiations, credentials, and personal contact information remain outside the public repository.

## Existing issue alignment

This wave is intended to complement, not replace, the existing control surfaces:
- Quant Cohort 01 operational contract: issue #35.
- Quant leakage-audit screen: issue #43.
- Program execution queue: issue #56.
- Repository governance/review: issue #50.
- Member portal production closure: tracked separately in `build-the-future-11/finance4all-global-reach`.

## Definition of done

The wave is complete when:
1. Quant Cohort has a reviewer-approved leakage-safe pre-result package;
2. FI-JEPA has a frozen real-data protocol and reproducible baseline path;
3. the falsification suite is used by at least one research lane;
4. FinanceMetaBench v0.1 is frozen before outcome collection;
5. each active researcher has one owner, one reviewer, one artifact, and one due date;
6. every active lane has a truthful release decision and evidence path.
