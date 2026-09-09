# FinanceMeta FinTech Studio Buildathon — FMP Technical Brief

**Status:** PRE-LAUNCH / MANAGEMENT APPROVAL PENDING  
**Event window:** November 2026  
**Exact dates:** NOT YET FROZEN  
**Format:** 7–10 day online buildathon  
**Latest bounded planning estimate:** 15–25 teams  

This brief turns the Financial Modeling Prep email thread into participant-facing technical rules without upgrading a pending sponsorship into a confirmed public partnership.

## 1. FMP claim boundary

Financial Modeling Prep has confirmed a proposed operating configuration for event access, but its management/leadership review remains the final public-claim gate.

Until explicit final approval is received and preserved:

- do **not** call FMP a confirmed sponsor, partner, or endorser;
- do **not** publish an FMP logo or brand asset;
- do **not** promise API keys to participants as guaranteed;
- do **not** imply that FMP has approved FinanceMeta, the event, participants, judging, or project quality.

The correct internal description is: **FMP technical/data-access terms confirmed; final management approval pending.**

## 2. Confirmed FMP technical constraints

If final approval is granted and keys are provisioned, the event must enforce all of the following:

| Constraint | Frozen value |
| --- | --- |
| Dataset coverage | FMP datasets excluding real-time quotes |
| Real-time quotes | Not permitted |
| Rate limit | 300 calls per minute |
| Data bandwidth cap | 20 GB |
| Key model | One event-only API key per team |
| Access duration | Event window only |
| Required attribution | `Financial Data Powered by FMP` |

No participant-facing material may weaken these limits.

## 3. Intended educational/prototype use

The buildathon is for student teams building bounded finance/economics prototypes, including examples such as:

- market-data visualizations;
- equity-research tooling;
- financial statement/fundamental analysis tools;
- educational finance applications;
- bounded quant/research prototypes;
- finance/economics data products.

The event does not request unrestricted production access or an ongoing commercial data license.

## 4. Minimum project discipline

Every submitted project must identify:

1. the user or problem;
2. the input data;
3. the system/output produced;
4. a validation plan;
5. a runnable or reviewable demo/artifact.

A project that only presents a pitch deck without a reviewable technical or analytical artifact is incomplete.

## 5. Data-use rules for participants

If FMP access is activated:

- Never request or proxy prohibited real-time quote endpoints.
- Never expose a team API key in a public repository, frontend bundle, screenshot, notebook output, demo recording, or submission artifact.
- Treat each key as team-scoped and event-only.
- Do not share keys across teams.
- Stay within the stated rate and bandwidth limits.
- Cache/reuse responses when reasonable rather than issuing unnecessary duplicate requests.
- Do not use the event key for unrelated personal/commercial work.
- Include the exact attribution phrase `Financial Data Powered by FMP` anywhere FMP-derived data is shown publicly in event materials or the final project.
- After the event, remove event credentials and verify that no key remains in repository history or published artifacts.

## 6. Secrets handling

Recommended participant implementation:

- load the API key from an environment variable;
- keep `.env` files ignored by version control;
- use server-side/proxy access for web applications when a client bundle would expose the credential;
- redact keys from logs and error traces;
- rotate/revoke immediately if a key is exposed.

FinanceMeta should not collect team keys in a public form or spreadsheet.

## 7. Judging boundary

The following judging dimensions have been used in FinanceMeta planning and may appear in the final rulebook:

- usefulness;
- correctness;
- finance/economics depth;
- technical execution;
- validation;
- communication;
- continuation potential.

**Weights are not frozen in this technical brief.** Do not invent or publish scoring weights from this file. The final event rulebook must freeze weights, tie handling, conflicts policy, judging records, and any finalist round before submissions open.

## 8. Participant-count boundary

The email thread contains multiple historical planning estimates, including an earlier 40+ target and a later bounded 15–25-team initial-edition estimate.

For current planning, use **15–25 teams** only as a capacity estimate. It is not a registration count.

Before provisioning:

- freeze the exact registered-team count from the registration source of truth;
- freeze education-stage and country breakdowns only from actual registration data;
- provision no more keys than the final approved team set;
- never report a planning target as attendance or completion.

## 9. Launch gates

Before any FMP-backed participant access is announced or issued, all must be true:

- [ ] explicit FMP management/leadership approval is received and preserved;
- [ ] exact event dates are frozen;
- [ ] public event/registration page is frozen;
- [ ] final eligibility and team rules are frozen;
- [ ] final registered-team count is frozen from source data;
- [ ] endpoint/access tier is confirmed if FMP provides any narrower implementation detail;
- [ ] provisioning and expiry process is documented;
- [ ] the exact attribution phrase is present in participant materials;
- [ ] any FMP logo/brand use has explicit permission;
- [ ] final judging rulebook is frozen before submissions open.

## 10. Post-event evidence

Track separately:

- registered teams;
- teams actually issued keys;
- teams that actually used FMP data;
- active teams;
- completed prototypes;
- submitted artifacts;
- finalists;
- failures, quota/rate-limit incidents, credential incidents, and complaints.

A sponsor offer, provisioned-but-unused key, or warm email is not counted as participant usage or delivered project impact.
