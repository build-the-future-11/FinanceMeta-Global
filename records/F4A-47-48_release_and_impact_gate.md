# F4A-47 / F4A-48 — release, privacy and verified-impact gate

## F4A-47 Trust, Privacy & Platform Release Gate

Keep three evidence classes separate:

1. **local static/prototype tools** — local browser/calculation checks;
2. **application source/build** — unit/type/lint/build/security checks;
3. **canonical production system** — real auth, backend policies, identities, email, account lifecycle, deployment and monitoring.

Passing a local/static check cannot certify production authentication or row-level access controls.

Before a production/member-system release record at minimum:

- canonical repository, commit, production domain and backend project;
- role model and authorised state transitions;
- two-identity cross-user isolation tests where user data exists;
- real sign-in/sign-out/session/redirect tests;
- database migration/verification state and row-level access policy tests;
- notification/email delivery receipts when the feature promises delivery;
- export/deletion/account-lifecycle workflow;
- secrets/logging/data-retention review;
- accessibility and negative-authorisation E2E;
- incident/rollback/maintenance owner.

The local RemitClear/MicroBiz tools have a different data surface and must not be used as evidence that the member portal passed these production checks.

## F4A-48 Verified Impact & Sustainability Ledger

Freeze the metric contract before participant actuals are entered. At minimum define:

### Unique learning completer

A deduplicated person who meets the programme's frozen completion definition. Preserve denominator, missing work and withdrawal status. Do not infer learning impact from completion alone.

### Active chapter

A chapter with a named accountable lead, verified host/safeguards when needed, at least one accepted delivery artifact in the current measurement window and a confirmed next-cycle commitment. A signup/location alone is not an active chapter.

### Partner contribution

A dated, evidenced contribution attached to a project and classified by contribution type. An expression of interest or logo is not a contribution.

### Repeat user / contributor

A deduplicated person with qualifying activity in at least two distinct measurement windows under a frozen activity definition. Automated/internal/test events are excluded.

## Data-control requirements

- documented lawful/consented basis appropriate to the activity;
- minimal participant identifier or privacy-preserving deduplication key;
- role-limited evidence store;
- separation of operational contact data from analysis extracts where feasible;
- denominator and attrition tracking;
- correction/version history;
- retention/deletion policy;
- metric steward and access reviewer.

Targets and actuals remain separate fields. Missing evidence yields `UNKNOWN`/not counted rather than a best guess.

No participant count, country count, chapter count or partner count should be published merely because it exists in an application form, CRM, spreadsheet or chat roster.
