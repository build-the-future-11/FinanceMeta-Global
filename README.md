# FinanceMeta Global

FinanceMeta Global is a public workspace for student work in finance, economics, quantitative research, markets, and financial technology.

This repository is intentionally evidence-first. It should not be read as proof of membership scale, partnerships, publications, internships, competition outcomes, or active programs unless those claims are backed by linked artifacts.

## Current repository contents

- `FI-JEPA/` — an existing research directory that must be evaluated on its own evidence and claim boundaries.
- `OPERATING_SYSTEM_2026.md` — operating standard for research, chapters, programs, participant outcomes, partnerships, and evidence reporting.
- `registry/programs.json` — machine-readable program registry; program families currently default to planned/unverified until evidence records satisfy launch gates.
- `registry/projects.json` — machine-readable research/project registry with explicit evidence boundaries.
- `templates/program_evidence_record.md` — reusable execution/evidence record for a bounded program cycle.
- `templates/chapter_monthly_evidence.md` — chapter activity and trailing-45-day status evidence template.
- `QUARTERLY_OUTPUT_REPORT_TEMPLATE.md` — outcome report that emphasizes completed work over audience or registration vanity metrics.
- `LICENSE` — repository license.

## Operating principle

FinanceMeta should optimize for completed, reviewable outcomes rather than registrations or promotional reach. Research and program claims move through explicit evidence levels from proposal to independent validation.

Read **[OPERATING_SYSTEM_2026.md](OPERATING_SYSTEM_2026.md)** before adding a new program, chapter, research project, public metric, partnership claim, or competition.

## Start a program or chapter record

A proposed program should first be registered in `registry/programs.json`, then receive a bounded evidence record based on `templates/program_evidence_record.md`. A chapter is counted as active only when its monthly evidence record satisfies the trailing-45-day qualifying-output rule in the operating system.

Do not change `PLANNED_UNVERIFIED`, `UNAUDITED`, or another conservative state merely because a launch is intended. Advance state only when the required linked artifacts exist.

## Research release standard

A serious research release should include:

1. a falsifiable question;
2. fair baselines;
3. dataset and provenance information;
4. exact reproduction instructions;
5. raw or traceable results;
6. limitations and negative findings;
7. an explicit release decision.

Synthetic experiments must be labeled synthetic. Simulated finance results must not be presented as realized returns. Failed or inconclusive experiments remain part of the record.

## Quarterly reporting

Use `QUARTERLY_OUTPUT_REPORT_TEMPLATE.md` to report completed outputs, negative findings, chapter health, externally verified outcomes, and portfolio decisions. Planned targets belong in the commitments section and must not be rolled into achieved metrics.

## Contributing

New work should begin with a bounded project or program record rather than a marketing description. Until a dedicated project template is added, use the required project record in `OPERATING_SYSTEM_2026.md` as the minimum research specification and record it in `registry/projects.json`.

The goal is that an external reviewer can quickly answer: **what exists, what ran, what evidence supports it, and what remains unproven?**
