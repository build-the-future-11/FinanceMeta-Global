# FinanceMeta Global

FinanceMeta Global is a public workspace for student work in finance, economics, quantitative research, markets, and financial technology.

This repository is intentionally evidence-first. It should not be read as proof of membership scale, partnerships, publications, internships, competition outcomes, or active programs unless those claims are backed by linked artifacts.

## Current repository contents

- `FI-JEPA/` — current proposal-stage research directory; its status is recorded conservatively in `registry/projects.json`.
- `OPERATING_SYSTEM_2026.md` — operating standard for research, chapters, programs, participant outcomes, partnerships, and evidence reporting.
- `registry/programs.json` — program launch-gate registry. Entries remain planned until evidence satisfies their gate.
- `registry/projects.json` — machine-readable project evidence registry and claim boundary.
- `templates/program_evidence_record.md` — bounded program-cycle execution and evidence record.
- `templates/research_project_evidence.md` — research question, baselines, provenance, result, reproducibility, review, and release record.
- `templates/chapter_monthly_evidence.md` — chapter evidence record enforcing the trailing-45-day activity definition.
- `QUARTERLY_OUTPUT_REPORT_TEMPLATE.md` — evidence-backed output report centered on completed work, failures, and external validation.
- `scripts/validate_registry.py` — conservative validator for registry schema and promotion-state contradictions.
- `LICENSE` — repository license.

## Operating principle

FinanceMeta should optimize for completed, reviewable outcomes rather than registrations or promotional reach. Research and program claims move through explicit evidence levels from proposal to independent validation.

Read **[OPERATING_SYSTEM_2026.md](OPERATING_SYSTEM_2026.md)** before adding a new program, chapter, research project, public metric, partnership claim, or competition.

## Evidence workflow

1. Register the program or project in `registry/`.
2. Create the appropriate evidence record from `templates/`.
3. Execute work and link actual artifacts rather than intended outputs.
4. Run the registry validator after status changes.
5. Advance maturity/evidence level only when the defined gate is satisfied.
6. Report completed outcomes through the quarterly report rather than rolling targets into achievement metrics.

```bash
python3 scripts/validate_registry.py
```

A chapter is counted as active only when a monthly evidence record satisfies the trailing-45-day qualifying-output rule. A research project should not move beyond M0/E0 because a paper idea or README exists; implementation, reproducibility, results, and external validation are separate gates.

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

Use `QUARTERLY_OUTPUT_REPORT_TEMPLATE.md` to report completed research/build artifacts, negative results, chapter health, externally verified outcomes, partnerships with concrete commitments, and portfolio decisions. Planned targets belong only in the commitments section.

## Contributing

New work should begin with a bounded evidence record rather than a marketing description. The goal is that an external reviewer can quickly answer: **what exists, what ran, what evidence supports it, and what remains unproven?**
