# RAIOps4Fin 2026: ledger-audit manuscript package

**Stage: manuscript and analysis prepared for human author review. Unsubmitted. The containing pull request is a draft and has not been merged. This is not whole-project or research-program completion.**

The package contains a complete four-page ACM `sigconf` anonymous tool manuscript, a one-page presentation handout, a ten-minute speaker script, reproducible figures, and the executed fault-enumeration evidence. The manuscript describes a dependency-free operational control on synthetic ledgers. It makes no observed trading, forecasting efficacy, production adoption or regulatory-certification claim.

## Completed analysis

- Fixed enumeration of **58 synthetic ledger controls**: 3 valid inputs (original plus two equivalent timezone representations) and 55 invalid inputs. Every expected classification matched; every invalid case withheld all performance paths.
- **15 Decimal-oracle comparisons**: strategy, buy-and-hold and cash at five fixed one-way cost settings. Maximum absolute total-return discrepancy is **3.44e-16**; turnover agrees exactly.
- **4 new oracle unit tests**, covering opening/closing costs, reversal costs, cash and fractional exposure.
- The existing **17 package tests** and frozen **70-fixture OHLCV corpus** remain separate evidence. They are not added to the 58-control denominator.

All inputs, reports and their SHA-256 manifest are in `evidence/ledger-fault-enumeration-v1/`. The study is an exhaustive enumeration of the specified fault locations in one six-row fixture; it is not an estimate of sensitivity on financial institutions' data.

## Reproduce from the repository root

Python 3.11+ is sufficient for analysis and tests; no runtime dependency is installed by these commands. Use a new output directory.

```bash
python -m unittest discover -s FinTech-Studio/point-in-time-integrity-toolkit/tests -v
python -m unittest discover -s FinTech-Studio/point-in-time-integrity-toolkit/publication/raiops4fin2026 -p 'test_*.py' -v
python FinTech-Studio/point-in-time-integrity-toolkit/publication/raiops4fin2026/analyze_ledger_controls.py \
  --package FinTech-Studio/point-in-time-integrity-toolkit \
  --output /tmp/financemeta-ledger-faults-new
cmp /tmp/financemeta-ledger-faults-new/summary.json \
  FinTech-Studio/point-in-time-integrity-toolkit/publication/raiops4fin2026/evidence/ledger-fault-enumeration-v1/summary.json
```

From this directory, `python build_figures_handout.py` rebuilds the PNG figure and handout using Matplotlib and ReportLab. Compile `manuscript.tex` twice with pdfLaTeX and a complete ACM-compatible TeX environment (Libertine, Inconsolata, newtx and kastrup/binhex dependencies). `python build_paper.py` runs the two ordinary pdfLaTeX passes. No shell escape is used. Rendering metadata can vary, so the package does not promise byte-identical PDFs across TeX/font versions.

`acmart.cls` is the unmodified generated ACM class obtained from the maintained ACM class source on 7 October 2026. Its source and license are recorded in `FORMAT_AND_VENUE.md`. It is not a locally invented approximation of `sigconf`.

## Venue fit and submission boundaries

The current workshop call welcomes reproducible tools and actionable methodologies in responsible financial-AI operations. This manuscript's fit is auditability and evaluation reliability; it is not an empirical financial model paper. The workshop allows up to six main pages, excluding references/appendices, and requires double-blind PDF review. The manuscript is four pages including references. **It does not meet the workshop's five-page minimum for possible CEUR inclusion, and no CEUR election is made.** It is deliberately not padded with unsupported results.

The deadline is **12 October 2026, 23:59 AoE** under the host conference's current workshop extension. The event is **14 November 2026 in Milan**. A presenting author must register and attend. The workshop site still gives 14 October notification while the host's extension gives 16 October; use the current portal/organizer instructions for that administrative detail. Accepted workshop authors are directed to register by 18 October AoE. No registration, submission, payment or message has been sent.

Before submission, human authors must review the code, facts and citations; settle the exact contributor list/order and affiliations; approve the disclosure of AI-assisted drafting, coding and figures; check overlapping/archival work; and confirm attendance. Public project links are intentionally outside the anonymous manuscript. If a supplement is shared, its anonymous access mechanism must also be checked. A publication-ready author list and accountability approval are not invented here.

## Protected evidence

The existing FI-JEPA v1 retained audit remains **CLOSED NEGATIVE / BOUNDARY**. It is neither rerun nor included as efficacy evidence in this manuscript. All previous ledger, OHLCV and closed-study files remain unchanged. Observed-data work, independent availability verification and external scientific review remain separate requirements for a broader finance research claim.

Presentation rendering requires the DejaVu Sans regular/bold fonts. The default path is `/usr/share/fonts/truetype/dejavu`; set `PRESENTATION_FONT_DIR` to the folder containing `DejaVuSans.ttf` and `DejaVuSans-Bold.ttf` on another installation. Both fonts are embedded in the handout.
