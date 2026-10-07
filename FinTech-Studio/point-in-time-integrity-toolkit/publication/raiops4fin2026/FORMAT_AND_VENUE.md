# Format and venue verification

Checked 7 October 2026.

- Official workshop: https://raiops4fin2026.github.io/ICAIF/
- Readable official Pages source: https://github.com/RAIOps4Fin2026/ICAIF/blob/main/index.html ; inspected Git blob `23a0217952b6700c491835333a1a5560ec824899`.
- Host workshop extension: https://icaif2026.org/workshop.html
- Workshop CFP PDF (older date; current extension takes precedence): https://icaif2026.org/calls/workshops/RAIOps4Fin-2026-Call-for-Papers.pdf
- Host formatting: https://icaif2026.org/call-for-papers.html

The workshop specifies six pages excluding references/appendices, English PDF, double-blind review and in-person presentation. Its own page excludes references/appendices while the older PDF says references; this manuscript contains no appendix and is four pages total, so fits either limit. The host requires ACM `sigconf,anonymous`. The current workshop deadline is 12 October 2026 at 23:59 AoE. The workshop site lists notification on 14 October; the central extension gives 16 October and accepted-author registration by 18 October AoE. Attendance is 14 November in Milan. These administrative conflicts remain visible rather than guessed away.

The prepared four-page paper is below the at-least-five-page condition for possible CEUR proceedings. No archival consent/election is made.

## Supplied ACM class

The official ACM template download returned HTTP 403. The maintained ACM class source was retrieved instead from:

- https://raw.githubusercontent.com/borisveytsman/acmart/master/acmart.dtx
- https://raw.githubusercontent.com/borisveytsman/acmart/master/acmart.ins

Running the unmodified `acmart.ins` generated `acmart.cls`. The class is distributed under the LaTeX Project Public License; its copyright and license header are retained. Full maintained source: https://github.com/borisveytsman/acmart . No local style edits were made.

- Supplied class SHA-256: `22d3bc6308fb8f64aa38b69bedc5ff5f311bbfd71f02f4af3746ae7a91ac3585`
- Downloaded DTX SHA-256: `8780f6b7763852a290b0fcbc5d53a7209e324569a082aaa95ade79d2e7f7db8f`
- Downloaded INS SHA-256: `fa7daaa9375775ce699f33f909667bdb83c0408fd8d67606926df86c346c4044`

The successful PDF uses Libertine, Inconsolata and newtx fonts, not the class's missing-font fallback. Final author fields and accountability approval remain outstanding. AI-assisted drafting, code and deterministic figure preparation are disclosed in the manuscript.
