# F4A-09 — Global Financial Inclusion Atlas source audit

**Date:** 2026-09-28  
**State:** DATA PRESENT / INDEPENDENT REVIEW PENDING / PUBLIC RELEASE NOT AUTHORISED

## What exists now

The Finance4All library contains a local Atlas dataset and HTML view derived from the World Bank's 2025 Little Data Book / Global Findex material. The local records use 2024 observations, 2025 publication edition, explicit missingness, source PDF page references and a transformation version.

The current local indicator keys are:

1. `account_all`
2. `account_women`
3. `digital_payments`
4. `formal_saving`
5. `formal_borrowing`
6. `resilience_30d`
7. `mobile_phone`
8. `smartphone`

Current records are explicitly marked `assistant_visual_transcription; independent_review_pending`. `source_indicator_id` is not yet populated.

## Authoritative source route located

The World Bank Global Findex 2025 download page exposes country-level data for the 2024 round and offers CSV/XLSX/DTA/DataBank downloads. The direct CSV route identified for reconciliation is:

`https://thedocs.worldbank.org/en/doc/be6615202d1f08a25855c8ac2d615122-0050012025/related/GlobalFindexDatabase2025.csv`

This path is a source locator, not evidence that the local values have already been independently matched to those bytes.

## M1 reconciliation procedure

1. Obtain the official CSV through an authorised accessible route.
2. Bind the exact downloaded bytes to an F4A-01 dataset card with source title, vintage, retrieval time, use/redistribution notes and SHA-256.
3. Inspect source headers and metadata before changing the local mapping.
4. Map every local indicator key to the exact source indicator ID/label where the official file supports one.
5. Compare every non-missing transcribed value against the authoritative source. Keep a mismatch log; do not silently overwrite.
6. Verify that explicit missingness such as `not_surveyed_in_source_2024` is consistent with the official source rather than treating missing as zero.
7. Freeze the intended 30-country coverage and document the selection rationale without claiming representativeness.
8. Have a second reviewer independently verify a sample spanning countries, indicators, missing values and page/source mappings.
9. Only after reconciliation, promote the first country cards from review draft to release candidate.

## Claim boundary

The Atlas may describe observed country-level indicators and documented gaps. It must not turn cross-country associations into causal claims, individual credit/risk scores, rankings of people, or claims that Finance4All caused any observed outcome.

## Exit evidence

- exact official source bytes + hash;
- adopted dataset card;
- local-to-source indicator map;
- full mismatch/reconciliation log;
- frozen coverage list;
- missingness policy;
- independent review receipt;
- reproducible transformation/export script or documented deterministic process.

Until those exist, local values remain review data rather than a released Finance4All statistical product.
