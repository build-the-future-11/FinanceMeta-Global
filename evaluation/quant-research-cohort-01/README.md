# FinanceMeta Quant Research Cohort 01 — pre-result contract

**State: FORMING / PRE-RESULT. This directory does not authorize held-out evaluation and does not claim a market edge.**

This is the Week-1 scientific contract proposed for FinanceMeta Quant Research Cohort 01 (issue #35). The purpose is to make the first cohort small enough to finish, strict enough that a null result is useful, and reproducible enough that another contributor can challenge the result.

## Frozen dataset family

Cohort 01 uses **BTCUSDT spot 1-hour klines from Binance Public Data / Binance Vision, monthly archives from 2020-01 through 2024-12**.

Why this family:

- one liquid instrument avoids universe-selection and survivorship ambiguity in the first cohort;
- the public archive exposes monthly files and official checksum companions;
- timestamps and kline fields are documented by the upstream public-data project;
- the period ends before 2025-01-01, when Binance documents a spot timestamp-unit change from milliseconds to microseconds;
- the exact raw archive bytes can be pinned before modeling.

This choice is about reproducibility, not an assumption that BTC is uniquely predictable.

## Research question

Using only information available by the close of hour `t`, does a random-forest signal improve held-out next-hour direction prediction over transparent persistence and logistic-regression baselines, and does any improvement survive prespecified simulated costs across frozen volatility/trend regimes?

A negative result is a valid completion.

## Chronology

- **Train:** 2020-01-01 through 2022-12-31
- **Validation:** 2023-01-01 through 2023-12-31
- **Held-out test:** 2024-01-01 through 2024-12-31

No random train/test split is permitted. Held-out access remains disabled until the data lock, leakage audit, selected configuration, source SHA, and independent reviewer sign-off are committed.

## Before anyone models

1. Download all 60 monthly ZIPs and their official `.CHECKSUM` companions.
2. Verify every archive.
3. Run the continuity/schema checks in `data_manifest.json`.
4. Produce and commit `data_manifest.lock.json` plus `data_quality_report.json`.
5. Implement frozen features/labels and write `leakage_audit.md`.
6. Run train/validation only.
7. Freeze the selected candidate configuration and source commit SHA.
8. Obtain independent reproducibility-reviewer sign-off.
9. Only then unlock the 2024 test period once.

## Primary claim rule

The candidate may be described as adding held-out predictive value only if:

- held-out ensemble ROC-AUC exceeds logistic-regression ROC-AUC by at least **0.01**, **and**
- the lower bound of the frozen 95% 168-hour moving-block-bootstrap CI for the AUC difference is **> 0**.

Otherwise the result is **no demonstrated improvement**. No fallback metric may replace this rule after the test is opened.

The simulated long/flat return analysis is secondary and cannot rescue a failed primary predictive result. It uses frozen 0, 2.5, 5, and 10 bps per-unit-turnover assumptions and must be described as a retrospective diagnostic, not realizable profit.

## Reviewer checklist

The independent reviewer should reject held-out unlock if any of these are unresolved:

- checksum or archive-version ambiguity;
- silent missing-bar filling;
- duplicate or non-monotonic timestamps;
- a feature that uses `t+1` or later;
- scaler/model selection fit using validation/test information outside the frozen procedure;
- changed regime thresholds, costs, probability threshold, feature set, seeds, or metric after seeing validation outcomes beyond the explicitly allowed grid selection;
- missing losing seeds/regimes;
- a public claim that implies live-trading profitability or general market predictability.

## Required evidence package

Week 1: `experiment_contract.json`, locked data manifest, data-quality report, leakage audit, reproducible baselines.

Final: per-seed, per-regime, cost-sensitivity and placebo tables; primary summary; short report with failure analysis; one-command reproduction path.

Related: #35.
