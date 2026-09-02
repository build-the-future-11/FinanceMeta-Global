# FinanceMeta FinTech Studio Buildathon × Financial Modeling Prep

**Status:** DRAFT — partner management approval pending. Do not publish FMP as a finalized sponsor/partner until written approval is received.

## Event
FinanceMeta is planning a **7–10 day online FinTech Studio Buildathon in November 2026** for student teams building finance/economics tools, research prototypes, data products, visualizations, or educational applications.

Planning targets such as **40+ teams, 20+ completed prototypes, and 8+ finalists are targets, not achieved figures**.

Each team should define:
1. a clear user/problem;
2. the data it will use;
3. the tool/model/analysis it will build;
4. a validation plan;
5. a reproducible demo.

## FMP data access — confirmed operating constraints
Subject to final management approval, FMP has confirmed the following event-access structure:

- **Datasets:** FMP datasets may be used **except real-time quotes**.
- **Rate limit:** maximum **300 API calls per minute**.
- **Bandwidth:** maximum **20 GB** of data bandwidth.
- **Provisioning:** **one API key per team**.
- **Duration:** access is active **only for the duration of the event**.
- **Attribution:** participant and public materials using FMP data must include the exact phrase **“Financial Data Powered by FMP”**.

These constraints are part of the event design, not optional recommendations.

## Team key rules
- Assign one named key owner per team.
- Never commit an API key to GitHub or another source-control system.
- Never put a key in a public notebook, screenshot, presentation, recorded demo, submitted report, or shared chat.
- Use environment variables/secrets for code that needs the key.
- Do not share one team’s key with another team.
- Do not attempt to circumvent rate, endpoint, bandwidth, or event-duration restrictions.
- Stop and contact the organizers if a project unexpectedly requires real-time quotes or usage beyond the allowed limits; redesign is preferable to bypassing the partner boundary.

## Suitable project/data directions
Examples include:
- company profiles and reference data;
- historical price analysis;
- financial statements and fundamentals;
- equity-research tooling;
- market-data visualizations;
- bounded quantitative/research prototypes;
- finance education applications.

Projects should be designed so they remain useful without real-time quote endpoints.

## Evaluation discipline
Projects will be judged on usefulness, correctness, finance/economics depth, technical execution, validation, communication, and continuation potential. Teams should disclose limitations and avoid presenting a backtest, prototype, or model output as guaranteed investment performance.

If a project uses historical market data, teams should explicitly consider leakage/look-ahead, train/test chronology where relevant, fair baselines, and transaction costs where trading performance is evaluated.

## Attribution
Where FMP data is used, include:

> **Financial Data Powered by FMP**

Add the attribution to the project README/report and to any public demo, showcase page, or presentation that materially uses FMP data.

## Post-event reporting
For each project using FMP, organizers should retain:
- team + project name;
- FMP data families/endpoints used;
- approximate usage level;
- demo/repository link where permitted;
- whether required attribution is present;
- finalist/continuation status.

## Approval boundary
This document encodes the operating terms already communicated by FMP, but **final management approval is still pending**. Until that approval is received, do not publish logos, announce a finalized sponsorship, distribute production event keys, or imply that final approval has already occurred.

Tracking: `build-the-future-11/FinanceMeta-Global#20`.
