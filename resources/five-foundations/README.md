# Five Foundations: A 35-Minute Personal Finance Lesson

**Version:** 1.0  
**Created:** September 8, 2026  
**Provider:** FinanceMeta  
**Target audience:** U.S. high-school learners, grades 9-12  
**Format:** classroom lesson, student handout, answer key, and standards map  
**Cost:** free  
**Account required:** no  

## Publication architecture

**Canonical learner-facing page:** `https://finance-meta.org/learn/five-foundations`  
**Canonical public-site repository:** `build-the-future-11/finance4all-global-reach`  
**Public route source:** `src/pages/learn/FiveFoundations.tsx`

This directory is the operations/audit package used to preserve standards mapping, submission state, and review evidence. It is **not** the canonical public educational surface. If learner-facing copy here and the canonical site differ, the public-site implementation must be reconciled before any external submission or review.

Until the canonical production URL is verified from a logged-out browser, do not substitute this GitHub directory or a Vercel preview/deployment URL as the resource link in an external listing.

## Purpose

Five Foundations is a short, general financial-education lesson covering five ideas that recur across saving, borrowing, and investing decisions:

1. compound interest;
2. inflation and purchasing power;
3. diversification;
4. borrowing cost, interest rate, and APR; and
5. risk and expected return.

It is designed for one 35-minute class. The lesson uses hypothetical numbers and general concepts. It does not recommend a security, financial product, lender, broker, account, portfolio, or individualized financial action.

## Audit package

- `TEACHER_GUIDE.md` preserves the reviewed run-of-show and worked examples.
- `STUDENT_HANDOUT.md` preserves the reviewed learner exercises.
- `ANSWER_KEY.md` preserves worked solutions and teaching notes.
- `STANDARDS_ALIGNMENT.md` records the mapping to the 2021 National Standards for Personal Financial Education.
- `CLEARINGHOUSE_AUDIT.md` records the fail-closed audit against the Jump$tart Clearinghouse listing criteria.
- `SUBMISSION_PACKET.md` contains prepared form fields and the canonical URL gate.
- `resource_manifest.json` records release state, canonical-publication ownership, and claim boundaries in machine-readable form.

## Learning objectives

By the end of the lesson, a learner should be able to:

- calculate two periods of annual compound growth;
- distinguish a nominal increase from a change in purchasing power;
- explain what diversification can and cannot do;
- distinguish an interest rate from APR and explain why borrowing terms affect total cost; and
- explain the general relationship between investment risk and expected return without treating return as guaranteed.

## Scope boundary

This resource is educational, not financial advice. The examples are intentionally hypothetical. Learners should not use the lesson as a recommendation to buy, sell, borrow, open an account, or choose a financial provider.

The resource contains no paid advertising, affiliate links, product placement, stock picks, lender recommendations, or calls to purchase FinanceMeta services.

## Relationship to the September 2026 pilot

This public resource is separate from the frozen `FINANCEMETA-LITERACY-SEP2026-v1` evaluation protocol. It does not change the frozen intervention or assessment. Do not substitute this public lesson for the pilot intervention without creating a new protocol version.

## Access and terms

The intended public resource is free at `https://finance-meta.org/learn/five-foundations`, with no account or personal financial information required. The final live access state remains a release gate until the production route is independently verified.

Repository content is governed by the repository's MIT License unless a file states otherwise.

## Primary references

The lesson was checked against primary or regulator-maintained educational sources:

- Jump$tart Coalition, *2021 National Standards for Personal Financial Education*: https://www.jumpstart.org/what-we-do/support-financial-education/standards/
- Consumer Financial Protection Bureau, *What is the difference between a loan interest rate and the APR?*: https://www.consumerfinance.gov/ask-cfpb/what-is-the-difference-between-a-loan-interest-rate-and-the-apr-en-733/
- U.S. Securities and Exchange Commission, Investor.gov, *What is Risk?*: https://www.investor.gov/introduction-investing/investing-basics/what-risk
- Investor.gov, *Diversify Your Investments*: https://www.investor.gov/introduction-investing/investing-basics/save-and-invest/diversify-your-investments

The standards mapping and source-by-claim audit are recorded in `STANDARDS_ALIGNMENT.md` and `CLEARINGHOUSE_AUDIT.md`.
