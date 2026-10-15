# F4A-17 / F4A-18 — calculation and release review

**Date:** 2026-09-28  
**Scope:** second-pass specification/calculation review of the current local tools. This is not an independent human audit, field validation or production certification.

## F4A-17 RemitClear

The local tool defines:

- principal converted `P`;
- sender fee `F`;
- offered FX rate `R` (receiving per sending);
- contemporaneous reference rate `M`;
- receiving fee `G` in receiving currency.

Its displayed arithmetic is internally coherent with the stated definitions:

- sender debit = `P + F`;
- gross receipt = `P × R`;
- net receipt = `P × R − G` when the receiving fee is known;
- entered cost in sending currency = `F + P(1 − R/M) + G/M`;
- cost percentage uses converted principal as denominator.

The interface already warns that unknown receiving fees make the result incomplete, that observation times/conditions must be comparable and that it does not fetch live rates or recommend a provider.

### Tests/review still required before release candidate

1. reference rate must be finite and strictly positive;
2. amount-mode conversion must preserve the stated meaning of `P` and sender budget;
3. fee/rate/receiving-fee edge cases: zero, decimal precision, very small values and unrealistic/extreme values;
4. differing quote timestamps must remain visibly non-comparable;
5. unknown `G` must never be silently treated as zero;
6. currency/rounding display must not imply payment-settlement precision;
7. dated real corridor examples require permitted source use and same-condition snapshots;
8. a reviewer should recompute at least ten fixtures independently.

## F4A-18 MicroBiz

The local tool's stated calculations are also coherent as educational scenarios:

- contribution per unit = price − variable cost;
- whole-unit break-even = ceiling(fixed costs / contribution) only when contribution is positive;
- closing cash = opening cash + collections − cash paid;
- the stress case applies the declared collection reduction, cash-payment increase and one-time first-month outflow;
- invoice aging uses due date/outstanding amount, rejects overpayment/duplicate IDs and keeps future/today items current.

### Tests/review still required before release candidate

1. exact-decimal regression around break-even boundaries;
2. contribution = 0 and contribution < 0 must fail with an explanatory warning;
3. zero/negative fixed costs and malformed dates require explicit policy;
4. cash-stress percentages at 0%, 100% and boundary-adjacent values;
5. multi-month carry-forward reconciliation;
6. partial payment, fully paid, overpaid, duplicate ID and due-today invoice fixtures;
7. import/export round-trip and malformed/oversize JSON tests;
8. printable/exported reports must retain “scenario, not advice” labels;
9. approved business-host testing must use minimal/non-identifying inputs and cannot be described as profit improvement.

## Release boundary for both tools

The current local implementation does not move money, connect bank accounts, provide credit decisions or automatically upload user data. Keep it that way for this four-month scope unless a separate legal/security/product review deliberately changes the project.

Production/public release additionally requires F4A-47 privacy/accessibility/security checks. Participant-facing usability work requires host approval and applicable consent. No savings, profit, resilience or welfare improvement claim is authorised by a functional calculator test.
