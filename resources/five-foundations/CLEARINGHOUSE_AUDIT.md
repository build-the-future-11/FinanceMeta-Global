# Jump$tart Clearinghouse Audit

**Audit date:** September 8, 2026  
**Resource:** Five Foundations v1.0  
**Resource type:** Lessons / Lesson Plans and Activities  
**Submission status:** **NOT SUBMITTED**  
**Overall state:** **RESOURCE QUALITATIVE CRITERIA SUBSTANTIALLY SATISFIED; CANONICAL DEPLOYMENT AND PROVIDER ELIGIBILITY REMAIN OPEN**

Criteria source: https://jumpstartclearinghouse.org/static/src/assets/criteria.pdf

This file is intentionally fail-closed. It distinguishes the quality of the resource from FinanceMeta's eligibility to submit it and from the production availability of the canonical resource URL.

## Canonical publication boundary

- Provider website: `https://finance-meta.org`
- Resource URL: `https://finance-meta.org/learn/five-foundations`
- Standards section: `https://finance-meta.org/learn/five-foundations#standards`
- Canonical public-site repository: `build-the-future-11/finance4all-global-reach`

This `FinanceMeta-Global` directory is an operations/audit package, not the learner-facing source of truth. A GitHub tree/blob URL, Vercel preview/deployment URL, or other microsite must not be substituted into the external listing if the canonical production route is unavailable.

## Provider criteria

Jump$tart states that a provider must own or control the resource and must be a National Partner, a current/previous Clearinghouse provider, an entity well-known by Jump$tart, or be supported by a reference letter if Jump$tart requires one. It also requires an established Internet presence with complete resource information and an access path.

| Provider criterion | State | Evidence / action |
| --- | --- | --- |
| FinanceMeta owns or controls the submitted resource | PASS | The public lesson is implemented in FinanceMeta's canonical site repository and the audit package is versioned here. |
| Eligible-provider category | **UNRESOLVED** | Current evidence does not establish that FinanceMeta is a National Partner, previous provider, or an entity already considered well-known by Jump$tart. Jump$tart must confirm whether a reference letter is required. |
| Established Internet presence and resource access path | **CONDITIONAL** | Canonical paths are defined, but the production domain/resource/standards URLs must be tested logged out after deployment before submission. |

**Provider gate:** Do not submit or describe the resource as Clearinghouse-eligible until Jump$tart confirms provider eligibility or supplies the required reference-letter instructions.

## Resource listing criteria

### 1. Predominantly personal finance content

**State: PASS**

The lesson covers compounding, inflation/purchasing power, diversification, borrowing cost, and risk/expected return. There is no unrelated content.

### 2. Consistent with the National Standards

**State: PASS**

`STANDARDS_ALIGNMENT.md` maps the lesson to Saving 8-5 and 12-4, Investing 8-5, 8-7, 12-3, 12-4, and 12-6, and Managing Credit 8-2, 8-3, 12-1, and 12-3. The lesson narrows rather than contradicts broader standards.

### 3. Materially accurate and up to date

**State: PASS, with source maintenance required**

The September 2026 version was checked against current primary or regulator-maintained sources. Important accuracy corrections are explicit:

- real purchasing-power change uses the growth-factor ratio when an exact calculation is shown;
- diversification is not presented as a guarantee against loss;
- APR is distinguished from interest rate;
- expected return is not presented as guaranteed return.

Source review should be repeated before any future version is submitted if the resource changes materially.

### 4. Well written, professionally packaged, good quality

**State: PASS FOR CONTENT; CANONICAL WEB RELEASE IN PROGRESS**

The reviewed package contains a teacher guide, student handout, answer key, standards alignment, and audit. The canonical site implementation combines the instructional pieces into a printable public lesson. Production availability must still be verified after deployment.

### 5. Balanced and unbiased

**State: PASS**

The lesson does not favor a provider, security, lender, account, or financial product. Claims about diversification and risk include limitations rather than sales framing.

### 6. Appropriate for the target audience

**State: PASS**

The resource is scoped to grades 9-12, uses short worked examples, defines APR and interest rate, and separates optional quantitative extensions from required understanding.

### 7. No disrespectful or discriminatory language/images

**State: PASS**

The package contains no demographic assumptions, stereotypes, discriminatory language, or disrespectful imagery. Examples use generic monetary units and hypothetical situations.

### 8. Broadly available and easily accessible nationwide

**State: CONDITIONAL ON CANONICAL DEPLOYMENT + FINAL URL TEST**

The intended access path is `https://finance-meta.org/learn/five-foundations`, with no account or payment required. Before submission, verify from a logged-out browser that the provider domain, resource route, and standards anchor resolve and that no regional, school, account, or payment restriction blocks access.

### 9. Transparent pricing/access conditions/terms

**State: PASS**

The resource is specified as free, requires no account, and requests no personal financial information. Final live behavior must match those terms before submission.

## Specifically ineligible categories

| Ineligibility risk | Audit result |
| --- | --- |
| Sells/promotes products, services, or specific financial accounts | CLEAR. No product sales or recommendations. |
| Drives commerce / paid advertising | CLEAR. No affiliate links, ads, or commercial CTA. |
| Unauthorized aggregation of copyrighted material | CLEAR. The package summarizes concepts and links to source materials rather than reproducing third-party curricula. |
| Standalone article/blog/vlog | CLEAR. This is a structured lesson package with educator and learner materials. |
| Financial advice beyond broad general education | CLEAR. The advice boundary is stated repeatedly and no individualized action is recommended. |

## Source-by-claim accuracy check

| Claim | Primary support | Audit conclusion |
| --- | --- | --- |
| Compound interest applies interest to principal plus previously earned interest | 2021 National Standards, Saving 8-5 | Accurate. |
| Inflation can reduce purchasing power when nominal growth lags price growth | 2021 National Standards, Saving 12-4 and Investing 12-4 | Accurate. |
| Diversification reduces concentration but does not guarantee against market loss | 2021 National Standards, Investing 8-5; SEC Investor.gov diversification guidance | Accurate with limitation included. |
| Interest rate and APR are different; APR includes interest and certain fees | CFPB loan interest rate vs APR guidance | Accurate. |
| Higher interest rates and longer terms generally increase borrowing cost | 2021 National Standards, Managing Credit 8-3 | Accurate. |
| Greater potential/expected return generally comes with greater risk | 2021 National Standards, Investing 12-3; SEC Investor.gov risk guidance | Accurate when stated as a general relationship, not a guarantee. |

## Final release gates before submission

Do not submit until all are complete:

- [x] Resource is complete enough for a teacher and learner to use without missing instructional pieces.
- [x] Standards alignment is explicit.
- [x] Advice/product-promotion audit passes.
- [x] Accuracy audit uses current primary/regulator sources.
- [x] Pricing/access terms are explicit.
- [ ] Canonical site change is merged and deployed.
- [ ] `https://finance-meta.org` is verified logged out.
- [ ] `https://finance-meta.org/learn/five-foundations` is verified logged out and contains the complete lesson.
- [ ] `https://finance-meta.org/learn/five-foundations#standards` resolves to the standards section.
- [ ] Jump$tart confirms FinanceMeta's provider-eligibility route, including whether a reference letter is required.
- [ ] Submission form fields are rechecked against the live canonical resource.
- [ ] Submission receipt is preserved after actual submission.

## Strongest defensible claim today

FinanceMeta has a complete, audited 35-minute personal-finance lesson whose content is mapped to the 2021 National Standards, and a canonical web implementation is being release-gated on `finance-meta.org`. The resource has **not** been submitted or accepted by Jump$tart; production access and provider eligibility remain unresolved gates.
