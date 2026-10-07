# Ten-minute research/tool presentation script

Prepared speaking material, not a record of an accepted or delivered talk. Use `presentation_handout.pdf` and the paper's tables as the visual aids. All values are synthetic engineering controls.

**0:00-1:00 - The distinction.** A return calculation can be reproducible while the information used to choose the exposure was unavailable. Conversely, a correctly timed ledger can lose money. This work makes the first question executable and keeps it separate from the second. The contribution is a small operational control, not a trading model.

**1:00-2:00 - The row contract.** Explain the six fields: decision time, latest feature availability, latest training-label availability, target end, position and asset return. Availability must precede or equal the decision; the target must end later; the next row must start exactly at the prior target end. All instants have explicit UTC offsets. Point at the contiguous-interval equality and explain that skipping an interval is rejected rather than silently accepted.

**2:00-4:00 - The one-second twin.** Present the original six-row ledger. Then describe changing only the first latest-feature time to one second after the decision. The positions and returns stay the same, so an arithmetic-only calculation would produce the same P&L. The audit instead emits a row-level availability failure and withholds strategy, buy-and-hold and cash paths. A baseline is not exempt from an invalid evidence contract. Explain that a producer can still falsify metadata: consistency is not authenticity.

**4:00-5:30 - The complete controlled corpus.** Show the family counts. Six error families are introduced independently at all six row locations. Gap, overlap and duplicate-decision cases cover all five internal boundaries. Four schema/empty cases complete 55 invalid controls. The valid inputs are the original plus UTC-7 and UTC+5:30 spellings of identical instants. All 58 expected classifications matched; all valid performance objects match exactly. Do not call this population sensitivity or real-world accuracy.

**5:30-7:00 - Costs without hidden assumptions.** Explain opening/rebalancing exposure changes and final liquidation. A -1 to +1 reversal trades two units. At the final interval an extra closing unit is charged. The independent Decimal calculation parses the original strings, uses 70-digit precision and repeats all three paths at five fixed costs. The maximum total-return discrepancy is 3.44e-16. The cost curve is a calculation check, not a search for a profitable setting.

**7:00-8:00 - The losing pass.** At the original 10 bps combined setting, the strategy is -2.446481%, buy-and-hold +1.158764%, and cash 0%. The strategy passes integrity and loses. This is intentional: an audit pass cannot be presented as an investment result. Mention the omitted drift, execution, financing and borrow effects.

**8:00-9:00 - Workflow.** The insertion point is after freezing a ledger and before publishing a performance report. Retain the consumed-byte hash, configuration hash, check findings and interval-level arithmetic. The analyst then supplies external source/vintage and timing evidence for human review. The code is dependency-free and produces a full failure corpus and manifest in one command.

**9:00-10:00 - What remains.** This is one constructed sequence, no observed dataset, and no adoption study. Closed forecasting evidence stays closed and outside the efficacy claims. The next useful empirical step would require a permissible source with independently supported release times and a frozen question. End by inviting scrutiny of the input contract and failure behavior rather than claiming financial superiority.

## Prepared Q&A

- **Does passing mean leakage-free?** No. Only the declared metadata and covered contract are checked. Omitted features or false times remain external evidence problems.
- **Why withhold baseline returns?** The report treats the ledger as one admissibility unit. Publishing a subset after failure can detach favorable numbers from invalid inputs.
- **Why only one asset?** A narrow explicit contract is reviewable. Portfolio accounting requires a separate contract rather than silently treating this approximation as a simulator.
- **Are the 70 OHLCV cases part of the 58?** No. They are a separate retained regression corpus on another input layer.
- **Why no CEUR claim?** The prepared manuscript is four pages including references and does not satisfy the workshop's at-least-five-page CEUR condition. No proceedings election or acceptance is claimed.
