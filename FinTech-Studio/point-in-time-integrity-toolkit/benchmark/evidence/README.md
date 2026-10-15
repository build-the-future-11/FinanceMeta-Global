# Frozen synthetic benchmark evidence

The retained summary in `benchmark_summary.json` was produced from the frozen 70-fixture benchmark implementation in `../run.py`.

Local verification on 25 September 2026:
- 70 fixtures total;
- 10 clean controls;
- 10 per fault family;
- 100% detection for duplicate, out-of-order, gap, non-finite, OHLC, and negative-volume fixtures;
- 0% clean false-positive rate;
- 100% deterministic repeat rate;
- frozen release thresholds passed.

The benchmark runner writes one raw JSON report per fixture into `reports/` when executed. Those generated reports should be retained by CI/reviewer evidence rather than hand-edited.

Reproduce:

```bash
python -m unittest discover -s tests -v
python benchmark/run.py benchmark/evidence
```

This is synthetic structural-test evidence only. It is not real-market validation.
