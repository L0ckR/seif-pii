# Second-pass private golden review

Implementation: `4e46c86a6f6c7bf91e9525968981ae47d7984aba`, branch
`codex/golden-improvements`; baseline `cbb628562f68afd764fb5a9f271c017d4c8fdeb1`.
Dataset and NER predictions are unchanged. This directory may contain original
dataset fragments; it belongs only in the authorized private repository and is
excluded from submission ZIPs and image builds.

- `annotation-audit.md` / `.json`: independent review of every original hybrid
  mismatch, without reading implementation or changing labels.
- `case-comparison.json`: before/after diagnostics for all 446 cases and the
  reasons/actions for 31 remaining hybrid mismatches.
- `cpu-baseline.json` / `cpu-final.json`: sequential local CPU measurements with
  frozen NER, 4014 calls per profile; these are not HTTP RPS measurements.
- `scanpatch-report.json`, `scanpatch-summary.json`, `scanpatch-validation.json`:
  unchanged external dataset/model-output replay, including original-prediction
  reproduction, all 14 source hashes and prior/new aggregate comparisons.
- `manifest.json`: hashes of the evidence files, dataset and NER cache.

The first parallel CPU trial overlapped with tests and was not used. A separate
sequential baseline/current pair produced the reported measurements. The earlier
Scanpatch preliminary run is preserved locally under `output/`; the committed
report uses the final implementation and matched its source hashes exactly.

The initial PERSON helper experiment added its candidates to the old detector;
it did not validate replacing the old cardholder rule. The real integrated
comparison found and fixed a foreign-name regression before this final result.
Final results here measure the actual detector with the old rule removed.

Reproduce quality:

```bash
python scripts/evaluate_golden.py --baseline benchmarks/golden/reference-v2.json \
  --output output/golden-quality/recheck-v2.json
```

For interpretation, see [the report](../../../docs/golden-improvements.md).
