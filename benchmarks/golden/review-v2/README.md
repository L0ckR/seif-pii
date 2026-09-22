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
- `pii-bench/`: frozen historical PERSON-only regression, both original splits,
  unchanged common4 and improved supported8 results, independent per-case audit.
- `redmadrobot/`: frozen PERSON+LOCATION and historical PERSON-only regression;
  includes the observed exact-case regressions and paired bootstrap interval.
- `authored-fixtures.json`: unchanged evaluator, 48/48 exact cases before/after.

The two additional external directories include exact archived replay tools.
Their READMEs record host-specific measurement paths and required cached inputs;
adapt these paths on another host. Original external texts/caches and PII-Bench
offset prediction files remain local and are not bundled here. The PII-Bench
artifact manifest also records hashes of those omitted local prediction files.
RedMadRobot compressed count files contain only numeric per-row diagnostics and
support aggregate-only reproduction without the original texts. External reports
measure changes within each fixed protocol; PII-Bench PERSON-only is not a
measurement of the deployed PERSON+LOCATION profile.

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
