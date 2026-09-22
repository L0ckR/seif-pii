# Frozen RedMadRobot regression replay

Compares source commits `cbb628562f68afd764fb5a9f271c017d4c8fdeb1` and `4e46c86a6f6c7bf91e9525968981ae47d7984aba`. The primary estimand is change in micro typed-character F1 for PERSON+LOCATION hybrid on the same 1237 common5 whole cases, retaining 371 negative cases. Character units include whitespace/punctuation inside spans. Secondary historical mode uses PERSON-only NER. Full corpus: 2841 offered rows, 2839 aligned, 2 frozen exclusions. Original labels, mappings and identical merge rules are validated before replay.

Runtime: CPython 3.13.7, NumPy 2.4.6, regex 2026.9.10. No model installation or inference is needed. `runner.py` blocks socket connections; detectors execute locally against frozen NER offsets at reconstructed score 0.85. The local environment used was `/tmp/seif-presidio-313/bin/python`.

## Archived files

- `runner.py`: reruns detector and merge code from an explicit source snapshot.
- `aggregate.py`: recomputes comparison, slices, case outcomes and paired bootstrap from saved count reports.
- `baseline.json.gz`, `current.json.gz`: **required for standalone aggregate replay**. They contain hashes, aggregate metrics and per-row TP/FP/FN counts; no source sentences, entity values, or error text. Uncompressed files are about 7 MB each; compressed files are much smaller.
- `report.json`, `summary.md`, `validation.json`: aggregate-only review artifacts. `validation.json` records SHA256 of the uncompressed measurement inputs.

The dataset CSV and original cache are not distributed here. End-to-end replay additionally needs the original pinned corpus/cache files, historical `docs/redmadrobot-comparison.json`, and code snapshots. Their SHA256 values are in `report.json`; source snapshot file manifests are included there. To reproduce the exact source manifests, export the listed relative paths at each recorded commit with `git show REVISION:PATH`; do not mix files from different revisions.

## Exact measurement commands

The commands below are the commands used on the measurement host, from `/home/lockr/projects/seif-pii-golden-improvements`. Paths are host-specific and should be adapted when running the archived tools elsewhere. `runner.py` refuses to overwrite an existing output; use fresh paths for a new replay.

```bash
/tmp/seif-presidio-313/bin/python output/golden-improvements/external-redmadrobot/runner.py --source output/golden-improvements/external-scanpatch/baseline-source --revision cbb628562f68afd764fb5a9f271c017d4c8fdeb1 --data /home/lockr/projects/seif-pii/output/external-bench-next --historical-report docs/redmadrobot-comparison.json --output output/golden-improvements/external-redmadrobot/baseline.json
/tmp/seif-presidio-313/bin/python output/golden-improvements/external-redmadrobot/runner.py --source output/golden-improvements/external-redmadrobot/current-source --revision 4e46c86a6f6c7bf91e9525968981ae47d7984aba --data /home/lockr/projects/seif-pii/output/external-bench-next --historical-report docs/redmadrobot-comparison.json --output output/golden-improvements/external-redmadrobot/current.json
/tmp/seif-presidio-313/bin/python output/golden-improvements/external-redmadrobot/aggregate.py
```

For a self-contained aggregate-only replay, copy the archived tools and compressed count reports to a fresh directory, unpack the count reports beside `aggregate.py`, then run it. NumPy is the only non-stdlib dependency of aggregation. This rewrites generated report/summary/validation in that directory, so retain the archived originals separately.

```bash
gzip -dk baseline.json.gz current.json.gz
python aggregate.py
```

The aggregate metrics and bootstrap samples are deterministic; generation timestamps and artifact hashes incorporating those timestamps differ on rerun. No error-text inspection or detector tuning occurred during this check. The benchmark had been used in prior development; this is regression evidence, not a fresh holdout claim.
