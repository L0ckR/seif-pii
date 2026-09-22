# RedMadRobot behavior parity after code-quality refactoring

Baseline: `12191083eea106edf85c7bbc6e60a140faf1b4c2` (accepted implementation).
Current: `1cde5ce3c6a70fabfe8f09b56f01de624d69633d`.

Result: PASS on all 2839 aligned rows in rules, PERSON-only and PERSON+LOCATION modes (8517 row/profile comparisons). Ordered raw type/start/end predictions, full masked-string SHA-256, exact masked alphanumeric offsets, mapped spans and every frozen metric scope are identical. Actual mask/restoration round trips pass for both revisions (17034 round trips). The baseline exactly reproduces the prior accepted report. The two original exclusions from 2841 offered rows are unchanged.

This is behavior parity, not a fresh quality estimate. Existing accepted-baseline limitations remain. Confidence/reason fields are not compared; type/start/end/order are. Actual masking uses mask mode; randomized token mode is not compared. No model inference, network, source edits, label edits or bootstrap was performed. All source/input hashes are included in `report.json`; frozen PERSON/LOCATION cache tuples use the original reconstructed score 0.85.

Runtime: CPython 3.13.7 at `/tmp/seif-presidio-313/bin/python`, regex 2026.9.10. The detector replay imports each exact source snapshot's own evaluator/detector helpers. Aggregation uses Python standard library only.

Full replay commands from `/home/lockr/projects/seif-pii-code-quality` (host-specific paths):

The baseline result and identical runner were reused byte-for-byte from `output/code-quality/rmr-parity-44df456`; the current-version replay and aggregation were executed anew. To rerun the baseline command below, export its exact Git snapshot into `baseline-source` in this directory first.

```bash
/tmp/seif-presidio-313/bin/python output/code-quality/rmr-parity-1cde5ce/runner.py --source output/code-quality/rmr-parity-1cde5ce/baseline-source --revision 12191083eea106edf85c7bbc6e60a140faf1b4c2 --data /home/lockr/projects/seif-pii/output/external-bench-next --historical-report docs/redmadrobot-comparison.json --output output/code-quality/rmr-parity-1cde5ce/baseline.json
/tmp/seif-presidio-313/bin/python output/code-quality/rmr-parity-1cde5ce/runner.py --source output/code-quality/rmr-parity-1cde5ce/current-source --revision 1cde5ce3c6a70fabfe8f09b56f01de624d69633d --data /home/lockr/projects/seif-pii/output/external-bench-next --historical-report docs/redmadrobot-comparison.json --output output/code-quality/rmr-parity-1cde5ce/current.json
/tmp/seif-presidio-313/bin/python output/code-quality/rmr-parity-1cde5ce/aggregate.py
```

Every `seif/*.py` and `scripts/*.py` snapshot file was exported with `git show <revision>:<path>` and independently checked against its Git revision after replay. The runner checks source and input hashes before/after computation and refuses to overwrite an existing replay result. Dataset, cached NER offsets, original protocol/card and historical report are needed for a full detector replay. They are not needed for numeric aggregation.

Archive these files: `README.md`, `runner.py`, `aggregate.py`, `reference.json`, `report.json`, `summary.md`, `validation.json`, `archive-validation.json`, `baseline.json.gz`, `current.json.gz`. Compressed reports contain types, numeric offsets/counts, masked-string hashes and metadata, without original text or masked text. Keep the source snapshots and large uncompressed replay JSON local.

Standalone replay: place `runner.py`, `aggregate.py`, `reference.json`, `baseline.json.gz` and `current.json.gz` in a fresh directory and run `python aggregate.py`. The script reads gzip directly and reproduces report, summary and validation byte-for-byte. `reference.json` contains canonical digests of the previous accepted report's source/protocol/metric/prediction fields, not replacement labels. Archive checksums and the standalone replay check are in `archive-validation.json`.
