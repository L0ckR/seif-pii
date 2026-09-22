# Scanpatch parity: accepted121 → final1cde5ce

Baseline Git source: `12191083eea106edf85c7bbc6e60a140faf1b4c2`.
Current Git source: `1cde5ce3c6a70fabfe8f09b56f01de624d69633d`.

Result: PASS. All 532 rows × rules/hybrid profiles have identical ordered raw type/start/end lists, full masked-string SHA-256, masked alphanumeric offsets, mapped predictions and five frozen metric scopes. Both revisions pass actual mask→restore_exact on every row/profile (2128 round trips). The baseline reproduces the accepted Scanpatch121 predictions and metrics exactly. Primary common whole-case selection remains 164 cases; all 532 input rows and labels are unchanged.

Runtime used: CPython 3.13.7, regex 2026.9.10, PyArrow and the already installed Presidio2.2.364 score-evidence source. No NLP model is loaded or run. The frozen original evaluator validates corpus/cache hashes and every original annotation; score0.85 is justified by the existing Presidio source/configuration. Fixed accepted121 evaluation helpers are used with both detectors. The selected snapshot supplies actual detector and transform code. Socket calls are blocked. No code, corpus, label, mapping or threshold changes are made by this parity run.

Commands executed from `/home/lockr/projects/seif-pii-code-quality` (host-specific paths):

```bash
/tmp/seif-presidio-313/bin/python output/code-quality/scanpatch-parity-1cde5ce/runner.py --source output/code-quality/rmr-parity-44df456/baseline-source --revision 12191083eea106edf85c7bbc6e60a140faf1b4c2 --support-source output/code-quality/rmr-parity-44df456/baseline-source --data /home/lockr/projects/seif-pii/output/external-bench-next --output output/code-quality/scanpatch-parity-1cde5ce/baseline.json
/tmp/seif-presidio-313/bin/python output/code-quality/scanpatch-parity-1cde5ce/runner.py --source output/code-quality/rmr-parity-1cde5ce/current-source --revision 1cde5ce3c6a70fabfe8f09b56f01de624d69633d --support-source output/code-quality/rmr-parity-44df456/baseline-source --data /home/lockr/projects/seif-pii/output/external-bench-next --output output/code-quality/scanpatch-parity-1cde5ce/current.json
/tmp/seif-presidio-313/bin/python output/code-quality/scanpatch-parity-1cde5ce/aggregate.py
```

The snapshots are exact Git exports of `seif/*.py` and `scripts/*.py`; each snapshot file was independently verified against its Git revision after replay. `original_evaluator.py` is the unchanged frozen evaluator from the previous measurement; `original-report.json` supplies the original cache hash and Presidio configuration. Full detector replay needs those files plus the local frozen Scanpatch parquet/card/protocol/cache, and refuses to overwrite an existing result.

Archive these files: `README.md`, `runner.py`, `aggregate.py`, `reference.json`, `report.json`, `summary.md`, `validation.json`, `archive-validation.json`, `original_evaluator.py`, `original-report.json`, `baseline.json.gz`, `current.json.gz`.

Standalone numeric replay needs only `runner.py`, `aggregate.py`, `reference.json`, `baseline.json.gz` and `current.json.gz` together in a fresh directory. Run `python aggregate.py`; it reads gzip directly and recreates report, summary and validation byte-for-byte. No corpus, model or source snapshot is required for aggregation. Archive checksums and reproduction checks are recorded in `archive-validation.json`.

Compressed predictions and reports contain numeric offsets/counts, entity types and hashes, without source document text or masked output text. Confidence/reason fields and randomized token-mode strings are not compared. This checks behavior parity on an already used corpus, not new quality or language coverage; existing baseline errors remain unchanged. No bootstrap is needed for identical outputs.
