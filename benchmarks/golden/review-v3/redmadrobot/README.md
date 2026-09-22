# RedMadRobot: final release measurement

Baseline source: `4e46c86a6f6c7bf91e9525968981ae47d7984aba`.
Release source: `12191083eea106edf85c7bbc6e60a140faf1b4c2`.
Document-fields source SHA-256: `b56592ad71b66a31ce8abeba82047a3818183d6f9ead81f19e9dbc58ea417c55`.

The actual-mask gate passes: no previously masked gold alphanumeric character is exposed and no new non-gold alphanumeric character is masked, across all 2839 aligned rows and all three modes. Aggregate F1 does not decrease in any reported scope/mode/metric. The strict typed-category gate FAILS: three generic identity-document pairs are newly protected but assigned PASSPORT where the reference expects DRIVER_LICENSE (33 typed FP characters). This disclosed tradeoff preserves consistent handling of nominative and inflected series/number labels. It is not a claim of zero errors or universal absence of regressions.

Primary estimand: common5 whole-case subset (1237 rows, including 371 negative rows), PERSON+LOCATION typed-character micro-F1. The supported8 whole-case subset contains 1830 rows. Per-type and per-position diagnostics cover all 2839 aligned rows. Historical PERSON-only and rules modes are separate. Typed character metrics include spaces/punctuation; actual mask metrics use `char.isalnum()` and real `mask(..., 'mask')` results followed by exact restoration checks. Actual masking evaluates all output candidate types against the same eight mapped gold types; off-scope gold types may affect absolute FP counts, while paired no-new-FP differences remain explicit. All metrics and exclusions are frozen; this previously used development corpus is not a fresh holdout.

Frozen input: `redmadrobot-rnd/pii_benchmark`, revision `f77ea831274daf980cc45c61a93c226be9d978d6`; 2841 offered rows, 2839 aligned, two original protocol exclusions. Corpus/card/protocol/cache SHA-256 are in `report.json` → `input_sha256`. Labels, mappings and exclusions match the preceding report. The baseline's complete scopes canonically reproduce the previous 4e report exactly.

Runtime used: CPython 3.13.7 at `/tmp/seif-presidio-313/bin/python`, regex 2026.9.10. The runner imports the frozen repository's detector/evaluation helpers. It blocks sockets; PERSON/LOCATION tuples come from the existing cache at reconstructed score 0.85. No live NER/model inference, training, label edits or new bootstrap interval occurs. Aggregation requires only Python standard-library modules.

Commands actually executed, from `/home/lockr/projects/seif-pii-golden-improvements` (paths are host-specific):

```bash
/tmp/seif-presidio-313/bin/python output/generalized-fixes-v3/redmadrobot-candidate-local-2/runner.py --source output/generalized-fixes-v3/baseline-source --revision 4e46c86a6f6c7bf91e9525968981ae47d7984aba --data /home/lockr/projects/seif-pii/output/external-bench-next --historical-report docs/redmadrobot-comparison.json --output output/generalized-fixes-v3/redmadrobot-candidate-local-2/baseline.json
/tmp/seif-presidio-313/bin/python output/generalized-fixes-v3/redmadrobot-release/runner.py --source output/generalized-fixes-v3/redmadrobot-release/current-source --revision 12191083eea106edf85c7bbc6e60a140faf1b4c2 --data /home/lockr/projects/seif-pii/output/external-bench-next --historical-report docs/redmadrobot-comparison.json --output output/generalized-fixes-v3/redmadrobot-release/current.json
/tmp/seif-presidio-313/bin/python output/generalized-fixes-v3/redmadrobot-release/aggregate.py
```

The candidate-local-2 baseline was copied byte-for-byte to this final directory. Its runner is byte-identical to the final runner. Both Python source snapshots were exported from their exact Git revisions; every snapshot file was checked against `git show`. Re-running the detector requires these `seif/*.py` and `scripts/*.py` source files plus the frozen CSV/card/protocol/cache and historical report paths. The runner refuses to overwrite existing numeric results; use a fresh directory. Source and input hashes are validated before and after detection.

For standalone aggregate replay, place `runner.py`, `aggregate.py`, `prior-reference.json`, `baseline.json.gz` and `current.json.gz` together in a fresh directory, then run `python aggregate.py`. It directly loads gzip when plain JSON is absent, does not read any corpus or source text, and reconstructs `report.json`, `summary.md`, `validation.json` and `paired-all8-character-counts.json` byte-for-byte. Deterministic gzip roundtrip checks and artifact SHA-256 values are recorded in `archive-validation.json`.

Publish/archive these 11 files:

- `README.md`
- `runner.py`
- `aggregate.py`
- `prior-reference.json`
- `report.json`
- `summary.md`
- `validation.json`
- `archive-validation.json`
- `baseline.json.gz`
- `current.json.gz`
- `paired-all8-character-counts.json.gz`

The compressed reports contain only counts, categories, numeric offsets, case identifiers, protocol/source metadata and hashes. They contain no original document text, masks, replacement values or reconstructed corpus sentences. Keep source snapshots and large uncompressed replay JSON local; they are unnecessary for aggregate replay. `prior-reference.json` contains canonical baseline-reference digests and numeric diagnostic references; it is not a label override.
