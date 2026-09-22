# Frozen PII-Bench integration parity

Accepted source `12191083eea106edf85c7bbc6e60a140faf1b4c2` and integrated source `1cde5ce3c6a70fabfe8f09b56f01de624d69633d` produced byte-identical recorded outputs for all 1,810 original cases (domain 900; entity 910) under both `rules` and `hybrid_person_only`. There were zero changed case/profile predictions, and all 7,240 exact restoration roundtrips passed. Both replays also reproduced every original accepted span from the prior release cache.

The comparison preserves the detector's ordered five-field spans (`start`, `end`, `type`, `confidence`, `reason`) and compares actual masked-string hashes and lengths, full replacement-record hashes, restored-string hashes and input hashes. Corpus text and replacement originals were kept in memory; the output files contain no raw corpus texts. The hybrid profile uses only the original cached Presidio PERSON candidates with score 0.85; LOCATION is excluded. There were no model or network calls.

## Reproduction

The actual command was run from `/home/lockr/projects/seif-pii-code-quality`:

```sh
/tmp/seif-presidio-313/bin/python output/code-quality/pii-parity-1cde5ce/parity_runner.py
```

The runner exclusively creates files and intentionally refuses to overwrite an existing run. To reproduce, copy `parity_runner.py` to a fresh, empty output directory and execute that copy with the same interpreter. Its output directory is derived from the copied script location. Its repository, accepted release and dataset locations are pinned absolute paths inside the script; preserve these local inputs and both Git revisions. Git snapshots are extracted and source hashes validated before and after replay.

Runtime: Python 3.13.7, pyarrow 25.0.1 and numpy 2.4.6, using `/tmp/seif-presidio-313/bin/python`. Git is required for snapshot extraction. Existing data are read from `/home/lockr/projects/seif-pii/output/external-bench`; original accepted provenance and cached predictions are read from `/home/lockr/projects/seif-pii-golden-improvements/output/generalized-fixes-v3/pii-bench-release`. No network access, model download or model runtime is required.

## Evidence

- `report.json`: full source, data, cache, runtime and output hashes; per-split/profile comparison; validation and limitations.
- `summary.json`: compact parity proof.
- `manifest-accepted.json`, `manifest-integrated.json`: pinned source file hashes.
- `outputs-accepted.json`, `outputs-integrated.json`: byte-identical per-case evidence, SHA256 `458518354713e4517c1b6860c7f565c1a8a35ccb381e926a3e487c2511fd84dd`.
- `worker-accepted.json`, `worker-integrated.json`: restoration and original-cache checks.
- `artifact-sha256.json`: integrity manifest for the archived artifacts, including this README and runner.

This demonstrates parity for the original frozen corpus and two profiles; it is not a claim about every possible input, HTTP behavior, live PERSON+LOCATION models or throughput. Metrics and bootstrap intervals were not recalculated because all compared predictions and actual masks are identical.
