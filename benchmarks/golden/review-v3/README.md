# Generalized PII field fixes: frozen evidence

Baseline detector: `4e46c86a6f6c7bf91e9525968981ae47d7984aba`.
Final detector: `12191083eea106edf85c7bbc6e60a140faf1b4c2`.
Branch: `codex/golden-improvements` in the authorized private repository.
Interpretation and results: [generalized-pii-fixes.md](../../../docs/generalized-pii-fixes.md).

This is post-diagnostic regression on previously used corpora. The labels,
NER outputs, mappings, exclusions and thresholds are unchanged. It is not a
fresh holdout or a measurement of the official hackathon score.

Actual-mask privacy gates pass on every evaluated case: no previously protected
gold alphanumeric position was exposed and no new non-gold alphanumeric position
was masked. The strict RedMadRobot typed-category gate **fails**: three newly
covered generic identity documents receive PASSPORT while gold says
DRIVER_LICENSE. Those 33 typed FP positions are disclosed, not removed from
scoring. All reported aggregate F1 values remain equal or improve. Acceptance
prioritizes actual masking with a consistent series/number grammar; it does not
claim that every prediction, entity category or exact case is unchanged.

## Evidence

- `golden-report.json`, `golden-validation.json`: 446 immutable cases, both
  profiles; all 892 masks identical to baseline. A number-field marker is split
  from one document entity without changing protected text.
- `scanpatch/`: fixed PERSON+LOCATION replay, all 532 cases and the original
  164-case primary subset; all actual masks unchanged. The historical initial
  period boundary mismatch remains.
- `pii-bench/`: both original splits, 1810 cases, frozen PERSON-only protocol;
  108 audit strata and actual mask coverage against complete per-case gold.
  See its README for source/cache hashes and omitted local prediction files.
- `redmadrobot/`: 2839 aligned cases, unchanged two exclusions, all three
  profiles. Includes all-case typed, untyped and actual-mask gates, the three
  residual category errors, and an unchanged ambiguous baseline miss. Its
  standalone aggregate tool reproduces numeric reports from compressed counts.
- `tests-summary.json`: 2618 tests passed without skips, isolated real Redis,
  CI lint scope `seif tests scripts`. Archived measurement scripts are preserved
  as executed; a repository-wide Ruff pass is not claimed.
- `authored-fixtures*.json`, `independent-review.md`, the two adversarial
  reviews: independent synthetic checks, including actual mask and restore.
- `cpu-baseline.json`, `cpu-final.json`, `cpu_benchmark.py`: sequential local
  CPU timing, 4014 calls/profile with cached NER. These do not measure HTTP RPS,
  tunnel performance or live NER inference.
- `punctuation-probe.json`: diagnostic counterfactual explaining why terminal
  initial punctuation was not trimmed universally; not a production change.
- `manifest.json`: source, input and artifact hashes.

Original external texts/caches, local prediction files and runtime source
snapshots are omitted. Compressed RedMadRobot reports contain numeric positions,
labels, counts and metadata, not original texts. This directory is excluded from
source submission ZIPs and image builds. Existing review-v2, reference-v2, golden
cases and the NER cache remain unchanged.

## Repeat the checks

```bash
python scripts/evaluate_golden.py --baseline benchmarks/golden/reference-v2.json \
  --output output/golden-quality/recheck-v3.json
ruff check seif tests scripts
pytest -q
python scripts/evaluate.py --output output/fixtures-v3.json
```

The full test run requires the real Redis/Sentinel setup documented in CI.
The external replay tools retain the host-specific paths used during measurement;
adapt those paths and provide the pinned original caches on another host.
The Scanpatch wrapper also uses the frozen helper scripts under the local
`output/golden-improvements/external-scanpatch` experiment and its parent runner.
Input/source hashes must match before comparing outputs. No replay downloads
corpora or calls a live NER model. Per-dataset READMEs record these requirements.

Intermediate candidates `5719ff5`, `d6b31a2` and `8224842` were not accepted as
final results: external checks found misses or category ambiguities despite unit
test passes. Their original artifacts remain under local
`output/generalized-fixes-v3/`. The final reports in this archive use only
`1219108`; earlier result files and annotation policies were not rewritten.
