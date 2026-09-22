# Code quality integration evidence

This archive preserves the native analyzer output for source revisions `547ed56` and `1cde5ce`, intermediate Sonar scans, and frozen detection-parity evidence. See `docs/code-quality-integration.json` for the machine-readable comparison and the integration report for interpretation.

Native Ruff reports were copied byte-for-byte. The final raw security report still contains all 969 findings, including 942 test assertions; reviewed CI policy and raw diagnostic counts are different measurements. Sonar's final seven findings, native severities, two reported vulnerabilities, security rating C and quality-gate result are also preserved. No finding was deleted or marked resolved in these artifacts.

`artifact-sha256.json` hashes the archived files and records original-byte hashes for gzip files. Gzip decompression reproduces their source bytes. `sonar/environment` contains server, scanner, plugin and quality-profile evidence; scanner logs and credentials are excluded. The all-source analysis includes test files as source and imported no coverage report, so reported Sonar coverage 0% is not a measured pytest coverage result.

`pii-bench` contains the original runner, source manifests, complete report, worker checks, and both identical ordered-output files compressed with gzip. They contain case IDs, hashes, numeric offsets/confidence, entity labels and fixed detector reason codes, not corpus text or replacement originals. The original README and artifact manifest describe the original local run, including source snapshots that can be recovered from the pinned Git revisions; those duplicated source snapshots are not included here. To reproduce, follow `pii-bench/original-README.md` and run the copied runner in a fresh output directory.

`redmadrobot` and `scanpatch` preserve the corresponding complete reports, replay runners and compressed original outputs; the integrated raw spans, actual masks and metric scopes are unchanged on 2,839 aligned RedMadRobot rows (three profiles) and 532 Scanpatch rows (two profiles). `web` records the checked demo behavior and mock-service browser checks.

`golden/report.json` contains the unchanged quality metrics. `golden/prediction-parity.json` verifies every rules/hybrid result against the accepted golden release using entity offsets and hashes of actual masked strings; text-bearing prediction files are deliberately not copied. `tests/junit.xml.gz` has 2,628 passing test records and no captured output. Regex parity reports retain their original counts.

These are local code-analysis and regression results, not the organizer's scoring implementation or a throughput measurement. Organizer annotations remain provisional AI labels. The corpora are previously inspected regression data, not new holdouts.
