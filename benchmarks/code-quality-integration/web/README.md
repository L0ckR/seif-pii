# Local browser validation

Real Chromium loaded the integration worktree through an ephemeral FastAPI server bound to 127.0.0.1. The module script completed initialization; `/health` and `/v1/types` returned 200, the UI showed online status and the translated entity label. A synthetic email with a leading emoji was protected through `/v1/mask` and restored exactly through `/v1/unmask`. A rejected browser clipboard promise displayed the fallback notice. No console errors or warnings occurred.

Source hashes, scope, and limitations are in `report.json`; CLI assertions, network statuses, and page snapshots are in this directory. The clipboard refusal was injected for deterministic branch coverage. No production files were edited and no public service was contacted.
