#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.14.7t
  uv pip install -r requirements.lock
  uv pip install -e '.[dev]'
fi
export SEIF_DEMO=1
export SEIF_PORT="${PORT:-8765}"
exec .venv/bin/python scripts/serve.py
