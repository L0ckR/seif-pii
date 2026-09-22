#!/usr/bin/env bash
set -euo pipefail
# Requires installed cloudflared and a running local server (make run).
# Quick Tunnel URL is temporary; a named tunnel is required for a stable endpoint.
exec cloudflared tunnel --url "http://127.0.0.1:${PORT:-8765}" --protocol quic --no-autoupdate
