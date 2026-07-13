#!/usr/bin/env bash
# safe_scan.sh — run a heavy analysis command (grep/python over output/) SAFELY on the
# GB10 (117GB unified mem, vLLM may be resident). Two guards:
#   1. oom_score_adj=+900: the scan is the OOM killer's FIRST victim (undoes the
#      session-wide -1000 exemption this shell inherits from oom_protect_session.sh —
#      without this, a runaway scan is UNKILLABLE and the killer eats systemd/gnome).
#   2. ulimit -v cap (default 20 GiB): a runaway allocation fails with ENOMEM inside
#      the scan instead of exhausting the host. Override: SAFE_SCAN_GB=n.
# CONTEXT (2026-07-13 OOM): bare `grep -r` over output/ walked into multi-GB binary
# .pt banks (1304 files >100MB); grep buffers whole binary "lines" -> ~110GB spike ->
# host OOM killed the session. ALWAYS scan output/ through this wrapper, and prefer
# `grep -rI --include='*.md' --include='*.json' --include='*.jsonl' --include='*.log'`.
#   usage: bash scripts/safe_scan.sh <command...>
set -uo pipefail
GB="${SAFE_SCAN_GB:-20}"
ulimit -v $((GB * 1024 * 1024)) 2>/dev/null || true
echo 900 > /proc/self/oom_score_adj 2>/dev/null || true
exec "$@"
