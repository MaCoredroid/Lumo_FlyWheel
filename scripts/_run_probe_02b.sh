#!/usr/bin/env bash
# _run_probe_02b.sh — fire-and-forget launcher for the attempt_02b probe.
# Usage: /path/to/_run_probe_02b.sh start
# Self-daemonises with nohup if called without the _inner flag so that the
# AppleScript caller does not have to embed a literal '&' (which is a reserved
# operator in AppleScript string interpolation).
set -u
MODE="${1:-start}"
LOG=/tmp/cnb55_probe_02b.log
DONE=/tmp/cnb55_probe_02b.done
if [ "$MODE" = "start" ]; then
  rm -f "$DONE" "$LOG"
  nohup "$0" _inner >/dev/null 2>&1 &
  echo $! > /tmp/cnb55_probe_02b.pid
  exit 0
fi
# _inner path: the actual probe work.
cd /Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel
{
  echo "=== attempt_02b probe start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  N=3 bash scripts/probe_family.sh
  rc=$?
  echo "=== attempt_02b probe end $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc ==="
  echo "EXIT=$rc" > "$DONE"
} >"$LOG" 2>&1
