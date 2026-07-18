#!/usr/bin/env bash
# fr13_gpu_clean.sh — one-shot GPU/campaign cleanup + GB10 unified-memory recovery.
# Fixes the recurring 2026-07-18 failure mode: a reaped campaign driver orphans its fr13-bigdenom-*
# vLLM container (holds ~90GB unified mem = wedge); leftover sample_dcgm samplers hold /dev/nvidia*
# and block recovery. Kills drivers by PIDFILE only (a pkill pattern can match the CALLER's command
# line and self-kill the invoking shell — never do that from an interactive call).
# Exit 0 = clean (avail>=90GB); exit 1 = still wedged.
set -uo pipefail
cd "$(dirname "$0")/.."
for pf in output/*/driver.*.pid; do
  [[ -f $pf ]] || continue
  pid=$(cat "$pf" 2>/dev/null)
  if [[ -n ${pid:-} ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[clean] stopping campaign driver pid $pid ($pf)"
    kill -TERM "$pid" 2>/dev/null; sleep 2; kill -KILL "$pid" 2>/dev/null
  fi
  rm -f "$pf"
done
# serve_variant arms + dcgm samplers: safe from a SCRIPT (this script's cmdline can't match them)
pkill -KILL -f 'scripts/fr13_bigdenom_swe_serve_variant.sh' 2>/dev/null
pkill -KILL -f 'sample_dcgm_during_task' 2>/dev/null
docker ps -a --format '{{.Names}}' 2>/dev/null | grep '^fr13-bigdenom-' | xargs -r docker rm -f
sleep 3
PYTHONPATH=src .venv/bin/python -c 'from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()' 2>/dev/null
avail=$(free -g | awk 'NR==2{print $7}')
echo "[clean] mem avail=${avail}GB, containers=$(docker ps -q 2>/dev/null | wc -l)"
if [[ ${avail:-0} -lt 90 ]]; then echo "[clean] FAIL: avail<90GB (still wedged)"; exit 1; fi
exit 0
