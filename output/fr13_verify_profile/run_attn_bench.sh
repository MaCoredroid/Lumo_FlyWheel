#!/usr/bin/env bash
# Verifier V2 microbench: runs INSIDE the cg_combo container after its gate
# probes finish (container still up), before teardown. Fallback: boots nothing.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
until grep -q "CG_COMBO_GATE_DONE" output/fr13_verify_profile/cg_combo/gate_console.log 2>/dev/null; do sleep 60; done
if docker ps --format '{{.Names}}' | grep -q fr13-cg-combo; then
  docker exec fr13-cg-combo python3 /workspace/scripts/fr13_attn_mgeom_bench.py \
    > output/fr13_verify_profile/attn_mgeom_bench.log 2>&1
  echo "ATTN_BENCH_DONE rc=$?"
else
  echo "ATTN_BENCH_SKIPPED container gone"
fi
