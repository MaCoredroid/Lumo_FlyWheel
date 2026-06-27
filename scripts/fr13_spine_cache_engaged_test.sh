#!/usr/bin/env bash
# SPINE + cache, RE-MEASURED with engagement assert (user 2026-06-27).
#
# chain5 (pure spine, NO off-spine branches) + cache-ON at the SAME deployed APC config as the
# eng/tree arms (SNAP_FIX, cap=1e6, conv, zeroaccept) PLUS FR13_APC_REQUIRE_* so variant.sh reads
# pid-N's live os.environ marker and HARD-FAILS if vacuous. This is the GATING measurement for
# "carrier is the COMMON cache path": if spine+cache also fails ENGAGED, the bug is upstream of all
# tree-specific machinery (reproduces with zero branches) -> focus the commons (GDN align boundary).
#   fails ~N/N (engaged) => carrier is common cache -> focus commons.
#   solves              => spine+cache is clean -> the carrier IS tree+cache-specific (reopens C).
# Prior (pre-engagement-marker, noisy): 2 failed + 1 NORESULT, 0 clean. This re-measures it cleanly.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
N=${N:-2}
SUBSET=output/fr13_b1_gold_swe/subset_astropy12907.json
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=output/fr13_spine_cache_engaged/run_$TS
mkdir -p "$ROOT"
echo "$ROOT" > /tmp/claude-1000/-home-mark-shared/46f03809-5059-4e30-936d-1adda7f44337/scratchpad/spine_cache_engaged_root.txt 2>/dev/null || true
export RUNROOT="$ROOT"
export MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 OFFLOAD_CODEX=0 DEPLOY_FORCE_TEMP=0.6 SEED=1313
export FR10_METRICS=0 BATCH_INVARIANT=0
# IDENTICAL deployed APC config:
export FR13_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 CUDAGRAPH_MODE=PIECEWISE
export FR13_APC_HIT_RECURRENT_SUFFIX=1 FR13_APC_HIT_SUFFIX_CAP=1000000
export FR13_APC_SNAP_FIX_ZEROACCEPT=1
# ENGAGEMENT ASSERT: fail-loud if the worker marker is vacuous.
export FR13_APC_REQUIRE_SNAP_FIX=1 FR13_APC_REQUIRE_HIT_SUFFIX_CAP=1000000
echo "SPINE+CACHE ENGAGED RE-MEASURE: chain5 cache-ON cap=1e6 + REQUIRE marker, astropy-12907, N=$N -> $ROOT" | tee "$ROOT/RESULTS.txt"

run_one() {
  local i=$1 A="spc_r$i"
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  docker ps -a --format '{{.Names}}' | grep -qi fr13 && docker ps -a --format '{{.Names}}' | grep -i fr13 | xargs -r docker rm -f >/dev/null 2>&1 || true
  echo "=== [$A] chain5 cache-ON ENGAGED-ASSERT $(date -u +%H:%M:%SZ) ===" | tee -a "$ROOT/RESULTS.txt"
  bash scripts/fr13_bigdenom_swe_serve_variant.sh "$A" chain5 "$SUBSET" > "$ROOT/${A}.serve.log" 2>&1 || true
  local SO="$ROOT/$A/swe_orchestrator.log"
  local mk; mk=$(cat "$ROOT/$A/apc_bridge_marker.txt" 2>/dev/null)
  local eng; eng=$(grep -qE "FAIL: APC (worker|engagement|bridge)" "$ROOT/${A}.serve.log" 2>/dev/null && echo "VACUOUS(exit4)" || echo "ENGAGED")
  local ru; ru=$(grep -lE "Unterminated string" "$ROOT/$A/swe_out/verified/per_task/"*/codex_trace*.jsonl 2>/dev/null | wc -l)
  local v; v=$(grep -oE "verdict=(resolved|failed)" "$SO" 2>/dev/null | tail -1)
  [ -z "$v" ] && v="verdict=NORESULT(see ${A}.serve.log)"
  echo "$A -> $v | engagement=$eng | marker=[$mk] | runaway=$ru" | tee -a "$ROOT/RESULTS.txt"
}
for i in $(seq 1 "$N"); do run_one "$i"; done
res=$(grep -cE "^spc_r[0-9]+ -> verdict=resolved" "$ROOT/RESULTS.txt" || true)
tot=$(grep -cE "^spc_r[0-9]+ -> verdict=" "$ROOT/RESULTS.txt" || true)
echo "=== SPINE+CACHE ENGAGED (chain5) resolved: ${res}/${tot}  [vs tree+cache eng 0/2, cacheoff 3/3] ===" | tee -a "$ROOT/RESULTS.txt"
echo "=== spine+cache engaged re-measure done -> $ROOT/RESULTS.txt ==="
