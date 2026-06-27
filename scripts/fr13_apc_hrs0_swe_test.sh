#!/usr/bin/env bash
# DECISIVE: cat6root cache-ON SWE solve-rate with HIT_RECURRENT_SUFFIX=0 (the subtractive fix).
# Greedy divergence (fr13_apc_greedy_divergence.sh) showed cache-ON+HRS=1 (the bake) collapses into a
# 5-token degenerate LOOP (uniq=5/122) while HRS=0 is near-normal (uniq=42, like cache-OFF 50-62) -->
# the sequential recurrent roll (HIT_RECURRENT_SUFFIX) is a NET CARRIER. If cache-ON+HRS=0 now SOLVES
# the SWE task (cache-OFF=3/3, cache-ON+HRS=1=0/2), the fix is to UN-BAKE HIT_RECURRENT_SUFFIX.
# NON-DESTRUCTIVE: pre-sets FR13_APC_HIT_RECURRENT_SUFFIX=0 via env; bake on disk untouched.
# Attempt-1 metric (SWE_EMPTY_PATCH_RETRIES=0) + engagement assert (REQUIRE marker).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
N=${N:-3}
SUBSET=output/fr13_b1_gold_swe/subset_astropy12907.json
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=output/fr13_apc_hrs0_swe/run_$TS
mkdir -p "$ROOT"
echo "$ROOT" > /tmp/claude-1000/-home-mark-shared/46f03809-5059-4e30-936d-1adda7f44337/scratchpad/hrs0_swe_root.txt 2>/dev/null || true
export RUNROOT="$ROOT"
export MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 OFFLOAD_CODEX=0 DEPLOY_FORCE_TEMP=0.6 SEED=1313
export FR10_METRICS=0 BATCH_INVARIANT=0 SWE_EMPTY_PATCH_RETRIES=0
# deployed APC config, but the SUBTRACTIVE change: HIT_RECURRENT_SUFFIX=0
export FR13_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 CUDAGRAPH_MODE=PIECEWISE
export FR13_APC_HIT_RECURRENT_SUFFIX=0
export FR13_APC_REQUIRE_SNAP_FIX=1
echo "HRS=0 SWE TEST: cat6root cache-ON HIT_RECURRENT_SUFFIX=0 (subtractive), attempt-1, astropy-12907, N=$N -> $ROOT" | tee "$ROOT/RESULTS.txt"

run_one() {
  local i=$1 A="hrs0_r$i"
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  docker ps -a --format '{{.Names}}' | grep -qi fr13 && docker ps -a --format '{{.Names}}' | grep -i fr13 | xargs -r docker rm -f >/dev/null 2>&1 || true
  echo "=== [$A] cat6root cache-ON HRS=0 attempt-1 $(date -u +%H:%M:%SZ) ===" | tee -a "$ROOT/RESULTS.txt"
  bash scripts/fr13_bigdenom_swe_serve_variant.sh "$A" cat6root "$SUBSET" > "$ROOT/${A}.serve.log" 2>&1 || true
  local SO="$ROOT/$A/swe_orchestrator.log"
  local mk; mk=$(cat "$ROOT/$A/apc_bridge_marker.txt" 2>/dev/null)
  local eng; eng=$(grep -qE "FAIL: APC (worker|engagement|bridge)" "$ROOT/${A}.serve.log" 2>/dev/null && echo "VACUOUS" || echo "ENGAGED")
  local hrs; hrs=$(echo "$mk" | grep -oE "HIT_RECURRENT_SUFFIX=[01]")
  local v; v=$(grep -oE "verdict=(resolved|failed)" "$SO" 2>/dev/null | tail -1)
  [ -z "$v" ] && v="verdict=NORESULT(see ${A}.serve.log)"
  echo "$A -> $v | engagement=$eng | $hrs | marker=[$mk]" | tee -a "$ROOT/RESULTS.txt"
}
for i in $(seq 1 "$N"); do run_one "$i"; done
res=$(grep -cE "^hrs0_r[0-9]+ -> verdict=resolved" "$ROOT/RESULTS.txt" || true)
tot=$(grep -cE "^hrs0_r[0-9]+ -> verdict=" "$ROOT/RESULTS.txt" || true)
echo "=== HRS=0 SWE (cat6root cache-ON) resolved: ${res}/${tot}  [vs HRS=1 baked: 0/2, cache-OFF: 3/3] ===" | tee -a "$ROOT/RESULTS.txt"
echo "    -> if HRS=0 solves, UN-BAKE HIT_RECURRENT_SUFFIX (subtractive fix). done -> $ROOT/RESULTS.txt"
