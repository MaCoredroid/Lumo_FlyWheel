#!/usr/bin/env bash
# MATCHED CONTROL for the big-N cache confound.
#
# The big-N showed cache-ON(tree) 0/3 vs cache-OFF(cacheoff) 2/2 solved on astropy-12907, BUT the
# arms differ in more than the cache: tree runs CUDAGRAPH_MODE=PIECEWISE (eager GDN) while cacheoff
# falls to the vLLM default (FULL+PIECEWISE). (--mamba-block-size / --mamba-ssm-cache-dtype are
# cache-INHERENT -- vLLM rejects them without prefix-caching -- so fp32-SSM/block1024 are legitimately
# part of "the cache" and need no separate control.) The ONLY controllable confound is the graph mode.
#
# This arm = cache-OFF + CUDAGRAPH_MODE=PIECEWISE (everything else default), the missing cell:
#   tree    = cache-ON  + PIECEWISE
#   cacheoff= cache-OFF + FULL(default)
#   matched = cache-OFF + PIECEWISE   <-- this
# VERDICT LOGIC:
#   matched solves like cacheoff (~N/N) => graph mode is NOT the cause => tree's failures ARE the cache.
#   matched fails like tree    (~0/N) => PIECEWISE (eager-GDN graph mode) is the cause, NOT the cache.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
N=${N:-3}
SUBSET=output/fr13_b1_gold_swe/subset_astropy12907.json
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=output/fr13_matched_control/run_$TS
mkdir -p "$ROOT"
echo "$ROOT" > /tmp/claude-1000/-home-mark-shared/46f03809-5059-4e30-936d-1adda7f44337/scratchpad/matched_root.txt 2>/dev/null || true
export RUNROOT="$ROOT"
export MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 OFFLOAD_AGENT=0 DEPLOY_FORCE_TEMP=0.6 SEED=1313
export FR10_METRICS=0 BATCH_INVARIANT=0
# cache OFF + PIECEWISE (the only matchable confound). fp32-SSM/block are cache-inherent -> not set.
export FR13_ENABLE_APC=0 CUDAGRAPH_MODE=PIECEWISE
unset FR13_APC_HIT_RECURRENT_SUFFIX FR13_APC_HIT_SUFFIX_CAP MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE APC_MAX_NUM_BATCHED_TOKENS 2>/dev/null || true
echo "MATCHED CONTROL: cache-OFF + CUDAGRAPH_MODE=PIECEWISE, astropy-12907, N=$N -> $ROOT" | tee "$ROOT/RESULTS.txt"

run_one() {
  local i=$1 A="matched_r$i"
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  docker ps -a --format '{{.Names}}' | grep -qi fr13 && docker ps -a --format '{{.Names}}' | grep -i fr13 | xargs -r docker rm -f >/dev/null 2>&1 || true
  echo "=== [$A] cache-OFF PIECEWISE $(date -u +%H:%M:%SZ) ===" | tee -a "$ROOT/RESULTS.txt"
  bash scripts/fr13_bigdenom_swe_serve_variant.sh "$A" cat6root "$SUBSET" > "$ROOT/${A}.serve.log" 2>&1 || true
  local SO="$ROOT/$A/swe_orchestrator.log"
  local ru; ru=$(grep -lE "Unterminated string" "$ROOT/$A/swe_out/verified/per_task/"*/{agent,codex}_trace*.jsonl 2>/dev/null | wc -l)
  local v; v=$(grep -oE "verdict=(resolved|failed)" "$SO" 2>/dev/null | tail -1)
  [ -z "$v" ] && v="verdict=NORESULT(see ${A}.serve.log)"
  echo "$A -> $v | runaway=$ru" | tee -a "$ROOT/RESULTS.txt"
}
for i in $(seq 1 "$N"); do run_one "$i"; done
res=$(grep -cE "^matched_r[0-9]+ -> verdict=resolved" "$ROOT/RESULTS.txt" || true)
tot=$(grep -cE "^matched_r[0-9]+ -> verdict=" "$ROOT/RESULTS.txt" || true)
echo "=== MATCHED (cache-OFF+PIECEWISE) resolved: ${res}/${tot} ===" | tee -a "$ROOT/RESULTS.txt"
echo "  compare: tree(cache-ON+PIECEWISE) 0/3 | cacheoff(cache-OFF+FULL) ~N/N" | tee -a "$ROOT/RESULTS.txt"
echo "=== matched control done -> $ROOT/RESULTS.txt ==="
