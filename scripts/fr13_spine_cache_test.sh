#!/usr/bin/env bash
# SPINE+CACHE discriminator AND candidate fix.
#
# chain5 (a pure spine, NO off-spine branch) + cache-ON at the SAME config as the failing tree arm
# (FR13_ENABLE_APC=1 + cap=1e6 recompute + PIECEWISE + fp32-SSM + block1024). The ONLY difference
# from tree(cat6root, which went 0/3) is the absence of the `(1,)` off-spine root-sibling branch.
#   spine+cache SOLVES ~3/3  => the off-spine `(1,)` branch IS the carrier (it's the only delta).
#                               This is ALSO a shippable lossless config (spine+cache) and confirms
#                               the fix target = the off-spine-branch state restore on a cache hit.
#   spine+cache FAILS  ~0/3  => off-spine branch EXONERATED => carrier is shared by spine+tree
#                               (full-attn KV restore / graph residual) => different fix.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
N=${N:-3}
SUBSET=output/fr13_b1_gold_swe/subset_astropy12907.json
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=output/fr13_spine_cache/run_$TS
mkdir -p "$ROOT"
echo "$ROOT" > /tmp/claude-1000/-home-mark-shared/46f03809-5059-4e30-936d-1adda7f44337/scratchpad/spine_root.txt 2>/dev/null || true
export RUNROOT="$ROOT"
export MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 OFFLOAD_AGENT=0 DEPLOY_FORCE_TEMP=0.6 SEED=1313
export FR10_METRICS=0 BATCH_INVARIANT=0
# IDENTICAL to the failing tree arm except the shape (chain5 not cat6root):
export FR13_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 CUDAGRAPH_MODE=PIECEWISE
export FR13_APC_HIT_RECURRENT_SUFFIX=1 FR13_APC_HIT_SUFFIX_CAP=1000000
echo "SPINE+CACHE: chain5 cache-ON cap=1e6 PIECEWISE, astropy-12907, N=$N -> $ROOT" | tee "$ROOT/RESULTS.txt"

run_one() {
  local i=$1 A="spine_r$i"
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  docker ps -a --format '{{.Names}}' | grep -qi fr13 && docker ps -a --format '{{.Names}}' | grep -i fr13 | xargs -r docker rm -f >/dev/null 2>&1 || true
  echo "=== [$A] chain5 cache-ON cap=1e6 $(date -u +%H:%M:%SZ) ===" | tee -a "$ROOT/RESULTS.txt"
  bash scripts/fr13_bigdenom_swe_serve_variant.sh "$A" chain5 "$SUBSET" > "$ROOT/${A}.serve.log" 2>&1 || true
  local SO="$ROOT/$A/swe_orchestrator.log"
  local ru; ru=$(grep -lE "Unterminated string" "$ROOT/$A/swe_out/verified/per_task/"*/{agent,codex}_trace*.jsonl 2>/dev/null | wc -l)
  local v; v=$(grep -oE "verdict=(resolved|failed)" "$SO" 2>/dev/null | tail -1)
  [ -z "$v" ] && v="verdict=NORESULT(see ${A}.serve.log)"
  echo "$A -> $v | runaway=$ru" | tee -a "$ROOT/RESULTS.txt"
}
for i in $(seq 1 "$N"); do run_one "$i"; done
res=$(grep -cE "^spine_r[0-9]+ -> verdict=resolved" "$ROOT/RESULTS.txt" || true)
tot=$(grep -cE "^spine_r[0-9]+ -> verdict=" "$ROOT/RESULTS.txt" || true)
echo "=== SPINE+CACHE (chain5) resolved: ${res}/${tot}  [vs tree(cat6root) 0/3, cacheoff 3/3] ===" | tee -a "$ROOT/RESULTS.txt"
echo "=== spine+cache done -> $ROOT/RESULTS.txt ==="
