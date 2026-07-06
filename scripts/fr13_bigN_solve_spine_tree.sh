#!/usr/bin/env bash
# BIG-N SOLVE RATE: spine+cache (chain5) vs tree+cache (cat6root), BOTH cache-ON,
# BOTH at their LOSSLESS config, on the real SWE-Verified astropy-12907. The temp-0.6
# agentic solve is flaky, so N repeats give a solve RATE per arm -> is tree+cache at
# spine+cache parity on the coding-quality axis? Constrained decoding OFF (paused).
#   tree : forked cat6root + recompute cap=1000000 (whole-suffix recurrent = 1/48 == spine, lossless)
#   spine: forked chain5 (5-spine) + NO recompute (snapshot matches native -> lossless w/o recompute)
# Both: APC on, MAMBA_BLOCK_SIZE=1024 fp32 SSM, CUDAGRAPH_MODE=PIECEWISE, B=1, temp 0.6, seed 1313, offloaded codex.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
N=${N:-5}
SUBSET=output/fr13_b1_gold_swe/subset_astropy12907.json
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=output/fr13_bigN_solve/run_$TS
mkdir -p "$ROOT"
echo "$ROOT" > /tmp/claude-1000/-home-mark-shared/46f03809-5059-4e30-936d-1adda7f44337/scratchpad/bigN_root.txt 2>/dev/null || true
export RUNROOT="$ROOT"
# deployment regime (matches the shipgate ARM-ON)
# OFFLOAD_AGENT default 0 (on-GB10): this is a SOLVE-RATE test, not a speed test, so
# the unified-memory contention the offload avoids is irrelevant here. The offload
# (codex on alienware) is currently unreachable from alienware->GB10 anyway. Override
# OFFLOAD_AGENT=1 only for a speed run once the alienware network is restored.
export MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 OFFLOAD_AGENT=${OFFLOAD_AGENT:-${OFFLOAD_CODEX:-0}} DEPLOY_FORCE_TEMP=0.6 SEED=1313
export FR10_METRICS=0 BATCH_INVARIANT=0
export FR13_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 CUDAGRAPH_MODE=PIECEWISE
unset APC_MAX_NUM_BATCHED_TOKENS 2>/dev/null || true
# constrained decoding PAUSED -> LUMO_PROXY_TOOL_STRICT unset (no grammar)
echo "BIG-N solve: spine(chain5) vs tree(cat6root) cache-ON LOSSLESS, astropy-12907, N=$N -> $ROOT" | tee "$ROOT/RESULTS.txt"

run_one() {  # arm kind i
  local arm=$1 kind=$2 i=$3 A="${1}_r${3}"
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  docker ps -a --format '{{.Names}}' | grep -qi fr13 && docker ps -a --format '{{.Names}}' | grep -i fr13 | xargs -r docker rm -f >/dev/null 2>&1 || true
  echo "=== [$A] KIND=$kind recompute=${FR13_APC_HIT_RECURRENT_SUFFIX} cap=${FR13_APC_HIT_SUFFIX_CAP:-none} $(date -u +%H:%M:%SZ) ===" | tee -a "$ROOT/RESULTS.txt"
  bash scripts/fr13_bigdenom_swe_serve_variant.sh "$A" "$kind" "$SUBSET" > "$ROOT/${A}.serve.log" 2>&1 || true
  local SO="$ROOT/$A/swe_orchestrator.log"
  local v rr
  v=$(grep -oE "verdict=(resolved|failed)" "$SO" 2>/dev/null | tail -1)
  rr=$(grep -oE "resolved_rate=[0-9.]+" "$SO" 2>/dev/null | tail -1)
  [ -z "$v" ] && v="verdict=NORESULT(see ${A}.serve.log)"
  echo "$A -> $v $rr" | tee -a "$ROOT/RESULTS.txt"
}

# INTERLEAVED tree (cache-ON, lossless cap=1e6) <-> cacheoff (cache-OFF control) per i, so the
# BINDING comparison rate(tree) vs rate(cacheoff) builds PAIRWISE (first read after 1 pair, not
# after 2N boots). The give-up flake floors both arms; the cache signal is the DIFFERENCE:
# tree >> cacheoff => residual-triggered runaway/give-up, ~= => general flake (cache effectively fine).
for i in $(seq 1 "$N"); do
  export FR13_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 CUDAGRAPH_MODE=PIECEWISE
  export FR13_APC_HIT_RECURRENT_SUFFIX=1 FR13_APC_HIT_SUFFIX_CAP=1000000
  run_one tree cat6root "$i"
  unset FR13_ENABLE_APC FR13_APC_HIT_RECURRENT_SUFFIX FR13_APC_HIT_SUFFIX_CAP \
        MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE CUDAGRAPH_MODE APC_MAX_NUM_BATCHED_TOKENS 2>/dev/null || true
  run_one cacheoff cat6root "$i"
done
# SPINE (secondary): re-arm APC, lossless without recompute. Re-set the APC env the
# cacheoff block unset above.
export FR13_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 CUDAGRAPH_MODE=PIECEWISE
export FR13_APC_HIT_RECURRENT_SUFFIX=0; unset FR13_APC_HIT_SUFFIX_CAP
for i in $(seq 1 "$N"); do run_one spine chain5 "$i"; done

echo "=== SOLVE RATES ===" | tee -a "$ROOT/RESULTS.txt"
for arm in tree spine cacheoff; do
  res=$(grep -cE "^${arm}_r[0-9]+ -> verdict=resolved" "$ROOT/RESULTS.txt" || true)
  tot=$(grep -cE "^${arm}_r[0-9]+ -> verdict=" "$ROOT/RESULTS.txt" || true)
  echo "$arm: ${res}/${tot} resolved" | tee -a "$ROOT/RESULTS.txt"
done
echo "=== DONE -> $ROOT/RESULTS.txt ==="
