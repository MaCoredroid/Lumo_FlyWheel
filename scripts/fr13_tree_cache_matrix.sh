#!/usr/bin/env bash
# FR13 TREE x CACHE matrix on the PROVEN-LOSSLESS config (cache-ON = EXACT_SEED=1 + block 1024).
# Trees: chain5 (E5 spine, no sibling) / cat6root (sibling @depth-1) / cat8 (siblings @depths 1-3,
# the accept-frontier middle ground). Cache: ON (FR13_ENABLE_APC=1 + FR13_APC_EXACT_SEED=1) vs OFF.
# Single task 12907, B=1, temp 0.6, full graph. Full accounting per arm (s_per_fwd, accept/event,
# decode/e2e tok/s, TTFT, prefix-hit%, resolve). Answers: (a) does the proven cache preserve resolve
# (ON vs OFF per tree), (b) does cat8's depth-1-2-3 siblings beat cat6's depth-1-only on accept/event.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_astropy12907.json}
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT=output/fr13_tree_cache_matrix/run_$TS; mkdir -p "$RUNROOT"; export RUNROOT
echo "$RUNROOT" > /home/mark/.claude/jobs/22c39bb9/tmp/matrix_root.txt
echo "=== FR13 TREE x CACHE MATRIX (EXACT_SEED=1 cache-ON) 12907 -> $RUNROOT ==="
export MAX_NUM_SEQS_OVR=1 OFFLOAD_CODEX=1 DEPLOY_FORCE_TEMP=0.6 DOCKER_MEM_CAP=105g \
  GPU_UTIL="${GPU_UTIL:-0.76}" GPU_GUARD_FLOOR_MIB="${GPU_GUARD_FLOOR_MIB:-3000}" FR10_METRICS=0

# tag:KIND:APC:EXSEED  (cache-OFF first per tree = no leak)
ARMS=${ARMS:-"e5_OFF:chain5:0:0 e5_ON:chain5:1:1 cat6_OFF:cat6root:0:0 cat6_ON:cat6root:1:1 cat8_OFF:cat8:0:0 cat8_ON:cat8:1:1"}

run_arm() {
  local tag=$1 kind=$2 apc=$3 exseed=$4
  local ARM="m_${tag}"
  echo "--- [$tag] KIND=$kind APC=$apc EXACT_SEED=$exseed @ $(date -u +%H:%M:%S) ---"
  docker ps -aq --filter "name=fr13-bigdenom-$ARM" | xargs -r docker rm -f >/dev/null 2>&1 || true
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  local EXTRA=""
  [ "$apc" = "1" ] && EXTRA="FR13_APC_EXACT_SEED=$exseed MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32"
  env FR13_ENABLE_APC=$apc FR13_APC_CONFIG_ONLY=0 $EXTRA \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" "$kind" "$SUBSET" > "$RUNROOT/${ARM}.log" 2>&1 </dev/null
  echo "  [$tag] serve_variant rc=$? @ $(date -u +%H:%M:%S)"
  docker ps -aq --filter "name=fr13-bigdenom-$ARM" | xargs -r docker rm -f >/dev/null 2>&1 || true
  if [ -d "$RUNROOT/$ARM/swe_out" ]; then
    .venv/bin/python scripts/fr13_decode_accounting.py "$RUNROOT/$ARM/swe_out" "$tag ($kind, APC=$apc)" 2>/dev/null
    for m in $(find "$RUNROOT/$ARM/swe_out" -name runner_metadata.json 2>/dev/null); do
      .venv/bin/python -c "import json,sys;m=json.load(open(sys.argv[1]));print('  [%s] VERDICT %s'%('$tag',m.get('eval_report',{}).get('verdict','?')))" "$m" 2>/dev/null
    done
  else
    echo "  [$tag] WARN no swe_out — tail:"; tail -10 "$RUNROOT/$ARM.log" | sed 's/^/    /'
  fi
}

for spec in $ARMS; do
  IFS=: read -r tag kind apc exseed <<< "$spec"
  run_arm "$tag" "$kind" "$apc" "$exseed"
done

echo ""
echo "=== TREE x CACHE MATRIX SUMMARY (12907) ==="
for spec in $ARMS; do
  IFS=: read -r tag kind apc exseed <<< "$spec"
  echo "  --- $tag ($kind, cache=$([ "$apc" = 1 ] && echo ON || echo OFF)) ---"
  .venv/bin/python scripts/fr13_decode_accounting.py "$RUNROOT/m_${tag}/swe_out" "$tag" 2>/dev/null | grep -E "s_per_fwd|accept/event|tok/s|hit rate|TTFT|e2e" | sed 's/^/  /'
done
echo "=== MATRIX DONE @ $(date -u +%H:%M:%S) -> $RUNROOT ==="
