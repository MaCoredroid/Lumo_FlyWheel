#!/usr/bin/env bash
# FR13 CLEAN 3-WAY (current build, apples-to-apples) -- replaces the config-drifted
# historic cache-OFF 23.88/18.80 with fresh re-runs on the SAME build as cache-ON.
# Arms (4-task b4_four, B=1 temp 0.6, full graph):
#   cat6_OFF   = cat6root tree, cache OFF (FR13_ENABLE_APC=0)         -- the +27% decode baseline, re-run
#   chain5_OFF = chain5 spine,  cache OFF                              -- the E5/spine baseline, re-run
#   cat6_ON    = cat6root tree, cache ON + EXACT_SEED + block 1024     -- the deployed lossless config
# Order: the no-leak cache-OFF arms FIRST (bank the baselines), then the leak-prone
# cat6_ON LAST at GPU_UTIL 0.65 (outrun the residual committer leak for ~90min).
# Reports TOKEN-WEIGHTED decode + e2e + mean TTFT + gen + cache-hit% + verdicts.
# Single-task is too noisy (the FB A/B saw 30%+ swings) -> multi-task TW only.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_b4_four.json}
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT=output/fr13_apc_3way/run_$TS; mkdir -p "$RUNROOT"; export RUNROOT
echo "$RUNROOT" > /home/mark/.claude/jobs/22c39bb9/tmp/threeway_root.txt
echo "=== FR13 CLEAN 3-WAY  b4_four(4-task)  B=1 temp0.6  FULL GRAPH  -> $RUNROOT ==="
export MAX_NUM_SEQS_OVR=1 OFFLOAD_CODEX=1 DEPLOY_FORCE_TEMP=0.6 DOCKER_MEM_CAP=105g FR10_METRICS=0

reduce() {  # $1 = swe_out root -> token-weighted decode/e2e + mean TTFT + gen + cache-hit%
  .venv/bin/python - "$1" <<'PY' 2>/dev/null
import sys,re,glob,os
def parse(f):
    d={}
    try:
        for ln in open(f):
            ln=ln.strip()
            if ln.startswith('#') or not ln: continue
            m=re.match(r'(\S+?)(\{[^}]*\})?\s+([0-9eE.+-]+)$',ln)
            if m: d[m.group(1)]=d.get(m.group(1),0.0)+float(m.group(3))
    except: pass
    return d
root=sys.argv[1]; tg=td=te=tft=tfc=ph=pq=0.0
for pre in glob.glob(os.path.join(root,'**','vllm_metrics_pre.txt'),recursive=True):
    post=pre.replace('_pre.txt','_post.txt')
    if not os.path.exists(post): continue
    a=parse(pre); b=parse(post)
    def df(k): return b.get(k,0)-a.get(k,0)
    tg+=df('vllm:generation_tokens_total'); td+=df('vllm:request_decode_time_seconds_sum')
    te+=df('vllm:e2e_request_latency_seconds_sum')
    tft+=df('vllm:time_to_first_token_seconds_sum'); tfc+=df('vllm:time_to_first_token_seconds_count')
    ph+=df('vllm:prefix_cache_hits_total'); pq+=df('vllm:prefix_cache_queries_total')
dec=f"{tg/td:.2f}" if td>0 else "NA"; e2e=f"{tg/te:.2f}" if te>0 else "NA"
ttft=f"{tft/tfc:.2f}s" if tfc>0 else "NA"; hit=f"{100*ph/pq:.0f}%" if pq>0 else "NA"
print(f"decode={dec} e2e={e2e} ttft={ttft} cache_hit={hit} gen={tg:.0f} dec_s={td:.0f}")
PY
}

run_arm() {
  local ARM=$1 KIND=$2 APC=$3 EXSEED=$4 GUTIL=$5 ETPD=$6
  echo "--- [$ARM] boot KIND=$KIND APC=$APC EXACT_SEED=$EXSEED util=$GUTIL @ $(date -u +%H:%M:%S) ---"
  docker ps -aq --filter "name=fr13-bigdenom-$ARM" | xargs -r docker rm -f >/dev/null 2>&1 || true
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  local EXTRA=""
  if [ "$APC" = "1" ]; then EXTRA="FR13_APC_EXACT_SEED=$EXSEED MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32"; fi
  env FR13_ENABLE_APC=$APC FR13_APC_CONFIG_ONLY=0 FR13_LEAK_PROBE=1 \
      GPU_UTIL="$GUTIL" GPU_GUARD_FLOOR_MIB=3000 $EXTRA \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" "$KIND" "$SUBSET" > "$RUNROOT/${ARM}.log" 2>&1 </dev/null
  echo "  [$ARM] serve_variant rc=$? @ $(date -u +%H:%M:%S)"
  docker ps -aq --filter "name=fr13-bigdenom-$ARM" | xargs -r docker rm -f >/dev/null 2>&1 || true
  if [ -d "$RUNROOT/$ARM/swe_out" ]; then
    echo "  [$ARM] TW: $(reduce "$RUNROOT/$ARM/swe_out")"
    [ "$APC" = "1" ] && echo "  [$ARM] ES_CHAIN_PUBLISH=$(grep -ac ES_CHAIN_PUBLISH "$RUNROOT/$ARM"/logs/fr13_apc_exact_seed_eng.log 2>/dev/null || echo 0)"
    for m in $(find "$RUNROOT/$ARM/swe_out" -name runner_metadata.json 2>/dev/null); do
      .venv/bin/python -c "import json,sys;m=json.load(open(sys.argv[1]));print('  [%s] %s %s'%('$ARM',m.get('instance_id'),m.get('eval_report',{}).get('verdict','?')))" "$m" 2>/dev/null
    done
  else
    echo "  [$ARM] WARN no swe_out — tail:"; tail -12 "$RUNROOT/$ARM.log" | sed 's/^/    /'
  fi
}

# no-leak cache-OFF baselines FIRST (the apples-to-apples replacements for historic 23.88/18.80)
# CAT6ON_UTIL default 0.65 keeps the 4-task cat6_ON arm under the leak ceiling; for a 1-task run
# (SUBSET=...astropy12907.json) the leak is a non-factor, so pass CAT6ON_UTIL=0.76 for a uniform-util
# apples-to-apples comparison across all three arms.
run_arm cat6_OFF   cat6root 0 0 0.76 6
run_arm chain5_OFF chain5   0 0 0.76 5
run_arm cat6_ON    cat6root 1 1 "${CAT6ON_UTIL:-0.65}" 6

echo ""
echo "=== CLEAN 3-WAY SUMMARY (b4_four 4-task TW, current build) ==="
for ARM in cat6_OFF chain5_OFF cat6_ON; do
  echo "  $ARM: $(reduce "$RUNROOT/$ARM/swe_out")"
done
echo "  (historic refs, DIFFERENT config: cat6_OFF~23.88 chain5/E5_OFF~18.80 cache-ON cat6~16.8)"
echo "=== CLEAN 3-WAY DONE @ $(date -u +%H:%M:%S) -> $RUNROOT ==="
