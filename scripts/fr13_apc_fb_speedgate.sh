#!/usr/bin/env bash
# FR13_APC_FIXED_BUFFER decode A/B on the LIVE spec-decode SWE path -- where the
# per-token .clone() append tax actually runs (and the committer drain can fire),
# UNLIKE the eager state-diff replay whose committer drain never publishes
# (ES_CHAIN_PUBLISH=0, prefill-capture-driven). cat6root, 1-task (astropy 12907,
# a reliable resolver), B=1 temp 0.6, full graph + EXACT_SEED. Two arms SERIAL:
#   FB=0 = shipped Python-list path   FB=1 = pre-alloc copy_ buffer
# Reports: TOKEN-WEIGHTED decode TPS delta (the .clone() tax removed), e2e TPS,
# resolve verdict (lossless proxy at temp 0.6), ES_CHAIN_PUBLISH (drain fired?).
# The unit test (fr13_apc_fixedbuf_invariant_test.py) already proved the drained
# chunk bit-exact; this is the production speed payoff + no-crash/lossless check.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_astropy12907.json}
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT=output/fr13_apc_fb_speedgate/run_$TS; mkdir -p "$RUNROOT"; export RUNROOT
echo "$RUNROOT" > /home/mark/.claude/jobs/22c39bb9/tmp/fb_speedgate_root.txt
echo "=== FR13 FB DECODE A/B  cat6root  12907 1-task  B=1 temp0.6  FULL GRAPH+EXACT_SEED  FB=0 vs FB=1 -> $RUNROOT ==="
export MAX_NUM_SEQS_OVR=1 OFFLOAD_CODEX=1 DEPLOY_FORCE_TEMP=0.6 DOCKER_MEM_CAP=105g \
  GPU_UTIL="${GPU_UTIL:-0.76}" GPU_GUARD_FLOOR_MIB="${GPU_GUARD_FLOOR_MIB:-3000}" FR10_METRICS=0

manual_decode() {  # $1 = swe_out root -> token-weighted decode TPS (sum gen / sum decode)
  .venv/bin/python - "$1" <<'PY' 2>/dev/null
import sys,re,glob,os
def p(f):
    d={}
    try:
        for ln in open(f):
            ln=ln.strip()
            if ln.startswith('#') or not ln: continue
            m=re.match(r'(\S+?)(\{[^}]*\})?\s+([0-9eE.+-]+)$',ln)
            if m: d[m.group(1)]=d.get(m.group(1),0.0)+float(m.group(3))
    except: pass
    return d
root=sys.argv[1]; tg=td=te=0.0
for pre in glob.glob(os.path.join(root,'**','vllm_metrics_pre.txt'),recursive=True):
    post=pre.replace('_pre.txt','_post.txt')
    if not os.path.exists(post): continue
    a=p(pre); b=p(post)
    g=b.get('vllm:generation_tokens_total',0)-a.get('vllm:generation_tokens_total',0)
    dc=b.get('vllm:request_decode_time_seconds_sum',0)-a.get('vllm:request_decode_time_seconds_sum',0)
    ec=b.get('vllm:e2e_request_latency_seconds_sum',0)-a.get('vllm:e2e_request_latency_seconds_sum',0)
    tg+=g; td+=dc; te+=ec
print(f"decode={tg/td:.2f} e2e={tg/te:.2f} gen={tg:.0f} dec_s={td:.0f}" if td>0 and te>0 else "no-decode")
PY
}

run_arm() {
  local FB=$1
  local ARM="sg_fb${FB}"
  echo "--- [$ARM] boot FB=$FB @ $(date -u +%H:%M:%S) ---"
  docker ps -aq --filter "name=fr13-bigdenom-$ARM" | xargs -r docker rm -f >/dev/null 2>&1 || true
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  FR13_ENABLE_APC=1 FR13_APC_CONFIG_ONLY=0 FR13_APC_EXACT_SEED=1 FR13_APC_FIXED_BUFFER=$FB FR13_LEAK_PROBE=1 \
    MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" cat6root "$SUBSET" > "$RUNROOT/${ARM}.log" 2>&1 </dev/null
  echo "  [$ARM] serve_variant rc=$? @ $(date -u +%H:%M:%S)"
  docker ps -aq --filter "name=fr13-bigdenom-$ARM" | xargs -r docker rm -f >/dev/null 2>&1 || true
  if [ -d "$RUNROOT/$ARM/swe_out" ]; then
    echo "  [$ARM] MANUAL token-weighted: $(manual_decode "$RUNROOT/$ARM/swe_out")"
    local pub; pub=$(grep -ac "ES_CHAIN_PUBLISH" "$RUNROOT/$ARM"/logs/fr13_apc_exact_seed_eng.log 2>/dev/null || echo 0)
    echo "  [$ARM] ES_CHAIN_PUBLISH=$pub (committer drain fired live iff >0)"
    for m in $(find "$RUNROOT/$ARM/swe_out" -name runner_metadata.json 2>/dev/null); do
      .venv/bin/python -c "import json,sys;m=json.load(open(sys.argv[1]));print('  [%s] verdict %s %s'%('$ARM',m.get('instance_id'),m.get('eval_report',{}).get('verdict','?')))" "$m" 2>/dev/null
    done
  else
    echo "  [$ARM] WARN no swe_out — tail:"; tail -14 "$RUNROOT/$ARM.log" | sed 's/^/    /'
  fi
}

run_arm 0
run_arm 1

echo ""
echo "=== FB DECODE A/B SUMMARY (cat6root 12907, token-weighted) ==="
for FB in 0 1; do
  echo "  FB=$FB: $(manual_decode "$RUNROOT/sg_fb${FB}/swe_out")"
done
echo "  (FB=0 shipped list path; FB=1 pre-alloc buffer; decode delta = .clone() tax removed; banked FB=0 ref 17.9)"
echo "=== FB DECODE A/B DONE @ $(date -u +%H:%M:%S) -> $RUNROOT ==="
