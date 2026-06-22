#!/usr/bin/env bash
# FR13 APC — REAL-task CACHE_AB shadow lossless gate on agentic 12907
# =================================================================================
# THE GATE (user 2026-06-22, "shadow + cache-off, check the cache-hit is lossless";
# standing rule: NEVER flee to a synthetic proxy -- use the REAL task + shadow):
#   Run the REAL agentic 12907 with APC-ON + the both-banks node-bank-leaf fix
#   (FR13_APC_SSM_LEAF_SRC) + the CACHE_AB shadow. On every real cache-HIT re-prefill,
#   CACHE_AB records h_on (the GDN boundary state APC RESTORES) and h_off (the SAME
#   logical-position state a FRESH cache-OFF chunk-scan over [0,boundary_pos) computes
#   -- the real fresh state, not a proxy). The reducer matches them by boundary_pos.
#   LOSSLESS iff per-element |h_on - h_off| ~ 0 within bf16 ULP (0.0078125) across all
#   GDN layers + boundaries.
# cuda-graph deploy regime (CACHE_AB's prefill path is eager anyway -> it fires on the
# real cache-hit re-prefills; decode is graphed).
#
# PER THE BANKED RULE: this drill VERIFIES THE PATH WAS EXERCISED before any verdict
# -- cache-hit-rate>0 AND CACHE_AB ON-arm records>0 -- else the gate is VACUOUS.
#
# VERDICT:
#   path exercised + all boundaries |h_on-h_off|<=ULP -> APC restore LOSSLESS (real).
#   path exercised + some boundary > ULP            -> the fix's restore still diverges
#       at that (layer,boundary): localize -> conv source / SSM leaf off-by-one.
#   NOT exercised (0 hits / 0 ON records)           -> VACUOUS, re-run (do NOT trust).
# =================================================================================
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
REPO=/home/mark/shared/lumoFlyWheel

SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_astropy12907.json}
KIND=cat6root
TS=$(date -u +%Y%m%dT%H%M%S%3NZ)
RUNROOT=output/fr13_apc_cacheab/$TS
export RUNROOT
mkdir -p "$RUNROOT"
ARM=${ARM:-cat6root_cacheab_b1}
echo "[run-key] artifacts -> $RUNROOT (REAL-task CACHE_AB shadow lossless gate, 12907)"

export MAX_NUM_SEQS_OVR=${MAX_NUM_SEQS_OVR:-1}
export SWE_CONCURRENCY=${SWE_CONCURRENCY:-1}
export OFFLOAD_CODEX=${OFFLOAD_CODEX:-1}
export DEPLOY_FORCE_TEMP=${DEPLOY_FORCE_TEMP:-0.6}
export SEED=${SEED:-1313}
export FR10_METRICS=0
export BATCH_INVARIANT=0

# clean APC disciplines + the fix + the CACHE_AB shadow; cuda-graph deploy
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
unset APC_MAX_NUM_BATCHED_TOKENS 2>/dev/null || true
export FR13_APC_CONV_FIX=1
export FR13_APC_CONV_SNAPSHOT=0
export FR13_APC_SSM_SNAPSHOT=0
export FR13_APC_VERBATIM=0
export FR13_APC_SSM_LEAF_SRC=1          # THE FIX (both banks -> node-bank leaf)
export FR13_APC_CACHE_AB=1              # THE SHADOW (h_on vs h_off per cache-hit boundary)
export FR13_APC_CACHE_AB_LOG=/logs/fr13_apc_cache_ab.jsonl
export FR13_APC_CACHE_AB_BLOCK=1024
export FR13_APC_SSM_DIAG=1              # leaf-src fired + publisher engagement needles
export ENFORCE_EAGER=${CACHEAB_EAGER:-0}   # cuda-graph deploy by default (CACHE_AB prefill is eager anyway)
echo "[cacheab] ENABLE_APC=1 block=1024 fp32 CONV_FIX=1 SSM_LEAF_SRC=1 CACHE_AB=1 EAGER=$ENFORCE_EAGER"

git rev-parse HEAD 2>/dev/null | tee "$RUNROOT/git_head_$TS.txt" || true
bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" "$KIND" "$SUBSET"
RC=$?
echo "ARM rc=$RC (armdir=$RUNROOT/$ARM)"

PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY

echo
echo "=== PATH-EXERCISED CHECK (banked rule: no verdict on a vacuous run) ==="
ABJSONL=$(find "$RUNROOT/$ARM" -name 'fr13_apc_cache_ab.jsonl' 2>/dev/null | head -1)
DLOG=$(find "$RUNROOT/$ARM" -name 'docker_full.log' 2>/dev/null | head -1)
ON_RECS=0; HITRATE="?"
[ -n "$ABJSONL" ] && ON_RECS=$(grep -c '"phase": "on"\|"phase":"on"' "$ABJSONL" 2>/dev/null || echo 0)
[ -n "$DLOG" ] && HITRATE=$(grep -oE 'Prefix cache hit rate: [0-9.]+%' "$DLOG" 2>/dev/null | grep -vE ': 0\.0%' | tail -1)
LEAF_FIRED=0; [ -n "$DLOG" ] && LEAF_FIRED=$(grep -cE 'FR13_APC_SSM_LEAF_SRC\] fired' "$DLOG" 2>/dev/null || echo 0)
echo "  CACHE_AB jsonl=$ABJSONL"
echo "  ON-arm (h_on, real cache-hit) records=$ON_RECS  leaf_src_fired=$LEAF_FIRED  cache_hit_rate=$HITRATE"
if [ "${ON_RECS:-0}" -lt 1 ]; then
  echo "  *** VACUOUS: 0 cache-hit h_on records -> NO VERDICT. (need a real cache-hit re-prefill.) ***"
  echo "CACHEAB_DONE rc=$RC VACUOUS"
  exit 0
fi

echo "=== CACHE_AB REDUCE: per-boundary |h_on - h_off| lossless verdict ==="
.venv/bin/python scripts/fr13_apc_cache_ab_reduce.py "$ABJSONL" --out "$RUNROOT/$ARM/cacheab_verdict.json" 2>&1 | tail -25

echo
echo "=== 12907 coding-quality (cache-ON solve under the fix) ==="
.venv/bin/python - "$RUNROOT/$ARM" <<'PY'
import json, glob, sys, os
arm=sys.argv[1]
for pt in sorted(glob.glob(arm+"/swe_out/*/per_task/*")):
    cjk=0
    for tf in glob.glob(pt+"/codex_trace*.jsonl"):
        for l in open(tf):
            try: it=json.loads(l).get('item',json.loads(l))
            except: continue
            if it.get('type')=='agent_message': cjk+=sum(1 for c in (it.get('text','')or'') if '一'<=c<='鿿')
    pf=pt+"/patch.diff"; sz=os.path.getsize(pf) if os.path.exists(pf) else 0
    er=pt+"/eval/eval_report.json"; v="none"
    if os.path.exists(er):
        try: v=json.load(open(er)).get("verdict","?")
        except Exception: pass
    print(f"  {os.path.basename(pt)}: patch={sz}B eval={v} cjk={cjk}")
PY
echo "CACHEAB_DONE rc=$RC"
exit $RC
