#!/usr/bin/env bash
# FR13 APC E4 — VERBATIM (tree SSM corrected write-through) validation on 12907
# =================================================================================
# The spine control (output/fr13_bigdenom_swe/nativeapc_spine: native MTP-5 + APC,
# same disciplines, cuda-graph) SOLVES 12907 (504B resolved). The cat6root TREE +
# APC EMPTIES 12907 (stock-align cuda-graph AND eager, plain-align) while cache-OFF
# solves it. => TREE-SPECIFIC committer defect, NOT cuda-graph, NOT task-noise:
# the tree committer writes the accepted-leaf recurrent state into the node-bank,
# but stock align restores the BLOCK-ALIGNED row on a cache-hit -> stale -> poison.
#
# FR13_APC_VERBATIM is the fix: at commit, copy the committed accepted-leaf conv+SSM
# INTO the exact block-aligned row align reads (block_ids[aligned_new_computed//bs-1],
# post-commit seqlen). This is the SGLang "snapshot-the-committed-state, restore
# verbatim, never reconstruct" discipline ported to vLLM's single-checkpoint align.
#
# EAGER REQUIRED: the publisher's GPU->CPU leaf-row sync is a no-op under cuda-graph
# capture (decouple agent a4e1ce949228ba819) -> run eager for the proof-of-concept.
# SSM_DIAG=1: VERIFY the publisher+write-through FIRE (risk#1: empty leaf-map = silent
# no-op). Look for [FR13_APC_VERBATIM]/[FR13_WT_DIAG] did=True / non-zero fired.
#
#   * VERBATIM SOLVES 12907 (resolved ~504B, DIAG shows fired) -> SSM fix WORKS (eager
#       proof); next = make publisher cuda-graph-safe so the tree ships at spine speed.
#   * VERBATIM EMPTIES + DIAG fired -> corrected-WT insufficient -> side-buffer snapshot.
#   * VERBATIM EMPTIES + DIAG NOT fired -> publisher still gated off / leaf-map empty -> debug wiring.
# =================================================================================
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
REPO=/home/mark/shared/lumoFlyWheel

SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_astropy12907.json}
KIND=cat6root
TS=$(date -u +%Y%m%dT%H%M%S%3NZ)
RUNROOT=output/fr13_apc_e4_verbatim/$TS
export RUNROOT
GATEDIR=$RUNROOT
mkdir -p "$GATEDIR"
ARM_ON=${ARM_ON:-cat6root_verbatim_b1}
echo "[run-key] artifacts -> $RUNROOT (VERBATIM tree SSM fix, 12907, EAGER, DIAG on)"

export MAX_NUM_SEQS_OVR=${MAX_NUM_SEQS_OVR:-1}
export SWE_CONCURRENCY=${SWE_CONCURRENCY:-1}
export OFFLOAD_CODEX=${OFFLOAD_CODEX:-1}
export DEPLOY_FORCE_TEMP=${DEPLOY_FORCE_TEMP:-0.6}
export SEED=${SEED:-1313}

export FR10_METRICS=0
export BATCH_INVARIANT=0
# NOTE: this is a CORRECTNESS validation, not a clean-speed read -> SSM_DIAG is REQUIRED
# (eager, so DIAG is graph-safe). Assert only the SPEED/argmax diag flags are off.
_DIAG_FLAGS=(FR13_SFWD_GPU_TIMER FR13_COMMIT_ARGMAX_GATE FR13_FORK_MARGIN_DUMP \
  FR13_CHASE_DIAG FR13_FIX1_SELFCHECK FR13_REPLAY_BOUNDARY_LOG FR13_GDN_SUBOP_MAB \
  FR13_FA2_MAB FR13_REPLAY_DURABLE_AB FR13_TCF_SELFCHECK)
for f in "${_DIAG_FLAGS[@]}"; do
  v="${!f:-0}"
  if [[ "$v" != "0" && -n "$v" ]]; then echo "FAIL: speed-diag flag $f=$v set"; exit 2; fi
done

echo "=== FR13 APC E4 VERBATIM (cat6root) ts=$TS subset=$SUBSET ==="
git rev-parse HEAD 2>/dev/null | tee "$GATEDIR/git_head_$TS.txt" || true

# --- VERBATIM tree SSM fix: clean disciplines + CONV_FIX=1 + VERBATIM=1 + EAGER + DIAG ---
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
unset APC_MAX_NUM_BATCHED_TOKENS 2>/dev/null || true
export FR13_APC_CONV_FIX=1            # stock conv (preprocess-scoped override, invertible snapshot)
export FR13_APC_CONV_SNAPSHOT=0
export FR13_APC_SSM_SNAPSHOT=0        # VERBATIM self-publishes the leaf; no wrong-row override
export FR13_APC_SSM_WRITE_THROUGH=0
export FR13_APC_HIT_RECURRENT_SUFFIX=0
export FR13_APC_DROP_FINAL_BLOCK=0
export FR13_APC_VERBATIM=1            # THE FIX: corrected write-through into the block-aligned row
export FR13_APC_SSM_DIAG=1            # verify publisher + write-through fire (risk#1)
export ENFORCE_EAGER=1               # publisher GPU->CPU sync is eager-only
echo "[verbatim] ENABLE_APC=1 block=1024 fp32 CONV_FIX=1 VERBATIM=1 SSM_DIAG=1 ENFORCE_EAGER=1;"
echo "          CONV_SNAPSHOT=SSM_SNAPSHOT=WRITE_THROUGH=HIT_SUFFIX=DROP_FINAL=0"

bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM_ON" "$KIND" "$SUBSET"
ON_RC=$?
echo "ARM_ON rc=$ON_RC (armdir=$RUNROOT/$ARM_ON)"

PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY

echo
echo "=== VERBATIM 12907 VERDICT ==="
.venv/bin/python - "$RUNROOT/$ARM_ON" <<'PY'
import json, glob, sys, os
arm=sys.argv[1]
def trig(s):
    s="".join(s.split())
    if len(s)<6: return 0.0
    t=[s[i:i+3] for i in range(len(s)-2)]
    from collections import Counter
    return max(Counter(t).values())/len(t)
for pt in sorted(glob.glob(arm+"/swe_out/*/per_task/*")):
    cjk=0; rt=0.0
    for tf in glob.glob(pt+"/codex_trace*.jsonl"):
        items=[json.loads(l).get('item',json.loads(l)) for l in open(tf)]
        msgs=[it.get('text','') or '' for it in items if it.get('type')=='agent_message']
        cjk+=sum(1 for m in msgs for ch in m if '一'<=ch<='鿿'); rt=max([rt]+[trig(m) for m in msgs])
    pf=pt+"/patch.diff"; sz=os.path.getsize(pf) if os.path.exists(pf) else 0
    er=pt+"/eval/eval_report.json"; v="?"
    if os.path.exists(er):
        try: v=json.load(open(er)).get("verdict","?")
        except Exception: pass
    g="GARBLE" if (cjk>5 or rt>0.18) else "coherent"
    print(f"  {os.path.basename(pt)}: patch={sz}B eval={v} cjk={cjk} -> {g}")
PY
echo "--- DIAG: did the VERBATIM write-through FIRE? (look for did=True / fired>0) ---"
grep -hoE '\[FR13_(APC_VERBATIM|WT_DIAG|COMMIT_SITE_WT)\][^\n]*' "$RUNROOT/$ARM_ON"/serve*.log "$RUNROOT/$ARM_ON"/docker*.log 2>/dev/null | grep -E 'did=True|fired=[1-9]' | head -5 || echo "  (no fired DIAG line found — check publisher gating / leaf-map population)"
echo
echo "  SOLVES + DIAG fired -> SSM fix WORKS (eager) -> make cuda-graph-safe for ship"
echo "  EMPTIES + DIAG fired -> corrected-WT insufficient -> side-buffer snapshot"
echo "  EMPTIES + DIAG NOT fired -> wiring/leaf-map -> debug"
echo "VERBATIM_DONE on_rc=$ON_RC"
exit $ON_RC
