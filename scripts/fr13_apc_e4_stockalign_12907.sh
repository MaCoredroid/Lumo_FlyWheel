#!/usr/bin/env bash
# FR13 APC E4-pre — TRUE STOCK-ALIGN validation on agentic 12907
# =================================================================================
# WHY: E1/the plain-align gate set FR13_APC_CONV_FIX=0 believing "0 = off". The
# decoupling audit (agent a4e1ce949228ba819) showed CONV_FIX=0 does the OPPOSITE:
# it RE-ENABLES the get_conv_copy_spec tree-node conv override (conv garble carrier
# "b") and DISABLES the stock raw-remainder return. So "plain-align" was NOT stock
# on the conv axis -> the empty-patch verdict is CONFOUNDED.
#
# This drill tests the REAL ship candidate: STOCK vLLM align under APC + the clean
# disciplines, with CONV_FIX=1 (= stock get_conv_copy_spec, the launcher default)
# and EVERY other tree fix-fn OFF, under the normal CUDA-GRAPH deploy regime.
#
#   * STOCK-ALIGN SOLVES 12907 (504B-ish, resolved) -> the empties were the CONV_FIX=0
#       self-inflicted error; stock-align APC is the ship candidate (cuda-graph, fast,
#       no VERBATIM). Extend to 4-task + per-token rescore.
#   * STOCK-ALIGN STILL EMPTIES 12907 -> the SSM/conv stale-block-pool-row defect is
#       real -> test FR13_APC_VERBATIM (eager) next.
#
# Control: cache-OFF 12907 solves (504B, banked + this-boot ARM_OFF). Garble read at
# codex_trace agent_messages, NOT engine logs. Single arm, single boot, cuda-graph.
# =================================================================================
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
REPO=/home/mark/shared/lumoFlyWheel

SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_astropy12907.json}
KIND=cat6root
TS=$(date -u +%Y%m%dT%H%M%S%3NZ)
RUNROOT=output/fr13_apc_e4_stockalign/$TS
export RUNROOT
GATEDIR=$RUNROOT
mkdir -p "$GATEDIR"
ARM_ON=${ARM_ON:-cat6root_stockalign_b1}
echo "[run-key] artifacts -> $RUNROOT (TRUE stock-align cache-ON, 12907, cuda-graph)"

export MAX_NUM_SEQS_OVR=${MAX_NUM_SEQS_OVR:-1}
export SWE_CONCURRENCY=${SWE_CONCURRENCY:-1}
export OFFLOAD_CODEX=${OFFLOAD_CODEX:-1}
export DEPLOY_FORCE_TEMP=${DEPLOY_FORCE_TEMP:-0.6}
export SEED=${SEED:-1313}

export FR10_METRICS=0
export BATCH_INVARIANT=0
_DIAG_FLAGS=(FR13_SFWD_GPU_TIMER FR13_COMMIT_ARGMAX_GATE FR13_FORK_MARGIN_DUMP \
  FR13_CHASE_DIAG FR13_FIX1_SELFCHECK FR13_APC_STATE_PROBE FR13_APC_SSM_DIAG \
  FR13_REPLAY_BOUNDARY_LOG FR13_GDN_SUBOP_MAB FR13_FA2_MAB FR13_REPLAY_DURABLE_AB \
  FR13_TCF_SELFCHECK)
for f in "${_DIAG_FLAGS[@]}"; do
  v="${!f:-0}"
  if [[ "$v" != "0" && -n "$v" ]]; then echo "FAIL: diag flag $f=$v set — needs OFF"; exit 2; fi
done
echo "[metrics-off] diag/state-probe OFF"

echo "=== FR13 APC E4-pre STOCK-ALIGN (cat6root) ts=$TS subset=$SUBSET ==="
git rev-parse HEAD 2>/dev/null | tee "$GATEDIR/git_head_$TS.txt" || true

# --- TRUE STOCK ALIGN: clean disciplines + CONV_FIX=1 (stock conv) + every other fix-fn OFF ---
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
unset APC_MAX_NUM_BATCHED_TOKENS 2>/dev/null || true   # launcher defaults block_size (#45238)
export FR13_APC_CONV_FIX=1            # STOCK get_conv_copy_spec (NOT 0 = the conv override carrier)
export FR13_APC_CONV_SNAPSHOT=0
export FR13_APC_SSM_SNAPSHOT=0
export FR13_APC_SSM_WRITE_THROUGH=0
export FR13_APC_HIT_RECURRENT_SUFFIX=0
export FR13_APC_DROP_FINAL_BLOCK=0
export FR13_APC_VERBATIM=0
# BLOCK_ALIGN_45477 left at launcher default (=1). CUDA-GRAPH (no ENFORCE_EAGER) = deploy regime.
echo "[stock-align] ENABLE_APC=1 block=1024 fp32 maxnb=block_size CONV_FIX=1 (stock) BLOCK_ALIGN=1;"
echo "             CONV_SNAPSHOT=SSM_SNAPSHOT=WRITE_THROUGH=HIT_SUFFIX=DROP_FINAL=VERBATIM=0; cuda-graph"

bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM_ON" "$KIND" "$SUBSET"
ON_RC=$?
echo "ARM_ON rc=$ON_RC (armdir=$RUNROOT/$ARM_ON)"

PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY

echo
echo "=== STOCK-ALIGN 12907 VERDICT (codex_trace agent_messages) ==="
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
    print(f"  {os.path.basename(pt)}: patch={sz}B eval={v} cjk={cjk} trig={rt:.3f} -> {g}")
PY
echo
echo "  STOCK-ALIGN SOLVES 12907 -> empties were CONV_FIX=0 self-inflicted; stock-align = ship candidate"
echo "  STOCK-ALIGN EMPTIES 12907 -> SSM stale-row defect real -> test FR13_APC_VERBATIM (eager) next"
echo "STOCKALIGN_DONE on_rc=$ON_RC"
exit $ON_RC
