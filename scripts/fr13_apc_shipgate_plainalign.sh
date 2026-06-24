#!/usr/bin/env bash
# FR13 APC 4-TASK GATE — PLAIN-ALIGN cache-ON vs cache-OFF
# =================================================================================
# E1 (output/fr13_apc_e1_plainalign) proved on agentic 12907 that PLAIN-ALIGN
# (APC on, every tree fix-fn OFF, only the clean disciplines) is COHERENT — cjk=0,
# trigram<=0.05 both attempts — i.e. the cjk word-salad / truncated-tool-call GARBLE
# that the full-pile fix-fn config produced (12907 cjk=125) is a BAKED FIX-FN, NOT
# align's reconstruct. The one wrinkle: 12907 still empty-patched coherently under
# plain-align (a different boot's cache-OFF solved it = 504B). This gate's ARM_OFF
# is the SAME-CONFIG cache-OFF control that resolves whether that empty-patch is a
# cache-ON effect or just a hard-task/temp-0.6 variance.
#
# ARM_ON  = PLAIN-ALIGN cache-ON: FR13_ENABLE_APC=1, --mamba-block-size 1024,
#           --mamba-ssm-cache-dtype float32, --max-num-batched-tokens=block_size
#           (#45238 fix), BLOCK_ALIGN_45477 (launcher default 1), and EVERY tree
#           fix-fn FORCED OFF (CONV_FIX/CONV_SNAPSHOT/SSM_SNAPSHOT/SSM_WRITE_THROUGH/
#           HIT_RECURRENT_SUFFIX/DROP_FINAL=0). DIAG off.
# ARM_OFF = cache-OFF baseline (FR13_ENABLE_APC unset -> byte-identical locked cat6root).
#
# Gives: (a) garble-elimination across all 4 (12907/13033/13236/13398), (b) coding-
# quality parity cache-ON vs cache-OFF same-boot, (c) the 12907 empty-patch confound
# resolution, (d) pair-dumps for the offline per-token lossless rescore (E5 floor 12.90%
# vs no-spec RECURRENT oracle). Metrics OFF (clean speed read). Two boots, serialized,
# recover between. SAME offload/seed/temp/subset as the original ship gate.
# =================================================================================
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
REPO=/home/mark/shared/lumoFlyWheel

SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_b4_four.json}   # 12907/13033/13236/13398
KIND=cat6root
EXPECT_TOK_PER_DRAFT=6
TS=$(date -u +%Y%m%dT%H%M%S%3NZ)
RUNROOT=output/fr13_apc_shipgate_plainalign/$TS
export RUNROOT
GATEDIR=$RUNROOT
mkdir -p "$GATEDIR"
echo "[run-key] artifacts -> $RUNROOT (PLAIN-ALIGN cache-ON FIRST, then cache-OFF bar)"

ARM_OFF=${ARM_OFF:-cat6root_apcoff_b1}
ARM_ON=${ARM_ON:-cat6root_plainalign_b1}

export MAX_NUM_SEQS_OVR=${MAX_NUM_SEQS_OVR:-1}
export SWE_CONCURRENCY=${SWE_CONCURRENCY:-1}
export OFFLOAD_CODEX=${OFFLOAD_CODEX:-1}
export DEPLOY_FORCE_TEMP=${DEPLOY_FORCE_TEMP:-0.6}
export SEED=${SEED:-1313}

export FR10_METRICS=0
export BATCH_INVARIANT=0
_DIAG_FLAGS=(FR13_SFWD_GPU_TIMER FR13_COMMIT_ARGMAX_GATE FR13_FORK_MARGIN_DUMP \
  FR13_CHASE_DIAG FR13_FIX1_SELFCHECK \
  FR13_REPLAY_BOUNDARY_LOG FR13_GDN_SUBOP_MAB FR13_FA2_MAB FR13_REPLAY_DURABLE_AB \
  FR13_TCF_SELFCHECK)
for f in "${_DIAG_FLAGS[@]}"; do
  v="${!f:-0}"
  if [[ "$v" != "0" && -n "$v" ]]; then
    echo "FAIL: diagnostic flag $f=$v is set — gate requires DIAG OFF"; exit 2
  fi
done
echo "[metrics-off] all diag/state-probe flags OFF — clean speed read"

echo "=== FR13 APC PLAIN-ALIGN GATE (cat6root) ts=$TS subset=$SUBSET ==="
echo "    ARM_ON=$ARM_ON (plain-align cache-ON)   ARM_OFF=$ARM_OFF (cache-OFF bar)"
git rev-parse HEAD 2>/dev/null | tee "$GATEDIR/git_head_$TS.txt" || true

# =================================================================================
# ARM 1 — PLAIN-ALIGN cache-ON. RUN FIRST. Every tree fix-fn FORCED OFF.
# =================================================================================
echo
echo "########## ARM 1: PLAIN-ALIGN cache-ON (every tree fix-fn OFF) ##########"
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
unset APC_MAX_NUM_BATCHED_TOKENS 2>/dev/null || true   # launcher defaults to block_size = #45238 fix
# PLAIN-ALIGN: explicit =0 -> launcher's `: "${VAR:=1}"` respects set-but-zero (no override)
export FR13_APC_CONV_FIX=0
export FR13_APC_CONV_SNAPSHOT=0
# BLOCK_ALIGN_45477 left at launcher default (=1) — a CLEAN discipline (kept).
echo "[plain-align] ENABLE_APC=1 block=1024 fp32 maxnb=block_size BLOCK_ALIGN=default(1);"
echo "             CONV_FIX=CONV_SNAPSHOT=SSM_SNAPSHOT=WRITE_THROUGH=HIT_RECURRENT_SUFFIX=DROP_FINAL=0"

bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM_ON" "$KIND" "$SUBSET"
ON_RC=$?
echo "ARM_ON rc=$ON_RC (armdir=$RUNROOT/$ARM_ON)"

PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
sleep 3

# =================================================================================
# ARM 2 — cache-OFF baseline (FR13_ENABLE_APC unset => byte-identical locked cat6root).
#         This is the SAME-CONFIG control for the 12907 empty-patch confound.
# =================================================================================
echo
echo "########## ARM 2: cache-OFF (no APC / baseline = same-config control) ##########"
unset FR13_ENABLE_APC MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE APC_MAX_NUM_BATCHED_TOKENS \
      FR13_APC_CONV_FIX FR13_APC_CONV_SNAPSHOT \
      FR13_APC_BLOCK_ALIGN_45477 2>/dev/null || true

bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM_OFF" "$KIND" "$SUBSET"
OFF_RC=$?
echo "ARM_OFF rc=$OFF_RC (armdir=$RUNROOT/$ARM_OFF)"

PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY

# =================================================================================
# SUMMARY — garble scan + coding-quality verdicts per arm per task.
# =================================================================================
echo
echo "=== PLAIN-ALIGN GATE SUMMARY (garble + verdicts per arm) ==="
.venv/bin/python - "$RUNROOT" "$ARM_ON" "$ARM_OFF" <<'PY'
import json, glob, sys, os
runroot, arm_on, arm_off = sys.argv[1], sys.argv[2], sys.argv[3]
def trig(s):
    s="".join(s.split())
    if len(s)<6: return 0.0
    t=[s[i:i+3] for i in range(len(s)-2)]
    from collections import Counter
    return max(Counter(t).values())/len(t)
for arm in (arm_on, arm_off):
    print(f"--- {arm} ---")
    for pt in sorted(glob.glob(f"{runroot}/{arm}/swe_out/*/per_task/*")):
        tid=os.path.basename(pt)
        cjk=0; rt=0.0
        for tf in glob.glob(pt+"/codex_trace*.jsonl"):
            items=[json.loads(l).get('item',json.loads(l)) for l in open(tf)]
            msgs=[it.get('text','') or '' for it in items if it.get('type')=='agent_message']
            cjk+=sum(1 for m in msgs for ch in m if '一'<=ch<='鿿')
            rt=max([rt]+[trig(m) for m in msgs])
        pf=pt+"/patch.diff"; sz=os.path.getsize(pf) if os.path.exists(pf) else 0
        er=pt+"/eval/eval_report.json"; v="?"
        if os.path.exists(er):
            try: v=json.load(open(er)).get("verdict","?")
            except Exception: pass
        g="GARBLE" if (cjk>5 or rt>0.18) else "coherent"
        print(f"  {tid}: patch={sz}B eval={v} cjk={cjk} trig={rt:.3f} -> {g}")
PY

echo
echo "=== GATE arms complete (off_rc=$OFF_RC on_rc=$ON_RC) ==="
echo "NEXT (offline, SEPARATE GPU boots): per-token lossless rescore — pair-dumps -> oracle src"
echo "  -> recurrent rescore -> consolidate (cache-ON as --cat9-*, cache-OFF as --native-*);"
echo "  LOSSLESS iff cache-ON clear-margin flip-rate <= cache-OFF AND within E5 floor 12.90%."
OVERALL=0
[[ "$OFF_RC" == "0" && "$ON_RC" == "0" ]] || OVERALL=1
echo "GATE_DONE overall=$OVERALL (off_rc=$OFF_RC on_rc=$ON_RC)"
exit $OVERALL
