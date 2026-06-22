#!/usr/bin/env bash
# FR13 APC E1 — PLAIN-ALIGN discriminator on the agentic 12907 garble repro
# =================================================================================
# Carrier workflow wy2r4elrs (medium-conf) says: the full-pile cache-ON config
# garbles agentically because the SSM fix-fns (WT/SSM_SNAPSHOT) write the committed
# leaf VALUE into the temporal-copy SNAPSHOT row (start_addr), but the cache-HIT
# restore reads the BLOCK-ALIGNED row (block_ids[(seq_len-1)//block_size]) — a
# DIFFERENT row (measured 658 vs 716) => dead write, stock align reads a stale
# block-pool row => GDN recurrent poison => garble over the long agentic context.
#
# E1 DISCRIMINATOR (cheapest, on the AGENTIC regime — greedy byte-AB is proven
# non-discriminating): APC ON, every tree fix-fn FORCED OFF (plain align), keep ONLY
# the clean disciplines that ship the spine within-floor. Run agentic 12907 one pass.
#   * plain-align CLEAN on 12907  -> a baked fix-fn is the carrier (bisect E2:
#       +SSM_SNAPSHOT, then +WT, then +CONV).
#   * plain-align GARBLES on 12907 -> align's reconstruct itself is wrong for the
#       tree committer's node-bank -> E3 cudagraph discriminator, then E4 verbatim.
#
# CONTROL: cache-OFF 12907 is banked LOSSLESS-SOLVE (504B patch,
# output/fr13_bigdenom_swe/cat6root_apcoff_b1/.../12907/patch.diff). Garble verdict
# is read at codex_trace agent_messages (CJK/repeated-trigram), NOT engine logs.
#
# Single arm, single boot. Same offload/metrics-off discipline as the ship gate.
# =================================================================================
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
REPO=/home/mark/shared/lumoFlyWheel

SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_astropy12907.json}   # 12907 only
KIND=cat6root
TS=$(date -u +%Y%m%dT%H%M%S%3NZ)
RUNROOT=output/fr13_apc_e1_plainalign/$TS
export RUNROOT
GATEDIR=$RUNROOT
mkdir -p "$GATEDIR"
ARM_ON=${ARM_ON:-cat6root_plainalign_b1}
echo "[run-key] artifacts -> $RUNROOT (plain-align cache-ON, 12907 agentic)"

# B=1 temp-0.6 deployment regime (identical to ship gate ARM_ON)
export MAX_NUM_SEQS_OVR=${MAX_NUM_SEQS_OVR:-1}
export SWE_CONCURRENCY=${SWE_CONCURRENCY:-1}
export OFFLOAD_CODEX=${OFFLOAD_CODEX:-1}
export DEPLOY_FORCE_TEMP=${DEPLOY_FORCE_TEMP:-0.6}
export SEED=${SEED:-1313}

# METRICS OFF (clean; DIAG OFF is REQUIRED — DIAG-induced garble is a known false positive)
export FR10_METRICS=0
export BATCH_INVARIANT=0
_DIAG_FLAGS=(FR13_SFWD_GPU_TIMER FR13_COMMIT_ARGMAX_GATE FR13_FORK_MARGIN_DUMP \
  FR13_CHASE_DIAG FR13_FIX1_SELFCHECK FR13_APC_STATE_PROBE FR13_APC_SSM_DIAG \
  FR13_REPLAY_BOUNDARY_LOG FR13_GDN_SUBOP_MAB FR13_FA2_MAB FR13_REPLAY_DURABLE_AB \
  FR13_TCF_SELFCHECK)
for f in "${_DIAG_FLAGS[@]}"; do
  v="${!f:-0}"
  if [[ "$v" != "0" && -n "$v" ]]; then
    echo "FAIL: diagnostic flag $f=$v is set — E1 requires DIAG OFF (probe garble = false positive)"; exit 2
  fi
done
echo "[metrics-off] all diag/state-probe flags OFF"

echo "=== FR13 APC E1 PLAIN-ALIGN (cat6root) ts=$TS subset=$SUBSET ==="
git rev-parse HEAD 2>/dev/null | tee "$GATEDIR/git_head_$TS.txt" || true

# --- PLAIN-ALIGN cache-ON: keep clean disciplines, FORCE every tree fix-fn OFF ---
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
unset APC_MAX_NUM_BATCHED_TOKENS 2>/dev/null || true   # launcher defaults to block_size=1024 (#45238 fix)
# explicit =0 -> launcher's `: "${VAR:=1}"` respects the set-but-zero value (no override)
export FR13_APC_CONV_FIX=0
export FR13_APC_CONV_SNAPSHOT=0
export FR13_APC_SSM_SNAPSHOT=0
export FR13_APC_SSM_WRITE_THROUGH=0
export FR13_APC_HIT_RECURRENT_SUFFIX=0
export FR13_APC_DROP_FINAL_BLOCK=0
# BLOCK_ALIGN_45477 left to launcher default (=1) — a CLEAN discipline (kept).
echo "[plain-align] ENABLE_APC=1 block=1024 fp32 maxnb=block_size BLOCK_ALIGN=default(1);"
echo "             CONV_FIX=CONV_SNAPSHOT=SSM_SNAPSHOT=WRITE_THROUGH=HIT_RECURRENT_SUFFIX=DROP_FINAL=0"

bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM_ON" "$KIND" "$SUBSET"
ON_RC=$?
echo "ARM_ON rc=$ON_RC (armdir=$RUNROOT/$ARM_ON)"

PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY

# --- GARBLE VERDICT at codex_trace agent_messages (CJK + repeated-trigram) ---
echo
echo "=== E1 GARBLE VERDICT (codex_trace agent_messages — NOT engine log) ==="
.venv/bin/python - "$RUNROOT/$ARM_ON" <<'PY'
import json, glob, sys, os
arm = sys.argv[1]
tfs = glob.glob(arm + "/swe_out/*/per_task/*/codex_trace*.jsonl")
if not tfs:
    print("  WARN: no codex_trace found under", arm); sys.exit(0)
def trig(s):
    s = "".join(s.split())
    if len(s) < 6: return 0.0
    t = [s[i:i+3] for i in range(len(s)-2)]
    from collections import Counter
    c = Counter(t)
    return max(c.values()) / len(t)
for tf in sorted(tfs):
    items = [json.loads(l).get('item', json.loads(l)) for l in open(tf)]
    msgs = [it.get('text','') or '' for it in items if it.get('type') == 'agent_message']
    blob = "".join(msgs)
    cjk = sum(1 for ch in blob if '一' <= ch <= '鿿')
    rt = max((trig(m) for m in msgs), default=0.0)
    types = {}
    for it in items: types[it.get('type','?')] = types.get(it.get('type','?'),0)+1
    pt = os.path.basename(os.path.dirname(tf))
    verdict = "GARBLE" if (cjk > 5 or rt > 0.18) else "coherent"
    print(f"  {pt} [{os.path.basename(tf)}]: cjk={cjk} max_trigram={rt:.3f} -> {verdict}  types={types}")
# patch sizes + eval verdicts
print("  --- patches/verdicts ---")
for m in sorted(glob.glob(arm + "/swe_out/*/per_task/*/patch.diff")):
    pd = os.path.dirname(m)
    sz = os.path.getsize(m)
    er = os.path.join(pd, "eval", "eval_report.json")
    v = "?"
    if os.path.exists(er):
        try: v = json.load(open(er)).get("verdict","?")
        except Exception: pass
    print(f"  {os.path.basename(pd)}: patch={sz}B eval={v}")
PY

echo
echo "=== E1 INTERPRETATION ==="
echo "  plain-align CLEAN + non-empty patch  -> a baked fix-fn was the carrier (run E2 bisection)"
echo "  plain-align GARBLE/empty             -> align reconstruct wrong -> E3 cudagraph, then E4 verbatim"
echo "E1_DONE on_rc=$ON_RC"
exit $ON_RC
