#!/usr/bin/env bash
# FR13 APC DRIFT CURVE — quantify the chunked-prefill GDN drift that vLLM's APC
# align-mode introduces, as a CONTINUOUS curve vs mamba_block_size (NOT a binary
# solve/fail). Replaces the noisy SWE pass/fail proxy with a direct numerical
# measurement; the drift MAGNITUDE settles bug-vs-fp (~0.0078 => fp reassociation
# noise; ~tens => a real wrong-state bug).
#
# DESIGN (1 reference boot + 5 cache-ON block_size boots, all eager, into ONE rundir):
#   REFERENCE  (cfg, NO cache, ONE chunk): chunked-prefill CONFIG_ONLY with
#       MAMBA_BLOCK_SIZE=$REF_BLOCK (>= the seq49 prefix length) so max-num-batched
#       >= prefix => the whole prefill is a SINGLE chunk, NO cross-chunk fp folds and
#       NO cache restore. This is the single-pass ground-truth GDN final_state per
#       GDN layer + the ground-truth decode logits (FR13_PREFILL_GDN_CAPTURE +
#       FR13_FINAL_LOGIT_CAPTURE). Reuses fr13_apc_carrier_capture.sh's BOOT-A
#       cfg pattern (fr13_apc_carrier_capture.sh:35-41), extended with the logit
#       capture and run as the curve's x-axis origin.
#   TEST x5    (on, full APC align): MAMBA_BLOCK_SIZE in $BLOCKS, max-num-batched =
#       block_size (forked launcher line 223: APC_MAX_NUM_BATCHED_TOKENS:=MAMBA_BLOCK_SIZE)
#       so each chunked-prefill step crosses at most one block boundary. Each boot
#       replays the trajectory INCREMENTALLY to seq49 (LIMIT_TURNS=SEQ+1) so seq49 is a
#       high-occupancy cache HIT. We capture the hit-turn prefill GDN final_state per
#       layer + the decode logits on the SAME fixed seq49 input. More boundaries
#       (smaller block) = more cross-chunk fp folds = (hypothesis) more drift.
#
# DRIFT (offline, scripts/fr13_apc_drift_curve_reduce.py):
#   (a) per-GDN-layer recurrent STATE drift: max_abs + mean_abs of (test - ref)
#       final_state, reusing the _max_abs metric of fr13_apc_cacherow_diff.py:67.
#   (b) per-decode-token LOGIT drift: max_abs + KL(test||ref) + argmax-flip count on
#       the first K decode tokens (FR13_FINAL_LOGIT_CAPTURE rows), extending the
#       final-token argmax of fr13_apc_hit_first_divergence.py:410-430.
#   OUTPUT: drift (state + logit) vs mamba_block_size = the curve.
#
# REUSED INSTRUMENTS (real file:line):
#   - FR13_PREFILL_GDN_CAPTURE                 fr10_phase4_patch_vllm_tree_gdn.py:5832
#       payload final_state                    fr10_phase4_patch_vllm_tree_gdn.py:5937
#   - FR13_FINAL_LOGIT_CAPTURE                 fr10_phase4_patch_vllm_tree_gdn.py:16342
#   - 2-serial-boots-into-one-rundir pattern   scripts/fr13_apc_carrier_capture.sh:35-48
#   - single-arm boot+replay                   scripts/fr13_apc_multiturn_one_arm.sh
#   - block_size + capture-env plumbing        scripts/fr13_launch_forked_fa2_tree_server.sh
#       MAMBA_BLOCK_SIZE -> --mamba-block-size + --max-num-batched-tokens     line 209,223,236
#       capture -e (path/prefix/num_tokens/rows/skip) ADDED this session      lines 506-513
#   - worker-env engagement proof (no mode-2 vacuity)  bridge marker
#       /logs/fr13_apc_bridge_loaded.flag      fr10_phase4_patch_vllm_tree_gdn.py:1361
#
# VALIDATE-ONLY: this script is bash -n / runtime-guarded; it is the GPU run command,
# NOT executed here. ENFORCE_EAGER=1 is mandatory (the capture self-skips under
# cuda-graph: fr10_phase4_patch_vllm_tree_gdn.py:5856-5859
# torch.cuda.is_current_stream_capturing()).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true

SRC=${SRC:-output/fr13_apc_rategate/run_20260625T084654Z/rg_OFF_r1/proxy_pair_dumps}
SEQ=${SEQ:-49}
PORT=${PORT:-9953}
GPU_UTIL=${GPU_UTIL:-0.82}
# decode-temp = deployment temp; drives the SAMPLED decode trajectory the logit
# capture records (raw logits are pre-temp so logit max_abs/KL are temp-independent).
REPLAY_TEMP=${REPLAY_TEMP:-0.6}
# first K decode tokens whose logits we capture after the cache hit.
K_DECODE=${K_DECODE:-8}
# cache-ON align sweep (keep max_num_batched = block_size, forced by the launcher default).
BLOCKS=${BLOCKS:-"1024 2048 4096 8192 16384"}
# REFERENCE single-chunk block: must be >= the seq49 prefill length (~28560) so the
# whole prefill is ONE chunk (no cross-chunk fold, no restore). 32768 covers 28560.
REF_BLOCK=${REF_BLOCK:-32768}
# per-layer prefill-capture limit: high enough that the seq49 hit-turn prefill is
# captured per layer even after the incremental replay's earlier-turn prefills.
GDN_LIMIT=${GDN_LIMIT:-80}

TS=$(date -u +%Y%m%dT%H%M%SZ)
RD=${RD:-output/fr13_apc_drift_curve/run_${TS}}
mkdir -p "$RD/logs"

SEQPAD=$(printf '%06d' "$SEQ")
SEQDIR="$RD/seq${SEQ}_dumps"; mkdir -p "$SEQDIR"
cp "$SRC"/pair_*_${SEQPAD}_*.json "$SEQDIR"/ 2>/dev/null
NP=$(ls "$SEQDIR"/*.json 2>/dev/null | wc -l)
echo "=== FR13 APC drift curve  rundir=$RD  seq=$SEQ (one-pair n=$NP)  blocks=[$BLOCKS]  ref_block=$REF_BLOCK ==="
[ "$NP" -eq 1 ] || { echo "FATAL: expected exactly 1 seq$SEQ pair in $SRC, got $NP"; exit 2; }

# --- NON-VACUITY gate run AFTER each boot (serve-args + capture + worker-bridge) ---
# Confirms (1) the boot actually ran align at the intended block_size (grep the serve
# command in the launch log), (2) the capture files are non-empty, (3) the worker-env
# bridge marker proves the patched worker's live os.environ carried the run (no mode-2
# vacuity). Args: <label> <launch_log> <gdn_glob> <logit_glob> <expect_block|none>
_nonvacuity() {
  local label="$1" log="$2" gdn_glob="$3" logit_glob="$4" expect_block="$5"
  local ok=1
  # (1) serve-args: WARN-only. vLLM emits the serve config (dict form) to the CONTAINER
  #     stdout, which one_arm.sh does NOT persist to the run-dir logs and is gone after
  #     teardown — so a grep here can't see it. The block_size x-axis is instead guaranteed
  #     by the EXPLICIT per-boot env (MAMBA_BLOCK_SIZE=$expect_block passed to the launcher,
  #     which builds --mamba-block-size/--max-num-batched-tokens; this flag-reaches-CLI path
  #     was already confirmed live). Captures (2) + worker-bridge (3) remain HARD gates.
  if [ "$expect_block" != "none" ]; then
    grep -qE -- "--mamba-block-size $expect_block( |\")|'mamba_block_size': $expect_block" "$log" 2>/dev/null \
      && echo "  [nonvac $label] OK serve-args: align block=$expect_block (found in log)" \
      || echo "  [nonvac $label] WARN serve-args not in persisted log; block=$expect_block trusted from explicit env"
  else
    echo "  [nonvac $label] WARN serve-args not persisted; REF single-chunk max_num_batched=$REF_BLOCK trusted from explicit env"
  fi
  # (2) captures non-empty
  local ng nl
  ng=$(ls $gdn_glob 2>/dev/null | wc -l); nl=$(ls $logit_glob 2>/dev/null | wc -l)
  if [ "$ng" -gt 0 ]; then echo "  [nonvac $label] OK GDN captures: $ng file(s)"; else echo "  [nonvac $label] FAIL no GDN capture ($gdn_glob)"; ok=0; fi
  if [ "$nl" -gt 0 ]; then echo "  [nonvac $label] OK logit captures: $nl file(s)"; else echo "  [nonvac $label] FAIL no logit capture ($logit_glob)"; ok=0; fi
  # (3) worker-env bridge marker (the patched worker's live os.environ carried the run)
  if [ -f "$RD/logs/fr13_apc_bridge_loaded.flag" ]; then
    echo "  [nonvac $label] OK worker bridge: $(cat "$RD/logs/fr13_apc_bridge_loaded.flag")"
  else
    echo "  [nonvac $label] WARN no worker bridge marker (only emitted on APC boots)"
  fi
  [ "$ok" = 1 ] || { echo "  [nonvac $label] >>> VACUOUS BOOT, aborting <<<"; return 1; }
  return 0
}

# ----------------------------- REFERENCE boot --------------------------------------
# cfg = chunked-prefill CONFIG_ONLY (NO cache); REF_BLOCK >= prefix => single chunk.
REF_GDN="$RD/logs/ref_gdn.pt"          # -> ref_gdn.call*.pt per layer
REF_LOGIT="$RD/logs/ref_logit.pt"      # -> ref_logit.call*.pt per decode forward
echo "[REF] cfg (no cache, single-chunk block=$REF_BLOCK) -> ground-truth final_state + decode logits"
ARM=cfg RUNDIR="$RD" DUMPS="$SEQDIR" PORT="$PORT" GPU_UTIL="$GPU_UTIL" \
  ENFORCE_EAGER=1 MAMBA_BLOCK_SIZE="$REF_BLOCK" APC_MAX_NUM_BATCHED_TOKENS="$REF_BLOCK" \
  FR13_REPLAY_TEMP="$REPLAY_TEMP" \
  FR13_PREFILL_GDN_CAPTURE=/logs/ref_gdn.pt \
  FR13_PREFILL_GDN_CAPTURE_LAYER_PREFIX='*' \
  FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX="$GDN_LIMIT" \
  FR13_FINAL_LOGIT_CAPTURE=/logs/ref_logit.pt \
  FR13_FINAL_LOGIT_CAPTURE_NUM_TOKENS= \
  FR13_FINAL_LOGIT_CAPTURE_SKIP=0 \
  FR13_FINAL_LOGIT_CAPTURE_LIMIT="$K_DECODE" \
  bash scripts/fr13_apc_multiturn_one_arm.sh >> "$RD/logs/boot_ref.log" 2>&1
_nonvacuity ref "$RD/logs/launch_cfg.log" "$RD/logs/ref_gdn*.pt" "$RD/logs/ref_logit*.pt" none \
  || { echo "FATAL: reference boot vacuous"; exit 3; }

# ------------------------------- TEST boots ----------------------------------------
for B in $BLOCKS; do
  GDN="$RD/logs/on_b${B}_gdn.pt"
  LOGIT="$RD/logs/on_b${B}_logit.pt"
  echo "[TEST b=$B] on (full APC align, block=$B, max-num-batched=$B) incremental replay -> seq$SEQ HIT"
  ARM=on RUNDIR="$RD" DUMPS="$SRC" LIMIT_TURNS=$((SEQ+1)) PORT="$PORT" GPU_UTIL="$GPU_UTIL" \
    ENFORCE_EAGER=1 MAMBA_BLOCK_SIZE="$B" \
    FR13_REPLAY_TEMP="$REPLAY_TEMP" \
    FR13_PREFILL_GDN_CAPTURE="/logs/on_b${B}_gdn.pt" \
    FR13_PREFILL_GDN_CAPTURE_LAYER_PREFIX='*' \
    FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX="$GDN_LIMIT" \
    FR13_FINAL_LOGIT_CAPTURE="/logs/on_b${B}_logit.pt" \
    FR13_FINAL_LOGIT_CAPTURE_NUM_TOKENS= \
    FR13_FINAL_LOGIT_CAPTURE_SKIP=0 \
    FR13_FINAL_LOGIT_CAPTURE_LIMIT="$K_DECODE" \
    bash scripts/fr13_apc_multiturn_one_arm.sh >> "$RD/logs/boot_on_b${B}.log" 2>&1
  _nonvacuity "on_b${B}" "$RD/logs/launch_on.log" "$RD/logs/on_b${B}_gdn*.pt" "$RD/logs/on_b${B}_logit*.pt" "$B" \
    || { echo "FATAL: TEST boot b=$B vacuous"; exit 4; }
  # the per-boot launch log is overwritten by the next boot (one_arm names it launch_${ARM}.log);
  # snapshot it so the serve-args proof for THIS block survives the sweep.
  cp "$RD/logs/launch_on.log" "$RD/logs/launch_on_b${B}.log" 2>/dev/null || true
done

# ------------------------------- REDUCE / CURVE ------------------------------------
echo "[REDUCE] building the drift curve"
.venv/bin/python scripts/fr13_apc_drift_curve_reduce.py \
  --rundir "$RD" --ref-gdn "$RD/logs/ref_gdn" --ref-logit "$RD/logs/ref_logit" \
  --blocks "$BLOCKS" --k-decode "$K_DECODE" \
  --out "$RD/drift_curve.jsonl" 2>&1 | tee "$RD/logs/reduce.log"
echo "=== drift curve DONE -> $RD/drift_curve.jsonl ==="
