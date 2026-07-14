#!/usr/bin/env bash
# Tree-size scaling sweep (B=1, isolates tree_n from batch-tile confound): for each
# tree, boot -> sfwd timer -> short probe -> record verify ms/step + cudagraph MODE
# (FULL vs PIECEWISE/eager = the cliff signal) + boot success (cap check). Answers
# "is 16 the limit?" — is the scaling smooth or does it cliff past 16 rows.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RR=output/fr13_slot_reorder/tree_scale_$STAMP; mkdir -p "$RR"
SUM="$RR/summary.txt"; : > "$SUM"
echo "tree_scale sweep $STAMP (B=1, sfwd verify ms/step + cudagraph mode)" | tee -a "$SUM"

sweep() {  # kind rows
  local KIND="$1" ROWS="$2" C="fr13-bigdenom-ts_$1"
  [[ -z "$(docker ps -q)" ]] || { echo "ABORT: docker busy" | tee -a "$SUM"; return 1; }
  rm -f output/fr13_sfwd_sidecar/ts_${KIND}_sfwd.json.*
  echo "----- $KIND ($ROWS rows) -----" | tee -a "$SUM"
  env FR13_SLOT_REORDER=1 FR13_ATTN_KV_REMAP=1 FR13_DEVICE_MULTIDRAFT=1 \
    FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
    FR13_SFWD_GPU_TIMER=1 FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/ts_${KIND}_sfwd.json \
    ACCEPT_SPEED_PROBE=1 OFFLOAD_AGENT=0 PROBE_N=256 MAX_NUM_SEQS_OVR=1 PROBE_MODES=temp06 \
    PROBE_CHAT_MESSAGES=output/fr13_matched_proof_swe_prompt.json RUNROOT="$RR" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh "ts_$KIND" "$KIND" subset_carrier_four.json \
    > "$RR/ts_$KIND.log" 2>&1
  local RC=$?
  local DL="$RR/ts_$KIND/docker_full.log"
  # cudagraph mode during decode (FULL = captured; PIECEWISE/NONE = eager cliff)
  local CG=$(grep -oiE "cudagraph_mode: (FULL|PIECEWISE|NONE)|Capturing cudagraphs|CUDAGraph mode" "$DL" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  local CGRUN=$(grep -oE "cudagraph_runtime_mode=[A-Z]+|Running batch with cudagraph_mode: [A-Za-z.]+" "$DL" 2>/dev/null | sort | uniq -c | tail -3 | tr '\n' ';')
  local S=$(ls output/fr13_sfwd_sidecar/ts_${KIND}_sfwd.json.* 2>/dev/null | head -1)
  local MS="n/a"
  [[ -n "$S" ]] && MS=$(.venv/bin/python -c "import json;d=json.load(open('$S'));print('%.1f (%d steps)'%(d['decode_forward_gpu_seconds']/max(d['n_pure_decode_steps_timed'],1)*1000,d['n_pure_decode_steps_timed']))" 2>/dev/null)
  local ACC=$(.venv/bin/python -c "import json,glob;fs=glob.glob('$RR/ts_$KIND/accept_speed_temp06.json');print('%.3f'%json.load(open(fs[0])).get('accept_per_forward',0)) if fs else print('na')" 2>/dev/null || echo na)
  echo "  rc=$RC  rows=$ROWS  verify_ms/step=$MS  accept=$ACC  cudagraph=[$CGRUN]" | tee -a "$SUM"
  docker rm -f "$C" >/dev/null 2>&1 || true
  bash scripts/recover_host_memory.sh >/dev/null 2>&1 || true
  sleep 5
}

sweep cat8 9
sweep t33333 16
sweep t55555 26
echo "=== SWEEP DONE ($RR) ===" | tee -a "$SUM"
cat "$SUM"
