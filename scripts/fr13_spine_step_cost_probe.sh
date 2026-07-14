#!/usr/bin/env bash
# Measure the per-spine-step DRAFTER cost (the mtp_k speed lever's size) with NO hot-path edit.
# Same-width trees, different depth: t333 (spine_steps=2) vs t33333 (spine_steps=4). The dfwd
# GPU timer brackets the whole drafter propose (all spine forwards + host gaps between them), so
#   (dfwd_ms(t33333) - dfwd_ms(t333)) / 2  =  per-spine-step drafter cost
# = exactly what MTP-k saves by drafting fewer spine steps (deep spine from Arctic suffix instead).
# sfwd (verify) captured for context (it grows with tree size, NOT spine_steps -- expected to differ).
# B=1 (MAX_NUM_SEQS_OVR=1). Detached boot per directive (setsid survives launcher SIGTERM).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RR=output/fr13_slot_reorder/spine_cost_$STAMP; mkdir -p "$RR"
SUM="$RR/summary.txt"; : > "$SUM"
echo "spine-step drafter cost probe $STAMP (B=1; dfwd brackets propose)" | tee -a "$SUM"

arm() {  # kind
  local KIND="$1" C="fr13-bigdenom-sc_$1"
  [[ -z "$(docker ps -q)" ]] || { echo "ABORT: docker busy" | tee -a "$SUM"; return 1; }
  rm -f "output/fr13_sfwd_sidecar/sc_${KIND}_dfwd.json."* "output/fr13_sfwd_sidecar/sc_${KIND}_sfwd.json."*
  echo "----- $KIND -----" | tee -a "$SUM"
  export FR13_SLOT_REORDER=1 FR13_ATTN_KV_REMAP=1 FR13_DEVICE_MULTIDRAFT=1 \
    FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
    FR13_DFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/sc_${KIND}_dfwd.json \
    FR13_SFWD_GPU_TIMER=1 FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/sc_${KIND}_sfwd.json \
    ACCEPT_SPEED_PROBE=1 OFFLOAD_AGENT=0 PROBE_N=256 MAX_NUM_SEQS_OVR=1 PROBE_MODES=temp06 \
    PROBE_CHAT_MESSAGES=output/fr13_matched_proof_swe_prompt.json RUNROOT="$RR"
  bash scripts/fr13_detached_boot.sh "sc_$KIND" \
    "bash scripts/fr13_bigdenom_swe_serve_variant.sh sc_$KIND $KIND subset_carrier_four.json > $RR/sc_$KIND.log 2>&1"
  # poll up to 20min for the accept_speed sidecar (probe done) or the dfwd sidecar
  local ok=0
  for _ in $(seq 1 240); do
    sleep 5
    if ls output/fr13_sfwd_sidecar/sc_${KIND}_dfwd.json.* >/dev/null 2>&1 \
       && ls "$RR/sc_$KIND"/accept_speed_temp06.json >/dev/null 2>&1; then ok=1; break; fi
    # bail early if the arm died
    grep -qiE "EngineCore failed|Traceback|NotImplementedError" "$RR/sc_$KIND/death.log" 2>/dev/null && break
  done
  local DF=$(ls output/fr13_sfwd_sidecar/sc_${KIND}_dfwd.json.* 2>/dev/null | head -1)
  local SF=$(ls output/fr13_sfwd_sidecar/sc_${KIND}_sfwd.json.* 2>/dev/null | head -1)
  local ACC=$(ls "$RR/sc_$KIND"/accept_speed_temp06.json 2>/dev/null | head -1)
  local DFMS="na" SFMS="na" ACCV="na"
  [[ -n "$DF" ]] && DFMS=$(.venv/bin/python -c "import json;d=json.load(open('$DF'));import sys;n=d.get('n_spans',0);print('%.2f (%d spans, %.3fs)'%(d['gpu_seconds']/max(n,1)*1000,n,d['gpu_seconds']))" 2>/dev/null)
  [[ -n "$SF" ]] && SFMS=$(.venv/bin/python -c "import json;d=json.load(open('$SF'));n=d.get('n_pure_decode_steps_timed',d.get('n_spans',0));print('%.2f/step'%(d.get('decode_forward_gpu_seconds',d.get('gpu_seconds',0))/max(n,1)*1000))" 2>/dev/null)
  [[ -n "$ACC" ]] && ACCV=$(.venv/bin/python -c "import json;print('%.3f'%json.load(open('$ACC')).get('accept_per_forward',0))" 2>/dev/null)
  echo "  ok=$ok  dfwd=$DFMS  sfwd=$SFMS  accept=$ACCV" | tee -a "$SUM"
  docker rm -f "$C" >/dev/null 2>&1 || true
  bash scripts/recover_host_memory.sh >/dev/null 2>&1 || true
  sleep 5
}

arm t333
arm t33333
echo "=== NOTE: per-spine-step drafter cost = (dfwd_ms(t33333)-dfwd_ms(t333))/2 ===" | tee -a "$SUM"
echo "=== PROBE DONE ($RR) ===" | tee -a "$SUM"
cat "$SUM"
