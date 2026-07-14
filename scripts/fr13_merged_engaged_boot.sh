#!/usr/bin/env bash
# FR13 merged drafter gate (d): boot the WIDE tree (t33333/cat33333) with FR13_DRAFT_SOURCE=merged
# and ASSERT ENGAGED. FR13_DRAFT_SOURCE=merged triggers (1) the launcher prelaunch arctic-inference
# install, (2) the /logs/fr13_draft_source_merged.arm sidecar. The seam then speculates per step and
# logs "[FR13_MERGED ENGAGED] ... match_full=N ...". GATE = arctic installed + speculate_fired>0 AND
# match_full>0 (match_full=0 with speculate>0 => the gappy-history trap; started/ingested>0 = lifecycle
# ran). Detached boot per directive. Single GPU job.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RR=output/fr13_slot_reorder/merged_engaged_$STAMP; mkdir -p "$RR"
SUM="$RR/summary.txt"; : > "$SUM"
echo "merged-drafter ENGAGED boot $STAMP (t33333 + FR13_DRAFT_SOURCE=merged)" | tee -a "$SUM"

[[ -z "$(docker ps -q)" ]] || { echo "ABORT: docker busy" | tee -a "$SUM"; exit 1; }

KIND=t33333
export FR13_DRAFT_SOURCE=merged \
  FR13_SLOT_REORDER=1 FR13_ATTN_KV_REMAP=1 FR13_DEVICE_MULTIDRAFT=1 \
  FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
  FR13_DFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/merged_${KIND}_dfwd.json \
  ACCEPT_SPEED_PROBE=1 OFFLOAD_AGENT=0 PROBE_N=512 MAX_NUM_SEQS_OVR=1 PROBE_MODES=temp06 \
  PROBE_CHAT_MESSAGES=output/fr13_matched_proof_swe_prompt.json RUNROOT="$RR"
bash scripts/fr13_detached_boot.sh "merged_$KIND" \
  "bash scripts/fr13_bigdenom_swe_serve_variant.sh merged_$KIND $KIND subset_carrier_four.json > $RR/merged_$KIND.log 2>&1"

DL="$RR/merged_$KIND/docker_full.log"
ARM="$RR/merged_$KIND"
echo "booting (poll up to 25min for probe + ENGAGED)..." | tee -a "$SUM"
for _ in $(seq 1 300); do
  sleep 5
  [[ -f "$ARM/accept_speed_temp06.json" ]] && break
  grep -qiE "EngineCore failed|Traceback.*fr13_merged|NotImplementedError|CUDA out of memory" "$DL" 2>/dev/null && { echo "boot error detected" | tee -a "$SUM"; break; }
done

echo "=== arctic install ===" | tee -a "$SUM"
grep -aE "FR13-PRELAUNCH|arctic-inference|Successfully installed arctic|ERROR: .*arctic|Failed to build.*arctic" "$DL" 2>/dev/null | tail -6 | tee -a "$SUM"
echo "=== ENGAGED needle (last) ===" | tee -a "$SUM"
grep -aE "FR13_MERGED ENGAGED" "$DL" 2>/dev/null | tail -3 | tee -a "$SUM"
echo "=== probe accept + dfwd ===" | tee -a "$SUM"
[[ -f "$ARM/accept_speed_temp06.json" ]] && .venv/bin/python -c "import json;d=json.load(open('$ARM/accept_speed_temp06.json'));print('accept_per_forward=%.3f'%d.get('accept_per_forward',0))" 2>/dev/null | tee -a "$SUM"
DF=$(ls output/fr13_sfwd_sidecar/merged_${KIND}_dfwd.json.* 2>/dev/null | head -1)
[[ -n "$DF" ]] && .venv/bin/python -c "import json;d=json.load(open('$DF'));print('dfwd=%.2f ms/step (%d spans)'%(d['gpu_seconds']/max(d['n_spans'],1)*1000,d['n_spans']))" 2>/dev/null | tee -a "$SUM"

# VERDICT
ENG=$(grep -aoE "match_full=[0-9]+" "$DL" 2>/dev/null | tail -1 | grep -oE "[0-9]+" || echo 0)
SPEC=$(grep -aoE "speculate_fired=[0-9]+" "$DL" 2>/dev/null | tail -1 | grep -oE "[0-9]+" || echo 0)
echo "=== VERDICT: speculate_fired=$SPEC match_full=$ENG ===" | tee -a "$SUM"
if [[ "${ENG:-0}" -gt 0 ]]; then echo "ENGAGED PASS (match_full>0, non-gappy)" | tee -a "$SUM";
elif [[ "${SPEC:-0}" -gt 0 ]]; then echo "PARTIAL: speculate fired but match_full=0 (gappy-history trap or no repetition) -- investigate" | tee -a "$SUM";
else echo "VACUOUS: no speculate (arctic install failed? merged not gated on? check above)" | tee -a "$SUM"; fi

docker rm -f "fr13-bigdenom-merged_$KIND" >/dev/null 2>&1 || true
bash scripts/recover_host_memory.sh >/dev/null 2>&1 || true
echo "=== DONE ($RR) ===" | tee -a "$SUM"; cat "$SUM"
