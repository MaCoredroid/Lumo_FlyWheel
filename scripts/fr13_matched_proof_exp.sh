#!/usr/bin/env bash
# MATCHED PROOF (user goal): does cat8-SPINE accept == native (superset), and do cat8 BRANCHES rescue
# native-misses, so cat8 > native STRICTLY? cat8 (branch-rescue diag) + native, SAME fixed prompt,
# greedy(deterministic=exactly matched) + temp06 + temp10. B=1 => ZERO batching effect (kills that confound).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
RR=output/fr13_matched_proof; rm -rf "$RR"; mkdir -p "$RR"
MODES="${MODES:-greedy temp06 temp10}"; N="${N:-512}"
# Real SWE-Verified task prompt (chat messages) — faithful ship-config content, not the synthetic prompt.
CHATMSG="${CHATMSG:-output/fr13_matched_proof_swe_prompt.json}"
echo "PROMPT: real SWE task = $(python3 -c "import json;print(json.load(open('$CHATMSG')).get('task'))" 2>/dev/null)"
run_cat8() {
  echo "===== cat8 (branch-rescue diag, remap ON) ====="
  ACCEPT_SPEED_PROBE=1 OFFLOAD_AGENT=0 PROBE_N=$N MAX_NUM_SEQS_OVR=1 PROBE_MODES="$MODES" RUNROOT=$RR \
    PROBE_CHAT_MESSAGES="$CHATMSG" \
    FR13_ATTN_KV_REMAP=1 FR13_DEVICE_MULTIDRAFT=1 \
    FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
    FR13_BRANCH_ACCEPT_DIAG=1 FR13_BRANCH_ACCEPT_JSON=/workspace/$RR/probe_cat8/branch_hist.json \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh probe_cat8 cat8 subset_b4_four.json > "$RR/cat8.log" 2>&1
  echo "[cat8] rc=$? containers: $(docker ps -q|wc -l)"
}
run_native() {
  echo "===== native (bar) ====="
  ACCEPT_SPEED_PROBE=1 OFFLOAD_AGENT=0 PROBE_N=$N MAX_NUM_SEQS_OVR=1 PROBE_MODES="$MODES" RUNROOT=$RR \
    PROBE_CHAT_MESSAGES="$CHATMSG" \
    FR13_ATTN_KV_REMAP=1 \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh probe_native nativemtp5_exseed subset_b4_four.json > "$RR/native.log" 2>&1
  echo "[native] rc=$? containers: $(docker ps -q|wc -l)"
}
run_cat8; run_native
echo "===== MATCHED PROOF DONE ====="; .venv/bin/python scripts/fr13_matched_proof_report.py "$RR"
