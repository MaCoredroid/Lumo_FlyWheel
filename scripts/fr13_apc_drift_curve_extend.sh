#!/usr/bin/env bash
# FR13 APC drift curve EXTEND: add the two missing curve points align@{8192,4096}
# (8192 FIRST = decisive: it's the WORKING e2e config — is its fixed-replay drift ~fp or
# still huge?), reusing the existing 193-capture continuous REF + on_b1024/on_b2048 captures
# in run_20260629T000358Z. Then reduce ALL FOUR blocks vs the REF into the full curve.
# NO PRE_SNAP_FIX (proven vacuous: drift 77.96->77.96 unchanged). Serial boots (concurrency=1),
# reduce only AFTER all boots (no compute overlapping inference).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
D=output/fr13_apc_drift_curve/run_20260629T000358Z            # existing run: REF + on_b1024/2048
SRC=output/fr13_apc_rategate/run_20260625T084654Z/rg_OFF_r1/proxy_pair_dumps
SEQ=49; PORT=9953; GPU_UTIL=0.82; REPLAY_TEMP=0.6; K_DECODE=8; GDN_LIMIT=80
echo "=== drift curve EXTEND  D=$D  blocks=[8192 4096]  reuse REF=$(ls "$D"/logs/ref_gdn*.pt 2>/dev/null|wc -l) gdn ==="
PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
for p in fr13 swe-agent swe-codex codex; do docker ps -aq --filter "name=$p" | xargs -r docker rm -f >/dev/null 2>&1; done

for B in 8192 4096; do
  echo "[EXTEND] boot align@b=$B (capture on_b${B}) @ $(date -u +%H:%M:%S)"
  ARM=on RUNDIR="$D" DUMPS="$SRC" LIMIT_TURNS=$((SEQ+1)) PORT="$PORT" GPU_UTIL="$GPU_UTIL" \
    ENFORCE_EAGER=1 MAMBA_BLOCK_SIZE="$B" \
    FR13_REPLAY_TEMP="$REPLAY_TEMP" \
    FR13_PREFILL_GDN_CAPTURE="/logs/on_b${B}_gdn.pt" \
    FR13_PREFILL_GDN_CAPTURE_LAYER_PREFIX='*' \
    FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX="$GDN_LIMIT" \
    FR13_FINAL_LOGIT_CAPTURE="/logs/on_b${B}_logit.pt" \
    FR13_FINAL_LOGIT_CAPTURE_NUM_TOKENS= \
    FR13_FINAL_LOGIT_CAPTURE_SKIP=0 \
    FR13_FINAL_LOGIT_CAPTURE_LIMIT="$K_DECODE" \
    bash scripts/fr13_apc_multiturn_one_arm.sh >> "$D/logs/boot_extend_b${B}.log" 2>&1
  echo "  boot b=$B rc=$? captures=$(find "$D" -name "on_b${B}_gdn*.pt" 2>/dev/null|wc -l) @ $(date -u +%H:%M:%S)"
  for p in fr13 swe-agent swe-codex codex; do docker ps -aq --filter "name=$p" | xargs -r docker rm -f >/dev/null 2>&1; done
done

echo "=== REDUCE all four blocks vs continuous REF @ $(date -u +%H:%M:%S) ==="
.venv/bin/python scripts/fr13_apc_drift_curve_reduce.py \
  --rundir "$D" --ref-gdn "$D/logs/ref_gdn" --ref-logit "$D/logs/ref_logit" \
  --blocks "1024 2048 4096 8192" --k-decode "$K_DECODE" --out "$D/drift_curve_full.jsonl" 2>&1 | tail -28
echo "=== EXTEND DONE -> $D/drift_curve_full.jsonl @ $(date -u +%H:%M:%S) ==="
