#!/usr/bin/env bash
# EXACT_SEED state-diff test (the UNCONFOUNDED metric — not the char-8/solve verdict that
# previously judged EXACT_SEED). Boot align@1024 + FR13_APC_EXACT_SEED=1, reuse the existing
# 193-capture continuous (cfg, no-cache) REF, reduce the per-layer GDN ssm_state diff.
# Baseline no-fix @1024 = state_max 77.96. If EXACT_SEED is the structural fix: the @1024 drift
# collapses toward fp (and is block-INDEPENDENT) — the realization-mismatch thin tail goes away.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
SRC=output/fr13_apc_rategate/run_20260625T084654Z/rg_OFF_r1/proxy_pair_dumps
OLD_REF=output/fr13_apc_drift_curve/run_20260629T000358Z/logs
SEQ=49; PORT=9953; GPU_UTIL=0.82; REPLAY_TEMP=0.6; K_DECODE=8; GDN_LIMIT=80
B=${B:-1024}
TS=$(date -u +%Y%m%dT%H%M%SZ)
RD=output/fr13_apc_exactseed_statediff/run_${TS}; mkdir -p "$RD/logs"
echo "$RD" > /home/mark/.claude/jobs/22c39bb9/tmp/eseed_root.txt
# symlink the REF (hardlinks are "Operation not permitted" on this fs; full cp = 100G/boot):
# instant, host-side reduce follows the symlinks (container only WRITES on_b*, never reads ref_*).
ln -s "$PWD/$OLD_REF"/ref_gdn*.pt "$RD/logs/" 2>/dev/null || cp "$OLD_REF"/ref_gdn*.pt "$RD/logs/" 2>/dev/null
ln -s "$PWD/$OLD_REF"/ref_logit*.pt "$RD/logs/" 2>/dev/null || cp "$OLD_REF"/ref_logit*.pt "$RD/logs/" 2>/dev/null
echo "=== EXACT_SEED STATE-DIFF  rundir=$RD  REF=$(ls "$RD/logs"/ref_gdn*.pt 2>/dev/null|wc -l) gdn  block=$B + EXACT_SEED=1 ==="
PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
for p in fr13 swe-codex codex; do docker ps -aq --filter "name=$p" | xargs -r docker rm -f >/dev/null 2>&1; done

echo "[ESEED] boot align@b=$B with FR13_APC_EXACT_SEED=1 @ $(date -u +%H:%M:%S)"
ARM=on RUNDIR="$RD" DUMPS="$SRC" LIMIT_TURNS=$((SEQ+1)) PORT="$PORT" GPU_UTIL="$GPU_UTIL" \
  ENFORCE_EAGER=1 MAMBA_BLOCK_SIZE="$B" FR13_APC_EXACT_SEED=1 \
  FR13_REPLAY_TEMP="$REPLAY_TEMP" \
  FR13_PREFILL_GDN_CAPTURE="/logs/on_b${B}_gdn.pt" \
  FR13_PREFILL_GDN_CAPTURE_LAYER_PREFIX='*' \
  FR13_PREFILL_GDN_CAPTURE_LIMIT_PER_PREFIX="$GDN_LIMIT" \
  FR13_FINAL_LOGIT_CAPTURE="/logs/on_b${B}_logit.pt" \
  FR13_FINAL_LOGIT_CAPTURE_NUM_TOKENS= \
  FR13_FINAL_LOGIT_CAPTURE_SKIP=0 \
  FR13_FINAL_LOGIT_CAPTURE_LIMIT="$K_DECODE" \
  bash scripts/fr13_apc_multiturn_one_arm.sh >> "$RD/logs/boot_eseed.log" 2>&1
echo "boot rc=$? captures=$(find "$RD" -name "on_b${B}_gdn*.pt" 2>/dev/null|wc -l) @ $(date -u +%H:%M:%S)"

echo "=== EXACT_SEED ENGAGEMENT canary (did the chunked-ckpt chain fire? the non-vacuity check) ==="
grep -aoE "EXACT_SEED=[A-Za-z0-9]+|chunked_ckpt|GAP-[abc]|exact.?seed.*(publish|fired|engage|chain)|EXACT_SEED_ENG" "$RD/logs/boot_eseed.log" 2>/dev/null | sort | uniq -c | head -12 || echo "  (none — check engagement)"

echo "=== REDUCE vs continuous REF (baseline no-fix @${B}=77.96 / 49.70@2048) ==="
.venv/bin/python scripts/fr13_apc_drift_curve_reduce.py \
  --rundir "$RD" --ref-gdn "$RD/logs/ref_gdn" --ref-logit "$RD/logs/ref_logit" \
  --blocks "$B" --k-decode "$K_DECODE" --out "$RD/eseed_statediff.jsonl" 2>&1 | tail -22
echo "=== ESEED STATE-DIFF DONE -> $RD/eseed_statediff.jsonl @ $(date -u +%H:%M:%S) ==="
