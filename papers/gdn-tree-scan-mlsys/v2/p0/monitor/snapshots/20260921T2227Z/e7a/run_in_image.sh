#!/usr/bin/env bash
# Run an E7a python entrypoint INSIDE the pinned vLLM image with the worktree, the read-only
# historical archive, and the read-only model directory mounted. Host-side GPU utilization is
# sampled at 1 Hz for the whole container lifetime (contention record; the unrelated server on
# this GB10 is never touched). Usage:
#   run_in_image.sh <out_dir> <python-file> [args...]
set -euo pipefail
IMAGE_DIGEST="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
WORKTREE=/home/mark/lumo-paper-v2-20260921
HIST=/home/mark/shared/lumoFlyWheel/output
MODELS=/home/mark/shared/models
OUT_DIR=$1; shift
ENTRY=$1; shift
mkdir -p "$OUT_DIR/triton_cache" "$OUT_DIR/telemetry"
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{
  echo "start_utc=$START_UTC"
  echo "--- compute apps (before) ---"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
  echo "--- procs (before) ---"; { ps -eo pid,etime,rss,args | grep -i "vllm\|EngineCore" | grep -v grep | cut -c1-240; } || echo "(no vllm/EngineCore process)"
  free -g || true
} > "$OUT_DIR/telemetry/contention_before.txt" 2>&1 || true
# 1 Hz GPU utilization / memory sampler for the whole window (GB10 reports memory as N/A; util is the signal)
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm --format=csv -l 1 > "$OUT_DIR/telemetry/gpu_util_1hz.csv" 2>&1 &
SAMPLER=$!
# timestamped COMPUTE-PROCESS inventory at 1 Hz: attributes device activity to our container's python vs the
# unrelated server (aggregate utilization alone cannot separate them)
( while true; do nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory --format=csv,noheader || echo "$(date -u +%Y/%m/%d\ %H:%M:%S.000), none, none, none"; sleep 1; done ) > "$OUT_DIR/telemetry/compute_procs_1hz.csv" 2>&1 &
PSAMPLER=$!
trap 'kill $SAMPLER $PSAMPLER 2>/dev/null || true' EXIT
REL_OUT=${OUT_DIR#"$WORKTREE"/}
REL_ENTRY=${ENTRY#"$WORKTREE"/}
set +e
docker run --rm -i --gpus all --ipc=host \
  --name "e7a-$(date -u +%H%M%S)" \
  -v "$WORKTREE":/work \
  -v "$HIST":/hist:ro \
  -v "$MODELS":/models:ro \
  -e TRITON_CACHE_DIR=/work/"$REL_OUT"/triton_cache \
  -e E7A_IMAGE_DIGEST="$IMAGE_DIGEST" \
  -e E7A_OUT_DIR=/work/"$REL_OUT" \
  -e E7A_START_UTC="$START_UTC" \
  -e PYTHONUNBUFFERED=1 \
  -w /work/papers/gdn-tree-scan-mlsys/v2/experiments/e7a \
  --entrypoint python3 "$IMAGE_DIGEST" /work/"$REL_ENTRY" "$@"
RC=$?
set -e
kill $SAMPLER $PSAMPLER 2>/dev/null || true
{
  echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$RC"
  echo "--- compute apps (after) ---"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
  echo "--- procs (after) ---"; { ps -eo pid,etime,rss,args | grep -i "vllm\|EngineCore" | grep -v grep | cut -c1-240; } || echo "(no vllm/EngineCore process)"
} > "$OUT_DIR/telemetry/contention_after.txt" 2>&1 || true
echo "run_in_image: rc=$RC out=$OUT_DIR"
exit $RC
