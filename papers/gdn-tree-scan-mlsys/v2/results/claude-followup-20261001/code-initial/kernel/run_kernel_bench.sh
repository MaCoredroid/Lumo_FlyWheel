#!/usr/bin/env bash
# Host launcher (GB10) for the GDN tree-verification kernel timing + memory benchmark.
# Starts ONE container of the pinned vLLM image (the M1 / deployment image), runs in_container.sh, writes everything to
#   /home/mark/shared/lumotree-v2exp-runs/kernel/<UTC timestamp>/
# Usage:
#   bash run_kernel_bench.sh                      # full run: 6 methods x {graph,eager} + tree-size sweep (100 repeats, 20 warmup)
#   bash run_kernel_bench.sh --smoke              # quick wiring check (5 repeats, 2 warmup, results labelled SMOKE)
#   bash run_kernel_bench.sh --methods "lumo_fixed32 treewy_author_default" --no-sweep --repeats 200
#   bash run_kernel_bench.sh --dry-run            # all host checks, print the docker command, launch nothing
# Refuses (exit 3) unless the GPU is idle by the same predicate the v2exp queue uses, no queue driver is running, no compute
# process holds the GPU and >= MIN_AVAIL_GIB host memory is available. Never stops or removes containers it did not create.
set -uo pipefail
WT=${KBENCH_WT:-/home/mark/shared/lumotree-v2exp-20260930}
REVIEW=${KBENCH_REVIEW:-/home/mark/lumotree-review-20260927}
MODEL_CONFIG=${KBENCH_MODEL_CONFIG_HOST:-/models/qwen3.8-27b-nvfp4/config.json}
IMAGE_ID=sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc            # = vllm/vllm-openai@sha256:3dbe092e... (M1 + deployment)
IMAGE_DIGEST=vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
ROOT=/home/mark/shared/lumotree-v2exp-runs/kernel
TIMEOUT_S=${KBENCH_TIMEOUT_S:-10800}
MIN_AVAIL_GIB=${KBENCH_MIN_AVAIL_GIB:-24}
METHODS="lumo_fixed32 weaver_author_default weaver_aligned_local treewy_author_default naive_native_paths naive_torch_node"
SWEEP="lumo_generic authors naive"
REPEATS=100; WARMUP=20; SWEEP_REPEATS=50; SWEEP_WARMUP=10; SMOKE=0; DRY=0; EXTRA=""; ALLOW_QUEUE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; REPEATS=5; WARMUP=2; SWEEP_REPEATS=5; SWEEP_WARMUP=2; shift ;;
    --methods) METHODS="$2"; shift 2 ;;
    --no-sweep) SWEEP=""; shift ;;
    --sweep) SWEEP="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --extra) EXTRA="$2"; shift 2 ;;          # extra kernel_bench.py args, e.g. "--stratum mixed-stress"
    --allow-queue-running) ALLOW_QUEUE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
[[ $SMOKE == 1 || $REPEATS -ge 50 ]] || { echo "REFUSED: --repeats $REPEATS < 50 (use --smoke for a labelled quick run)"; exit 2; }
SMOKE_FLAG=""; [[ $SMOKE == 1 ]] && SMOKE_FLAG="--smoke"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN=$ROOT/$TS
NAME=lumo-kbench-$TS            # must NOT contain "v2exp": queue_v2.sh removes containers matching name=v2exp
refuse() { echo "REFUSED: $*"; exit 3; }

# ---- idle / contention checks (read-only) -------------------------------------------------------------------------------
[[ -z "$(docker ps -q)" ]] || refuse "a container is running: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"
DRV=$(ps -eo args | awk '$1=="bash" && $2 ~ /(run_native_arm|run_tree_arm|run_swe_native_arm|run_swe_tree_arm|promoab_[a-z0-9_]*)\.sh$/' | wc -l)
[[ $DRV == 0 ]] || refuse "$DRV v2exp arm driver(s) running"
if [[ $ALLOW_QUEUE == 0 ]]; then
  QN=$(ps -eo args | awk '$1=="bash" && $2 ~ /(queue|queue_v2)\.sh$/' | wc -l)
  [[ $QN == 0 ]] || refuse "a v2exp queue driver is running (it starts its next step 20 s after seeing an idle GPU); stop it or pass --allow-queue-running"
fi
APPS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v '^$' || true)
[[ -z "$APPS" ]] || refuse "GPU compute processes present: $APPS"
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
[[ -z "$UTIL" || $UTIL -le 5 ]] || refuse "GPU utilization ${UTIL}% > 5%"
AVAIL=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo)
[[ $AVAIL -ge $MIN_AVAIL_GIB ]] || refuse "MemAvailable ${AVAIL} GiB < ${MIN_AVAIL_GIB} GiB (GB10 unified memory; no reclaim is attempted here)"
GOT_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE_ID" 2>/dev/null)
[[ "$GOT_ID" == "$IMAGE_ID" ]] || refuse "pinned image $IMAGE_ID not present (got '$GOT_ID')"
docker image inspect -f '{{json .RepoDigests}}' "$IMAGE_ID" | grep -q "${IMAGE_DIGEST#*@}" || refuse "image $IMAGE_ID does not carry $IMAGE_DIGEST"
for f in "$WT/scripts/v2exp/kernel/in_container.sh" "$WT/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py" "$MODEL_CONFIG" \
         "$REVIEW/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/m1/m1_adapters_v3.py" \
         "$REVIEW/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/q1_component_runner_v2_2.py"; do
  [[ -e "$f" ]] || refuse "missing $f"
done

ENVS=(-e FR13_FIXED32_MODE=hydra27_fixed32 -e FR13_SUBTREE_PARALLEL=1 -e FR13_SCAN_ALIGN=0 -e FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 -e FR13_RING_EXPORT=1
      -e FR13_TREE_RUNROW_INIT=1 -e FR13_FLAGS_INKERNEL=1 -e FR13_FIXED32_COMMIT_DEVICE_FILL=1 -e FR13_FIXED32_KV_REMAP16=1
      -e FR13_FIXED32_COMMITTER_LAYER_BATCH=0 -e FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION=0 -e FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=0
      -e FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 -e FR13_FIXED32_CONV_COMMIT_ZERO_TAIL=0
      -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e PYTHONDONTWRITEBYTECODE=1 -e KBENCH_IMAGE_ID="$IMAGE_ID"
      -e KBENCH_WORKSPACE=/workspace -e KBENCH_REVIEW_ROOT=/review -e KBENCH_MODEL_CONFIG=/model_config.json -e KBENCH_OUT=/out
      -e KBENCH_METHODS="$METHODS" -e KBENCH_SWEEP="$SWEEP" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)"
      -e KBENCH_ARGS="--repeats $REPEATS --warmup $WARMUP $SMOKE_FLAG $EXTRA"
      -e KBENCH_SWEEP_ARGS="--repeats $SWEEP_REPEATS --warmup $SWEEP_WARMUP $SMOKE_FLAG")
RUN_ARGV=(docker run -d --name "$NAME" --label lumo.kbench="$TS" --gpus all --ipc=host --network none "${ENVS[@]}"
          -v "$WT:/workspace:ro" -v "$REVIEW:/review:ro" -v "$MODEL_CONFIG:/model_config.json:ro" -v "$RUN:/out"
          --entrypoint bash "$IMAGE_ID" /workspace/scripts/v2exp/kernel/in_container.sh)
if [[ $DRY == 1 ]]; then printf '%q ' "${RUN_ARGV[@]}"; echo; echo "dry run: all host checks passed; nothing launched"; exit 0; fi

mkdir -p "$RUN"
exec > >(tee -a "$RUN/launcher.log") 2>&1
echo "run=$RUN container=$NAME image=$IMAGE_ID smoke=$SMOKE methods=[$METHODS] sweep=[$SWEEP] repeats=$REPEATS warmup=$WARMUP"
{
  echo "{\"utc\": \"$TS\", \"worktree_head\": \"$(git -C "$WT" rev-parse HEAD)\", \"review_head\": \"$(git -C "$REVIEW" rev-parse HEAD 2>/dev/null)\","
  echo " \"image_id\": \"$IMAGE_ID\", \"image_digest\": \"$IMAGE_DIGEST\", \"host_mem_available_gib\": $AVAIL, \"gpu_util_before\": \"$UTIL\","
  echo " \"kernel_dir_sha256\": \"$(cd "$WT/scripts/v2exp/kernel" && find . -type f \( -name '*.py' -o -name '*.sh' \) | sort | xargs sha256sum | sha256sum | cut -d' ' -f1)\"}"
} > "$RUN/launch.json"
printf '%q ' "${RUN_ARGV[@]}" > "$RUN/docker_run.txt"; echo >> "$RUN/docker_run.txt"
CID=$("${RUN_ARGV[@]}") || { echo "docker run failed"; exit 4; }
echo "$CID" > "$RUN/container.cid"
cleanup() {
  docker logs "$CID" > "$RUN/container.log" 2>&1
  docker inspect -f '{{.State.ExitCode}} {{.State.Status}}' "$CID" > "$RUN/container_state.txt" 2>&1
  docker rm -f "$CID" > /dev/null 2>&1 && echo "removed own container $NAME"
}
trap cleanup EXIT
t0=$(date +%s)
while [[ "$(docker inspect -f '{{.State.Running}}' "$CID" 2>/dev/null)" == "true" ]]; do
  sleep 15
  if (( $(date +%s) - t0 > TIMEOUT_S )); then echo "TIMEOUT after ${TIMEOUT_S}s; stopping own container"; docker stop -t 30 "$CID" > /dev/null; break; fi
done
RC=$(docker inspect -f '{{.State.ExitCode}}' "$CID" 2>/dev/null || echo 99)
echo "container exit=$RC elapsed_s=$(( $(date +%s) - t0 ))"
cat "$RUN/logs/rc.txt" 2>/dev/null
[[ -f "$RUN/summary.md" ]] && echo "summary: $RUN/summary.md"
exit "$RC"
