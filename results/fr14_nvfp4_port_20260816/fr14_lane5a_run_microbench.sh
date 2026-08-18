#!/bin/bash
# FR14 lane 5A: run the head-GEMM microbench inside the pinned image.
#
# No engine, no KV reservation, no model body -- only the two heads are
# resident, so the GPU window is seconds rather than minutes and the number is
# not contaminated by anything else the serve does.  Container is removed on
# exit unconditionally (GPU discipline: zero containers left).
set -u
REPO=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
RES_DIR=$REPO/results/fr14_nvfp4_port_20260816
OUT=/home/mark/shared/tmp-scratch/fr14_lane5a
IMAGE=vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
NAME=fr14_lane5a_microbench

mkdir -p "$OUT"
cleanup() { docker rm -f $NAME >/dev/null 2>&1; }
trap cleanup EXIT

if [ -n "$(docker ps -q)" ]; then
  echo "REFUSED: containers still running: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"
  exit 1
fi

docker rm -f $NAME >/dev/null 2>&1
docker run --rm --name $NAME --gpus all --network host \
  -v /home/mark/shared/models:/models:ro \
  -v "$RES_DIR":/ovl:ro \
  -v "$OUT":/cap \
  --entrypoint python3 $IMAGE \
  /ovl/fr14_lane5a_head_gemm_microbench.py "$@" 2>&1 | tee "$OUT/microbench.log"
echo "containers now: $(docker ps -aq | wc -l)"
