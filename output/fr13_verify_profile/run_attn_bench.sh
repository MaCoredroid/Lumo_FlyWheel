#!/usr/bin/env bash
# Verifier V2 microbench v2: after cg_combo gate + teardown, run the bench in a
# bare container (no vLLM serve, no model load — just the fork's attn binding).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
until grep -q "CG_COMBO_GATE_DONE" output/fr13_verify_profile/cg_combo/gate_console.log 2>/dev/null; do sleep 60; done
while docker ps --format '{{.Names}}' | grep -q fr13; do sleep 30; done
IMAGE="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
mkdir -p output/fr13_verify_profile/attn_bench_logs
docker run --rm --gpus all --entrypoint python3 \
  -v "$PWD:/workspace" -v "$PWD/output/fr13_verify_profile/attn_bench_logs:/logs" \
  --name fr13-attn-bench "$IMAGE" \
  /workspace/scripts/fr13_attn_mgeom_bench.py \
  > output/fr13_verify_profile/attn_mgeom_bench.log 2>&1
echo "ATTN_BENCH_DONE rc=$?"
