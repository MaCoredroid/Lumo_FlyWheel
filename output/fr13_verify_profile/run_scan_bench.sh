#!/usr/bin/env bash
# Verifier scan-row bench: tree GDN kernel at n=6 vs n=22 (row-scaling verdict,
# same question the attn bench answered: is scan row-work real or is the cost
# state-stream-bound?). Bare container, existing harness, production scale.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
until grep -q "DVKDUMP_DONE" output/fr13_msr/dvkdump_console.log 2>/dev/null; do sleep 300; done
while docker ps --format '{{.Names}}' | grep -q fr13; do sleep 30; done
IMAGE="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
docker run --rm --gpus all --entrypoint bash \
  -v "$PWD:/workspace" -w /workspace \
  --name fr13-scan-bench "$IMAGE" -c "
    export PYTHONPATH=/workspace/src
    python3 scripts/fr10_phase2_triton_tree_gdn_microbench.py --nodes 6 22 --production-scale" \
  > output/fr13_verify_profile/scan_rowbench.log 2>&1
echo "SCAN_BENCH_DONE rc=$?"
