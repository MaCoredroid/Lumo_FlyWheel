#!/usr/bin/env bash
# FR14 ABLATION ARM A / STEP 2: drive the SAME 4 SWE tasks with the SAME qwen-code
# agent harness, but at THEIR engine (sglang + RadixArk as-shipped + their parsers).
# Topology is identical to the stock B1 arm: agent+proxy on alienware, GB10 serves
# the engine only, so the decode TPS is uncontended.
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
cd "$REPO"
OUT=/home/mark/shared/tmp-scratch/fr14_ablation_a/step2
mkdir -p "$OUT/swe_out"

export SWE_AGENT=qwen_code
export SWE_AGENT_ENV=instance_image
export SWE_EMPTY_PATCH_RETRIES=0
export LUMO_SWE_AUTOCOMMIT=0
export LUMO_SWE_STALL_KILL_S=900
export HF_HUB_OFFLINE=0

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/swe_started_at.txt"
curl -fsS http://127.0.0.1:9950/metrics > "$OUT/sglang_metrics_pre.txt" 2>/dev/null

.venv/bin/python scripts/run_swe_bench_q36_a.py \
  --subset config/fr13_fixed32/subset_b4_four.json \
  --out-root "$OUT/swe_out" \
  --concurrency 1 \
  --agent-wall-s 5400 \
  --eval-timeout-s 1800 \
  --model qwen3.8-27b-nvfp4-radixark \
  --model-name "qwen3.8-27b-nvfp4-radixark::sglang-eagle3-1-4::qwen-code-0.19.4::fr14-ablA" \
  --agent-host alienware --agent-endpoint http://127.0.0.1:8023/v1 \
  --eval-host alienware \
  > "$OUT/swe_orchestrator.log" 2>&1
echo "swe rc=$?" | tee "$OUT/swe_rc.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/swe_ended_at.txt"
curl -fsS http://127.0.0.1:9950/metrics > "$OUT/sglang_metrics_post.txt" 2>/dev/null
