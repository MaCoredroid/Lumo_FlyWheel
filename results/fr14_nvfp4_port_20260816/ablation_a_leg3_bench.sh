#!/usr/bin/env bash
# FR14 leg 3: drive OUR fixed32 engine with THEIR bench client on THEIR shape.
set -uo pipefail
OUT=/home/mark/shared/tmp-scratch/fr14_ablation_a/leg3
PORT=9950
IMG=lmsysorg/sglang:qwen38-27b
BENCH=(python3 -m sglang.bench_serving --backend vllm
  --host 127.0.0.1 --port "$PORT"
  --model qwen3.8-27b-nvfp4 --tokenizer /models/qwen3.8-27b-nvfp4
  --dataset-name random --random-input-len 1024 --random-output-len 1024
  --random-range-ratio 1 --disable-tqdm
  --extra-request-body '{"temperature":0.6,"top_p":0.95,"top_k":20}')

curl -fsS "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics_pre_bs1.txt"
echo "[bench] bs=1 start $(date -u +%H:%M:%SZ)"
docker run --rm --network host -v /home/mark/shared/models:/models "$IMG" \
  "${BENCH[@]}" --num-prompts 8 --max-concurrency 1 > "$OUT/bench_bs1.log" 2>&1
echo "[bench] bs=1 rc=$? $(date -u +%H:%M:%SZ)"
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics_post_bs1.txt"

echo "[bench] bs=8 start $(date -u +%H:%M:%SZ)"
docker run --rm --network host -v /home/mark/shared/models:/models "$IMG" \
  "${BENCH[@]}" --num-prompts 16 --max-concurrency 8 > "$OUT/bench_bs8.log" 2>&1
echo "[bench] bs=8 rc=$? $(date -u +%H:%M:%SZ)"
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics_post_bs8.txt"
echo "[bench] done"
