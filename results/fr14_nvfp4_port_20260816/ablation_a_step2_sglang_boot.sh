#!/usr/bin/env bash
# FR14 ABLATION ARM A / STEP 2 (diagnostic, non-citable): serve THEIR stack
# (lmsysorg/sglang:qwen38-27b + RadixArk AS-SHIPPED + their Spark recipe + their
# parsers) on port 9950 so the SAME SWE harness that drove our engine can drive
# theirs. Recipe copied verbatim from results/fr14_nvfp4_port_20260816/sglang_calibration.sh
# with three deltas, all required by the harness topology and documented:
#   1. --host 0.0.0.0 (was 127.0.0.1) so the alienware agent-offload proxy can reach it
#   2. --port 9950    (was 30000)     so the harness's DEFAULT_METRICS_URL brackets it
#   3. --enable-metrics                so per-task token counters exist to bracket
set -uo pipefail
OUT=${OUT:-/home/mark/shared/tmp-scratch/fr14_ablation_a/step2}
mkdir -p "$OUT"
AS=/home/mark/shared/models/qwen3.8-27b-nvfp4-radixark-asshipped
IMG=lmsysorg/sglang:qwen38-27b
NAME=fr14_step2_sglang
PORT=9950

[[ -d "$AS" ]] || { echo "FAIL: as-shipped view missing: $AS"; exit 2; }
echo "[step2] as-shipped view: $(ls "$AS" | wc -l) files"

sudo -n bash -lc "sync; echo 3 > /proc/sys/vm/drop_caches; swapoff -a || true; swapon -a || true; echo 3 > /proc/sys/vm/drop_caches" >/dev/null 2>&1
free -g | tee "$OUT/free_before_boot.txt"
docker rm -f "$NAME" >/dev/null 2>&1

docker run -d --name "$NAME" --gpus all --network host --shm-size 16g \
  -v /home/mark/shared/models:/models \
  "$IMG" python3 -m sglang.launch_server \
  --model-path /models/qwen3.8-27b-nvfp4-radixark-asshipped \
  --served-model-name qwen3.8-27b-nvfp4-radixark \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  --attention-backend flashinfer \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.70 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-metrics > "$OUT/docker_run.log" 2>&1

BOOT=0
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { BOOT=1; break; }
  docker ps -q -f "name=$NAME" | grep -q . || break
  sleep 5
done
docker logs "$NAME" > "$OUT/sglang_boot.log" 2>&1
echo "[step2] boot=$BOOT after $((i*5))s"
if (( BOOT != 1 )); then tail -40 "$OUT/sglang_boot.log"; exit 3; fi
docker exec "$NAME" bash -lc 'tr "\0" " " < /proc/1/cmdline' > "$OUT/sglang_cmdline.txt" 2>&1
curl -s "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json"
echo "[step2] sglang HEALTHY on :$PORT"; cat "$OUT/models.json"; echo
echo "[step2] smoke chat:"
curl -s -m 180 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-nvfp4-radixark","messages":[{"role":"user","content":"Say exactly: FR14 sglang alive."}],"max_tokens":64,"temperature":0.6,"top_p":0.95}' \
  | tee "$OUT/smoke_chat.json" | head -c 500; echo
