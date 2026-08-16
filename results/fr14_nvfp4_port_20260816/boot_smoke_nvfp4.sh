#!/bin/bash
# FR14 stock-boot smoke: first-ever Qwen3.8-27B NVFP4 serve on this box.
# Waits for the unsloth download to finish, boots the PINNED production image
# with plain `vllm serve` (no fixed32, no spec config, eager mode), sends real
# chat completions, records everything, tears down. Evidence-first.
set -u
OUT=/home/mark/shared/tmp-scratch/nvfp4_port
MODEL_DIR=/home/mark/shared/models/qwen3.8-27b-nvfp4
IMAGE=vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
PORT=8199
NAME=fr14_nvfp4_smoke
RES=$OUT/boot_smoke_result.json
log() { echo "[$(date -u +%H:%M:%S)] $*" >> $OUT/boot_smoke.log; }

log "waiting for unsloth download to finish"
for i in $(seq 1 240); do
  pgrep -f 'download unsloth' > /dev/null || break
  sleep 30
done
if pgrep -f 'download unsloth' > /dev/null; then
  echo '{"verdict":"DOWNLOAD_TIMEOUT"}' > $RES; exit 1
fi
sleep 5
if [ ! -f $MODEL_DIR/model.safetensors ] || find $MODEL_DIR/.cache -name '*.incomplete' 2>/dev/null | grep -q .; then
  echo '{"verdict":"DOWNLOAD_INCOMPLETE_OR_FAILED"}' > $RES; exit 1
fi
log "download complete: $(stat -c%s $MODEL_DIR/model.safetensors) bytes main shard"

log "booting $NAME (eager, gpu-mem 0.70, max-len 32768)"
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --gpus all --network host \
  -v /models:/models \
  --entrypoint vllm $IMAGE serve /models/qwen3.8-27b-nvfp4 \
  --served-model-name qwen3.8-27b-nvfp4-smoke \
  --host 127.0.0.1 --port $PORT \
  --max-model-len 32768 --max-num-seqs 4 \
  --gpu-memory-utilization 0.70 \
  --enforce-eager >> $OUT/boot_smoke.log 2>&1

BOOT_OK=0
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { BOOT_OK=1; break; }
  docker ps -q -f name=$NAME | grep -q . || break
  sleep 10
done
docker logs $NAME > $OUT/boot_smoke_container.log 2>&1

if [ $BOOT_OK -ne 1 ]; then
  log "BOOT FAILED"
  echo '{"verdict":"BOOT_FAILED_OR_TIMEOUT","container_log":"boot_smoke_container.log"}' > $RES
  docker rm -f $NAME >/dev/null 2>&1
  exit 1
fi
log "health OK, sending smoke requests"

R1=$(curl -s -m 420 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen3.8-27b-nvfp4-smoke",
  "messages":[{"role":"user","content":"Reply with exactly: FR14 NVFP4 alive. Then state 17*23."}],
  "max_tokens":512,"temperature":0.6,"top_p":0.95,"top_k":20}')
R2=$(curl -s -m 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen3.8-27b-nvfp4-smoke",
  "messages":[{"role":"user","content":"Write a python function that reverses a linked list, then explain its invariant in two sentences."}],
  "max_tokens":4096,"temperature":0.6,"top_p":0.95,"top_k":20}')
echo "$R1" > $OUT/smoke_r1.json
echo "$R2" > $OUT/smoke_r2.json
docker logs $NAME > $OUT/boot_smoke_container.log 2>&1
docker rm -f $NAME >/dev/null 2>&1
log "teardown complete"

python3 - "$RES" <<'EOF'
import json, sys
out = {"verdict": "UNKNOWN"}
try:
    r1 = json.load(open("/home/mark/shared/tmp-scratch/nvfp4_port/smoke_r1.json"))
    r2 = json.load(open("/home/mark/shared/tmp-scratch/nvfp4_port/smoke_r2.json"))
    c1 = r1["choices"][0]["message"]
    c2 = r2["choices"][0]["message"]
    t1 = (c1.get("content") or "") + (c1.get("reasoning_content") or "")
    t2 = (c2.get("content") or "") + (c2.get("reasoning_content") or "")
    out["r1_len"], out["r2_len"] = len(t1), len(t2)
    out["r1_tail"] = t1[-200:]
    out["r1_has_391"] = "391" in t1
    words = t2.split()
    out["r2_degenerate"] = len(words) > 40 and len(set(words)) < len(words) * 0.25
    out["usage_r2"] = r2.get("usage")
    ok = out["r1_len"] > 0 and out["r2_len"] > 0 and not out["r2_degenerate"]
    out["verdict"] = "PASS" if ok else "SUSPECT_OUTPUT"
except Exception as e:
    out["verdict"] = "SMOKE_REQUEST_FAILED"
    out["error"] = str(e)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(out["verdict"])
EOF
log "verdict written"
