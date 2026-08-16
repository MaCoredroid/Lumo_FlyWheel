#!/bin/bash
# FR14 sglang native calibration: reproduce the ~38 TPS DGX Spark claim on this machine.
# Serves RadixArk AS-SHIPPED (pre-surgery configs) in their container with their recipe,
# then runs their bench_serving at ISL/OSL 1024/1024, bs=1 and bs-8 sweeps. Evidence-first.
set -u
OUT=/home/mark/shared/tmp-scratch/nvfp4_port
SRC=/home/mark/shared/models/qwen3.8-27b-nvfp4-radixark
AS=/home/mark/shared/models/qwen3.8-27b-nvfp4-radixark-asshipped
IMG=lmsysorg/sglang:qwen38-27b
NAME=fr14_sglang_calib
PORT=30000
log(){ echo "[$(date -u +%H:%M:%S)] $*" >> $OUT/sglang_calib.log; }

# as-shipped view: hardlink everything, then restore pre-surgery configs
rm -rf $AS && mkdir -p $AS
for f in $SRC/*; do ln "$f" "$AS/$(basename $f)" 2>/dev/null || cp -a "$f" "$AS/"; done
[ -f $SRC/config.json.pre_kv_surgery.bak ] && cp -a $SRC/config.json.pre_kv_surgery.bak $AS/config.json
[ -f $SRC/hf_quant_config.json.pre_kv_surgery.bak ] && cp -a $SRC/hf_quant_config.json.pre_kv_surgery.bak $AS/hf_quant_config.json
rm -f $AS/*.bak $AS/.lumo_* 2>/dev/null
log "as-shipped view built: $(ls $AS | wc -l) files"

sudo -n sysctl vm.drop_caches=3 >/dev/null 2>&1
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --gpus all --network host --shm-size 16g \
  -v /home/mark/shared/models:/models \
  $IMG python3 -m sglang.launch_server \
  --model-path /models/qwen3.8-27b-nvfp4-radixark-asshipped \
  --served-model-name qwen38-calib \
  --host 127.0.0.1 --port $PORT \
  --trust-remote-code \
  --attention-backend flashinfer \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.70 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder >> $OUT/sglang_calib.log 2>&1

BOOT=0
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { BOOT=1; break; }
  docker ps -q -f name=$NAME | grep -q . || break
  sleep 10
done
docker logs $NAME > $OUT/sglang_calib_container.log 2>&1
if [ $BOOT -ne 1 ]; then
  log "SGLANG BOOT FAILED"; echo '{"verdict":"SGLANG_BOOT_FAILED"}' > $OUT/sglang_calib_result.json
  docker rm -f $NAME >/dev/null 2>&1; exit 1
fi
log "sglang healthy; bench bs=1 then bs=8"

docker exec $NAME python3 -m sglang.bench_serving --backend sglang \
  --host 127.0.0.1 --port $PORT --model /models/qwen3.8-27b-nvfp4-radixark-asshipped \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --random-range-ratio 1 --num-prompts 8 --max-concurrency 1 \
  --output-file /tmp/bench_bs1.jsonl > $OUT/sglang_bench_bs1.log 2>&1
docker exec $NAME python3 -m sglang.bench_serving --backend sglang \
  --host 127.0.0.1 --port $PORT --model /models/qwen3.8-27b-nvfp4-radixark-asshipped \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --random-range-ratio 1 --num-prompts 32 --max-concurrency 8 \
  --output-file /tmp/bench_bs8.jsonl > $OUT/sglang_bench_bs8.log 2>&1
docker cp $NAME:/tmp/bench_bs1.jsonl $OUT/ 2>/dev/null
docker cp $NAME:/tmp/bench_bs8.jsonl $OUT/ 2>/dev/null
docker logs $NAME > $OUT/sglang_calib_container.log 2>&1
grep -A20 'Serving Benchmark Result' $OUT/sglang_bench_bs1.log | head -24 > $OUT/sglang_calib_summary.txt
grep -A20 'Serving Benchmark Result' $OUT/sglang_bench_bs8.log | head -24 >> $OUT/sglang_calib_summary.txt
docker rm -f $NAME >/dev/null 2>&1
log "calibration complete, teardown done"
echo '{"verdict":"DONE","summary":"sglang_calib_summary.txt"}' > $OUT/sglang_calib_result.json
