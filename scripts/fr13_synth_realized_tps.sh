#!/usr/bin/env bash
# FR13 Stage D: SYNTHETIC realized decode-TPS probe (no proxy, direct to vLLM).
# Measures the REALIZED streaming decode rate (includes the inter-step GAP) +
# the /metrics-derived s/fwd + accept/event -> wall/step + gap. The deploy showed
# cat6 wall/step 0.260 but s/fwd 0.138 => 0.122 GAP (where the +28ms vs E5 lives).
# This probe measures that gap WITHOUT the proxy: if the synthetic gap is small,
# the deploy gap is the proxy/streaming (per-committed-token), not vLLM-internal.
# Usage: fr13_synth_realized_tps.sh <cat6|e5>
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
REPO=/home/mark/shared/lumoFlyWheel
ARM=${1:?usage: <cat6|e5>}
PORT=9952
CONTAINER=fr13-synth-$ARM
OUT=output/fr13_synth_tps
mkdir -p "$OUT"
CAT6ROOT_TREE="[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]"

recover(){ PYTHONPATH="$REPO/src" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}
teardown(){ docker logs "$CONTAINER" >"$OUT/${ARM}_docker.log" 2>&1 || true; docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; recover; }
trap teardown EXIT

echo "[1/4] hygiene"; recover; [ -z "$(docker ps -q)" ] || { echo FAIL-not-empty; docker ps; exit 2; }

echo "[2/4] boot $ARM"
if [ "$ARM" = cat6 ]; then
  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS=1 \
    TREE="$CAT6ROOT_TREE" FR10_METRICS=0 BATCH_INVARIANT=0 \
    LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
    LOG_DIR="$REPO/$OUT/logs_$ARM" \
    scripts/fr13_launch_forked_fa2_tree_server.sh > "$OUT/${ARM}_launch.log" 2>&1
else
  # native E5: MTP-5 chain, FLASH_ATTN, tree path OFF (mirrors fr13_bigdenom_swe_serve.sh native)
  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS=1 \
    ATTENTION_BACKEND=FLASH_ATTN FR10_ENABLE_TREE_GDN=0 \
    SPEC_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":5}' \
    LOG_DIR="$REPO/$OUT/logs_$ARM" \
    scripts/fr10_launch_speed_server.sh > "$OUT/${ARM}_launch.log" 2>&1
fi
RC=$?; (( RC==0 )) || { echo "FAIL launcher rc=$RC"; tail -25 "$OUT/${ARM}_launch.log"; exit 2; }

echo "[3/4] wait health"
T0=$(date +%s); H=0
while (( $(date +%s) < T0+1200 )); do
  curl -fsS -m3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { H=1; break; }
  [ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" = running ] || { echo FAIL-died; docker logs "$CONTAINER" 2>&1|tail -30; exit 2; }
  sleep 5
done
(( H==1 )) || { echo FAIL-health; exit 2; }
echo "healthy after $(( $(date +%s)-T0 ))s"
docker exec "$CONTAINER" env 2>/dev/null | grep -iE '^(FR13_DEVICE_MULTIDRAFT|FR10_ENABLE_TREE_GDN|SPEC_CONFIG|FR13_FA2_TREE_BIAS)=' | sed 's/^/  env: /'

echo "[4/4] measure realized streaming decode-TPS + gap"
.venv/bin/python - "http://127.0.0.1:$PORT" "qwen3.6-27b" "$ARM" <<'PY' | tee "$OUT/${ARM}_result.txt"
import time, json, urllib.request, sys
URL, MODEL, ARM = sys.argv[1], sys.argv[2], sys.argv[3]
KEYS=["vllm:spec_decode_num_drafts_total","vllm:spec_decode_num_accepted_tokens_total",
      "vllm:generation_tokens_total","vllm:request_decode_time_seconds_sum"]
def metrics():
    txt=urllib.request.urlopen(URL+"/metrics",timeout=15).read().decode()
    m={k:0.0 for k in KEYS}
    for line in txt.splitlines():
        for k in KEYS:
            if line.startswith(k+" ") or line.startswith(k+"{"):
                try: m[k]+=float(line.rsplit(" ",1)[1])
                except: pass
    return m
def stream(maxtok):
    body=json.dumps({"model":MODEL,"prompt":"Explain in detail, step by step, how a modern out-of-order superscalar CPU fetches, renames, schedules, executes and retires instructions, with concrete examples:","max_tokens":maxtok,"temperature":0.6,"seed":1313,"stream":True}).encode()
    req=urllib.request.Request(URL+"/v1/completions",data=body,headers={"Content-Type":"application/json"})
    tf=None;n=0
    with urllib.request.urlopen(req,timeout=180) as r:
        for raw in r:
            s=raw.decode("utf-8","ignore").strip()
            if not s.startswith("data: "): continue
            p=s[6:]
            if p=="[DONE]": break
            try: d=json.loads(p)
            except: continue
            if d["choices"][0].get("text",""):
                if tf is None: tf=time.time()
                n+=1
    return n,(time.time()-tf) if tf else 0.0
stream(40)  # warmup
a=metrics(); N=0; DT=0.0
for _ in range(4):
    n,dt=stream(400); N+=n; DT+=dt
b=metrics()
dD=b[KEYS[0]]-a[KEYS[0]]; dA=b[KEYS[1]]-a[KEYS[1]]; dG=b[KEYS[2]]-a[KEYS[2]]; dRT=b[KEYS[3]]-a[KEYS[3]]
rtps=N/DT if DT else 0
acc=dA/dD if dD else 0; comm=dG/dD if dD else 0; sfwd=dRT/dD if dD else 0
wall=comm/rtps if rtps else 0; gap=wall-sfwd
print(f"ARM={ARM}")
print(f"  realized_decode_tps = {rtps:.2f}   (client-side streamed tokens/sec, INCLUDES the gap)")
print(f"  accept/event        = {acc:.3f}")
print(f"  committed/step      = {comm:.3f}")
print(f"  s_fwd (rdt/draft)   = {sfwd:.4f}")
print(f"  wall/step (realized)= {wall:.4f}   = committed/step / realized_tps")
print(f"  GAP (wall - s_fwd)  = {gap:.4f}   <-- the non-forward per-step time")
print(f"  drafts={dD:.0f} accepted={dA:.0f} gen_tok={dG:.0f} rdt={dRT:.2f}s streamed_tok={N}")
PY
echo "DONE $ARM"
