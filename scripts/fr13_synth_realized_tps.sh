#!/usr/bin/env bash
# FR13 Stage D: SYNTHETIC realized decode-TPS probe (no proxy, STREAM=FALSE so the
# client never backpressures vLLM). Measures the realized decode rate WITHOUT a slow
# consumer + the idle-INDEPENDENT GPU forward time (FR13_SFWD_GPU_TIMER) + accept.
# DECISIVE: if stream=false cat6 realized_tps is ~+17% over E5 (matching its committed
# ratio) while the DEPLOY is only +4%, the deploy gap is CONSUMER-PACED (agent-loop /
# proxy streaming), NOT our kernel. s_per_fwd_gpu confirms the forward is ~equal.
# Usage: fr13_synth_realized_tps.sh <cat6|e5>
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
REPO=/home/mark/shared/lumoFlyWheel
ARM=${1:?usage: <cat6|e5>}
PORT=9952
CONTAINER=fr13-synth-$ARM
OUT=output/fr13_synth_tps
mkdir -p "$OUT" "$OUT/sfwd"
CAT6ROOT_TREE="[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]"

recover(){ PYTHONPATH="$REPO/src" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}
teardown(){ docker logs "$CONTAINER" >"$OUT/${ARM}_docker.log" 2>&1 || true; docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; recover; }
trap teardown EXIT

echo "[1/4] hygiene"; recover; [ -z "$(docker ps -q)" ] || { echo FAIL-not-empty; docker ps; exit 2; }

echo "[2/4] boot $ARM (+ FR13_SFWD_GPU_TIMER)"
SFWD_JSON="/workspace/output/fr13_synth_tps/sfwd/${ARM}.json"
if [ "$ARM" = cat6 ]; then
  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS=1 \
    TREE="$CAT6ROOT_TREE" FR10_METRICS=0 BATCH_INVARIANT=0 \
    LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
    FR13_SFWD_GPU_TIMER=1 FR13_SFWD_GPU_TIMER_JSON="$SFWD_JSON" \
    LOG_DIR="$REPO/$OUT/logs_$ARM" \
    scripts/fr13_launch_forked_fa2_tree_server.sh > "$OUT/${ARM}_launch.log" 2>&1
else
  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS=1 \
    ATTENTION_BACKEND=FLASH_ATTN FR10_ENABLE_TREE_GDN=0 \
    SPEC_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":5}' \
    FR13_SFWD_GPU_TIMER=1 FR13_SFWD_GPU_TIMER_JSON="$SFWD_JSON" \
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

echo "[4/4] measure stream=FALSE realized decode-TPS + s_per_fwd_gpu"
.venv/bin/python - "http://127.0.0.1:$PORT" "qwen3.6-27b" "$ARM" "$REPO/output/fr13_synth_tps/sfwd/${ARM}.json" <<'PY' | tee "$OUT/${ARM}_result.txt"
import time, json, urllib.request, sys, os
URL, MODEL, ARM, SFWD = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
KEYS=["vllm:spec_decode_num_drafts_total","vllm:spec_decode_num_accepted_tokens_total",
      "vllm:generation_tokens_total","vllm:request_decode_time_seconds_sum"]
def metrics():
    txt=urllib.request.urlopen(URL+"/metrics",timeout=15).read().decode()
    m={k:0.0 for k in KEYS}
    for ln in txt.splitlines():
        for k in KEYS:
            if ln.startswith(k+" ") or ln.startswith(k+"{"):
                try: m[k]+=float(ln.rsplit(" ",1)[1])
                except: pass
    return m
def req(maxtok):
    body=json.dumps({"model":MODEL,"prompt":"Explain in detail, step by step, how a modern out-of-order superscalar CPU fetches, renames, schedules, executes and retires instructions, with concrete examples:","max_tokens":maxtok,"temperature":0.6,"seed":1313,"stream":False}).encode()
    r=urllib.request.Request(URL+"/v1/completions",data=body,headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(r,timeout=300) as resp:
        d=json.loads(resp.read().decode())
    return time.time()-t0, len(d["choices"][0].get("text","")), d
def sfwd_read():
    import glob
    fs=sorted(glob.glob(SFWD+"*"))
    if not fs: return {}
    try: return json.load(open(fs[-1]))
    except: return {}
req(40)  # warmup
# prefill baseline (1 token ~ prefill+overhead)
tp,_,_=req(1); tp2,_,_=req(1); t_prefill=min(tp,tp2)
a=metrics()
TFULL=0.0; NREQ=4; MT=400
for _ in range(NREQ):
    tf,_,_=req(MT); TFULL+=tf
b=metrics(); sf=sfwd_read()
avg_full=TFULL/NREQ
decode_wall=avg_full-t_prefill                 # per request, prefill removed, NO consumer backpressure (stream=false)
realized_tps=(MT-1)/decode_wall if decode_wall>0 else 0
dD=b[KEYS[0]]-a[KEYS[0]]; dA=b[KEYS[1]]-a[KEYS[1]]; dG=b[KEYS[2]]-a[KEYS[2]]; dRT=b[KEYS[3]]-a[KEYS[3]]
acc=dA/dD if dD else 0; comm=dG/dD if dD else 0; sfwd=dRT/dD if dD else 0
# s_per_fwd_gpu from the async-cuda-event sidecar (idle-INDEPENDENT pure GPU forward)
sec=sf.get("decode_forward_gpu_seconds",0); steps=sf.get("n_pure_decode_steps_timed",0)
s_per_fwd_gpu = sec/steps if steps else 0
gpu_basis_tps = comm/s_per_fwd_gpu if s_per_fwd_gpu else 0
print(f"ARM={ARM}  (stream=FALSE, no consumer backpressure)")
print(f"  realized_decode_tps  = {realized_tps:.2f}   (vLLM-internal rate, prefill-removed)")
print(f"  accept/event         = {acc:.3f}")
print(f"  committed/step       = {comm:.3f}")
print(f"  s_fwd (rdt/draft)    = {sfwd:.4f}")
print(f"  s_per_fwd_gpu        = {s_per_fwd_gpu:.4f}   (idle-INDEPENDENT pure GPU forward/step)")
print(f"  GPU_basis_tps        = {gpu_basis_tps:.2f}   = committed/step / s_per_fwd_gpu")
print(f"  t_prefill={t_prefill:.3f}s avg_full(400)={avg_full:.3f}s decode_wall={decode_wall:.3f}s drafts={dD:.0f} accepted={dA:.0f}")
PY
echo "DONE $ARM"
