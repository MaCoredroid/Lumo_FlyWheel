#!/bin/bash
# FR13 THROUGHPUT re-measure (standalone Stage A) — corrects the fr13_serialization_shot.sh
# Stage-A bug: its filler range(24000) built ~24K WORDS => ~100-120K+ TOKENS/request =>
# exceeded max_model_len=131072 => immediate HTTP 400 (no load applied). Here N_FILLER words
# tokenize to a SAFE, substantial prefill and the load FAILS LOUD if requests error.
# Tests: does the fix (max_num_batched=4096 + long_prefill_threshold=1024, mamba_block=1024)
# un-serialize decode to true B>1?  PASS = Running mean>1.6 (baseline 80-85% R1, mean ~1.3).
# GPU-SOLO. Usage: bash scripts/fr13_throughput_remeasure.sh
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
RUNDIR=output/fr13_serialization_shot; mkdir -p "$RUNDIR"
C=fr13-shot-serialfix; PORT=9950
N_FILLER=${N_FILLER:-6000}      # ~6000 words => ~24-30K tokens/req (< 131K, ~26 prefill steps at 4096)
CONC=${CONC:-4}
CAT6ROOT_TREE='[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]'
SPEC='{"method":"qwen3_5_mtp","num_speculative_tokens":6,"speculative_token_tree":"'"$CAT6ROOT_TREE"'"}'

if [[ -n "$(docker ps -q --filter name=fr13 2>/dev/null)" ]]; then
  echo "ABORT: fr13 container running (GPU-solo)."; docker ps --format '{{.Names}}' | grep fr13; exit 1
fi

echo "=== [tput] boot FIX server (max_num_batched=4096, threshold=1024, mamba_block=1024) ==="
docker rm -f "$C" 2>/dev/null || true
CONTAINER="$C" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS=4 \
  FR13_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 \
  APC_MAX_NUM_BATCHED_TOKENS=4096 APC_BLOCK_SIZE=1024 LUMO_LONG_PREFILL_THRESHOLD=1024 \
  ATTENTION_BACKEND=TREE_ATTN SPEC_CONFIG="$SPEC" \
  bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$RUNDIR/launch_tput.log" 2>&1 &
LPID=$!
for i in $(seq 1 400); do
  curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "  healthy after ${i}x3s"; break; }
  [[ -n "$(docker ps -aq -f name=$C -f status=exited)" ]] && { echo "FAIL: container died"; tail -30 "$RUNDIR/launch_tput.log"; exit 2; }
  sleep 3
done
curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "FAIL: /health"; exit 2; }
docker logs "$C" 2>&1 | grep -oE "max_num_batched_tokens': [0-9]+|long_prefill_token_threshold': [0-9]+|mamba_block_size': [0-9]+" | sort -u | sed 's/^/  engine-arg: /'

echo "=== [tput] token-length preflight (1 short probe must succeed) ==="
PRE=$(curl -fsS -m 30 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"say ok"}],"max_tokens":5,"temperature":0.6,"seed":1}' 2>&1 | head -c 200)
echo "  preflight: $PRE" | head -c 220; echo

echo "=== [tput] firing $CONC concurrent requests (N_FILLER=$N_FILLER words) + sampling Running for ~90s ==="
MARK_TS=$(date +%s)
.venv/bin/python - "$PORT" "$RUNDIR" "$N_FILLER" "$CONC" <<'PY' &
import sys,json,urllib.request,concurrent.futures,time
port,rundir,nf,conc=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
def req(i):
    filler=" ".join(f"tok{i}_{j}" for j in range(nf))
    msg=[{"role":"user","content":f"Request {i} distinct prefix {i}. {filler}\nSummarize in one word."}]
    data=json.dumps({"model":"qwen3.6-27b","messages":msg,"temperature":0.6,"max_tokens":256,"seed":i}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=data,headers={"Content-Type":"application/json"})
    t=time.time()
    try:
        urllib.request.urlopen(r,timeout=600).read(); return (i,round(time.time()-t,1),"ok")
    except Exception as e: return (i,round(time.time()-t,1),str(e)[:80])
with concurrent.futures.ThreadPoolExecutor(conc) as ex:
    res=list(ex.map(req,range(conc)))
json.dump(res,open(f"{rundir}/loadA_results.json","w"))
ok=sum(1 for _,_,s in res if s=="ok")
print(f"  load done ok={ok}/{conc}:",res)
if ok==0: print("  !!! STAGE A LOAD FAILED — all requests errored (likely still too long, or server issue). Throughput NOT measured.")
PY
LOADPID=$!
wait $LOADPID
echo "=== [tput] Running histogram DURING load (FIX config) ==="
python3 -c "
import re,subprocess,json
r=subprocess.run(['docker','logs','--since','$MARK_TS','$C'],capture_output=True,text=True)
out=r.stderr+r.stdout
rows=[int(m.group(1)) for m in re.finditer(r'Running: (\d+) reqs',out)]
try: ld=json.load(open('$RUNDIR/loadA_results.json')); ok=sum(1 for x in ld if x[2]=='ok')
except: ok=-1
if ok==0:
    print('  LOAD FAILED (ok=0) -> throughput INVALID. Reduce N_FILLER and retry.'); raise SystemExit
if rows:
    from collections import Counter; c=Counter(rows); n=len(rows)
    print(f'  ok_reqs={ok}  FIX Running dist={dict(sorted(c.items()))} n={n} mean={sum(rows)/n:.2f} pctR1={100*c.get(1,0)/n:.0f}%')
    print(f'  BASELINE was 80-85%% R1, mean ~1.3')
    print('  THROUGHPUT VERDICT:', 'FIX UN-SERIALIZES (mean>1.6)' if sum(rows)/n>1.6 else ('PARTIAL (mean 1.3-1.6)' if sum(rows)/n>1.3 else 'NO CHANGE (~1) -> fix ineffective'))
else: print(f'  ok_reqs={ok} but no Running samples — load too short or log level; raise N_FILLER')
"
docker rm -f "$C" 2>/dev/null; wait $LPID 2>/dev/null || true
echo "=== [tput] DONE ==="
