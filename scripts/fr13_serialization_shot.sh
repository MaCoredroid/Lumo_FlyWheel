#!/bin/bash
# FR13 SERIALIZATION-FIX SHOT (user 2026-07-08 "give a shot"). Tests the source-verified fix
# from FR13_B4_CACHE_MATRIX_RESULTS.md §7 (wf_1c4af669): does max_num_batched_tokens=4096 +
# long_prefill_token_threshold=1024 (mamba_block_size=1024 UNCHANGED) un-serialize decode to
# true B>1 WITHOUT breaking APC losslessness?
#   Root cause: max_num_batched==mamba_block(1024) + align-split => any decode leaves <1 block
#   budget => waiting prefill chunk rounds to 0 => break => decode+prefill mutually exclusive => ~B1.
#   Fix keeps per-request chunks <=1 block (threshold), so #45238 overshoot is NOT reintroduced.
# GPU-SOLO: run ONLY when GPU free (0 fr13 containers). Two stages, sequential.
#   Stage A THROUGHPUT: boot fix server, fire 4 concurrent ~30K-token requests, sample the
#     scheduler Running histogram. PASS = shifts from ~80% R1 toward R2-R4 + prompt_tput>0 w/ Running>0.
#   Stage B LOSSLESS: fr13_apc_temp06_precheck.sh with the SAME fix env (recurrent-oracle gate).
#     PASS = cache-ON clear-margin argmax-flip rate <= cache-OFF (within-floor).
# BASELINE for comparison already measured (cat8/native): Running==1 80-85% (dist R1~1466/1826).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
RUNDIR=output/fr13_serialization_shot; mkdir -p "$RUNDIR"
C=fr13-shot-serialfix; PORT=9950
CAT6ROOT_TREE='[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]'
SPEC='{"method":"qwen3_5_mtp","num_speculative_tokens":6,"speculative_token_tree":"'"$CAT6ROOT_TREE"'"}'

# ---- guard: GPU must be free ----
if [[ -n "$(docker ps -q --filter name=fr13 2>/dev/null)" ]]; then
  echo "ABORT: an fr13 container is running — shot is GPU-solo. Run after cat6 completes."; docker ps --format '{{.Names}}' | grep fr13; exit 1
fi

# ============ STAGE A: THROUGHPUT ============
echo "=== [shot] STAGE A THROUGHPUT: boot FIX server (max_num_batched=4096, threshold=1024, mamba_block=1024) ==="
docker rm -f "$C" 2>/dev/null || true
CONTAINER="$C" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS=4 \
  FR13_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 \
  APC_MAX_NUM_BATCHED_TOKENS=4096 APC_BLOCK_SIZE=1024 LUMO_LONG_PREFILL_THRESHOLD=1024 \
  ATTENTION_BACKEND=TREE_ATTN SPEC_CONFIG="$SPEC" \
  bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$RUNDIR/launchA.log" 2>&1 &
LPID=$!
# wait /health (up to 1200s), fail if container dies
for i in $(seq 1 400); do
  if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "  healthy after ${i}x3s"; break; fi
  if [[ -n "$(docker ps -aq -f name=$C -f status=exited)" ]]; then echo "FAIL: container died on boot"; tail -30 "$RUNDIR/launchA.log"; exit 2; fi
  sleep 3
done
curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "FAIL: /health not up"; tail -30 "$RUNDIR/launchA.log"; exit 2; }
# confirm the fix flags are live in the engine args
docker logs "$C" 2>&1 | grep -oE "max_num_batched_tokens': [0-9]+|long_prefill_token_threshold': [0-9]+|mamba_block_size': [0-9]+" | sort -u | sed 's/^/  engine-arg: /'

echo "=== [shot] firing 4 concurrent ~30K-token distinct-prefix requests (temp 0.6, 256 max_tokens) + sampling Running ==="
MARK_TS=$(date +%s)
.venv/bin/python - "$PORT" "$RUNDIR" <<'PY' &
import sys,json,urllib.request,concurrent.futures,time
port,rundir=sys.argv[1],sys.argv[2]
def req(i):
    # distinct long prompt (~24K tokens) so each must fully prefill (no shared cache)
    filler=" ".join(f"tok{i}_{j}" for j in range(6000))
    msg=[{"role":"user","content":f"Request {i} distinct prefix {i}*{i}. {filler}\nSummarize in one word."}]
    data=json.dumps({"model":"qwen3.6-27b","messages":msg,"temperature":0.6,"max_tokens":256,"seed":i}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=data,headers={"Content-Type":"application/json"})
    t=time.time()
    try:
        urllib.request.urlopen(r,timeout=600).read(); return (i,round(time.time()-t,1),"ok")
    except Exception as e: return (i,round(time.time()-t,1),str(e)[:60])
with concurrent.futures.ThreadPoolExecutor(4) as ex:
    res=list(ex.map(req,range(4)))
json.dump(res,open(f"{rundir}/loadA_results.json","w"))
print("  load done:",res)
PY
LOADPID=$!
# sample Running/Waiting during the load
for i in $(seq 1 40); do sleep 3; done &
wait $LOADPID
echo "=== [shot] Running histogram DURING the 4-concurrent load (FIX config) ==="
docker logs --since "$((MARK_TS))" "$C" 2>&1 | grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+" | sort | uniq -c | sort -rn | head | sed 's/^/  /'
python3 -c "
import re,subprocess
out=subprocess.run(['docker','logs','--since','$MARK_TS','$C'],capture_output=True,text=True).stderr+subprocess.run(['docker','logs','--since','$MARK_TS','$C'],capture_output=True,text=True).stdout
rows=[int(m.group(1)) for m in re.finditer(r'Running: (\d+) reqs',out)]
if rows:
    from collections import Counter; c=Counter(rows); n=len(rows)
    print(f'  FIX Running dist={dict(sorted(c.items()))} mean={sum(rows)/n:.2f} pctR1={100*c.get(1,0)/n:.0f}% (BASELINE was 80-85% R1, mean ~1.3)')
    print('  THROUGHPUT VERDICT:', 'FIX UN-SERIALIZES (mean>1.6 / R2-4 materially up)' if sum(rows)/n>1.6 else 'NO CHANGE (still ~1) -> fix ineffective, see fallback')
else: print('  no Running samples captured — widen sampling window')
"
docker rm -f "$C" 2>/dev/null; wait $LPID 2>/dev/null || true
sleep 5

# ============ STAGE B: LOSSLESS ============
echo "=== [shot] STAGE B LOSSLESS: recurrent-oracle gate with the SAME fix env ==="
CONTAINER=fr13-bigdenom-apc_temp06_shot RUNDIR="$RUNDIR/lossless" \
  MAX_NUM_SEQS_OVR=4 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 \
  APC_MAX_NUM_BATCHED_TOKENS=4096 APC_BLOCK_SIZE=1024 LUMO_LONG_PREFILL_THRESHOLD=1024 \
  ATTENTION_BACKEND=TREE_ATTN SPEC_CONFIG="$SPEC" \
  bash scripts/fr13_apc_temp06_precheck.sh 2>&1 | tee "$RUNDIR/losslessB.log" | tail -25
echo "=== [shot] DONE. Throughput: $RUNDIR (histogram above). Lossless: $RUNDIR/lossless (PASS/FAIL in log). ==="
echo "    Interpret: PASS both => bake APC_MAX_NUM_BATCHED_TOKENS=4096 + LUMO_LONG_PREFILL_THRESHOLD=1024 on cache-ON arms."
echo "    Lossless FAIL but throughput PASS => try LUMO_BATCH_INVARIANT_VLLM=1 (batch-comp GEMM), or conservative 2048, or cache-OFF scope."
