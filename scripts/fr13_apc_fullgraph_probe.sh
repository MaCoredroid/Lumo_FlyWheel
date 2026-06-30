#!/usr/bin/env bash
# FR13 APC FULL-GRAPH PROBE — reproduce the cache-ON full-graph garble + measure the decode-TPS upside.
# Mirrors fr13_apc_multiturn_one_arm.sh's "on" arm (full APC + EXACT_SEED + block 1024 + cat6root, the
# deployed lossless config) but boots under CUDAGRAPH_MODE (full graph) instead of ENFORCE_EAGER.
# This is the eager-vs-graph A/B regime the launcher comment used (graph GGGG at the cat-blob turn),
# now WITH EXACT_SEED on. STEP 1 of the full-graph fix: confirm garble + quantify the TPS prize.
#   CGMODE=FULL_AND_PIECEWISE (default, the poisoned regime) | PIECEWISE (control, known on-task)
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
CGMODE=${CGMODE:-FULL_AND_PIECEWISE}
PORT=${PORT:-9953}
GPU_UTIL=${GPU_UTIL:-0.82}
MAX_OUT=${MAX_OUT:-384}
export FR13_REPLAY_TEMP="${FR13_REPLAY_TEMP:-0.6}"   # STANDING RULE: temp 0.6, never greedy/temp-0
DUMPS=${DUMPS:-output/fr13_apc_rategate/run_20260625T084654Z/rg_OFF_r1/proxy_pair_dumps}
CAT6ROOT_TREE="[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]"
CONTAINER="fr13-apc-fullgraph-probe"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RD=output/fr13_apc_fullgraph_probe/run_${TS}; mkdir -p "$RD/logs"
echo "$RD" > /home/mark/.claude/jobs/22c39bb9/tmp/fullgraph_probe_root.txt
echo "=== FULL-GRAPH PROBE  CGMODE=$CGMODE  EXACT_SEED=1  block=1024  APC=on  -> $RD ==="
trap 'docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
docker ps -a --format '{{.Names}}'|grep -i fr13|xargs -r docker rm -f >/dev/null 2>&1 || true
sleep 2
LAUNCH="$RD/logs/launch.log"
echo "[probe] booting CGMODE=$CGMODE ..."
CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=$GPU_UTIL MAX_NUM_SEQS=1 \
  TREE="$CAT6ROOT_TREE" FR10_METRICS=0 BATCH_INVARIANT="${BATCH_INVARIANT:-1}" FR13_BI_TREE_ATTN=1 \
  GPU_GUARD_FLOOR_MIB="${GPU_GUARD_FLOOR_MIB:-2000}" \
  LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
  FR13_ENABLE_APC=1 FR13_APC_CONFIG_ONLY=0 \
  FR13_APC_EXACT_SEED=1 \
  MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 \
  CUDAGRAPH_MODE="$CGMODE" \
  FR13_RUN_DIR="$PWD/$RD" LOG_DIR="$PWD/$RD/logs" \
  setsid bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$LAUNCH" 2>&1 &

T0=$SECONDS; HEALTHY=0
while [ $((SECONDS-T0)) -lt 1200 ]; do
  curl -fsS -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { HEALTHY=1; break; }
  grep -qiE "CUDA out of memory|NVRM_NO_MEMORY" "$LAUNCH" 2>/dev/null && { echo "[probe] OOM"; tail -15 "$LAUNCH"; exit 3; }
  sleep 10
done
[ "$HEALTHY" = 1 ] || { echo "[probe] FAIL not healthy"; tail -30 "$LAUNCH"; exit 4; }
echo "[probe] healthy at $((SECONDS-T0))s"
echo "--- DEFINITIVE: did FULL GRAPH engage? (from docker container logs, not launch.log) ---"
docker logs "$CONTAINER" > "$RD/logs/engine_boot.log" 2>&1 || true
grep -aoE "Capturing cudagraph[s]?|cudagraph_mode[=:' ]+[A-Za-z_]+|CUDAGraphMode\.[A-Z_]+|enforce_eager[=: ]+(True|False)|Graph capturing finished|Capturing CUDA graph shapes|capturing [0-9]+ .*graph|full_cuda_graph|FULL_AND_PIECEWISE|PIECEWISE" "$RD/logs/engine_boot.log" 2>/dev/null | sort | uniq -c | sort -rn | head -12 | sed 's/^/  /'
CG_ENGAGED=$(grep -aciE "Capturing cudagraph|Graph capturing finished|cudagraph_mode.*(FULL|PIECEWISE)" "$RD/logs/engine_boot.log" 2>/dev/null)
EAGER_ON=$(grep -aciE "enforce_eager[=: ]+True|--enforce-eager" "$RD/logs/engine_boot.log" 2>/dev/null)
echo "  >>> CG capture lines=$CG_ENGAGED  enforce_eager_on=$EAGER_ON  (need CG>0 AND eager=0 for a valid full-graph test)"

curl -s "http://127.0.0.1:$PORT/metrics" > "$RD/metrics_pre.txt" 2>/dev/null
echo "[probe] replaying 12907 ..."
.venv/bin/python scripts/fr13_apc_multiturn_replay.py \
  --port "$PORT" --dumps-dir "$DUMPS" --arm "probe_${CGMODE}" \
  --out "$RD/replay.json" --max-output-tokens "$MAX_OUT" 2>&1 | tee "$RD/logs/replay.log"
# robust metrics_post: retry + timeout + container-alive check (prev runs raced teardown)
for _i in 1 2 3; do
  curl -fsS -m 20 "http://127.0.0.1:$PORT/metrics" > "$RD/metrics_post.txt" 2>/dev/null && [ -s "$RD/metrics_post.txt" ] && break
  echo "  [probe] metrics_post empty, retry $_i (container=$(docker ps --filter name=$CONTAINER --format '{{.Names}}' 2>/dev/null|head -1||echo GONE))"
  sleep 4
done

echo "=== GARBLE CHECK (CJK runs / char 8 / Unterminated on the replay output) ==="
.venv/bin/python - "$RD/replay.json" <<'PY'
import json,sys,re
try: d=json.load(open(sys.argv[1]))
except Exception as e: print("replay parse fail:",e); raise SystemExit
txt=json.dumps(d, ensure_ascii=False)
cjk=len(re.findall(r'[一-鿿぀-ヿ가-힯]{4,}', txt))
print(f"CJK runs(>=4): {cjk} | 'char 8': {txt.count('char 8')} | Unterminated: {txt.count('Unterminated')}")
print("VERDICT:", "GARBLE PRESENT (full-graph poisons cache-ON)" if cjk>0 else "NO CJK garble — clean")
PY

echo "=== DECODE TPS (CGMODE=$CGMODE; PIECEWISE serving floor ~17.5 tok/s) ==="
.venv/bin/python - "$RD/metrics_pre.txt" "$RD/metrics_post.txt" <<'PY'
import sys,re
def p(f):
    d={}
    for ln in open(f):
        ln=ln.strip()
        if ln.startswith('#') or not ln: continue
        m=re.match(r'(\S+?)(\{[^}]*\})?\s+([0-9eE.+-]+)$',ln)
        if m: d[m.group(1)]=d.get(m.group(1),0.0)+float(m.group(3))
    return d
a=p(sys.argv[1]); b=p(sys.argv[2])
def df(k): return b.get(k,0)-a.get(k,0)
gen=df('vllm:generation_tokens_total'); dt=df('vllm:request_decode_time_seconds_sum')
acc=df('vllm:spec_decode_num_accepted_tokens_total'); drf=df('vllm:spec_decode_num_draft_tokens_total')
print(f"gen toks: {int(gen)}  decode s: {dt:.1f}")
if dt>0: print(f"DECODE TPS = {gen/dt:.1f} tok/s")
if drf>0: print(f"spec accept: {acc:.0f}/{drf:.0f} = {acc/drf*100:.0f}%")
PY
echo "=== FULL-GRAPH PROBE DONE  CGMODE=$CGMODE -> $RD ==="
