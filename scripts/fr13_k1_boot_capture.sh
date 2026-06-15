#!/usr/bin/env bash
# FR13 K1 BootCapture (PHASE 2) -- boot the LOCKED cat9 spec server with
# FR13_SCAN_ALIGN=1 MODE=body (=> body seams d/e/K1 all ON = full (2)-oracle
# alignment of the per-node rank-1 update), ENFORCE_EAGER=1. Assert tok/draft==9
# (cat9 engaged), within-boot det, capture served streams, bridge-needle the
# worker /proc/<pid>/environ for FR13_SCAN_ALIGN=1 (FLAG LIVE), then teardown +
# recover. Flip rescoring vs the RECURRENT oracle is done SEPARATELY (next boot)
# by scripts/fr13_recur_rescore_in_container.sh on the saved capture JSON.
set -euo pipefail
REPO="${REPO:-/home/mark/shared/lumoFlyWheel}"
CONTAINER="${CONTAINER:-fr13-forked-fa2-tree}"
PORT="${PORT:-9950}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:${PORT}}"
MODEL="${MODEL:-qwen3.6-27b}"
OUT_DIR="${OUT_DIR:-$REPO/output/fr13_scan_align_rerun/logs}"
PROMPTS="${PROMPTS:-$REPO/output/fr13_acceptance_ladder/prompts_swe4.json}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1200}"
MAX_TOKENS="${MAX_TOKENS:-128}"; TOP_K="${TOP_K:-20}"; SEED="${SEED:-1313}"; MODE="${MODE:-tree_mtp}"
LAUNCHER="$REPO/scripts/fr13_launch_locked.sh"
CAP_OUT="$OUT_DIR/probe_us_k1.json"
BOOT_LOG="$OUT_DIR/k1_boot.log"
NEEDLE="$OUT_DIR/k1_flag_live.needle"
SUMMARY="$OUT_DIR/k1_capture_summary.json"
NUM_NODES=9
mkdir -p "$OUT_DIR"
fail(){ echo "K1_BOOT FAIL: $*" >&2; }

recover_host(){ PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}
assert_host_free(){ python3 - <<'PY'
from pathlib import Path
f={}
for l in Path("/proc/meminfo").read_text().splitlines():
    k,v=l.split(":",1); f[k]=int(v.strip().split()[0])
avail=f.get("MemAvailable",0)/1024/1024
swap=f.get("SwapTotal",0)-f.get("SwapFree",0)
if avail<100 or swap!=0:
    raise SystemExit(f"hygiene FAIL MemAvailable={avail:.1f}GiB swap_used={swap/1024/1024:.2f}GiB")
print(f"[hygiene] MemAvailable={avail:.1f}GiB swap_used=0 OK")
PY
}
metric(){ curl -fsS --max-time 10 "${ENDPOINT}/metrics" 2>/dev/null | awk -v n="$1" '$0 ~ ("^" n " ")||$0 ~ ("^" n "{"){v=$NF} END{if(v!="")print v}'; }
teardown(){ echo "[teardown] docker rm -f $CONTAINER"; docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; echo "[teardown] recover_host_memory"; recover_host||true; }

echo "[k1_boot] LOCKED cat9 + FR13_SCAN_ALIGN=1 MODE=body ENFORCE_EAGER=1"
recover_host; assert_host_free
[[ "$(docker ps -q | wc -l | tr -d ' ')" != "0" ]] && { fail "docker ps not empty"; docker ps >&2; exit 3; }
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "[boot] locked server (bg) -> $BOOT_LOG"
ENFORCE_EAGER=1 \
FR13_SCAN_ALIGN=1 FR13_SCAN_ALIGN_MODE=body \
CONTAINER="$CONTAINER" PORT="$PORT" \
  bash "$LAUNCHER" > "$BOOT_LOG" 2>&1 &
trap 'teardown' EXIT

echo "[boot] wait /health (timeout ${HEALTH_TIMEOUT_S}s)"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_S )); healthy=0
while [[ $(date +%s) -lt $deadline ]]; do
  cstate="$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo absent)"
  if [[ "$cstate" == "exited" || "$cstate" == "dead" ]]; then
    fail "container $cstate before health; boot tail:"; docker logs --tail 80 "$CONTAINER" >&2 2>/dev/null || tail -80 "$BOOT_LOG" >&2; exit 4
  fi
  curl -fsS --max-time 5 "${ENDPOINT}/health" >/dev/null 2>&1 && { healthy=1; break; }
  sleep 5
done
[[ "$healthy" != "1" ]] && { fail "not healthy in ${HEALTH_TIMEOUT_S}s; tail:"; tail -100 "$BOOT_LOG" >&2; exit 4; }
echo "[boot] /health ready"

# ===== NON-VACUITY (i): FLAG LIVE -- bridge-needle the worker /proc/<pid>/environ =====
echo "[flag_live] bridge-needle /proc/<pid>/environ for FR13_SCAN_ALIGN"
docker exec "$CONTAINER" bash -lc '
set -e
hit_align=""; hit_mode=""; pids=""
for p in $(ls /proc | grep -E "^[0-9]+$"); do
  e="/proc/$p/environ"
  [ -r "$e" ] || continue
  cmd="$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)"
  case "$cmd" in
    *vllm*|*VllmWorker*|*EngineCore*|*python*) ;;
    *) continue ;;
  esac
  a="$(tr "\0" "\n" < "$e" | grep -E "^FR13_SCAN_ALIGN=" || true)"
  m="$(tr "\0" "\n" < "$e" | grep -E "^FR13_SCAN_ALIGN_MODE=" || true)"
  if [ -n "$a" ]; then
    pids="$pids $p"
    echo "PID $p  cmd=[$cmd]"
    echo "   $a"
    echo "   $m"
    hit_align="$a"; hit_mode="$m"
  fi
done
echo "SUMMARY hit_align=[$hit_align] hit_mode=[$hit_mode] pids=[$pids]"
' | tee "$NEEDLE"

if ! grep -q "FR13_SCAN_ALIGN=1" "$NEEDLE"; then
  fail "FLAG NOT LIVE: FR13_SCAN_ALIGN=1 absent in worker /proc environ (bug-class #9/#10)"; exit 6
fi
if ! grep -q "FR13_SCAN_ALIGN_MODE=body" "$NEEDLE"; then
  fail "MODE NOT LIVE: FR13_SCAN_ALIGN_MODE=body absent in worker /proc environ"; exit 6
fi
echo "[flag_live] CONFIRMED FR13_SCAN_ALIGN=1 + MODE=body in worker environ"

curl -fsS --max-time 30 -X POST "${ENDPOINT}/reset_prefix_cache" >/dev/null 2>&1 || true

echo "[engage] warmup request"
WARM="$(python3 -c 'import json,sys;p=json.load(open(sys.argv[1]));print(p[0][:400])' "$PROMPTS")"
warm_payload="$(python3 - <<PY
import json
print(json.dumps({"model":"$MODEL","messages":[{"role":"user","content":$(python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))' <<<"$WARM")}],"max_tokens":24,"temperature":0.0,"seed":$SEED}))
PY
)"
curl -fsS --max-time 180 -H 'Content-Type: application/json' -d "$warm_payload" "${ENDPOINT}/v1/chat/completions" >/dev/null 2>&1 || {
  fail "warmup failed (drafter disengaged?); tail:"; docker logs --tail 80 "$CONTAINER" >&2 2>/dev/null; exit 5; }

DRAFT_TOKS="$(metric vllm:spec_decode_num_draft_tokens_total)"
NUM_DRAFTS="$(metric vllm:spec_decode_num_drafts_total)"
[[ -z "$DRAFT_TOKS" || -z "$NUM_DRAFTS" || "$NUM_DRAFTS" == "0" ]] && { fail "spec counters absent/zero (draft_toks='$DRAFT_TOKS' drafts='$NUM_DRAFTS')"; exit 5; }
TOK_PER_DRAFT="$(python3 -c 'import sys;print(repr(float(sys.argv[1])/float(sys.argv[2])))' "$DRAFT_TOKS" "$NUM_DRAFTS")"
ENGAGED="$(python3 -c 'import sys;print("true" if abs(float(sys.argv[1])-float(sys.argv[2]))<1e-6 else "false")' "$TOK_PER_DRAFT" "$NUM_NODES")"
echo "[engage] tok/draft=$TOK_PER_DRAFT len(TREE)=$NUM_NODES engaged=$ENGAGED"
[[ "$ENGAGED" != "true" ]] && { fail "ENGAGEMENT GATE tok/draft($TOK_PER_DRAFT)!=9; recording NOTHING"; exit 5; }

echo "[capture] gold_margin_probe -> $CAP_OUT"
python3 "$REPO/scripts/fr13_gold_margin_probe.py" capture \
  --arm us_tree_K1 --out "$CAP_OUT" \
  --endpoint "$ENDPOINT" --model "$MODEL" --prompts-file "$PROMPTS" \
  --max-tokens "$MAX_TOKENS" --top-k "$TOP_K" --mode "$MODE" --seed "$SEED"

DET_OK="$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));w=d.get("within_boot_det_rep1_eq_rep2");print("true" if isinstance(w,list) and len(w)>0 and all(bool(x) for x in w) else "false")' "$CAP_OUT")"
WITHIN="$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(json.dumps(d.get("within_boot_det_rep1_eq_rep2")))' "$CAP_OUT")"
echo "[capture] within_boot_det=$WITHIN det_ok=$DET_OK"

# ===== served-diverges-from-OFF: compare K1 served ids vs banked probe_us_off =====
DIVERGE="$(python3 - "$CAP_OUT" "$OUT_DIR/probe_us_off.json" <<'PY'
import json,sys
k1=json.load(open(sys.argv[1])); off=json.load(open(sys.argv[2]))
def ids(d):
    return [r["served_token_ids"] for r in d["records"]]
a=ids(k1); b=ids(off)
n=min(len(a),len(b))
diverge=False; firstpos=[]
for i in range(n):
    if a[i]!=b[i]:
        diverge=True
        # first diverging pos
        m=min(len(a[i]),len(b[i])); fp=next((j for j in range(m) if a[i][j]!=b[i][j]), m)
        firstpos.append(fp)
    else:
        firstpos.append(-1)
print(json.dumps({"diverges":diverge,"first_diverge_pos_per_prompt":firstpos,
                  "k1_lens":[len(x) for x in a],"off_lens":[len(x) for x in b]}))
PY
)"
echo "[diverge] $DIVERGE"

ACCEPTED="$(metric vllm:spec_decode_num_accepted_tokens_total)"
NUM_DRAFTS2="$(metric vllm:spec_decode_num_drafts_total)"
ACCEPT_PER_EVENT="$(python3 -c 'import sys;print(repr(float(sys.argv[1])/float(sys.argv[2])))' "$ACCEPTED" "$NUM_DRAFTS2" 2>/dev/null || echo null)"
echo "[accept] accepted=$ACCEPTED drafts=$NUM_DRAFTS2 accept/event=$ACCEPT_PER_EVENT"

python3 - <<PY
import json
json.dump({
 "name":"us_tree_K1","scan_align":"1","mode":"body","num_nodes":$NUM_NODES,
 "tok_per_draft":float("$TOK_PER_DRAFT"),"engaged":("$ENGAGED"=="true"),
 "within_boot_det":json.loads('$WITHIN'),"det_ok":("$DET_OK"=="true"),
 "diverge_vs_off":json.loads('''$DIVERGE'''),
 "accepted":float("$ACCEPTED"),"drafts":float("$NUM_DRAFTS2"),
 "accept_per_event":float("$ACCEPT_PER_EVENT"),
 "capture":"$CAP_OUT","boot_log":"$BOOT_LOG","needle":"$NEEDLE"
}, open("$SUMMARY","w"), indent=2)
print(open("$SUMMARY").read())
PY
echo "[done] capture=$CAP_OUT summary=$SUMMARY"
