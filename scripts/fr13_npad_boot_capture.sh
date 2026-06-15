#!/usr/bin/env bash
# FR13 NPAD-INVARIANT BootCapture (PHASE 2) -- boot the FORKED cat9 spec server
# with FR13_NPAD_INVARIANT=1 + FR13_SCAN_ALIGN=1 MODE=body (K1 ON), ENFORCE_EAGER=1.
#   * NON-VACUITY (i): bridge-needle worker /proc/<pid>/environ for BOTH
#     FR13_NPAD_INVARIANT=1 AND FR13_SCAN_ALIGN=1 (fail loud if absent, #9).
#   * engagement: tok/draft==9 (cat9 engaged).
#   * GATE 1: run scripts/fr13_npad_invariant_gate1.py inside the container
#     (spine states cat9 vs chain5, flag ON expect int-view 0.0, OFF expect 0.0289).
#   * GATE 2 (part 1): capture the served gold-margin probe + accept/event.
# The recurrent-oracle rescore (GATE 2 part 2) is a SEPARATE boot.
set -euo pipefail
REPO="${REPO:-/home/mark/shared/lumoFlyWheel}"
CONTAINER="${CONTAINER:-fr13-forked-fa2-tree}"
PORT="${PORT:-9950}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:${PORT}}"
MODEL="${MODEL:-qwen3.6-27b}"
OUT_DIR="${OUT_DIR:-$REPO/output/fr13_npad_invariant/logs}"
PROMPTS="${PROMPTS:-$REPO/output/fr13_acceptance_ladder/prompts_swe4.json}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"
MAX_TOKENS="${MAX_TOKENS:-128}"; TOP_K="${TOP_K:-20}"; SEED="${SEED:-1313}"; MODE="${MODE:-tree_mtp}"
LAUNCHER="$REPO/scripts/fr13_launch_forked_fa2_tree_server.sh"
PAYLOAD="${PAYLOAD:-$REPO/output/fr13_replay_gpu_gates/boot1a_eager_capture_logs/tree_gdn_capture_payload.pt}"
CAP_OUT="$OUT_DIR/probe_us_npad.json"
BOOT_LOG="$OUT_DIR/npad_boot.log"
NEEDLE="$OUT_DIR/npad_flag_live.needle"
GATE1_OUT="$OUT_DIR/npad_gate1_spine.json"
GATE1_LOG="$OUT_DIR/npad_gate1.log"
SUMMARY="$OUT_DIR/npad_capture_summary.json"
NUM_NODES=9
mkdir -p "$OUT_DIR"
fail(){ echo "NPAD_BOOT FAIL: $*" >&2; }

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

echo "[npad_boot] FORKED cat9 + FR13_NPAD_INVARIANT=1 + FR13_SCAN_ALIGN=1 MODE=body ENFORCE_EAGER=1"
recover_host; assert_host_free
[[ "$(docker ps -q | wc -l | tr -d ' ')" != "0" ]] && { fail "docker ps not empty"; docker ps >&2; exit 3; }
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "[boot] forked server (detached) -> $BOOT_LOG"
ENFORCE_EAGER=1 \
FR13_NPAD_INVARIANT=1 \
FR13_SCAN_ALIGN=1 FR13_SCAN_ALIGN_MODE=body \
CONTAINER="$CONTAINER" PORT="$PORT" \
  bash "$LAUNCHER" > "$BOOT_LOG" 2>&1
trap 'teardown' EXIT

echo "[boot] wait /health (timeout ${HEALTH_TIMEOUT_S}s)"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_S )); healthy=0
while [[ $(date +%s) -lt $deadline ]]; do
  cstate="$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo absent)"
  if [[ "$cstate" == "exited" || "$cstate" == "dead" ]]; then
    fail "container $cstate before health; boot tail:"; docker logs --tail 100 "$CONTAINER" >&2 2>/dev/null; exit 4
  fi
  curl -fsS --max-time 5 "${ENDPOINT}/health" >/dev/null 2>&1 && { healthy=1; break; }
  sleep 5
done
[[ "$healthy" != "1" ]] && { fail "not healthy in ${HEALTH_TIMEOUT_S}s; tail:"; docker logs --tail 120 "$CONTAINER" >&2 2>/dev/null; exit 4; }
echo "[boot] /health ready"

# ===== NON-VACUITY (i): FLAG LIVE -- bridge-needle worker /proc/<pid>/environ for BOTH flags =====
echo "[flag_live] bridge-needle /proc/<pid>/environ for FR13_NPAD_INVARIANT + FR13_SCAN_ALIGN"
docker exec "$CONTAINER" bash -lc '
set -e
hit_npad=""; hit_align=""; hit_mode=""; pids=""
for p in $(ls /proc | grep -E "^[0-9]+$"); do
  e="/proc/$p/environ"
  [ -r "$e" ] || continue
  cmd="$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)"
  case "$cmd" in
    *vllm*|*VllmWorker*|*EngineCore*|*python*) ;;
    *) continue ;;
  esac
  np="$(tr "\0" "\n" < "$e" | grep -E "^FR13_NPAD_INVARIANT=" || true)"
  a="$(tr "\0" "\n" < "$e" | grep -E "^FR13_SCAN_ALIGN=" || true)"
  m="$(tr "\0" "\n" < "$e" | grep -E "^FR13_SCAN_ALIGN_MODE=" || true)"
  if [ -n "$np" ] || [ -n "$a" ]; then
    pids="$pids $p"
    echo "PID $p  cmd=[$cmd]"
    echo "   $np"
    echo "   $a"
    echo "   $m"
    [ -n "$np" ] && hit_npad="$np"
    [ -n "$a" ] && hit_align="$a"
    [ -n "$m" ] && hit_mode="$m"
  fi
done
echo "SUMMARY hit_npad=[$hit_npad] hit_align=[$hit_align] hit_mode=[$hit_mode] pids=[$pids]"
' | tee "$NEEDLE"

if ! grep -q "FR13_NPAD_INVARIANT=1" "$NEEDLE"; then
  fail "FLAG NOT LIVE: FR13_NPAD_INVARIANT=1 absent in worker /proc environ (bug-class #9/#10)"; exit 6
fi
if ! grep -q "FR13_SCAN_ALIGN=1" "$NEEDLE"; then
  fail "FLAG NOT LIVE: FR13_SCAN_ALIGN=1 absent in worker /proc environ (bug-class #9)"; exit 6
fi
if ! grep -q "FR13_SCAN_ALIGN_MODE=body" "$NEEDLE"; then
  fail "MODE NOT LIVE: FR13_SCAN_ALIGN_MODE=body absent"; exit 6
fi
echo "[flag_live] CONFIRMED FR13_NPAD_INVARIANT=1 + FR13_SCAN_ALIGN=1 + MODE=body in worker environ"

# ===== GATE 1 (MECHANISM): spine states cat9 vs chain5, flag ON vs OFF, int-view =====
echo "[gate1] spine-state A/B (cat9 vs chain5) inside container -> $GATE1_OUT"
REL_PAYLOAD="/workspace/${PAYLOAD#$REPO/}"
REL_GATE1_OUT="/workspace/${GATE1_OUT#$REPO/}"
docker exec "$CONTAINER" bash -lc "
set -e
cd /workspace
PYTHONPATH=/workspace/src python3 /workspace/scripts/fr13_npad_invariant_gate1.py \
  --payload '$REL_PAYLOAD' --out '$REL_GATE1_OUT'
" 2>&1 | tee "$GATE1_LOG" || { fail "GATE1 script errored; see $GATE1_LOG"; }

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
  --arm us_tree_npad --out "$CAP_OUT" \
  --endpoint "$ENDPOINT" --model "$MODEL" --prompts-file "$PROMPTS" \
  --max-tokens "$MAX_TOKENS" --top-k "$TOP_K" --mode "$MODE" --seed "$SEED"

DET_OK="$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));w=d.get("within_boot_det_rep1_eq_rep2");print("true" if isinstance(w,list) and len(w)>0 and all(bool(x) for x in w) else "false")' "$CAP_OUT")"
WITHIN="$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(json.dumps(d.get("within_boot_det_rep1_eq_rep2")))' "$CAP_OUT")"
echo "[capture] within_boot_det=$WITHIN det_ok=$DET_OK"

# served-diverges-from-OFF / K1 (sanity, cross-trajectory)
DIVERGE="$(python3 - "$CAP_OUT" "$REPO/output/fr13_scan_align_rerun/logs/probe_us_k1.json" <<'PY'
import json,sys
try:
    npad=json.load(open(sys.argv[1])); k1=json.load(open(sys.argv[2]))
    def ids(d): return [r["served_token_ids"] for r in d["records"]]
    a=ids(npad); b=ids(k1); n=min(len(a),len(b))
    diverge=False; firstpos=[]
    for i in range(n):
        if a[i]!=b[i]:
            diverge=True
            m=min(len(a[i]),len(b[i])); fp=next((j for j in range(m) if a[i][j]!=b[i][j]), m); firstpos.append(fp)
        else: firstpos.append(-1)
    print(json.dumps({"diverges_vs_k1":diverge,"first_diverge_pos":firstpos,"npad_lens":[len(x) for x in a],"k1_lens":[len(x) for x in b]}))
except Exception as e:
    print(json.dumps({"error":str(e)}))
PY
)"
echo "[diverge] $DIVERGE"

ACCEPTED="$(metric vllm:spec_decode_num_accepted_tokens_total)"
NUM_DRAFTS2="$(metric vllm:spec_decode_num_drafts_total)"
ACCEPT_PER_EVENT="$(python3 -c 'import sys;print(repr(float(sys.argv[1])/float(sys.argv[2])))' "$ACCEPTED" "$NUM_DRAFTS2" 2>/dev/null || echo null)"
echo "[accept] accepted=$ACCEPTED drafts=$NUM_DRAFTS2 accept/event=$ACCEPT_PER_EVENT"

python3 - <<PY
import json
g1=None
try: g1=json.load(open("$GATE1_OUT"))
except Exception as e: g1={"error":str(e)}
json.dump({
 "name":"us_tree_npad","npad_invariant":"1","scan_align":"1","mode":"body","num_nodes":$NUM_NODES,
 "tok_per_draft":float("$TOK_PER_DRAFT"),"engaged":("$ENGAGED"=="true"),
 "within_boot_det":json.loads('$WITHIN'),"det_ok":("$DET_OK"=="true"),
 "diverge_vs_k1":json.loads('''$DIVERGE'''),
 "accepted":float("$ACCEPTED"),"drafts":float("$NUM_DRAFTS2"),
 "accept_per_event":float("$ACCEPT_PER_EVENT"),
 "gate1":{"flag_on_int_view_equal":g1.get("flag_on",{}).get("int_view_equal"),
          "flag_on_max_abs":g1.get("flag_on",{}).get("max_abs"),
          "flag_off_int_view_equal":g1.get("flag_off",{}).get("int_view_equal"),
          "flag_off_max_abs":g1.get("flag_off",{}).get("max_abs"),
          "gate1_pass":g1.get("gate1_pass"),
          "neg_control_powered":g1.get("neg_control_powered")} if "error" not in g1 else g1,
 "capture":"$CAP_OUT","boot_log":"$BOOT_LOG","needle":"$NEEDLE","gate1_out":"$GATE1_OUT"
}, open("$SUMMARY","w"), indent=2)
print(open("$SUMMARY").read())
PY
echo "[done] capture=$CAP_OUT summary=$SUMMARY gate1=$GATE1_OUT"
