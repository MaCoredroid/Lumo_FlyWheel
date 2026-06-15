#!/usr/bin/env bash
# FR13 ForkMargin BootCapture (ProbeClassify PHASE 1) -- boot the LOCKED cat9
# spec server with FR13_SCAN_ALIGN=1 MODE=body (K1 ON = candidate config) +
# FR13_FORK_MARGIN_DUMP=1 (READ-ONLY committer-fork classifier) + ENFORCE_EAGER=1.
# Assert: (i) DUMP FLAG LIVE -- worker /proc/<pid>/environ has
# FR13_FORK_MARGIN_DUMP=1 AND the dump jsonl is NON-EMPTY with per-node margins;
# (ii) K1 live -- FR13_SCAN_ALIGN=1 in worker environ; (iii) tok/draft==9
# (cat9 engaged); (iv) within-boot det. Capture served streams, then teardown +
# recover. Recurrent-oracle rescore + A/B classify is a SEPARATE phase.
set -euo pipefail
REPO="${REPO:-/home/mark/shared/lumoFlyWheel}"
CONTAINER="${CONTAINER:-fr13-forked-fa2-tree}"
PORT="${PORT:-9950}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:${PORT}}"
MODEL="${MODEL:-qwen3.6-27b}"
RUN_DIR="${RUN_DIR:-$REPO/output/fr13_fork_margin_probe}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/logs}"
PROMPTS="${PROMPTS:-$REPO/output/fr13_acceptance_ladder/prompts_swe4.json}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1200}"
MAX_TOKENS="${MAX_TOKENS:-128}"; TOP_K="${TOP_K:-20}"; SEED="${SEED:-1313}"; MODE="${MODE:-tree_mtp}"
LAUNCHER="$REPO/scripts/fr13_launch_locked.sh"
CAP_OUT="$OUT_DIR/probe_us_k1_forkmargin.json"
DUMP="$OUT_DIR/fr13_fork_margin_dump.jsonl"
BOOT_LOG="$OUT_DIR/fork_margin_boot.log"
NEEDLE="$OUT_DIR/fork_margin_flag_live.needle"
SUMMARY="$OUT_DIR/fork_margin_capture_summary.json"
NUM_NODES=9
mkdir -p "$OUT_DIR"
# clean any prior dump so non-emptiness proves THIS boot wrote it
rm -f "$DUMP"
fail(){ echo "FORK_MARGIN_BOOT FAIL: $*" >&2; }

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

echo "[fork_margin] LOCKED cat9 + FR13_SCAN_ALIGN=1 MODE=body + FR13_FORK_MARGIN_DUMP=1 ENFORCE_EAGER=1"
recover_host; assert_host_free
[[ "$(docker ps -q | wc -l | tr -d ' ')" != "0" ]] && { fail "docker ps not empty"; docker ps >&2; exit 3; }
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "[boot] locked server (bg) -> $BOOT_LOG"
ENFORCE_EAGER=1 \
FR13_SCAN_ALIGN=1 FR13_SCAN_ALIGN_MODE=body \
FR13_FORK_MARGIN_DUMP=1 FR13_FORK_MARGIN_DUMP_PATH=/logs/fr13_fork_margin_dump.jsonl \
LOG_DIR="$OUT_DIR" FR13_RUN_DIR="$RUN_DIR" \
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

# ===== NON-VACUITY (i+ii): FLAGS LIVE -- bridge-needle worker /proc/<pid>/environ =====
echo "[flag_live] bridge-needle /proc/<pid>/environ for FR13_FORK_MARGIN_DUMP + FR13_SCAN_ALIGN"
docker exec "$CONTAINER" bash -lc '
set -e
hit_fmd=""; hit_align=""; hit_mode=""; pids=""
for p in $(ls /proc | grep -E "^[0-9]+$"); do
  e="/proc/$p/environ"
  [ -r "$e" ] || continue
  cmd="$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)"
  case "$cmd" in
    *vllm*|*VllmWorker*|*EngineCore*|*python*) ;;
    *) continue ;;
  esac
  fmd="$(tr "\0" "\n" < "$e" | grep -E "^FR13_FORK_MARGIN_DUMP=" || true)"
  a="$(tr "\0" "\n" < "$e" | grep -E "^FR13_SCAN_ALIGN=" || true)"
  m="$(tr "\0" "\n" < "$e" | grep -E "^FR13_SCAN_ALIGN_MODE=" || true)"
  if [ -n "$fmd" ]; then
    pids="$pids $p"
    echo "PID $p  cmd=[$cmd]"
    echo "   $fmd"; echo "   $a"; echo "   $m"
    hit_fmd="$fmd"; hit_align="$a"; hit_mode="$m"
  fi
done
echo "SUMMARY hit_fmd=[$hit_fmd] hit_align=[$hit_align] hit_mode=[$hit_mode] pids=[$pids]"
' | tee "$NEEDLE"

if ! grep -q "FR13_FORK_MARGIN_DUMP=1" "$NEEDLE"; then
  fail "DUMP FLAG NOT LIVE: FR13_FORK_MARGIN_DUMP=1 absent in worker /proc environ (bug-class #9/#10)"; exit 6
fi
if ! grep -q "FR13_SCAN_ALIGN=1" "$NEEDLE"; then
  fail "K1 FLAG NOT LIVE: FR13_SCAN_ALIGN=1 absent in worker /proc environ (bug-class #9)"; exit 6
fi
if ! grep -q "FR13_SCAN_ALIGN_MODE=body" "$NEEDLE"; then
  fail "MODE NOT LIVE: FR13_SCAN_ALIGN_MODE=body absent in worker /proc environ"; exit 6
fi
echo "[flag_live] CONFIRMED FR13_FORK_MARGIN_DUMP=1 + FR13_SCAN_ALIGN=1 + MODE=body in worker environ"

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
  --arm us_tree_K1_forkmargin --out "$CAP_OUT" \
  --endpoint "$ENDPOINT" --model "$MODEL" --prompts-file "$PROMPTS" \
  --max-tokens "$MAX_TOKENS" --top-k "$TOP_K" --mode "$MODE" --seed "$SEED"

DET_OK="$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));w=d.get("within_boot_det_rep1_eq_rep2");print("true" if isinstance(w,list) and len(w)>0 and all(bool(x) for x in w) else "false")' "$CAP_OUT")"
WITHIN="$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(json.dumps(d.get("within_boot_det_rep1_eq_rep2")))' "$CAP_OUT")"
echo "[capture] within_boot_det=$WITHIN det_ok=$DET_OK"

# ===== NON-VACUITY (i, second half): DUMP NON-EMPTY with per-node margins =====
echo "[dump_live] verify $DUMP non-empty with per-node verify margins"
DUMP_STATS="$(python3 - "$DUMP" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
    print(json.dumps({"exists":False,"n":0})); raise SystemExit(0)
n=0; n_fork=0; n_with_margin=0; n_clear=0; n_neartie=0
for line in p.read_text().splitlines():
    line=line.strip()
    if not line: continue
    r=json.loads(line); n+=1
    if r.get("is_fork"): n_fork+=1
    # a per-node margin present on at least one divergence slot
    for k in ("winner_div_margin","spine_div_margin","split_node_margin"):
        m=r.get(k)
        if m and "verify_top2_margin_nat" in m:
            n_with_margin+=1
            break
print(json.dumps({"exists":True,"n_records":n,"n_fork_records":n_fork,
                  "n_records_with_node_margin":n_with_margin}))
PY
)"
echo "[dump_live] $DUMP_STATS"
DUMP_OK="$(python3 -c 'import json,sys;d=json.loads(sys.argv[1]);print("true" if d.get("exists") and d.get("n_records",0)>0 and d.get("n_records_with_node_margin",0)>0 else "false")' "$DUMP_STATS")"
[[ "$DUMP_OK" != "true" ]] && { fail "DUMP VACUOUS: $DUMP empty or has no per-node margins (bug-class #9)"; exit 7; }
echo "[dump_live] CONFIRMED dump non-empty with per-node verify margins"

ACCEPTED="$(metric vllm:spec_decode_num_accepted_tokens_total)"
NUM_DRAFTS2="$(metric vllm:spec_decode_num_drafts_total)"
ACCEPT_PER_EVENT="$(python3 -c 'import sys;print(repr(float(sys.argv[1])/float(sys.argv[2])))' "$ACCEPTED" "$NUM_DRAFTS2" 2>/dev/null || echo null)"
echo "[accept] accepted=$ACCEPTED drafts=$NUM_DRAFTS2 accept/event=$ACCEPT_PER_EVENT"

python3 - <<PY
import json
json.dump({
 "name":"us_tree_K1_forkmargin","scan_align":"1","mode":"body",
 "fork_margin_dump":"1","num_nodes":$NUM_NODES,
 "tok_per_draft":float("$TOK_PER_DRAFT"),"engaged":("$ENGAGED"=="true"),
 "within_boot_det":json.loads('$WITHIN'),"det_ok":("$DET_OK"=="true"),
 "dump_stats":json.loads('''$DUMP_STATS'''),
 "accepted":float("$ACCEPTED"),"drafts":float("$NUM_DRAFTS2"),
 "accept_per_event":float("$ACCEPT_PER_EVENT"),
 "capture":"$CAP_OUT","dump":"$DUMP","boot_log":"$BOOT_LOG","needle":"$NEEDLE"
}, open("$SUMMARY","w"), indent=2)
print(open("$SUMMARY").read())
PY
echo "[done] capture=$CAP_OUT dump=$DUMP summary=$SUMMARY"
