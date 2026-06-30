#!/usr/bin/env bash
# FR13 ReshapeBoot capture (PHASE 1) — boot ONE reshaped tree, prove reshape
# applied (tok/draft == len(TREE)), capture served streams + within-boot det +
# accept/event, teardown+recover. Flip rescoring is done SEPARATELY by the
# RECURRENT oracle (fr13_recurrent_decode_oracle.py) on the saved capture JSON
# (the HTTP teacher-force instrument is the WRONG/chunked oracle per directive).
#
#   scripts/fr13_reshape_boot_capture.sh <name> "<TREE>"
set -euo pipefail
NAME="${1:-}"; TREE_ARG="${2:-}"
[[ -z "$NAME" || -z "$TREE_ARG" ]] && { echo "usage: $0 <name> \"<TREE>\"" >&2; exit 2; }

REPO="${REPO:-/home/mark/shared/lumoFlyWheel}"
CONTAINER="${CONTAINER:-fr13-forked-fa2-tree}"
PORT="${PORT:-9950}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:${PORT}}"
MODEL="${MODEL:-qwen3.6-27b}"
OUT_DIR="${OUT_DIR:-$REPO/output/fr13_reshape_boot}"
PROMPTS="${PROMPTS:-$REPO/output/fr13_acceptance_ladder/prompts_swe4.json}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1200}"
MAX_TOKENS="${MAX_TOKENS:-128}"; TOP_K="${TOP_K:-20}"; SEED="${SEED:-1313}"; MODE="${MODE:-tree_mtp}"
LAUNCHER="$REPO/scripts/fr13_launch_forked_fa2_tree_server.sh"
CAP_OUT="$OUT_DIR/${NAME}_capture.json"; BOOT_LOG="$OUT_DIR/${NAME}_boot.log"
SUMMARY="$OUT_DIR/${NAME}_summary.json"
mkdir -p "$OUT_DIR"
fail(){ echo "RESHAPE_BOOT FAIL [$NAME]: $*" >&2; }

NUM_NODES="$(TREE="$TREE_ARG" python3 -c 'import ast,os;print(len(ast.literal_eval(os.environ["TREE"])))')"
MAX_DEPTH="$(TREE="$TREE_ARG" python3 -c 'import ast,os;t=ast.literal_eval(os.environ["TREE"]);print(max((len(p) for p in t),default=0))')"

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
if avail<95 or swap!=0:
    raise SystemExit(f"hygiene FAIL MemAvailable={avail:.1f}GiB swap_used={swap/1024/1024:.2f}GiB")
print(f"[hygiene] MemAvailable={avail:.1f}GiB swap_used=0 OK")
PY
}
metric(){ curl -fsS --max-time 10 "${ENDPOINT}/metrics" 2>/dev/null | awk -v n="$1" '$0 ~ ("^" n " ")||$0 ~ ("^" n "{"){v=$NF} END{if(v!="")print v}'; }
teardown(){ echo "[teardown] docker rm -f $CONTAINER"; docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; echo "[teardown] recover_host_memory"; recover_host||true; }

echo "[reshape_boot] name=$NAME num_nodes=$NUM_NODES max_depth=$MAX_DEPTH TREE=$TREE_ARG"
recover_host; assert_host_free
[[ "$(docker ps -q | wc -l | tr -d ' ')" != "0" ]] && { fail "docker ps not empty"; docker ps >&2; exit 3; }
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "[boot] forked server (bg) -> $BOOT_LOG"
TREE="$TREE_ARG" ENFORCE_EAGER=1 \
FR10_DECODE_MODE_DEFAULT=tree_mtp FR13_DRAFTER_SINGLE_LOGITS=1 FR13_EAGER_PACK=1 \
FR13_TREE_CONV_FUSED=1 FR13_TREE_SAMPLE_ROW=1 FR13_REPLAY_ROUTE=1 FR13_FA2_TREE_BIAS=1 \
FR13_FA2_PREFILL_NATIVE=1 FR13_TREE_ATTN_EXP2_SOFTMAX=1 FR13_CONV_COMMITTED_PATH=1 \
BATCH_INVARIANT=0 FR13_BI_TREE_ATTN=0 FR10_METRICS=0 \
LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
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

# reset spec counters baseline (reset prefix cache too)
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
[[ "$ENGAGED" != "true" ]] && { fail "ENGAGEMENT GATE tok/draft($TOK_PER_DRAFT)!=len(TREE)($NUM_NODES); recording NOTHING"; exit 5; }

# verify the engine actually logged the reshaped speculative_token_tree (not cat9)
echo "[engage] grep boot log for served tree shape tag"
grep -iE "FR13_RESHAPE_DEPTH3|shape.*(chain3|cat3w)|speculative_token_tree" "$BOOT_LOG" | tail -5 || true

echo "[capture] gold_margin_probe -> $CAP_OUT"
python3 "$REPO/scripts/fr13_gold_margin_probe.py" capture \
  --arm tree --out "$CAP_OUT" \
  --endpoint "$ENDPOINT" --model "$MODEL" --prompts-file "$PROMPTS" \
  --max-tokens "$MAX_TOKENS" --top-k "$TOP_K" --mode "$MODE" --seed "$SEED"

DET_OK="$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));w=d.get("within_boot_det_rep1_eq_rep2");print("true" if isinstance(w,list) and len(w)>0 and all(bool(x) for x in w) else "false")' "$CAP_OUT")"
WITHIN="$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(json.dumps(d.get("within_boot_det_rep1_eq_rep2")))' "$CAP_OUT")"
echo "[capture] within_boot_det=$WITHIN det_ok=$DET_OK"

ACCEPTED="$(metric vllm:spec_decode_num_accepted_tokens_total)"
NUM_DRAFTS2="$(metric vllm:spec_decode_num_drafts_total)"
ACCEPT_PER_EVENT="$(python3 -c 'import sys;print(repr(float(sys.argv[1])/float(sys.argv[2])))' "$ACCEPTED" "$NUM_DRAFTS2" 2>/dev/null || echo null)"
echo "[accept] accepted=$ACCEPTED drafts=$NUM_DRAFTS2 accept/event=$ACCEPT_PER_EVENT"

python3 - <<PY
import json
json.dump({
 "name":"$NAME","tree":"$TREE_ARG","num_nodes":$NUM_NODES,"max_depth":$MAX_DEPTH,
 "tok_per_draft":float("$TOK_PER_DRAFT"),"engaged":("$ENGAGED"=="true"),
 "within_boot_det":json.loads('$WITHIN'),"det_ok":("$DET_OK"=="true"),
 "accepted":float("$ACCEPTED"),"drafts":float("$NUM_DRAFTS2"),
 "accept_per_event":float("$ACCEPT_PER_EVENT"),
 "capture":"$CAP_OUT","boot_log":"$BOOT_LOG"
}, open("$SUMMARY","w"), indent=2)
print(open("$SUMMARY").read())
PY
echo "[done] capture=$CAP_OUT summary=$SUMMARY"
