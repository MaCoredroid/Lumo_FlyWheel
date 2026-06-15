#!/usr/bin/env bash
# FR13 SPEED PHASE 0 (2026-06-15): B=1 decode s/fwd baselines, decode_seconds basis.
# Arms: native E5 / E4 / E3 (fr10_launch_speed_server, naive_mtp) + cat9 (fr13_launch_locked, tree_mtp).
# GPU SERIALIZED -> one boot at a time on PORT 9950, recover_host_memory + docker-empty between.
# s/fwd = delta(vllm:request_decode_time_seconds_sum) / delta(vllm:spec_decode_num_drafts_total)
# (RAW /metrics counter delta over the probe load; NEVER TPS/accept/wall). Lean Phase 0:
# s/fwd + depth references (E3/E4 unmeasured); the lossless BAR is already confirmed at scale
# (big-denom cat9 13.55% ~= native 13.99%), the per-change lossless gate is applied per-lever later.
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel
cd "$REPO"
OUT=output/fr13_speed_phase0
mkdir -p "$OUT"
PROMPTS="$REPO/output/fr13_acceptance_ladder/prompts_swe4.json"
SEED=1313
PORT=9950
ENDPOINT="http://127.0.0.1:$PORT"
MODEL=qwen3.6-27b
RESULTS="$OUT/phase0_results.jsonl"
: > "$RESULTS"

recover() { PYTHONPATH="$REPO/src" python3 -c 'from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()' >/dev/null 2>&1 || true; }
assert_free() {
  python3 - <<'PY'
from pathlib import Path
f={}
for l in Path("/proc/meminfo").read_text().splitlines():
    k,v=l.split(":",1); f[k]=int(v.strip().split()[0])
avail=f.get("MemAvailable",0)/1024/1024
swap=(f.get("SwapTotal",0)-f.get("SwapFree",0))/1024/1024
if avail<95 or swap!=0:
    raise SystemExit(f"HYGIENE FAIL MemAvailable={avail:.1f}GiB(<95) swap_used={swap:.2f}GiB")
print(f"[hygiene] MemAvailable={avail:.1f}GiB swap=0 OK")
PY
}
metric() { curl -fsS --max-time 10 "$ENDPOINT/metrics" 2>/dev/null | awk -v n="$1" '$0 ~ ("^" n " ")||$0 ~ ("^" n "{"){v=$NF} END{if(v!="")print v}'; }
wait_health() {
  local cont="$1" deadline; deadline=$(( $(date +%s)+900 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    local st; st="$(docker inspect -f '{{.State.Status}}' "$cont" 2>/dev/null || echo absent)"
    if [ "$st" = exited ] || [ "$st" = dead ]; then
      echo "[boot] $cont $st before health; log tail:"; docker logs --tail 50 "$cont" 2>&1 | tail -50; return 1
    fi
    curl -fsS --max-time 5 "$ENDPOINT/health" >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "[boot] $cont health TIMEOUT"; return 1
}
teardown() { docker rm -f "$1" >/dev/null 2>&1 || true; recover; }

# measure_arm <name> <container> <mode> <expect_tok_per_draft>  (server already booted+healthy)
measure_arm() {
  local name="$1" cont="$2" mode="$3" expect="$4"
  # engage warmup
  local wprompt
  wprompt="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[0][:400])' "$PROMPTS")"
  local wpayload
  wpayload="$(python3 - "$wprompt" <<'PY'
import json,sys
print(json.dumps({"model":"qwen3.6-27b","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":24,"temperature":0.0,"seed":1313}))
PY
)"
  curl -fsS --max-time 120 -H 'Content-Type: application/json' -d "$wpayload" "$ENDPOINT/v1/chat/completions" >/dev/null 2>&1 || { echo "[$name] warmup FAIL"; return 1; }
  local dt nd
  dt="$(metric vllm:spec_decode_num_draft_tokens_total)"; nd="$(metric vllm:spec_decode_num_drafts_total)"
  local tpd; tpd="$(python3 -c 'print(float("'"${dt:-0}"'")/float("'"${nd:-1}"'") if float("'"${nd:-0}"'")>0 else -1)')"
  echo "[$name] tok/draft=$tpd expect=$expect"
  if ! python3 -c "import sys; sys.exit(0 if abs($tpd-$expect)<1e-6 else 1)"; then
    echo "[$name] ENGAGEMENT FAIL tok/draft=$tpd != $expect"; return 1
  fi
  # decode_seconds + counters BEFORE the load
  local ds0 sd0 ac0; ds0="$(metric vllm:request_decode_time_seconds_sum)"; sd0="$(metric vllm:spec_decode_num_drafts_total)"; ac0="$(metric vllm:spec_decode_num_accepted_tokens_total)"
  # drive B=1 greedy decode load via the tps probe
  python3 "$REPO/scripts/fr10_quick_decode_tps_probe.py" \
    --endpoint "$ENDPOINT" --model "$MODEL" --prompts-file "$PROMPTS" \
    --modes "$mode" --out "$OUT/${name}_tps.json" \
    --max-tokens 128 --temperature 0.0 --batch-size 1 --samples-per-prompt 4 \
    --seed "$SEED" --wait-health 0 >/dev/null 2>&1 || echo "[$name] tps probe nonzero exit (continuing, using metric delta)"
  local ds1 sd1 ac1; ds1="$(metric vllm:request_decode_time_seconds_sum)"; sd1="$(metric vllm:spec_decode_num_drafts_total)"; ac1="$(metric vllm:spec_decode_num_accepted_tokens_total)"
  python3 - "$name" "$expect" "$tpd" "$ds0" "$ds1" "$sd0" "$sd1" "$ac0" "$ac1" >> "$RESULTS" <<'PY'
import json,sys
name,expect,tpd,ds0,ds1,sd0,sd1,ac0,ac1=sys.argv[1:10]
def f(x):
    try: return float(x)
    except: return None
dds=f(ds1)-f(ds0) if f(ds1) is not None and f(ds0) is not None else None
dsd=f(sd1)-f(sd0) if f(sd1) is not None and f(sd0) is not None else None
dac=f(ac1)-f(ac0) if f(ac1) is not None and f(ac0) is not None else None
spf = dds/dsd if dds is not None and dsd not in (None,0) else None
ape = dac/dsd if dac is not None and dsd not in (None,0) else None
print(json.dumps({"arm":name,"expect_tpd":float(expect),"tok_per_draft":f(tpd),
  "s_per_fwd":spf,"accept_per_event":ape,"delta_decode_seconds":dds,"delta_drafts":dsd,"delta_accepted":dac}))
PY
  echo "[$name] recorded: $(tail -1 "$RESULTS")"
}

# ---- arm runner: boots, waits, measures, tears down ----
run_native() {  # name N
  local name="$1" N="$2"
  echo "===== $name (native MTP-$N) $(date -u +%FT%TZ) ====="
  recover; assert_free || { echo "[$name] hygiene SKIP"; return 1; }
  docker ps -q | grep -q . && { echo "[$name] docker not empty, cleaning"; docker rm -f fr10-speed-start fr13-forked-fa2-tree >/dev/null 2>&1 || true; }
  NUM_SPECULATIVE_TOKENS="$N" \
  SPEC_CONFIG="{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$N}" \
  CONTAINER=fr10-speed-start PORT="$PORT" FR10_METRICS=0 \
    bash "$REPO/scripts/fr10_launch_speed_server.sh" > "$OUT/${name}_boot.log" 2>&1 &
  sleep 8
  if wait_health fr10-speed-start; then measure_arm "$name" fr10-speed-start naive_mtp "$N" || echo "[$name] measure FAIL"; fi
  teardown fr10-speed-start
}
run_cat9() {
  echo "===== cat9 (tree_mtp, locked) $(date -u +%FT%TZ) ====="
  recover; assert_free || { echo "[cat9] hygiene SKIP"; return 1; }
  docker rm -f fr10-speed-start fr13-forked-fa2-tree >/dev/null 2>&1 || true
  CONTAINER=fr13-forked-fa2-tree PORT="$PORT" FR10_METRICS=0 \
    bash "$REPO/scripts/fr13_launch_locked.sh" > "$OUT/cat9_boot.log" 2>&1 &
  sleep 8
  if wait_health fr13-forked-fa2-tree; then measure_arm cat9 fr13-forked-fa2-tree tree_mtp 9 || echo "[cat9] measure FAIL"; fi
  teardown fr13-forked-fa2-tree
}

echo "######## FR13 SPEED PHASE 0 START $(date -u +%FT%TZ) ########"
run_native native_e5 5
run_native native_e4 4
run_native native_e3 3
run_cat9
echo "######## PHASE 0 CONSOLIDATE ########"
python3 - "$RESULTS" <<'PY'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(json.dumps({"schema":"fr13.speed_phase0.v1","arms":rows},indent=2))
PY
echo "######## PHASE 0 DONE $(date -u +%FT%TZ) ########"
