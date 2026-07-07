#!/usr/bin/env bash
# fr13_rp2_order_probe.sh — FR13 §47 ORDER-vs-SEED discriminator (2026-07-05).
#
# FINDING IT TESTS (campaign §47): within one cat8+cache boot, token-1 logprob of the
# byte-identical cold turn-1 request VARIES across requests (-0.001 .. -1.374) while
# cat8+nocache is -0.001 flat 16/16 (seeded2turn artifacts). Every first-request-of-boot
# ever measured was clean. Hypothesis: the "per-seed route table" was never seed-keyed —
# it is REQUEST-ORDER/HISTORY-keyed state contamination (bug-class 1), because in every
# prior probe seed==send-position.
#
# DESIGN (discriminates order vs seed by pinning the seed):
#   arm cat8_cache:
#     P1  seed=5 x10, reset_prefix_cache before EACH  (fixed seed; any variation = ORDER)
#     P2  seed=5 x6,  NO reset (full-prefix HITs; fixed-seed miss-vs-hit repeat axis)
#     P3  seeds 1..8, reset before each               (ties to the old per-seed tables)
#   arm cat8_nocache (control):
#     C1  seed=5 x6, reset before each
#     C2  seeds 1..4, reset before each
# All requests: logprobs=true, top_logprobs=20 (token-1 chosen-token logprob = the metric;
# reliable: token-1 is a normal-sampler token, raw pre-temperature log_softmax, read-only
# — source audit wf_20eeac71 Q2). temp 0.6 (standing rule), MAX_TOKENS default 448
# (bounds garble runaways; routes classify by first tool_call name / NO_TOOL+finish).
# /metrics prefix_cache + spec counters bracketed per request (cold verification).
#
# PREDICTIONS (banked §47): P1 sample1 clean ('Let'~-0.001, route=agent) then position-
# dependent drift => ORDER CONFIRMED, seed exonerated. If P1 flat: order refuted, P3
# spread would imply a genuine seed->forward leak. C1/C2 flat -0.001 throughout.
#
# Usage: bash scripts/probes/fr13_rp2_order_probe.sh          # both arms
#        ARMS="cat8_cache" bash scripts/probes/fr13_rp2_order_probe.sh
# Env:   MAX_TOKENS=448 TEMPERATURE=0.6 GPU_UTIL=0.82 GPU_GUARD_FLOOR_MIB=6500
#        RR=output/fr13_rp2_order (results root; output/ is gitignored)
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true

STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$STAGE/route_probe_payload.json"
PORT=9950
MODEL=qwen3.6-27b
RR="${RR:-output/fr13_rp2_order}"
GPU_UTIL="${GPU_UTIL:-0.82}"
GPU_GUARD_FLOOR_MIB="${GPU_GUARD_FLOOR_MIB:-6500}"
MAX_TOKENS="${MAX_TOKENS:-448}"
TEMPERATURE="${TEMPERATURE:-0.6}"
ARMS="${ARMS:-cat8_cache cat8_nocache}"
export GPU_UTIL GPU_GUARD_FLOOR_MIB

EXPECT_TOOLS_SHA="9d12fd4bf2527b969da87523581cce38744ccdd9ab45eea7c433e3e8d44b9667"
EXPECT_MSGS_SHA="698ce22713371c4d9764b570e6f5b87915653df7b1415dd9caf528b51128f5df"
CAT8_TREE="[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0)]"

mkdir -p "$RR"
echo "=== rp2_order_probe: arms=[$ARMS] max_tokens=$MAX_TOKENS -> $RR ==="
date -u +%Y-%m-%dT%H:%M:%SZ

# ---- FAIL-LOUD 0: payload integrity (same request as every prior probe) ----
[[ -f "$PAYLOAD" ]] || { echo "FAIL: payload not found: $PAYLOAD"; exit 2; }
python3 - "$PAYLOAD" "$EXPECT_TOOLS_SHA" "$EXPECT_MSGS_SHA" "$MODEL" <<'PY'
import json, sys, hashlib
payload, exp_tools, exp_msgs, model = sys.argv[1:5]
d = json.load(open(payload))
def canon(o): return json.dumps(o, sort_keys=True, separators=(',',':'), ensure_ascii=False).encode()
assert d.get("model") == model
assert hashlib.sha256(canon(d["tools"])).hexdigest() == exp_tools, "TOOL SCHEMA CHANGED"
assert hashlib.sha256(canon(d["messages"])).hexdigest() == exp_msgs, "MESSAGES CHANGED"
assert "seed" not in d, "base payload must not carry a seed"
print("[integrity] OK payload matches the canonical turn-1 request")
PY
(( $? == 0 )) || { echo "FAIL: payload integrity"; exit 2; }

# ---- FAIL-LOUD 1: co-residency guard ----
if [[ -n "$(docker ps -q 2>/dev/null)" ]]; then
  echo "FAIL: a container is already running (GPU serialized):"; docker ps; exit 2
fi

recover_host(){ PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}

build_send_payload(){ # $1=dest ; injects logprobs + overrides
  python3 - "$PAYLOAD" "$1" "$MAX_TOKENS" "$TEMPERATURE" <<'PY'
import json, sys
src, dst, mx, tp = sys.argv[1:5]
d = json.load(open(src))
d["max_tokens"] = int(mx); d["temperature"] = float(tp)
d["logprobs"] = True; d["top_logprobs"] = 20
json.dump(d, open(dst, "w"), ensure_ascii=False)
PY
}

arm_env(){ case "$1" in
    cat8_cache) cat <<E
TREE=$CAT8_TREE
FR13_ENABLE_APC=1
FR13_APC_EXACT_SEED=1
MAMBA_BLOCK_SIZE=1024
MAMBA_SSM_CACHE_DTYPE=float32
E
      ;;
    cat8_nocache) cat <<E
TREE=$CAT8_TREE
FR13_ENABLE_APC=1
FR13_APC_CONFIG_ONLY=1
MAMBA_BLOCK_SIZE=1024
MAMBA_SSM_CACHE_DTYPE=float32
E
      ;;
    cat8_cache_hrs) cat <<E
TREE=$CAT8_TREE
FR13_ENABLE_APC=1
FR13_APC_EXACT_SEED=0
FR13_APC_HIT_RECURRENT_SUFFIX=1
FR13_APC_HIT_SUFFIX_CAP=64
MAMBA_BLOCK_SIZE=1024
MAMBA_SSM_CACHE_DTYPE=float32
E
      ;;
    *) echo "FAIL_UNKNOWN_ARM"; return 1 ;;
  esac
}
arm_cache_expect(){ case "$1" in cat8_cache) echo "True";; cat8_cache_hrs) echo "True";; cat8_nocache) echo "False";; esac; }

# battery lines: "tag seed reset". BATTERY_PHASES (space-sep, e.g. "P1") filters
# to matching phases (default: all) — used by short diagnostic boots (§51 E4CAP).
BATTERY_PHASES="${BATTERY_PHASES:-}"
arm_battery(){ _arm_battery_raw "$1" | { if [[ -n "$BATTERY_PHASES" ]]; then
    grep -E "^($(echo "$BATTERY_PHASES" | tr ' ' '|'))_"; else cat; fi; }; }
_arm_battery_raw(){ case "$1" in
  cat8_cache|cat8_cache_hrs)
    for k in 1 2 3 4 5 6 7 8 9 10; do echo "P1_$k 5 1"; done
    for k in 1 2 3 4 5 6; do echo "P2_$k 5 0"; done
    for s in 1 2 3 4 5 6 7 8; do echo "P3_s$s $s 1"; done ;;
  cat8_nocache)
    for k in 1 2 3 4 5 6; do echo "C1_$k 5 1"; done
    for s in 1 2 3 4; do echo "C2_s$s $s 1"; done ;;
esac; }

run_arm(){
  local arm="$1"
  local cdir="$RR/$arm"; local container="fr13-rp2-$arm"
  mkdir -p "$cdir/logs"
  echo ""; echo "########## ARM $arm  ($(date -u +%H:%M:%S)) ##########"
  (
    set -uo pipefail
    export FR10_METRICS=0 BATCH_INVARIANT=0 LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16
    export CONTAINER="$container" PORT="$PORT" MAX_NUM_SEQS=1
    export GPU_UTIL GPU_GUARD_FLOOR_MIB
    export LOG_DIR="$PWD/$cdir/logs" FR13_RUN_DIR="$PWD/$cdir"
    export FR13_SERVE_LOG=1
    local kv
    while IFS= read -r kv; do
      [[ -z "$kv" ]] && continue
      [[ "$kv" == "FAIL_UNKNOWN_ARM" ]] && { echo "FAIL: unknown arm $arm"; exit 2; }
      export "$kv"
    done < <(arm_env "$arm")

    docker rm -f "$container" >/dev/null 2>&1 || true
    recover_host
    if [[ -n "$(docker ps -q 2>/dev/null)" ]]; then
      echo "FAIL: $arm — container running before boot (co-residency):"; docker ps; exit 3
    fi

    echo "[$arm] boot via fr13_launch_forked_fa2_tree_server.sh (GPU_UTIL=$GPU_UTIL floor=$GPU_GUARD_FLOOR_MIB)"
    bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$cdir/launch.log" 2>&1
    local rc=$?
    (( rc == 0 )) || { echo "FAIL: launcher rc=$rc"; tail -30 "$cdir/launch.log"; exit 3; }

    local t0; t0=$(date +%s); local healthy=0
    while (( $(date +%s) < t0 + 1200 )); do
      if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then healthy=1; break; fi
      if [[ "$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null)" != "running" ]]; then
        echo "FAIL: container died before health"; docker logs "$container" 2>&1 | tail -40; exit 3
      fi
      sleep 5
    done
    (( healthy == 1 )) || { echo "FAIL: health not up in 1200s"; docker logs "$container" 2>&1 | tail -40; exit 3; }
    echo "[$arm] healthy after $(( $(date +%s) - t0 ))s"

    docker logs "$container" > "$cdir/boot_log_snapshot.txt" 2>&1
    docker exec "$container" env 2>/dev/null | sort > "$cdir/container_env.txt" || true

    local models_json; models_json=$(curl -fsS -m 10 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
    echo "$models_json" > "$cdir/models.json"
    echo "$models_json" | grep -q "\"$MODEL\"" || { echo "FAIL: model != $MODEL"; exit 4; }

    local exp_cache; exp_cache=$(arm_cache_expect "$arm")
    grep -q "enable_prefix_caching=$exp_cache" "$cdir/boot_log_snapshot.txt" \
      || { echo "FAIL: expected enable_prefix_caching=$exp_cache not in boot log"; exit 4; }
    case "$arm" in
      cat8_cache)
        grep -q "^FR13_APC_EXACT_SEED=1$" "$cdir/container_env.txt" || { echo "FAIL: EXACT_SEED!=1"; exit 4; }
        grep -q "^ATTENTION_BACKEND=TREE_ATTN$" "$cdir/container_env.txt" || { echo "FAIL: backend!=TREE_ATTN"; exit 4; }
        grep -q "speculative_token_tree" "$cdir/container_env.txt" || { echo "FAIL: no tree in SPEC_CONFIG"; exit 4; };;
      cat8_nocache)
        grep -q "^FR13_APC_EXACT_SEED=0$" "$cdir/container_env.txt" || { echo "FAIL: EXACT_SEED!=0"; exit 4; }
        grep -q "^ATTENTION_BACKEND=TREE_ATTN$" "$cdir/container_env.txt" || { echo "FAIL: backend!=TREE_ATTN"; exit 4; }
        grep -q "speculative_token_tree" "$cdir/container_env.txt" || { echo "FAIL: no tree in SPEC_CONFIG"; exit 4; };;
      cat8_cache_hrs)
        grep -q "^FR13_APC_EXACT_SEED=0$" "$cdir/container_env.txt" || { echo "FAIL: EXACT_SEED!=0"; exit 4; }
        grep -q "^FR13_APC_HIT_RECURRENT_SUFFIX=1$" "$cdir/container_env.txt" || { echo "FAIL: HRS!=1"; exit 4; }
        grep -q "^ATTENTION_BACKEND=TREE_ATTN$" "$cdir/container_env.txt" || { echo "FAIL: backend!=TREE_ATTN"; exit 4; }
        grep -q "speculative_token_tree" "$cdir/container_env.txt" || { echo "FAIL: no tree in SPEC_CONFIG"; exit 4; };;
    esac
    echo "[$arm] boot-config asserts OK (cache=$exp_cache)"

    local send="$cdir/send_payload.json"; build_send_payload "$send"
    : > "$cdir/reset_log.txt"
    local n_total=0 ok=0 first_logprobs_checked=0
    local tag seed do_reset
    while read -r tag seed do_reset; do
      n_total=$((n_total+1))
      if [[ "$do_reset" == "1" ]]; then
        local rcode
        rcode=$(curl -s -o /dev/null -m 20 -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" 2>/dev/null)
        echo "$tag reset_prefix_cache http=$rcode" >> "$cdir/reset_log.txt"
        [[ "$rcode" == "200" ]] || echo "WARN: $arm $tag reset http=$rcode (cold NOT guaranteed)"
      fi
      curl -s -m 20 "http://127.0.0.1:$PORT/metrics" 2>/dev/null \
        | grep -E 'prefix_cache|spec_decode|num_accepted' > "$cdir/metrics_${tag}_pre.txt" || true
      python3 - "$send" "$cdir/send_${tag}.json" "$seed" <<'PYSEED'
import json, sys
d = json.load(open(sys.argv[1])); d["seed"] = int(sys.argv[3])
json.dump(d, open(sys.argv[2], "w"))
PYSEED
      curl -sS -m 900 -H 'Content-Type: application/json' -H 'Authorization: Bearer EMPTY' \
        --data-binary @"$cdir/send_${tag}.json" "http://127.0.0.1:$PORT/v1/chat/completions" \
        > "$cdir/sample_${tag}.json" 2>"$cdir/sample_${tag}.err"
      curl -s -m 20 "http://127.0.0.1:$PORT/metrics" 2>/dev/null \
        | grep -E 'prefix_cache|spec_decode|num_accepted' > "$cdir/metrics_${tag}_post.txt" || true
      if python3 - "$cdir/sample_${tag}.json" "$tag" <<'PY'
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: print("  PARSE_FAIL", str(e)[:120]); sys.exit(1)
ch = (d.get("choices") or [])
if not ch: print("  NO_CHOICES", json.dumps(d)[:200]); sys.exit(1)
c0 = ch[0]; m = c0.get("message") or {}
tc = m.get("tool_calls") or []
route = (tc[0].get("function",{}).get("name") if tc else "NO_TOOL")
lp = c0.get("logprobs") or {}
cont = lp.get("content") or []
t1 = (f"{cont[0].get('token','?')[:8]}@{cont[0].get('logprob'):.4f}" if cont else "NO_LOGPROBS")
print(f"  {sys.argv[2]} route={route} finish={c0.get('finish_reason')} ct={(d.get('usage') or {}).get('completion_tokens')} tok1={t1}")
sys.exit(0 if (tc or (m.get('content') or '').strip() or (m.get('reasoning') or '').strip()) else 1)
PY
      then ok=$((ok+1)); else echo "  WARN: $arm $tag empty/failed"; fi
      # FAIL-LOUD: logprobs must actually be present (first sample, class-9 vacuity guard)
      if (( first_logprobs_checked == 0 )); then
        first_logprobs_checked=1
        python3 - "$cdir/sample_${tag}.json" <<'PY' || { echo "FAIL: logprobs ABSENT in first response — instrument vacuous"; exit 5; }
import json, sys
d = json.load(open(sys.argv[1]))
lp = (d.get("choices") or [{}])[0].get("logprobs") or {}
assert lp.get("content"), "no logprobs content"
assert lp["content"][0].get("top_logprobs"), "no top_logprobs"
print(f"[engagement] logprobs live: {len(lp['content'])} positions, top-{len(lp['content'][0]['top_logprobs'])}")
PY
      fi
    done < <(arm_battery "$arm")
    echo "[$arm] non-empty samples: $ok/$n_total"

    docker logs "$container" > "$cdir/docker_full.log" 2>&1 || true
    docker rm -f "$container" >/dev/null 2>&1 || true
    recover_host
    sleep 2
    (( ok > 0 )) || { echo "FAIL: $arm ZERO non-empty completions"; exit 5; }
    echo "[$arm] DONE ($(date -u +%H:%M:%S))"
  )
  local arc=$?
  docker rm -f "$container" >/dev/null 2>&1 || true
  return $arc
}

RC=0
for arm in $ARMS; do
  run_arm "$arm" || { echo "ARM $arm FAILED (rc=$?)"; RC=1; }
done

echo ""
echo "=== rp2 boots complete (rc=$RC) — inline reduce ==="
python3 "$STAGE/fr13_rp2_order_reduce.py" --run-root "$RR" || echo "WARN: reduce failed (artifacts intact in $RR)"
exit $RC
