#!/usr/bin/env bash
# ============================================================================
# seeded2turn_run.sh — FR13 tree+cache DECODE-losslessness 2-turn seeded probe
# (campaign §30 instrument; builds on staged/route_probe.sh VERBATIM presets).
#
# GOAL (crisp, §28): make tree+cache-ON DECODE bit-identical to tree+cache-OFF on
# identical input, and MEASURE it with on-distribution SAME-SEED PAIRED STREAMS
# (NOT teacher-forcing, per user §30). For each arm and each seed k:
#     [reset_prefix_cache ONCE]  -> TURN-1 (seed=k, cold prefill)  [NO reset]
#                                -> TURN-2 (seed=k, HITS the cached prefix)
# Three questions from ONE probe (see seeded2turn_reduce.py):
#   1. COLD carrier (cross-boot, floor-bracketed): TURN-1 cache-ON vs cache-OFF
#      first-divergent TOKEN, bracketed by the cache-OFF-vs-cache-OFF floor (A').
#   2. RESTORE carrier (SAME-BOOT, confound-FREE): within a cache-ON boot,
#      TURN-1(miss) vs TURN-2(hit) — identical input, identical kernels/layout,
#      the ONLY diff is recompute-vs-restore. A fork here is a pure restore-
#      losslessness failure with ZERO cross-boot autotune confound (item 1d).
#   3. REFOLD value: does arm C (refold) push the restore fork later / vanish vs
#      arm B (conv-only)?  + the mechanical redirect_used>0 liveness gate.
#
# ARMS (serial, one GPU on GB10; all cat8 TREE, ENFORCE_EAGER=1, temp 0.6):
#   A  cat8_nocache        LOSSLESS REFERENCE   (CONFIG_ONLY => cache OFF)
#   A' cat8_nocache_b      FLOOR SELF-CHECK     (2nd nocache boot; brackets the
#                                                cross-boot autotune floor — item 1c)
#   B  cat8_cache          GIVE-UP config       (EXACT_SEED cache, refold OFF)
#   C  cat8_cache_refold   B + BLOCK_REFOLD=1 REFOLD_TO_SNAPSHOT=1 (MUST prove
#                          redirect_used>0 on turn-2 or the arm is VACUOUS)
#
# WHY paired-boot and not same-boot for the config A/B: enable_prefix_caching is an
# ENGINE-CONSTRUCTION flag, not per-request (launcher:224-254). cache-ON vs cache-OFF
# is irreducibly two boots => the tokens ~11-71 cross-boot autotune floor
# (feedback_no_cross_boot_byte_gate.md) can masquerade as signal. MITIGATION is
# LOAD-BEARING: the A' floor arm measures that floor directly, and the SAME-BOOT
# miss-vs-hit readout (#2 above) needs NO cross-boot comparison at all.
#
# DEFAULT (TURN2_MODE=resend): turn-2 re-sends turn-1's EXACT prompt (§30 literal:
# "turn-2 re-sends turn-1 and HITS the cached prefix"). This maximises the hit (the
# WHOLE prompt hits), keeps turn-2 input byte-identical across arms (clean paired
# diff), and makes the same-boot miss-vs-hit gate exact. TURN2_MODE=conversation
# builds a genuine 2nd user turn on a FROZEN shared assistant turn (see below).
#
# PASS-1 (default, CAPTURE=0): token streams only. NO capture patch, NO launcher
# edit, byte-identical served path. Yields the first-divergent-token readouts and
# the hit/es_seed/redirect asserts. RUN THIS FIRST.
# PASS-2 (CAPTURE=1): windowed per-decode-step GDN-state capture around the fork
# step D learned from pass-1. REQUIRES the default-OFF decode-capture patch applied
# first (staged apply_decode_capture_patch.py) — the run HARD-ASSERTS the capture
# env threaded into the container or fails loud (the H3 lesson).
#
# STRICTLY SERIAL. Refuses to boot on top of any running container (co-residency).
#
# Usage:  bash seeded2turn_run.sh                      # pass-1, all 4 arms
#         ARMS="cat8_cache" bash seeded2turn_run.sh    # a single arm
#         CAPTURE=1 STEP_LO=<D-2> STEP_HI=<D> bash seeded2turn_run.sh   # pass-2
# Env:    N=16 MAX_TOKENS=1024 TEMPERATURE=0.6 TURN2_MODE=resend|conversation
#         GPU_UTIL=0.78 GPU_GUARD_FLOOR_MIB=6500 PORT=9950
#         RR=output/fr13_seeded2turn   (results root; output/ is gitignored)
# ============================================================================
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true

STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The pinned turn-1 payload + integrity shas live with the proven route probe.
PAYLOAD="${PAYLOAD:-$STAGE/../staged/route_probe_payload.json}"
PORT="${PORT:-9950}"
MODEL=qwen3.6-27b
N="${N:-16}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TURN2_MODE="${TURN2_MODE:-resend}"          # resend | conversation
FIXED_TURN2="${FIXED_TURN2:-Given what you found, briefly state your single next concrete step.}"
RR="${RR:-output/fr13_seeded2turn}"
GPU_UTIL="${GPU_UTIL:-0.78}"
GPU_GUARD_FLOOR_MIB="${GPU_GUARD_FLOOR_MIB:-6500}"
# ref arm (A) MUST be the first token in ARMS so conversation-mode can freeze its
# turn-1 completion as the shared assistant turn before B/C run.
ARMS="${ARMS:-cat8_nocache cat8_nocache_b cat8_cache cat8_cache_refold}"
FREEZE_SEED="${FREEZE_SEED:-0}"             # which ref seed becomes the frozen assistant turn
# ---- PASS-2 windowed decode-state capture (default OFF) ----
CAPTURE="${CAPTURE:-0}"
STEP_LO="${STEP_LO:-0}"
STEP_HI="${STEP_HI:-1000000000000}"
CAPTURE_LIMIT="${CAPTURE_LIMIT:-4}"
CAPTURE_LAYER_PREFIX="${CAPTURE_LAYER_PREFIX:-*}"
export GPU_UTIL GPU_GUARD_FLOOR_MIB

EXPECT_TOOLS_SHA="9d12fd4bf2527b969da87523581cce38744ccdd9ab45eea7c433e3e8d44b9667"
EXPECT_MSGS_SHA="698ce22713371c4d9764b570e6f5b87915653df7b1415dd9caf528b51128f5df"
CAT8_TREE="[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0)]"

mkdir -p "$RR"
echo "=== seeded2turn: N=$N turn2_mode=$TURN2_MODE capture=$CAPTURE arms=[$ARMS] -> $RR ==="
date -u +%Y-%m-%dT%H:%M:%SZ

# ---- FAIL-LOUD 0: payload present + tools/messages byte-identical to the request ----
[[ -f "$PAYLOAD" ]] || { echo "FAIL: payload not found: $PAYLOAD"; exit 2; }
python3 - "$PAYLOAD" "$EXPECT_TOOLS_SHA" "$EXPECT_MSGS_SHA" "$MODEL" <<'PY'
import json, sys, hashlib
payload, exp_tools, exp_msgs, model = sys.argv[1:5]
d = json.load(open(payload))
def canon(o): return json.dumps(o, sort_keys=True, separators=(',',':'), ensure_ascii=False).encode()
assert d.get("model") == model, f"payload model {d.get('model')!r} != {model!r}"
assert d.get("tools"), "payload has no tools"
assert d.get("messages"), "payload has no messages"
ts = hashlib.sha256(canon(d["tools"])).hexdigest()
ms = hashlib.sha256(canon(d["messages"])).hexdigest()
assert ts == exp_tools, f"TOOL SCHEMA CHANGED: {ts} != {exp_tools}"
assert ms == exp_msgs,  f"MESSAGES CHANGED: {ms} != {exp_msgs}"
print(f"[integrity] OK model={model} n_tools={len(d['tools'])} tools_sha={ts[:12]} msgs_sha={ms[:12]}")
PY
(( $? == 0 )) || { echo "FAIL: payload integrity"; exit 2; }

if [[ "$CAPTURE" == "1" ]]; then
  echo "[pass-2] CAPTURE=1 STEP_LO=$STEP_LO STEP_HI=$STEP_HI LIMIT=$CAPTURE_LIMIT prefix=$CAPTURE_LAYER_PREFIX"
  echo "[pass-2] this REQUIRES the default-OFF decode-capture patch already applied"
  echo "         (staged apply_decode_capture_patch.py). Env-thread is HARD-ASSERTED below."
  if (( STEP_HI - STEP_LO > 8 )); then
    echo "FAIL: pass-2 window STEP_HI-STEP_LO=$(( STEP_HI - STEP_LO )) > 8. This is UNWINDOWED and"
    echo "  will explode disk (48 layers x all rows x every step). Set STEP_LO=<D-2> STEP_HI=<D>"
    echo "  from the pass-1 first-divergent decode step D (see runbook). Refusing to boot."
    exit 2
  fi
fi

# ---- FAIL-LOUD 1: co-residency guard ----
if [[ -n "$(docker ps -q 2>/dev/null)" ]]; then
  echo "FAIL: a docker container is already running — refusing to boot (GPU serialized). Current:"
  docker ps; exit 2
fi

recover_host(){ PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}

# canonical model-only base payload (seed/max_tokens/temperature/logprobs applied per-send)
base_send(){ # $1=dest
  python3 - "$PAYLOAD" "$1" "$MAX_TOKENS" "$TEMPERATURE" <<'PY'
import json, sys
src, dst, mx, tp = sys.argv[1:5]
d = json.load(open(src))
d.pop("seed", None)
d["max_tokens"] = int(mx)
d["temperature"] = float(tp)
d["logprobs"] = True            # exact per-token sequence for first-divergence (post-hoc; does NOT perturb sampling)
d["top_logprobs"] = 0
d["stream"] = False
json.dump(d, open(dst, "w"), ensure_ascii=False)
PY
}

arm_env(){ # VERBATIM route_probe.sh cat8_nocache / cat8_cache presets (+refold on C)
  case "$1" in
    cat8_nocache|cat8_nocache_b) cat <<E
TREE=$CAT8_TREE
FR13_ENABLE_APC=1
FR13_APC_CONFIG_ONLY=1
MAMBA_BLOCK_SIZE=1024
MAMBA_SSM_CACHE_DTYPE=float32
E
      ;;
    cat8_cache) cat <<E
TREE=$CAT8_TREE
FR13_ENABLE_APC=1
FR13_APC_EXACT_SEED=1
MAMBA_BLOCK_SIZE=1024
MAMBA_SSM_CACHE_DTYPE=float32
FR13_APC_BLOCK_REFOLD=0
FR13_APC_REFOLD_TO_SNAPSHOT=0
E
      ;;
    cat8_cache_refold) cat <<E
TREE=$CAT8_TREE
FR13_ENABLE_APC=1
FR13_APC_EXACT_SEED=1
MAMBA_BLOCK_SIZE=1024
MAMBA_SSM_CACHE_DTYPE=float32
FR13_APC_BLOCK_REFOLD=1
FR13_APC_REFOLD_TO_SNAPSHOT=1
E
      ;;
    *) echo "FAIL_UNKNOWN_ARM"; return 1 ;;
  esac
}
arm_cache_expect(){ case "$1" in
  cat8_cache|cat8_cache_refold) echo "True";;
  cat8_nocache|cat8_nocache_b)  echo "False";;
esac; }
arm_is_cache(){ case "$1" in cat8_cache|cat8_cache_refold) return 0;; *) return 1;; esac; }

# ---- prometheus prefix-cache counters (queries,hits) from a /metrics dump ----
metrics_qh(){ # $1=metrics-file ; echoes "Q H"
  python3 - "$1" <<'PY'
import sys, re
try:
    txt = open(sys.argv[1], errors="ignore").read()
except Exception:
    print("0 0"); sys.exit(0)
def tot(name):
    s = 0.0; got = False
    for ln in txt.splitlines():
        ln = ln.strip()
        if ln.startswith("#") or not ln.startswith(name): continue
        # match "name" or "name{labels}" then a float
        m = re.match(re.escape(name) + r'(\{[^}]*\})?\s+([-+0-9.eE]+)$', ln)
        if m:
            s += float(m.group(2)); got = True
    return s if got else 0.0
print(f"{tot('vllm:prefix_cache_queries_total')} {tot('vllm:prefix_cache_hits_total')}")
PY
}
snap_metrics(){ # $1=dest-file
  curl -fsS -m 10 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || true
  curl -fsS -m 10 "http://127.0.0.1:$PORT/metrics" > "$1" 2>/dev/null || : > "$1"
}

# ---- read FR13_OBS counters (prefer live periodic FR13_OBS_SUMMARY eng line; fall
#      back to the atexit fr13_obs_final.json). §24: atexit races teardown, so the
#      periodic FR13_OBS_SUMMARY (>=60s, EXACT_SEED+SERVE_LOG) is authoritative. ----
read_obs_key(){ # $1=cdir $2=key ; echoes int (0 if absent)
  python3 - "$1" "$2" <<'PY'
import sys, os, json, re
cdir, key = sys.argv[1], sys.argv[2]
val = 0; src = None
eng = os.path.join(cdir, "logs", "fr13_apc_exact_seed_eng.log")
if os.path.exists(eng):
    last = None
    for ln in open(eng, errors="ignore"):
        if "FR13_OBS_SUMMARY" in ln:
            m = re.search(r"FR13_OBS_SUMMARY\s+(\{.*\})", ln)
            if m: last = m.group(1)
    if last:
        try:
            d = json.loads(last); val = int(d.get(key, 0)); src = "obs_summary"
        except Exception: pass
if src is None:
    fj = os.path.join(cdir, "logs", "fr13_obs_final.json")
    if os.path.exists(fj):
        try:
            d = json.load(open(fj)); val = int((d.get("obs") or {}).get(key, 0)); src = "obs_final"
        except Exception: pass
print(val)
PY
}

# ---- write arm_meta.json for the reducer (BEFORE the hard asserts) ----
write_meta(){ # $1=cdir $2=arm $3=exp_cache $4=ok1 $5=ok2
  python3 - "$1" "$2" "$MODEL" "$3" "$N" "$4" "$5" "$TURN2_MODE" "$CAPTURE" <<'PY'
import json, sys, os, re
cdir, arm, model, exp_cache, N, ok1, ok2, mode, cap = sys.argv[1:10]
bl = open(os.path.join(cdir,"boot_log_snapshot.txt"),errors="ignore").read() if os.path.exists(os.path.join(cdir,"boot_log_snapshot.txt")) else ""
pc = re.search(r"enable_prefix_caching=(True|False)", bl)
def obs(k):
    eng = os.path.join(cdir,"logs","fr13_apc_exact_seed_eng.log"); last=None
    if os.path.exists(eng):
        for ln in open(eng,errors="ignore"):
            m=re.search(r"FR13_OBS_SUMMARY\s+(\{.*\})",ln)
            if m: last=m.group(1)
    if last:
        try: return int(json.loads(last).get(k,0))
        except Exception: return 0
    fj=os.path.join(cdir,"logs","fr13_obs_final.json")
    if os.path.exists(fj):
        try: return int((json.load(open(fj)).get("obs") or {}).get(k,0))
        except Exception: return 0
    return 0
meta = {"arm":arm,"model_served":model,"cache_expected":exp_cache,
        "cache_boot_log":(pc.group(1) if pc else None),
        "N":int(N),"nonempty_turn1":int(ok1),"nonempty_turn2":int(ok2),
        "turn2_mode":mode,"capture":cap,
        "obs":{k:obs(k) for k in ("es_seed_applied","redirect_engaged","redirect_used",
                                  "refold_published","conv_snapshot_events","snapshot_events")}}
json.dump(meta, open(os.path.join(cdir,"arm_meta.json"),"w"), indent=1)
print("[meta]", json.dumps(meta)[:400])
PY
}

# ---- build the turn-2 send payload ----
build_turn2(){ # $1=arm $2=turn1_response.json $3=turn1_send.json $4=dest $5=seed
  local arm="$1" t1resp="$2" t1send="$3" dst="$4" seed="$5"
  if [[ "$TURN2_MODE" == "resend" ]]; then
    # identical body to turn-1 (same messages) -> full-prompt hit on turn-2.
    python3 - "$t1send" "$dst" "$seed" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d["seed"] = int(sys.argv[3])
json.dump(d, open(sys.argv[2], "w"), ensure_ascii=False)
PY
  else
    # conversation: payload.messages + FROZEN shared assistant turn (+tool results
    # if it carried tool_calls) + FIXED user turn-2. The assistant turn is frozen
    # from ref arm A seed=$FREEZE_SEED so turn-2 INPUT is byte-identical across arms.
    local frozen="$RR/fixed_assistant.json"
    [[ -f "$frozen" ]] || { echo "FAIL: conversation mode needs $frozen (run arm cat8_nocache first)"; return 3; }
    python3 - "$PAYLOAD" "$frozen" "$dst" "$seed" "$MAX_TOKENS" "$TEMPERATURE" "$FIXED_TURN2" <<'PY'
import json, sys
payload, frozen, dst, seed, mx, tp, u2 = sys.argv[1:8]
d = json.load(open(payload))
d.pop("seed", None)
d["seed"] = int(seed); d["max_tokens"] = int(mx); d["temperature"] = float(tp)
d["logprobs"] = True; d["top_logprobs"] = 0; d["stream"] = False
asst = json.load(open(frozen))          # {"message": {...}} captured from ref turn-1
am = asst["message"]
msgs = list(d["messages"])
asst_msg = {"role": "assistant"}
if am.get("content"): asst_msg["content"] = am["content"]
tcs = am.get("tool_calls") or []
if tcs: asst_msg["tool_calls"] = tcs
if not am.get("content") and not tcs: asst_msg["content"] = am.get("reasoning_content") or ""
msgs.append(asst_msg)
# well-formed tool results so the template renders (avoids dangling tool_calls)
for i, tc in enumerate(tcs):
    msgs.append({"role": "tool",
                 "tool_call_id": tc.get("id") or f"call_{i}",
                 "content": "[fixed probe tool result: proceed.]"})
msgs.append({"role": "user", "content": u2})
d["messages"] = msgs
json.dump(d, open(dst, "w"), ensure_ascii=False)
PY
  fi
}

run_arm(){
  local arm="$1"
  local cdir="$RR/$arm"; local container="fr13-s2t-$arm"
  mkdir -p "$cdir/logs"
  echo ""
  echo "########## ARM $arm  ($(date -u +%H:%M:%S)) ##########"
  (
    set -uo pipefail
    export FR10_METRICS=0 BATCH_INVARIANT=0 LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16
    export CONTAINER="$container" PORT="$PORT" MAX_NUM_SEQS=1
    export GPU_UTIL GPU_GUARD_FLOOR_MIB
    export LOG_DIR="$PWD/$cdir/logs" FR13_RUN_DIR="$PWD/$cdir"
    export FR13_SERVE_LOG=1
    # EAGER (host-evaluated at launcher:673 — the §29 regime; removes graph-capture
    # nondeterminism and lets the pass-2 decode capture fire, guard is eager-only).
    export ENFORCE_EAGER=1
    # ---- PASS-2 windowed decode-state capture (default OFF => byte-identical) ----
    if [[ "$CAPTURE" == "1" ]]; then
      export FR13_DECODE_GDN_CAPTURE="/logs/decode_gdn/step.pt"
      export FR13_DECODE_GDN_CAPTURE_LAYER_PREFIX="$CAPTURE_LAYER_PREFIX"
      export FR13_DECODE_GDN_CAPTURE_STEP_LO="$STEP_LO"
      export FR13_DECODE_GDN_CAPTURE_STEP_HI="$STEP_HI"
      export FR13_DECODE_GDN_CAPTURE_LIMIT_PER_PREFIX="$CAPTURE_LIMIT"
      # per-decode-step route logits (fp32), windowed the same way
      export FR13_FINAL_LOGIT_CAPTURE="/logs/decode_gdn/logit.pt"
      export FR13_FINAL_LOGIT_CAPTURE_SKIP="$STEP_LO"
      export FR13_FINAL_LOGIT_CAPTURE_LIMIT="$(( STEP_HI - STEP_LO + 2 ))"
      mkdir -p "$cdir/logs/decode_gdn"
    fi
    # per-arm boot config
    local kv
    while IFS= read -r kv; do
      [[ -z "$kv" ]] && continue
      [[ "$kv" == "FAIL_UNKNOWN_ARM" ]] && { echo "FAIL: unknown arm $arm"; exit 2; }
      export "$kv"
    done < <(arm_env "$arm")

    docker rm -f "$container" >/dev/null 2>&1 || true
    recover_host
    if [[ -n "$(docker ps -q 2>/dev/null)" ]]; then
      echo "FAIL: $arm — a container is still running before boot (co-residency):"; docker ps; exit 3
    fi

    echo "[$arm] boot via fr13_launch_forked_fa2_tree_server.sh (EAGER, GPU_UTIL=$GPU_UTIL floor=$GPU_GUARD_FLOOR_MIB)"
    bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$cdir/launch.log" 2>&1
    local rc=$?
    if (( rc != 0 )); then echo "FAIL: launcher rc=$rc"; tail -30 "$cdir/launch.log"; exit 3; fi

    # ---- health wait (<=1200s) ----
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
    docker exec "$container" sh -c 'tr "\0" " " < /proc/1/cmdline' > "$cdir/vllm_cmdline.txt" 2>/dev/null || true

    # ---- FAIL-LOUD 2: served model ----
    curl -fsS -m 10 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null > "$cdir/models.json"
    grep -q "\"$MODEL\"" "$cdir/models.json" \
      || { echo "FAIL: served model does not advertise $MODEL"; head -c 400 "$cdir/models.json"; exit 4; }

    # ---- FAIL-LOUD 3: ENFORCE_EAGER threaded end-to-end (the H3 lesson) ----
    grep -q -- '--enforce-eager' "$cdir/vllm_cmdline.txt" \
      || { echo "FAIL: $arm — --enforce-eager NOT in live vLLM argv (/proc/1/cmdline):"; cat "$cdir/vllm_cmdline.txt"; exit 4; }
    echo "[$arm] --enforce-eager present in live argv OK"

    # ---- FAIL-LOUD 4: right boot config (cache state + backend + tree) ----
    local exp_cache; exp_cache=$(arm_cache_expect "$arm")
    grep -qa "enable_prefix_caching=$exp_cache" "$cdir/boot_log_snapshot.txt" \
      || { echo "FAIL: expected enable_prefix_caching=$exp_cache not in boot log for $arm";
           grep -aoE "enable_prefix_caching=(True|False)" "$cdir/boot_log_snapshot.txt" | head; exit 4; }
    grep -q "^ATTENTION_BACKEND=TREE_ATTN$" "$cdir/container_env.txt" || { echo "FAIL: $arm backend!=TREE_ATTN"; exit 4; }
    grep -q "speculative_token_tree" "$cdir/container_env.txt" || { echo "FAIL: $arm SPEC_CONFIG has no tree"; exit 4; }
    case "$arm" in
      cat8_cache|cat8_cache_refold)
        grep -q "^FR13_APC_EXACT_SEED=1$" "$cdir/container_env.txt" || { echo "FAIL: $arm EXACT_SEED!=1"; exit 4; }
        grep -q -- '--enable-prefix-caching' "$cdir/vllm_cmdline.txt" || { echo "FAIL: $arm serve argv missing --enable-prefix-caching"; cat "$cdir/vllm_cmdline.txt"; exit 4; };;
      cat8_nocache|cat8_nocache_b)
        grep -q "^FR13_APC_EXACT_SEED=0$" "$cdir/container_env.txt" || { echo "FAIL: $arm EXACT_SEED!=0 (config-only)"; exit 4; }
        if grep -q -- '--enable-prefix-caching' "$cdir/vllm_cmdline.txt"; then echo "FAIL: $arm serve argv HAS --enable-prefix-caching (config_only must be cache-OFF)"; cat "$cdir/vllm_cmdline.txt"; exit 4; fi;;
    esac
    if [[ "$arm" == "cat8_cache_refold" ]]; then
      grep -q "^FR13_APC_BLOCK_REFOLD=1$" "$cdir/container_env.txt" || { echo "FAIL: $arm BLOCK_REFOLD!=1"; exit 4; }
      grep -q "^FR13_APC_REFOLD_TO_SNAPSHOT=1$" "$cdir/container_env.txt" || { echo "FAIL: $arm REFOLD_TO_SNAPSHOT!=1"; exit 4; }
    fi
    # ---- FAIL-LOUD 4b (pass-2 only): decode-capture env actually threaded ----
    if [[ "$CAPTURE" == "1" ]]; then
      grep -q "^FR13_DECODE_GDN_CAPTURE=/logs/decode_gdn/" "$cdir/container_env.txt" \
        || { echo "FAIL: $arm — FR13_DECODE_GDN_CAPTURE NOT in container_env.txt."; \
             echo "  The decode-capture instrument is in HEAD (commit 482bb382) and the launcher"; \
             echo "  forwards it — if this failed, the launcher -e list drifted. Verify:"; \
             echo "    python3 $STAGE/apply_decode_capture_patch.py   (reports present/absent)"; exit 4; }
      echo "[$arm] decode-capture env threaded OK"
    fi
    echo "[$arm] boot-config asserts OK (cache=$exp_cache, TREE, $arm)"

    # ---- sample loop: N seeds x {turn-1 cold, turn-2 hit} ----
    local send1="$cdir/base_send.json"; base_send "$send1"
    : > "$cdir/reset_log.txt"; : > "$cdir/hits.jsonl"
    local i ok1=0 ok2=0
    for (( i=1; i<=N; i++ )); do
      # RESET ONCE before turn-1 for this seed => turn-1 is a genuine cold prefill.
      local rcode
      rcode=$(curl -s -o /dev/null -m 20 -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" 2>/dev/null)
      echo "seed $i reset_prefix_cache http=$rcode" >> "$cdir/reset_log.txt"
      [[ "$rcode" == "200" ]] || echo "WARN: $arm seed $i reset http=$rcode (turn-1 cold NOT guaranteed)"

      # ---- TURN 1 (cold) ----
      snap_metrics "$cdir/m_s${i}_pre.txt"
      python3 - "$send1" "$cdir/turn1_send_$i.json" "$i" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d["seed"] = int(sys.argv[3])
json.dump(d, open(sys.argv[2], "w"), ensure_ascii=False)
PY
      curl -sS -m 900 -H 'Content-Type: application/json' -H 'Authorization: Bearer EMPTY' \
        --data-binary @"$cdir/turn1_send_$i.json" "http://127.0.0.1:$PORT/v1/chat/completions" \
        > "$cdir/turn1_$i.json" 2>"$cdir/turn1_$i.err"
      snap_metrics "$cdir/m_s${i}_post1.txt"

      # freeze the shared assistant turn from the ref arm (A) at FREEZE_SEED
      if [[ "$arm" == "cat8_nocache" && "$i" == "$FREEZE_SEED" && "$TURN2_MODE" == "conversation" ]]; then
        python3 - "$cdir/turn1_$i.json" "$RR/fixed_assistant.json" <<'PY' || echo "WARN: freeze assistant failed"
import json, sys
d = json.load(open(sys.argv[1])); ch = (d.get("choices") or [{}])[0]
json.dump({"message": ch.get("message") or {}}, open(sys.argv[2], "w"), ensure_ascii=False)
print("[freeze] wrote", sys.argv[2])
PY
      fi

      # ---- TURN 2 (hit; NO reset) ----
      build_turn2 "$arm" "$cdir/turn1_$i.json" "$cdir/turn1_send_$i.json" "$cdir/turn2_send_$i.json" "$i" \
        || { echo "FAIL: $arm build_turn2 seed $i"; exit 6; }
      curl -sS -m 900 -H 'Content-Type: application/json' -H 'Authorization: Bearer EMPTY' \
        --data-binary @"$cdir/turn2_send_$i.json" "http://127.0.0.1:$PORT/v1/chat/completions" \
        > "$cdir/turn2_$i.json" 2>"$cdir/turn2_$i.err"
      snap_metrics "$cdir/m_s${i}_post2.txt"

      # per-turn hit deltas
      read q0 h0 < <(metrics_qh "$cdir/m_s${i}_pre.txt")
      read q1 h1 < <(metrics_qh "$cdir/m_s${i}_post1.txt")
      read q2 h2 < <(metrics_qh "$cdir/m_s${i}_post2.txt")
      python3 - "$cdir/hits.jsonl" "$i" "$q0" "$h0" "$q1" "$h1" "$q2" "$h2" <<'PY'
import json, sys
f, i, q0, h0, q1, h1, q2, h2 = sys.argv[1:9]
q0,h0,q1,h1,q2,h2 = map(float,(q0,h0,q1,h1,q2,h2))
rec = {"seed": int(i),
       "t1_q": q1-q0, "t1_h": h1-h0,
       "t2_q": q2-q1, "t2_h": h2-h1}
open(f,"a").write(json.dumps(rec)+"\n")
PY
      # non-empty asserts + route note per turn
      for t in 1 2; do
        if python3 - "$cdir/turn${t}_$i.json" "$t" "$i" <<'PY'
import json, sys
p, t, i = sys.argv[1], sys.argv[2], sys.argv[3]
try: d = json.load(open(p))
except Exception as e: print(f"  turn{t} seed{i} PARSE_FAIL {str(e)[:100]}"); sys.exit(1)
ch = (d.get("choices") or [])
if not ch: print(f"  turn{t} seed{i} NO_CHOICES {json.dumps(d)[:160]}"); sys.exit(1)
m = ch[0].get("message") or {}
tc = m.get("tool_calls") or []
txt = (m.get("content") or "").strip(); rsn = (m.get("reasoning_content") or "").strip()
if not (tc or txt or rsn): print(f"  turn{t} seed{i} EMPTY"); sys.exit(1)
route = (tc[0].get("function",{}).get("name") if tc else "NO_TOOL")
print(f"  turn{t} seed{i} OK route={route} finish={ch[0].get('finish_reason')} rsn={len(rsn)}c content={len(txt)}c")
PY
        then [[ "$t" == "1" ]] && ok1=$((ok1+1)) || ok2=$((ok2+1)); else echo "  WARN: $arm turn$t seed$i empty/failed"; fi
      done
    done
    echo "[$arm] non-empty: turn1=$ok1/$N turn2=$ok2/$N"

    # ---- HIT / OBS asserts ----
    docker logs "$container" > "$cdir/docker_full.log" 2>&1 || true
    # per-turn hit summary
    python3 - "$cdir/hits.jsonl" <<'PY' | tee "$cdir/hit_summary.txt"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])] if __import__("os").path.exists(sys.argv[1]) else []
t1h = sum(1 for r in rows if r["t1_h"] > 0); t2h = sum(1 for r in rows if r["t2_h"] > 0)
print(f"hit_summary seeds={len(rows)} turn1_hit_seeds={t1h} turn2_hit_seeds={t2h} "
      f"turn2_hit_total={sum(r['t2_h'] for r in rows):.0f}")
PY
    read T1HIT T2HIT < <(python3 - "$cdir/hits.jsonl" <<'PY'
import json, sys, os
rows = [json.loads(l) for l in open(sys.argv[1])] if os.path.exists(sys.argv[1]) else []
print(sum(1 for r in rows if r["t1_h"]>0), sum(1 for r in rows if r["t2_h"]>0))
PY
)
    # arm meta for the reducer — WRITTEN BEFORE the hard asserts so a vacuous-arm
    # abort (e.g. arm C redirect_used=0) still leaves full meta for the reducer.
    write_meta "$cdir" "$arm" "$exp_cache" "$ok1" "$ok2"

    if arm_is_cache "$arm"; then
      # turn-2 MUST hit (else the restore/refold path never fires => vacuous)
      (( T2HIT > 0 )) || { echo "FAIL: $arm — turn-2 prefix cache NEVER hit ($T2HIT/$N). Probe is VACUOUS."; \
                           echo "  Check: reset only-before-turn-1, template re-render of turn-2 prefix, block coverage."; exit 7; }
      echo "[$arm] turn-2 hit seeds=$T2HIT/$N (turn-1 hit seeds=$T1HIT) OK"
      local ES; ES=$(read_obs_key "$cdir" "es_seed_applied")
      echo "[$arm] es_seed_applied=$ES"
      (( ES > 0 )) || echo "WARN: $arm es_seed_applied=0 (EXACT_SEED restore may not have engaged on the hit; check obs)"
      if [[ "$arm" == "cat8_cache_refold" ]]; then
        local RU; RU=$(read_obs_key "$cdir" "redirect_used")
        local RE; RE=$(read_obs_key "$cdir" "redirect_engaged")
        echo "[$arm] redirect_engaged=$RE redirect_used=$RU"
        (( RU > 0 )) || { echo "FAIL: $arm — redirect_used=0 => REFOLD NEVER EXECUTED. Arm C == arm B; the refold A/B is VACUOUS (§27)."; \
                          echo "  Do NOT report 'refold no help' from this run — refold did not run. See runbook risk 'VACUOUS REFOLD ARM'."; exit 7; }
        echo "[$arm] refold LIVE (redirect_used=$RU) OK"
      fi
    else
      # config-only must NOT hit (sanity: caching truly off)
      (( T2HIT == 0 )) || echo "WARN: $arm (config-only) reported turn-2 hits=$T2HIT (expected 0 — caching should be OFF)"
    fi

    # ---- graceful teardown (SIGTERM first so atexit OBS flush can win, §24) ----
    docker stop -t 25 "$container" >/dev/null 2>&1 || true
    docker logs "$container" > "$cdir/docker_full.log" 2>&1 || true
    docker rm -f "$container" >/dev/null 2>&1 || true
    recover_host
    sleep 2
    (( ok1 > 0 && ok2 > 0 )) || { echo "FAIL: $arm produced ZERO non-empty completions on a turn"; exit 5; }
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
echo "=== seeded2turn boots complete (rc=$RC). Reduce with: ==="
echo "    .venv/bin/python $STAGE/seeded2turn_reduce.py --run-root $RR --tokenizer /models/qwen3.6-27b-fp8"
exit $RC
