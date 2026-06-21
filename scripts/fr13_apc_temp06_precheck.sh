#!/usr/bin/env bash
# FR13 APC TEMP-0.6 WITHIN-FLOOR PRECHECK (cache-ON serve capture)
# =================================================================================
# WRITE-ONLY scaffold (validated with `bash -n`); DO NOT RUN here. The orchestrator
# runs the GPU stage (GPU-solo / serialized on GB10; never a third concurrent
# workflow). This script boots ONE container, captures ONE temp-0.6 cache-ON served
# token-id stream, and tears down. The within-floor verdict is computed OFFLINE by
# the rescore + consolidate documented at the bottom (a SEPARATE GPU boot, no-spec
# recurrent oracle), NEVER inside this script.
#
# PURPOSE
# -------
# The gross APC prefix-cache poison on the Qwen3-Next-27B fp8 GDN-hybrid cat9
# tree-spec serve is FIXED (committed d228c76b): the launcher now defaults
# APC_MAX_NUM_BATCHED_TOKENS = MAMBA_BLOCK_SIZE so each chunked-prefill scheduler
# step crosses <=1 mamba block boundary (kills the align one-checkpoint-per-step
# overshoot, vLLM #45238). The cache-ON run is now coherent + on-task. A small
# RESIDUAL remains (greedy char-1 fork, both coherent). The ONLY valid way to judge
# that residual is at temp 0.6 within the E5 floor:
#
#   within-floor  <=>  cache-ON per-token CLEAR-MARGIN argmax-flip rate
#                      (vs the NO-SPEC RECURRENT oracle, computed offline) has a
#                      Wilson-95 upper bound <= the E5 floor 12.90%.
#
#   NEVER greedy byte-exact.  NEVER a backend-name / chunked-prefill proxy.
#   The flip reference is the no-spec RECURRENT decode oracle (deployment-correct),
#   NOT streamed logprobs and NOT a serial-torch ref.
#
# WHAT THIS SCRIPT DOES (boot-side, capture only)
# -----------------------------------------------
#   1. Boot the FIXED-APC cat9 tree server via
#      scripts/fr13_launch_forked_fa2_tree_server.sh with FR13_ENABLE_APC=1,
#      MAMBA_BLOCK_SIZE=1024, MAMBA_SSM_CACHE_DTYPE=float32. We DO NOT override
#      APC_MAX_NUM_BATCHED_TOKENS — the launcher (fr13_launch_forked_fa2_tree_server.sh
#      L205) now defaults it to $MAMBA_BLOCK_SIZE, which IS the fix.
#   2. WARM the long (>2000-token) prefix so the prefix cache actually fires (reuse
#      the prefill-after-hit prompt build: warm prefix P, then send full P+SUFFIX so
#      the request HITS P and the cache is live), then send ONE temp-0.6 cache-ON
#      /v1/completions request (temperature 0.6, seed 1313, top_k 20, max_tokens 300)
#      and capture the SERVED TOKEN IDS.
#   3. Write the oracle --src json {prompts:[<prompt_str>], records:[{served_token_ids:[...]}]}
#      (index-aligned, ONE entry) to output/fr13_apc_temp06/cat9_apc_on_src.json.
#   4. Tear down cleanly (docker rm -f + recover_host_memory).
#
# CAPTURE MECHANISM (load-bearing — see header note in fr13_oracle_capture.py):
#   token ids come straight from the vLLM /v1/completions response when the request
#   sets  "return_token_ids": true  -> choice["token_ids"] is the exact served id
#   list (fr13_oracle_capture.py::_stream_prompt L84-105 uses exactly this in THIS
#   image). This is a first-class served-side feature: no retokenization (the #12
#   round-trip fiction the SSE reducer fr13_swe_stream_to_oracle_src.py has to guard
#   against), no logprobs-channel reconstruction. It is the SAME mechanism the
#   recurrent oracle's own capture path uses, so the ids are exactly what the
#   recurrent rescore will force-decode against.
#
# Hygiene / boot / teardown / long-prompt build are copied from
#   scripts/fr13_apc_prefill_after_hit.sh
# and the boot invocation + APC flag block from
#   scripts/fr13_launch_forked_fa2_tree_server.sh
# =================================================================================
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel

CONTAINER=${CONTAINER:-fr13-bigdenom-apc_temp06}
PORT=${PORT:-9950}
MODEL=${MODEL:-qwen3.6-27b}
SEED=${SEED:-1313}
TOP_K=${TOP_K:-20}
TEMP=${TEMP:-0.6}
MAX_TOKENS=${MAX_TOKENS:-512}
RUNROOT=output/fr13_apc_temp06
SRC_OUT="$RUNROOT/cat9_apc_on_src.json"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNDIR="$RUNROOT/run_$TS"
mkdir -p "$RUNDIR/logs"

# ---- APC knobs: the FIXED defaults. DO NOT set APC_MAX_NUM_BATCHED_TOKENS — the
#      launcher defaults it to $MAMBA_BLOCK_SIZE (the #45238 fix). ----
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=${MAMBA_BLOCK_SIZE:-1024}
export MAMBA_SSM_CACHE_DTYPE=${MAMBA_SSM_CACHE_DTYPE:-float32}

# ---- forked cat9 TREE flags (exactly the prefill-after-hit / serve_variant pinset) ----
export FR10_METRICS=0
export BATCH_INVARIANT=0
export LUMO_FB_KERNEL_ROWS=1
export LUMO_FB_PROJ_PAD_ROWS=16
MAX_NUM_SEQS_OVR=${MAX_NUM_SEQS_OVR:-1}

echo "=== FR13 APC TEMP-0.6 PRECHECK container=$CONTAINER port=$PORT rundir=$RUNDIR ==="
echo "    APC: enable=1 mamba_block_size=$MAMBA_BLOCK_SIZE ssm_cache_dtype=$MAMBA_SSM_CACHE_DTYPE max_num_batched_tokens=DEFAULT(=block_size, the fix)"
echo "    sample: temp=$TEMP seed=$SEED top_k=$TOP_K max_tokens=$MAX_TOKENS"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$RUNDIR/started_at.txt"
git rev-parse HEAD 2>/dev/null | tee "$RUNDIR/git_head.txt" || true

# ---- recover_host_memory helper (copied from prefill-after-hit) ----
recover_host(){ PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY'
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}

# ---- pre-boot hygiene: recover + assert MemAvailable>95GiB + swap=0 + docker empty ----
echo "[hygiene] recover_host_memory + assert free"
recover_host || true
.venv/bin/python - <<'PY'
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
(( $? == 0 )) || { echo "FAIL: pre-boot hygiene"; exit 2; }
if [[ -n "$(docker ps -q)" ]]; then echo "FAIL: docker ps not empty before boot"; docker ps; exit 2; fi
free -g | tee "$RUNDIR/free_before_boot.txt"

# ---- teardown trap: ALWAYS docker rm -f + recover + free check, even on failure ----
teardown(){
  echo "[teardown] docker logs dump + docker rm -f $CONTAINER + recover_host_memory"
  docker logs "$CONTAINER" > "$RUNDIR/docker_full.log" 2>&1 || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  recover_host || true
  sleep 2
  free -g | tee "$RUNDIR/free_after_teardown.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ | tee "$RUNDIR/ended_at.txt"
}
trap teardown EXIT

# ---- boot the FORKED server with FR13_ENABLE_APC=1 + the forked cat9 flags ----
# (mirrors prefill-after-hit's forked-arm invocation; default cat9 TREE = launcher
# default => num_speculative_tokens=9, so no TREE override is passed. We deliberately
# do NOT pass APC_MAX_NUM_BATCHED_TOKENS so the launcher's =block_size default fires.)
CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS="$MAX_NUM_SEQS_OVR" \
  FR10_METRICS=0 BATCH_INVARIANT=0 \
  LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
  FR13_ENABLE_APC=1 \
  MAMBA_BLOCK_SIZE="$MAMBA_BLOCK_SIZE" \
  MAMBA_SSM_CACHE_DTYPE="$MAMBA_SSM_CACHE_DTYPE" \
  FR13_RUN_DIR="$PWD/$RUNDIR" LOG_DIR="$PWD/$RUNDIR/logs" \
  scripts/fr13_launch_forked_fa2_tree_server.sh > "$RUNDIR/launch.log" 2>&1
RC=$?
if (( RC != 0 )); then echo "FAIL: launcher rc=$RC"; tail -40 "$RUNDIR/launch.log"; exit 2; fi

# ---- wait up to 1200s for /health; FAIL-LOUD if the container dies ----
fail_scan(){
  docker logs "$CONTAINER" > "$RUNDIR/boot_log_snapshot.txt" 2>&1 || true
  if grep -nE "NotImplementedError|does not yet support|does not support|Traceback \(most recent call last\)|EngineDeadError|EngineCore .*died" \
       "$RUNDIR/boot_log_snapshot.txt" | head -20 | tee "$RUNDIR/boot_error_hits.txt" | grep -q .; then
    return 0
  fi
  return 1
}
T0=$(date +%s)
HEALTHY=0
while (( $(date +%s) < T0 + 1200 )); do
  if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
  if [[ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" != "running" ]]; then
    echo "FAIL: container died before health (APC+GDN-hybrid init crash)"
    fail_scan && { echo "--- fatal signature(s): ---"; cat "$RUNDIR/boot_error_hits.txt"; }
    docker logs "$CONTAINER" 2>&1 | tail -60; exit 3
  fi
  sleep 5
done
if (( HEALTHY != 1 )); then
  echo "FAIL: /health not up in 1200s"
  fail_scan && { echo "--- fatal signature(s): ---"; cat "$RUNDIR/boot_error_hits.txt"; }
  docker logs "$CONTAINER" 2>&1 | tail -60; exit 3
fi
echo "[boot] healthy after $(( $(date +%s) - T0 ))s"
docker logs "$CONTAINER" > "$RUNDIR/boot_log_snapshot.txt" 2>&1 || true
if fail_scan; then echo "FAIL: fatal signature in boot log despite /health"; cat "$RUNDIR/boot_error_hits.txt"; exit 3; fi

# ---- confirm APC engaged (non-vacuity: APC must actually be ON for a cache-ON capture) ----
APC_ON=0
grep -qiE "enable_prefix_caching=True|'enable_prefix_caching': True" "$RUNDIR/boot_log_snapshot.txt" && APC_ON=1
grep -qiE "mamba_block_size|mamba_cache_mode" "$RUNDIR/boot_log_snapshot.txt" && APC_ON=1
if (( APC_ON != 1 )); then
  echo "FAIL: no enable_prefix_caching=True / mamba evidence in boot log (APC silently dropped?)"
  exit 4
fi
echo "[apc] APC engaged in boot log"

# ---- build the long (>2000-token) prompt (copied from prefill-after-hit) ----
PAIR_DUMP=${PAIR_DUMP:-output/fr13_bigdenom_swe/cat9_b1/proxy_pair_dumps/pair_01781570708536496433_000002_initial.json}
PROMPT_FILE="$RUNDIR/long_prompt.txt"
.venv/bin/python - "$PAIR_DUMP" "$PROMPT_FILE" <<'PY'
import json, sys
from pathlib import Path
src, out = sys.argv[1], sys.argv[2]
text = None
p = Path(src)
if p.is_file():
    try:
        d = json.loads(p.read_text())
        req = d.get("request", {})
        parts = []
        instr = req.get("instructions")
        if isinstance(instr, str):
            parts.append(instr)
        for msg in req.get("input", []) or []:
            c = msg.get("content")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):
                for seg in c:
                    if isinstance(seg, dict):
                        for v in seg.values():
                            if isinstance(v, str):
                                parts.append(v)
        blob = "\n\n".join(x for x in parts if x)
        if len(blob) > 8000:  # ~>2000 tokens
            text = blob
    except Exception as e:
        print(f"[prompt] banked-dump parse failed: {e}", file=sys.stderr)
if text is None:
    base = ("The following is a long technical specification for a speculative "
            "decoding verifier on a gated DeltaNet hybrid language model. ")
    text = (base * 400)
Path(out).write_text(text)
print(f"[prompt] wrote {len(text)} chars to {out} (source={'banked' if p.is_file() else 'synth'})")
PY
[[ -s "$PROMPT_FILE" ]] || { echo "FAIL: long prompt not produced"; exit 4; }

# ---- WARM the prefix so the cache actually fires (prefill-after-hit geometry) ----
# Warm prefix P = first 80% of the long prompt, then the temp-0.6 capture request
# sends the FULL prompt (P+SUFFIX) which HITS P's cached blocks (incl the conv window
# at P's boundary) and prefills the suffix from the restored SSM state. This is the
# exact geometry the fixed APC path must be lossless on.
P_FILE="$RUNDIR/prefix_P.txt"
.venv/bin/python - "$PROMPT_FILE" "$P_FILE" <<'PY'
import sys
full = open(sys.argv[1]).read()
open(sys.argv[2], "w").write(full[: int(len(full) * 0.80)])  # P = first 80%
PY
echo "[warm] reset prefix cache + warm prefix P (greedy max_tokens=1)"
curl -fsS -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" >/dev/null 2>&1 || true
.venv/bin/python - "$PORT" "$MODEL" "$P_FILE" <<'PY'
import json, sys, urllib.request, urllib.error
port, model, pfile = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.dumps({"model": model, "prompt": open(pfile).read(),
                   "temperature": 0.0, "max_tokens": 1, "stream": False}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                             data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        print(f"[warm] HTTP {r.status}")
except Exception as e:
    print(f"[warm] WARN warmup error: {e}", file=sys.stderr)
PY

# ---- CAPTURE: temp-0.6 cache-ON request, served token ids via return_token_ids=True ----
# Mechanism: vLLM /v1/completions with return_token_ids=True returns choice["token_ids"]
# = the exact served id stream (identical to fr13_oracle_capture.py::_stream_prompt).
# We snapshot prefix-cache counters before/after to PROVE the cache fired on this request
# (non-vacuity: a cache-ON capture that hit 0 cached tokens is not a cache-ON capture).
echo "[capture] temp-$TEMP cache-ON request (full P+SUFFIX) -> served_token_ids"
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$RUNDIR/metrics_before_capture.txt" 2>&1 || true
.venv/bin/python - "$PORT" "$MODEL" "$PROMPT_FILE" "$TEMP" "$SEED" "$TOP_K" "$MAX_TOKENS" "$SRC_OUT" "$RUNDIR/capture_raw.json" <<'PY'
import json, sys, urllib.request, urllib.error
from pathlib import Path
port, model, pfile = sys.argv[1], sys.argv[2], sys.argv[3]
temp, seed, top_k, max_tokens = float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
src_out, raw_out = sys.argv[8], sys.argv[9]
prompt = open(pfile).read()
body = json.dumps({
    "model": model,
    "prompt": prompt,
    "temperature": temp,
    "seed": seed,
    "top_k": top_k,
    "max_tokens": max_tokens,
    "stream": False,
    "return_token_ids": True,   # <-- load-bearing: choice["token_ids"] = served ids
    "vllm_xargs": {"fr10_decode_mode": "tree_mtp"},
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                             data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        data = json.loads(r.read().decode())
except urllib.error.HTTPError as e:
    sys.stderr.write(f"capture HTTP {e.code}: {e.read().decode()[:600]}\n"); sys.exit(11)
except Exception as e:
    sys.stderr.write(f"capture client error: {e}\n"); sys.exit(11)
Path(raw_out).write_text(json.dumps(data, indent=2))
choice = data["choices"][0]
ids = choice.get("token_ids") or []
if not ids:
    sys.stderr.write("FAIL: choice['token_ids'] empty -> return_token_ids not honored "
                     "(wrong image/feature?). Cannot capture served ids.\n"); sys.exit(12)
src = {"prompts": [prompt], "records": [{"served_token_ids": [int(t) for t in ids]}]}
Path(src_out).parent.mkdir(parents=True, exist_ok=True)
Path(src_out).write_text(json.dumps(src))
print(f"[capture] served_token_ids n={len(ids)} finish={choice.get('finish_reason')} -> {src_out}")
PY
CAP_RC=$?
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$RUNDIR/metrics_after_capture.txt" 2>&1 || true
if (( CAP_RC != 0 )); then echo "FAIL: capture rc=$CAP_RC"; exit 5; fi

# ---- non-vacuity: confirm the prefix cache actually fired on the capture request ----
.venv/bin/python - "$RUNDIR/metrics_before_capture.txt" "$RUNDIR/metrics_after_capture.txt" <<'PY'
import re, sys
from pathlib import Path
def cached(fp):
    s = 0.0
    for line in Path(fp).read_text().splitlines():
        if line.startswith("#"): continue
        m = re.match(r"^(vllm:prompt_tokens_cached_total)(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", line)
        if m: s += float(m.group(3))
    return s
b, a = cached(sys.argv[1]), cached(sys.argv[2])
print(f"[non-vacuity] prompt_tokens_cached_total before={b} after={a} delta={a-b}")
if a - b <= 0:
    print("WARN: prompt_tokens_cached did NOT increase on the capture request -> "
          "cache did not fire (geometry/0-hit). The capture is still a valid served "
          "stream but is NOT cache-ON; investigate before trusting the within-floor verdict.")
else:
    print("[non-vacuity] OK: cache fired on the capture request (cache-ON confirmed)")
PY

# ---- cache-OFF arm (SAME boot, reset cache, full prefill, identical temp/seed) ----
# The binding gate is APPLES-TO-APPLES: cat9 has its own inherent spec-decode flip rate
# (~13% vs the no-spec recurrent oracle), so lossless = cache-ON does NOT push that rate
# higher than cache-OFF. Same boot => same autotune/conditions (no cross-boot caveat).
OFF_SRC_OUT="$RUNROOT/cat9_apc_off_src.json"
echo "[capture-off] reset prefix cache -> temp-$TEMP cache-OFF (full prefill, no hit) -> served_token_ids"
curl -fsS -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" >/dev/null 2>&1 || true
.venv/bin/python - "$PORT" "$MODEL" "$PROMPT_FILE" "$TEMP" "$SEED" "$TOP_K" "$MAX_TOKENS" "$OFF_SRC_OUT" "$RUNDIR/capture_off_raw.json" <<'PY'
import json, sys, urllib.request, urllib.error
from pathlib import Path
port, model, pfile = sys.argv[1], sys.argv[2], sys.argv[3]
temp, seed, top_k, max_tokens = float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
src_out, raw_out = sys.argv[8], sys.argv[9]
prompt = open(pfile).read()
body = json.dumps({
    "model": model, "prompt": prompt, "temperature": temp, "seed": seed, "top_k": top_k,
    "max_tokens": max_tokens, "stream": False, "return_token_ids": True,
    "vllm_xargs": {"fr10_decode_mode": "tree_mtp"},
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                             data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        data = json.loads(r.read().decode())
except Exception as e:
    sys.stderr.write(f"cache-OFF capture error: {e}\n"); sys.exit(11)
Path(raw_out).write_text(json.dumps(data, indent=2))
ids = (data["choices"][0].get("token_ids")) or []
if not ids:
    sys.stderr.write("WARN: cache-OFF token_ids empty\n"); sys.exit(12)
Path(src_out).write_text(json.dumps({"prompts": [prompt], "records": [{"served_token_ids": [int(t) for t in ids]}]}))
print(f"[capture-off] served_token_ids n={len(ids)} -> {src_out}")
PY
OFF_RC=$?
if (( OFF_RC != 0 )); then echo "WARN: cache-OFF capture rc=$OFF_RC (cache-ON arm still valid; fall back to banked cat9 ~13.28%)"; fi

echo "[done] src written: $SRC_OUT  (cache-OFF: $OFF_SRC_OUT)"
echo
# =================================================================================
# OFFLINE RESCORE -> WITHIN-FLOOR VERDICT  (SEPARATE GPU boot; run AFTER teardown)
# =================================================================================
# The flip reference is the NO-SPEC RECURRENT decode oracle (deployment-correct),
# NOT streamed logprobs, NOT a backend name. The rescore --src contract is exactly
# what this script wrote: {prompts:[str], records:[{served_token_ids:[int...]}]}
# (fr13_recurrent_decode_oracle.py::_load_served_streams L338-343; the rescore
# subcommand --arm/--src L680-694).
#
#   # 1) recurrent-oracle rescore of the cache-ON served stream (GPU, in-container):
#   scripts/fr13_recur_rescore_in_container.sh \
#       cat9_apc_on \
#       output/fr13_apc_temp06/cat9_apc_on_src.json \
#       output/fr13_apc_temp06/cat9_apc_on_rescore.json
#
#   # 2) consolidate -> clear-margin flip rate + Wilson-95 CI for the cat9-APC-ON arm.
#   #    NOTE: fr13_bigdenom_rescore_consolidate.py REQUIRES a paired native arm
#   #    (--cat9-* AND --native-*) and reports CI separation. For the single-prompt
#   #    WITHIN-FLOOR precheck you only need cat9-APC-ON's own Wilson-95 hi; pass the
#   #    banked native-E5 rescore+src as the --native-* pair so it loads, then read
#   #    out.cat9.wilson95_hi:
#   scripts/fr13_bigdenom_rescore_consolidate.py \
#       --cat9-rescore   output/fr13_apc_temp06/cat9_apc_on_rescore.json \
#       --cat9-src       output/fr13_apc_temp06/cat9_apc_on_src.json \
#       --native-rescore <BANKED native-E5 rescore json> \
#       --native-src     <BANKED native-E5 src json> \
#       --out            output/fr13_apc_temp06/cat9_apc_on_consolidated.json
#
#   # PASS CRITERION (within-floor):
#   #   cat9-APC-ON Wilson-95 upper bound (out.cat9.wilson95_hi) <= 0.1290 (E5 floor 12.90%).
#   #   If only the rate is needed, read out.cat9.clear_margin_rate / wilson95_hi.
# =================================================================================
echo "NEXT (offline, separate GPU boot — DO NOT run here):"
echo "  scripts/fr13_recur_rescore_in_container.sh cat9_apc_on $SRC_OUT $RUNROOT/cat9_apc_on_rescore.json"
echo "  scripts/fr13_bigdenom_rescore_consolidate.py \\"
echo "    --cat9-rescore $RUNROOT/cat9_apc_on_rescore.json --cat9-src $SRC_OUT \\"
echo "    --native-rescore <BANKED native-E5 rescore> --native-src <BANKED native-E5 src> \\"
echo "    --out $RUNROOT/cat9_apc_on_consolidated.json"
echo "  WITHIN-FLOOR PASS iff out.cat9.wilson95_hi <= 0.1290 (E5 floor 12.90%)"
# teardown runs via the EXIT trap
exit 0
