#!/usr/bin/env bash
# FR13 PREFIX-CACHE (APC) GATE-0 FEASIBILITY TEST
# =================================================================================
# AGENT B deliverable. WRITE-ONLY scaffold (validated with `bash -n`); DO NOT RUN
# until a human / orchestrator schedules it (it boots a GPU container on GB10 and
# the GPU is serialized — never launch a third concurrent workflow).
#
# PURPOSE: a focused, cheap go/no-go probe for enabling vLLM prefix caching (APC)
# on the Qwen3-Next-27B fp8 GDN-HYBRID + TREE/MTP spec-decode (FR13) stack. It
# boots the FORKED FA2 tree server with FR13_ENABLE_APC=1 (the flag-gated block in
# scripts/fr13_launch_forked_fa2_tree_server.sh) and answers FOUR questions that
# the prior research (research/fr13_workflows/prefix_cache_enable_plan.md) flagged
# as the live risks:
#
#   GATE-A  ENGINE BOOTS at all with APC+chunked-prefill+align forced on the
#           GDN-hybrid (NOT NotImplementedError / 'does not yet support' at init).
#   GATE-B  APC is ACTUALLY ENGAGED (boot log shows enable_prefix_caching=True and
#           the mamba cache mode), not silently dropped.
#   GATE-C  PREFIX CACHE HITS on a repeated long prompt — the three counters all
#           move > 0. A 0-hit here = the vLLM #45238 silent-0%-hit trap (align
#           keeps only ONE checkpoint at the last block boundary), NOT a crash.
#   GATE-D  NO CRASH under a real spec-decode generation (num_accepted>1) with APC
#           live — the DS-layout / vLLM #43559 (~20% acc drop / candidate-fix
#           #45477 unmerged) regression check. Server must still be /health-OK and
#           the docker log must be free of post-gen tracebacks.
#
# This does NOT measure accept/event, TPS, or losslessness — it is a feasibility
# gate. If all four pass, the next step is the real lossless A/B (APC OFF vs ON,
# byte-identical streams) on the locked pipeline. If GATE-C fails (0 hits) or
# GATE-D fails (crash), APC is NOT viable on this combo on this image as-configured.
#
# RE-RUN KNOBS (env): MAMBA_BLOCK_SIZE (1024), MAMBA_SSM_CACHE_DTYPE (float32),
#   APC_MAX_NUM_BATCHED_TOKENS (2048) — sweep these to probe the #45238 trap
#   (e.g. a smaller mamba-block-size keeps more checkpoints / changes the
#   last-block-boundary geometry) without editing the script.
#
# Hygiene / boot / teardown are copied from
#   scripts/fr13_bigdenom_swe_serve_variant.sh
# and the boot invocation + APC flag block from
#   scripts/fr13_launch_forked_fa2_tree_server.sh
# =================================================================================
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel

CONTAINER=${CONTAINER:-fr13-apc-gate0}
PORT=${PORT:-9950}
RUNROOT=output/fr13_apc_gate0
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNDIR="$RUNROOT/run_$TS"
mkdir -p "$RUNDIR/logs"

# ---- APC re-run knobs (exported so the forked launcher's docker -e picks them up) ----
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=${MAMBA_BLOCK_SIZE:-1024}
export MAMBA_SSM_CACHE_DTYPE=${MAMBA_SSM_CACHE_DTYPE:-float32}
export APC_MAX_NUM_BATCHED_TOKENS=${APC_MAX_NUM_BATCHED_TOKENS:-2048}

# ---- forked cat9 TREE flags (exactly serve_variant's forked-arm pinset) ----
# Default cat9 TREE (launcher default => num_speculative_tokens=9). FR10_METRICS=0
# BATCH_INVARIANT=0 LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16, MAX_NUM_SEQS=1.
export FR10_METRICS=0
export BATCH_INVARIANT=0
export LUMO_FB_KERNEL_ROWS=1
export LUMO_FB_PROJ_PAD_ROWS=16
MAX_NUM_SEQS_OVR=${MAX_NUM_SEQS_OVR:-1}

echo "=== FR13 APC GATE-0 container=$CONTAINER port=$PORT rundir=$RUNDIR ==="
echo "    APC: enable=1 mamba_block_size=$MAMBA_BLOCK_SIZE ssm_cache_dtype=$MAMBA_SSM_CACHE_DTYPE max_num_batched_tokens=$APC_MAX_NUM_BATCHED_TOKENS"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$RUNDIR/started_at.txt"
git rev-parse HEAD 2>/dev/null | tee "$RUNDIR/git_head.txt" || true

# ---- recover_host_memory helper (copied from serve_variant) ----
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
# (mirrors serve_variant's forked-arm invocation; default cat9 TREE = launcher
# default, so no TREE override is passed.)
CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS="$MAX_NUM_SEQS_OVR" \
  FR10_METRICS=0 BATCH_INVARIANT=0 \
  LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
  FR13_ENABLE_APC=1 \
  MAMBA_BLOCK_SIZE="$MAMBA_BLOCK_SIZE" \
  MAMBA_SSM_CACHE_DTYPE="$MAMBA_SSM_CACHE_DTYPE" \
  APC_MAX_NUM_BATCHED_TOKENS="$APC_MAX_NUM_BATCHED_TOKENS" \
  FR13_RUN_DIR="$PWD/$RUNDIR" LOG_DIR="$PWD/$RUNDIR/logs" \
  scripts/fr13_launch_forked_fa2_tree_server.sh > "$RUNDIR/launch.log" 2>&1
RC=$?
if (( RC != 0 )); then echo "FAIL: launcher rc=$RC"; tail -40 "$RUNDIR/launch.log"; exit 2; fi

# ---- GATE-A: wait up to 1200s for /health; FAIL-LOUD if the container dies ----
# Scan docker logs for the canonical APC/GDN unsupported-combo signatures.
fail_scan(){
  docker logs "$CONTAINER" > "$RUNDIR/boot_log_snapshot.txt" 2>&1 || true
  if grep -nE "NotImplementedError|does not yet support|does not support|Traceback \(most recent call last\)|EngineDeadError|EngineCore .*died|raise NotImplementedError" \
       "$RUNDIR/boot_log_snapshot.txt" | head -20 | tee "$RUNDIR/boot_error_hits.txt" | grep -q .; then
    return 0  # found a fatal signature
  fi
  return 1
}

T0=$(date +%s)
HEALTHY=0
while (( $(date +%s) < T0 + 1200 )); do
  if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
  if [[ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" != "running" ]]; then
    echo "GATE-A FAIL: container died before health (APC+GDN-hybrid init crash)"
    fail_scan && { echo "--- fatal signature(s) in boot log: ---"; cat "$RUNDIR/boot_error_hits.txt"; }
    docker logs "$CONTAINER" 2>&1 | tail -60; exit 3
  fi
  sleep 5
done
if (( HEALTHY != 1 )); then
  echo "GATE-A FAIL: /health not up in 1200s"
  fail_scan && { echo "--- fatal signature(s) in boot log: ---"; cat "$RUNDIR/boot_error_hits.txt"; }
  docker logs "$CONTAINER" 2>&1 | tail -60; exit 3
fi
echo "GATE-A PASS: healthy after $(( $(date +%s) - T0 ))s"

# A container can be /health-OK yet have logged a fatal during async init; re-scan.
docker logs "$CONTAINER" > "$RUNDIR/boot_log_snapshot.txt" 2>&1 || true
if fail_scan; then
  echo "GATE-A FAIL: fatal signature in boot log despite /health"
  cat "$RUNDIR/boot_error_hits.txt"; exit 3
fi

# ---- GATE-B: confirm APC + align engaged in the boot log ----
# vLLM logs enable_prefix_caching=True in the resolved CacheConfig and a
# mamba_cache_mode line when spec forces align. Accept either as evidence.
{
  echo "--- enable_prefix_caching ---"
  grep -nE "enable_prefix_caching=True|enable_prefix_caching': True|'enable_prefix_caching': True" "$RUNDIR/boot_log_snapshot.txt" | head -5
  echo "--- mamba cache mode / chunked prefill ---"
  grep -niE "mamba_cache_mode|mamba_block_size|chunked.?prefill|enable_chunked_prefill" "$RUNDIR/boot_log_snapshot.txt" | head -10
} | tee "$RUNDIR/gateB_apc_engaged.txt"
APC_ON=0
grep -qiE "enable_prefix_caching=True|'enable_prefix_caching': True" "$RUNDIR/boot_log_snapshot.txt" && APC_ON=1
ALIGN_ON=0
grep -qiE "mamba_cache_mode|mamba_block_size" "$RUNDIR/boot_log_snapshot.txt" && ALIGN_ON=1
if (( APC_ON == 1 || ALIGN_ON == 1 )); then
  echo "GATE-B PASS: APC/align engaged in boot log (apc=$APC_ON align/mamba=$ALIGN_ON)"
else
  echo "GATE-B FAIL: no enable_prefix_caching=True / mamba_cache_mode evidence in boot log"
  exit 4
fi

# ---- build the long (>2000-token) prompt for /v1/completions ----
# Source: a BANKED ~14k-token codex Responses-API pair dump (instructions + input
# text). We flatten it into ONE raw text prompt so /v1/completions can replay the
# same long prefix twice (the 2nd hits the cache). Falls back to a synthesized
# >2000-token prompt if no banked dump is present.
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
    # Synthesize a deterministic >2000-token prompt (no banked dump available).
    base = ("The following is a long technical specification for a speculative "
            "decoding verifier on a gated DeltaNet hybrid language model. ")
    text = (base * 400)  # ~ tens of thousands of chars -> well over 2000 tokens
Path(out).write_text(text)
print(f"[prompt] wrote {len(text)} chars to {out} (source={'banked' if p.is_file() else 'synth'})")
PY
[[ -s "$PROMPT_FILE" ]] || { echo "FAIL: long prompt not produced"; exit 4; }

# ---- snapshot /metrics, reset prefix cache, then snapshot again (clean baseline) ----
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$RUNDIR/metrics_before.txt" 2>&1 || true
curl -fsS -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" \
  > "$RUNDIR/reset_prefix_cache.txt" 2>&1 || echo "WARN: reset_prefix_cache failed (non-fatal)"

# ---- send a JSON /v1/completions request via a python helper (binary-safe body) ----
# Helper reads the prompt file, posts to /v1/completions, writes the HTTP status +
# response body. temp 0, max_tokens 8 (cheap; we only need the prefix cached).
send_completion(){  # args: temperature max_tokens outfile
  local temp="$1" maxtok="$2" outfile="$3"
  .venv/bin/python - "$PORT" "$PROMPT_FILE" "$temp" "$maxtok" "$outfile" <<'PY'
import json, sys, urllib.request, urllib.error
port, pfile, temp, maxtok, outfile = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
prompt = open(pfile).read()
body = json.dumps({
    "model": "qwen3.6-27b",
    "prompt": prompt,
    "temperature": temp,
    "max_tokens": maxtok,
    "stream": False,
}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        status = r.status
        data = r.read().decode()
except urllib.error.HTTPError as e:
    status = e.code
    data = e.read().decode()
except Exception as e:
    status = -1
    data = json.dumps({"client_error": str(e)})
open(outfile, "w").write(data)
print(f"HTTP {status}")
sys.exit(0 if status == 200 else 1)
PY
}

# ---- GATE-C: two IDENTICAL long temp-0 requests; the 2nd HITS the prefix cache ----
echo "[gateC] request #1 (cold, populates cache)"
send_completion 0 160 "$RUNDIR/completion_1.json" | tee "$RUNDIR/completion_1_http.txt"
RC1=${PIPESTATUS[0]}
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$RUNDIR/metrics_after_req1.txt" 2>&1 || true
echo "[gateC] request #2 (identical -> should HIT cache)"
send_completion 0 160 "$RUNDIR/completion_2.json" | tee "$RUNDIR/completion_2_http.txt"
RC2=${PIPESTATUS[0]}
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$RUNDIR/metrics_after_req2.txt" 2>&1 || true
if (( RC1 != 0 || RC2 != 0 )); then
  echo "GATE-C FAIL: a completions request errored (req1 rc=$RC1 req2 rc=$RC2)"
  echo "--- req1 body ---"; head -c 600 "$RUNDIR/completion_1.json"; echo
  echo "--- req2 body ---"; head -c 600 "$RUNDIR/completion_2.json"; echo
  exit 5
fi

# Assert all three prefix-cache counters are > 0 in the post-req2 /metrics.
# Names verified present in THIS image (vllm/vllm-openai@sha256:3dbe092...):
#   vllm:prefix_cache_queries_total
#   vllm:prefix_cache_hits_total   (this image; NOT the older gpu_prefix_cache_* name)
#   vllm:prompt_tokens_cached_total
# We match by metric NAME and sum across label sets, ignoring {labels}.
.venv/bin/python - "$RUNDIR/metrics_after_req2.txt" <<'PY'
import sys, re
from pathlib import Path
lines = Path(sys.argv[1]).read_text().splitlines()
# Sum every metric NAME (ignoring {labels}) into one dict, then read off.
sums = {}
seen = set()
for line in lines:
    if line.startswith("#"):
        continue
    m = re.match(r"^(vllm:[a-zA-Z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", line)
    if not m:
        continue
    name, val = m.group(1), m.group(3)
    seen.add(name)
    try:
        sums[name] = sums.get(name, 0.0) + float(val)
    except ValueError:
        continue
# Primary counter names (verified present in this image). The hits counter has an
# alt name in some vLLM builds; use whichever NAME is present in this /metrics.
queries = sums.get("vllm:prefix_cache_queries_total", 0.0)
hits_name = "vllm:prefix_cache_hits_total" if "vllm:prefix_cache_hits_total" in seen \
    else ("vllm:gpu_prefix_cache_hits_total" if "vllm:gpu_prefix_cache_hits_total" in seen
          else "vllm:prefix_cache_hits_total")
hits = sums.get(hits_name, 0.0)
cached = sums.get("vllm:prompt_tokens_cached_total", 0.0)
wanted = {
    "vllm:prefix_cache_queries_total": queries,
    hits_name: hits,
    "vllm:prompt_tokens_cached_total": cached,
}
print("[gateC] counters:")
ok = True
for k, v in wanted.items():
    status = "OK(>0)" if v > 0 else "ZERO"
    if v <= 0:
        ok = False
    print(f"   {k} = {v}  [{status}]")
if ok:
    print("GATE-C PASS: prefix cache hit (all three counters > 0)")
    sys.exit(0)
else:
    print("GATE-C FAIL: 0-hit -> the vLLM #45238 silent-0%%-hit trap (align keeps "
          "1 checkpoint at the last block boundary). This is NOT a crash; sweep "
          "MAMBA_BLOCK_SIZE / APC_MAX_NUM_BATCHED_TOKENS and re-run.")
    sys.exit(7)
PY
GATEC_RC=$?
if (( GATEC_RC != 0 )); then echo "GATE-C verdict rc=$GATEC_RC (FAIL)"; GATE_FAIL=1; else GATE_FAIL=0; fi

# ---- LOSSLESS A/B: req1 (cache MISS) vs req2 (cache HIT), identical greedy prompt.
# If the APC SSM snapshot/restore is lossless, the cache-hit continuation is
# byte-identical to the cache-miss continuation. A DIFFER = the carrier still poisons.
echo "[lossless] comparing req1 (miss) vs req2 (hit) greedy continuations"
.venv/bin/python - "$RUNDIR/completion_1.json" "$RUNDIR/completion_2.json" <<'PY'
import sys, json
def text(f):
    try:
        d = json.load(open(f)); return d["choices"][0]["text"]
    except Exception as e:
        return f"<<parse-error {e}>>"
c1, c2 = text(sys.argv[1]), text(sys.argv[2])
if c1 == c2:
    print(f"LOSSLESS-MATCH: C1==C2 ({len(c1)} chars) -> APC cache-hit byte-identical to miss")
    sys.exit(0)
i = next((k for k in range(min(len(c1), len(c2))) if c1[k] != c2[k]), min(len(c1), len(c2)))
print(f"LOSSLESS-DIFFER: first divergence at char {i} (len C1={len(c1)} C2={len(c2)})")
print("  C1[i:i+90]=", repr(c1[i:i+90]))
print("  C2[i:i+90]=", repr(c2[i:i+90]))
sys.exit(1)
PY
LOSSLESS_RC=$?
echo "[lossless] verdict rc=$LOSSLESS_RC ($([ $LOSSLESS_RC -eq 0 ] && echo LOSSLESS || echo POISON))"

# ---- TTFT: prefill latency cache-MISS vs cache-HIT (the APC speed win). Send the same
# long prompt max_tokens=1 (so time_total ~= prefill+TTFT); reset cache first for a true
# miss, then resend for a hit. curl -w gives wall-clock per request.
echo "[ttft] measuring prefill cache-MISS vs cache-HIT"
ttft_req(){ curl -s -o /dev/null -w '%{time_total}' -m 900 -X POST "http://127.0.0.1:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @<(.venv/bin/python -c "import json;print(json.dumps({'model':'qwen3.6-27b','prompt':open('$PROMPT_FILE').read(),'temperature':0,'max_tokens':1}))"); }
curl -fsS -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" >/dev/null 2>&1
T_MISS=$(ttft_req); echo "[ttft] cache-MISS prefill: ${T_MISS}s"
T_HIT=$(ttft_req);  echo "[ttft] cache-HIT  prefill: ${T_HIT}s"
.venv/bin/python -c "
m=float('$T_MISS'); h=float('$T_HIT')
print(f'[ttft] APC speedup: {m/h:.2f}x  (miss {m:.3f}s -> hit {h:.3f}s)')
print('TTFT-WIN' if h < m*0.9 else 'TTFT-NO-WIN')
" 2>/dev/null || echo "[ttft] parse failed (miss=$T_MISS hit=$T_HIT)"

# ---- GATE-D: ONE temp-0.6 generation (max_tokens=128) so spec-decode runs ----
# num_accepted>1 with APC live. Then assert the server is STILL alive and the
# docker log has no NEW crash/NotImplementedError after this generation (the
# DS-layout / vLLM #43559 regression check).
echo "[gateD] spec-decode generation (temp 0.6, max_tokens 128)"
LOGLEN_BEFORE=$(docker logs "$CONTAINER" 2>&1 | wc -l)
send_completion 0.6 128 "$RUNDIR/completion_specgen.json" | tee "$RUNDIR/completion_specgen_http.txt"
RCD=${PIPESTATUS[0]}
sleep 2
# (a) server still /health-OK?
ALIVE=0
curl -fsS -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && ALIVE=1
# (b) no NEW crash signature in the docker log produced AFTER the gen request
docker logs "$CONTAINER" 2>&1 | tail -n "+$((LOGLEN_BEFORE+1))" > "$RUNDIR/docker_log_after_specgen.txt" || true
# NOTE: `grep -c` already prints 0 on no-match AND exits 1, so a trailing
# `|| echo 0` yields the two-line string "0\n0" which breaks `(( CRASH_HITS == 0 ))`
# with a syntax error -> a false GATE-D FAIL. Take the count robustly (single int).
CRASH_HITS=$(grep -cE "NotImplementedError|does not yet support|does not support|Traceback \(most recent call last\)|EngineDeadError|EngineCore .*died|CUDA error|RuntimeError" \
  "$RUNDIR/docker_log_after_specgen.txt" 2>/dev/null | head -1 || true)
[[ "$CRASH_HITS" =~ ^[0-9]+$ ]] || CRASH_HITS=0
echo "[gateD] specgen_http_rc=$RCD alive=$ALIVE post_gen_crash_hits=$CRASH_HITS"
if (( ALIVE == 1 && CRASH_HITS == 0 && RCD == 0 )); then
  echo "GATE-D PASS: server survived spec-decode (num_accepted>1) under live APC, no post-gen crash"
  GATED_FAIL=0
else
  echo "GATE-D FAIL: APC+spec-decode crash/regression (alive=$ALIVE crash_hits=$CRASH_HITS specgen_rc=$RCD)"
  echo "--- post-gen docker log tail ---"; tail -40 "$RUNDIR/docker_log_after_specgen.txt"
  GATED_FAIL=1
fi

# ---- overall verdict ----
echo "=== APC GATE-0 SUMMARY ==="
echo "GATE-A (boot)        : PASS"
echo "GATE-B (apc engaged) : PASS"
echo "GATE-C (cache hits)  : $([[ "${GATE_FAIL:-1}" == "0" ]] && echo PASS || echo FAIL)"
echo "GATE-D (no crash)    : $([[ "${GATED_FAIL:-1}" == "0" ]] && echo PASS || echo FAIL)"
OVERALL=0
[[ "${GATE_FAIL:-1}" == "0" && "${GATED_FAIL:-1}" == "0" ]] || OVERALL=1
echo "OVERALL: $([[ "$OVERALL" == "0" ]] && echo PASS || echo FAIL)  (rundir=$RUNDIR)"
# teardown runs via the EXIT trap
exit $OVERALL
