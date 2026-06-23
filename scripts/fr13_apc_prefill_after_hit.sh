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
# GENREGION gate: cap MUST equal the mamba block (1024) -- the #45238/#45477
# fix. Do NOT raise it (a larger cap reintroduces the silent-0%-hit trap and
# breaks the R3 chunk-end-on-boundary geometry).
export APC_MAX_NUM_BATCHED_TOKENS=${APC_MAX_NUM_BATCHED_TOKENS:-1024}

# ---- FR13_APC_CACHE_AB: the CORRECT cache-ON vs cache-OFF losslessness probe.
# Default off (inert). When set =1 the launcher's docker -e plumbs it into the
# GDN forward; the req1(cache-ON)/req2(cache-OFF) geometry below then dumps
# h_on / h_off to /logs/fr13_apc_cache_ab.jsonl. block size for the OFF-arm
# boundary enumeration follows MAMBA_BLOCK_SIZE.
export FR13_APC_CACHE_AB=${FR13_APC_CACHE_AB:-0}
export FR13_APC_CACHE_AB_LOG=${FR13_APC_CACHE_AB_LOG:-/logs/fr13_apc_cache_ab.jsonl}
export FR13_APC_CACHE_AB_BLOCK=${FR13_APC_CACHE_AB_BLOCK:-$MAMBA_BLOCK_SIZE}

# ---- FR13_APC_CACHE_AB_GENREGION: pinned phase req_ids (purely req-derived
# phase; NO env phase toggle). The arms read these from the container env and
# match each row's req_id (plumbed via _LUMO_FA_SAMPLER_ROW_REQ_IDS) -> R2=ON
# (h_on), R3=OFF (h_off); R0 (warm) and unpinned reqs record nothing. The
# driver pins the engine req_id of each /v1/completions call via the
# X-Request-Id header (vLLM OpenAIServing._base_request_id promotes it to the
# engine-side req_id == input_batch.req_ids[i]).
export FR13_APC_AB_R0_REQ=${FR13_APC_AB_R0_REQ:-R0}
export FR13_APC_AB_R2_REQ=${FR13_APC_AB_R2_REQ:-R2}
export FR13_APC_AB_R3_REQ=${FR13_APC_AB_R3_REQ:-R3}

# ---- deployed-lossless config for the GENREGION gate (design BOOT block) ----
# FR13_APC_SSM_LEAF_SRC=1 tests the FIXED restore (the value the carrier
# redirects). ENFORCE_EAGER=1 (diagnostics eager). APC_MAX_NUM_BATCHED_TOKENS
# == block (the #45238/#45477 fix; do NOT raise). FR10_DECODE_MODE_DEFAULT=
# tree_mtp so R0 actually tree-generates. FR13_APC_BLOCK_ALIGN_45477=1 so R3's
# chunk ends land exactly on 1024/2048/3072.
export FR13_APC_SSM_LEAF_SRC=${FR13_APC_SSM_LEAF_SRC:-1}
export ENFORCE_EAGER=${ENFORCE_EAGER:-1}
export FR13_APC_CONV_FIX=${FR13_APC_CONV_FIX:-1}
export FR13_APC_BLOCK_ALIGN_45477=${FR13_APC_BLOCK_ALIGN_45477:-1}
export FR10_DECODE_MODE_DEFAULT=${FR10_DECODE_MODE_DEFAULT:-tree_mtp}
export APC_MAX_NUM_BATCHED_TOKENS=${APC_MAX_NUM_BATCHED_TOKENS:-1024}
# target generated-region block boundary to certify (k*block, k>k_P). With
# P=2048 (k_P=2) and G>=1024, B*=3072 is in the generated region.
export FR13_APC_AB_PROMPT_TOKENS=${FR13_APC_AB_PROMPT_TOKENS:-2048}
export FR13_APC_AB_TARGET_BOUNDARY=${FR13_APC_AB_TARGET_BOUNDARY:-3072}

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
  FR13_APC_DROP_FINAL_BLOCK="${FR13_APC_DROP_FINAL_BLOCK:-0}" \
  FR13_APC_CACHE_AB="${FR13_APC_CACHE_AB:-0}" \
  FR13_APC_CACHE_AB_LOG="${FR13_APC_CACHE_AB_LOG:-/logs/fr13_apc_cache_ab.jsonl}" \
  FR13_APC_CACHE_AB_BLOCK="${FR13_APC_CACHE_AB_BLOCK:-$MAMBA_BLOCK_SIZE}" \
  FR13_APC_AB_R2_REQ="${FR13_APC_AB_R2_REQ:-R2}" \
  FR13_APC_AB_R3_REQ="${FR13_APC_AB_R3_REQ:-R3}" \
  FR13_APC_SSM_LEAF_SRC="${FR13_APC_SSM_LEAF_SRC:-1}" \
  FR13_APC_CONV_FIX="${FR13_APC_CONV_FIX:-1}" \
  FR13_APC_BLOCK_ALIGN_45477="${FR13_APC_BLOCK_ALIGN_45477:-1}" \
  FR10_DECODE_MODE_DEFAULT="${FR10_DECODE_MODE_DEFAULT:-tree_mtp}" \
  ENFORCE_EAGER="${ENFORCE_EAGER:-1}" \
  MAMBA_BLOCK_SIZE="$MAMBA_BLOCK_SIZE" \
  MAMBA_SSM_CACHE_DTYPE="$MAMBA_SSM_CACHE_DTYPE" \
  APC_MAX_NUM_BATCHED_TOKENS="$APC_MAX_NUM_BATCHED_TOKENS" \
  FR13_RUN_DIR="$PWD/$RUNDIR" LOG_DIR="$PWD/$RUNDIR/logs" \
  "${LAUNCHER_SCRIPT:-scripts/fr13_launch_forked_fa2_tree_server.sh}" > "$RUNDIR/launch.log" 2>&1
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
send_completion(){  # args: temperature max_tokens outfile [promptfile]
  local temp="$1" maxtok="$2" outfile="$3" pfile="${4:-$PROMPT_FILE}"
  .venv/bin/python - "$PORT" "$pfile" "$temp" "$maxtok" "$outfile" <<'PY'
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

# =================================================================================
# FR13_APC_CACHE_AB_GENREGION two-phase geometry (R0 / R2 / R3).
# R0 (warm): prefix-prefill P (token_count(P)%1024==0) then TREE-DECODE G>=1024
#   so a block boundary B*=3072 falls in the GENERATED region. Capture R0's
#   EXACT generated token-ids (PG_IDS = P_ids + G_ids) for byte-identical replay.
# R2 (cache-ON, h_on): prompt=[P_ids + G_ids[:G2]] so it HITS cached blocks up to
#   and INCLUDING B*=3072 and re-prefills a <=1024 tail in ONE step.
# R3 (cache-OFF, h_off): reset cache, fresh prefill the FULL [P_ids+G_ids]; the
#   chunk completing [2048,3072) emits h_off@3072 (native chunked incumbent).
# Phase is purely req-derived in the GDN arms (X-Request-Id R0/R2/R3).
# =================================================================================
PROMPT_TOKENS=${FR13_APC_AB_PROMPT_TOKENS:-2048}
TARGET_BOUNDARY=${FR13_APC_AB_TARGET_BOUNDARY:-3072}
GEN_TOKENS=${FR13_APC_AB_GEN_TOKENS:-1024}
BLOCK=${MAMBA_BLOCK_SIZE:-1024}

# ---- /tokenize-pin P to exactly PROMPT_TOKENS tokens on the FULL completions-
# wrapped string (NOT char-fraction). Iteratively trim the raw prompt text until
# POST /tokenize reports exactly PROMPT_TOKENS ids; assert %BLOCK==0. Emits
# P_IDS_FILE (json list of token ids).
P_IDS_FILE="$RUNDIR/P_ids.json"
.venv/bin/python - "$PORT" "$PROMPT_FILE" "$P_IDS_FILE" "$PROMPT_TOKENS" "$BLOCK" <<'PY'
import json, sys, urllib.request, urllib.error
port, pfile, out, want, block = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
full = open(pfile).read()
def tok(text):
    body = json.dumps({"model": "qwen3.6-27b", "prompt": text}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/tokenize",
        data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode())
    # vLLM /tokenize returns {"count":N,"tokens":[ids...],...}
    ids = d.get("tokens")
    if ids is None:
        ids = d.get("token_ids")
    if ids is None:
        raise SystemExit(f"FAIL: /tokenize returned no token list: keys={list(d.keys())}")
    return [int(x) for x in ids]
ids_full = tok(full)
if len(ids_full) < want:
    raise SystemExit(f"FAIL: prompt only {len(ids_full)} tokens < wanted {want}; pick a longer source")
if want % block != 0:
    raise SystemExit(f"FAIL: PROMPT_TOKENS {want} %% block {block} != 0")
# Exact id-space pin: take the first `want` ids. We feed R0's prompt as TOKEN
# IDS directly (vLLM /v1/completions accepts prompt=[int,...]) so the count is
# exact regardless of detokenize round-trips.
p_ids = ids_full[:want]
if len(p_ids) != want:
    raise SystemExit(f"FAIL: could not pin P to {want} ids (got {len(p_ids)})")
json.dump(p_ids, open(out, "w"))
print(f"[tokenize] P pinned to {len(p_ids)} ids (want={want}, %{block}=={want % block})")
PY
(( $? == 0 )) || { echo "FAIL: P token-pin"; exit 4; }

# ---- a token-id completion helper: posts prompt=<id list> with a pinned
# X-Request-Id; captures the EXACT generated token-ids via logprobs. Fails loud
# if the response does not expose generated token-ids (no silent re-tokenize).
send_ids(){  # args: req_id prompt_ids_json temp maxtok seed outfile [gen_ids_out]
  local rid="$1" pidsf="$2" temp="$3" maxtok="$4" seed="$5" outfile="$6" genout="${7:-}"
  .venv/bin/python - "$PORT" "$rid" "$pidsf" "$temp" "$maxtok" "$seed" "$outfile" "$genout" <<'PY'
import json, sys, urllib.request, urllib.error
port, rid, pidsf, temp, maxtok, seed, outfile, genout = (
    sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]),
    int(sys.argv[5]), int(sys.argv[6]), sys.argv[7], sys.argv[8])
p_ids = json.load(open(pidsf))
payload = {
    "model": "qwen3.6-27b",
    "prompt": p_ids,
    "temperature": temp,
    "max_tokens": maxtok,
    "stream": False,
    "return_token_ids": True,
}
if temp > 0:
    payload["seed"] = seed
body = json.dumps(payload).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=body,
    headers={"Content-Type": "application/json", "X-Request-Id": rid},
    method="POST")
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
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
if status != 200:
    sys.exit(1)
if genout:
    d = json.loads(data)
    ch = d["choices"][0]
    gen_ids = None
    # return_token_ids=True puts the served (generated) token-ids at the choice
    # level (vLLM /v1/completions, same as fr13_gold_margin_probe.py L102).
    if isinstance(ch.get("token_ids"), list):
        gen_ids = [int(x) for x in ch["token_ids"]]
    elif isinstance((ch.get("logprobs") or {}).get("token_ids"), list):
        gen_ids = [int(x) for x in ch["logprobs"]["token_ids"]]
    if gen_ids is None:
        raise SystemExit(
            "FAIL: cannot capture EXACT generated token-ids (return_token_ids gave "
            "no choice.token_ids). Re-tokenizing text is NOT byte-safe across P|G. "
            "choice keys=" + repr(list(ch.keys())))
    # If the image returns prompt+generated under token_ids, slice off the prompt.
    _pt = ch.get("prompt_token_ids")
    if isinstance(_pt, list) and len(gen_ids) > maxtok and gen_ids[:len(_pt)] == [int(x) for x in _pt]:
        gen_ids = gen_ids[len(_pt):]
    if len(gen_ids) < 1:
        raise SystemExit("FAIL: zero generated token-ids captured")
    json.dump(gen_ids, open(genout, "w"))
    print(f"[capture] req={rid} captured {len(gen_ids)} generated token-ids")
sys.exit(0)
PY
}

G_IDS_FILE="$RUNDIR/G_ids.json"
PG_IDS_FILE="$RUNDIR/PG_ids.json"
R2_IDS_FILE="$RUNDIR/R2_ids.json"

echo "[R0] warm: reset cache, prefill P then TREE-DECODE G>=$GEN_TOKENS (temp 0.6, seed)"
curl -fsS -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" >/dev/null 2>&1
send_ids "$FR13_APC_AB_R0_REQ" "$P_IDS_FILE" 0.6 "$GEN_TOKENS" 1234 \
  "$RUNDIR/R0.json" "$G_IDS_FILE" | tee "$RUNDIR/R0_http.txt"
RC0=${PIPESTATUS[0]}
if (( RC0 != 0 )); then echo "FAIL: R0 errored (rc=$RC0)"; head -c 800 "$RUNDIR/R0.json"; echo; exit 5; fi

# Build PG_IDS = P_ids + G_ids; R2 prompt = P_ids + G_ids[:G2] (G2 s.t. R2 hits
# blocks incl TARGET_BOUNDARY and re-prefills a <=BLOCK tail in one step:
# TARGET < len(R2) <= TARGET+BLOCK). Assert TARGET in generated region.
.venv/bin/python - "$P_IDS_FILE" "$G_IDS_FILE" "$PG_IDS_FILE" "$R2_IDS_FILE" \
  "$PROMPT_TOKENS" "$TARGET_BOUNDARY" "$BLOCK" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); g = json.load(open(sys.argv[2]))
pgout, r2out = sys.argv[3], sys.argv[4]
ptok, target, block = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
pg = list(p) + list(g)
if not (ptok < target):
    raise SystemExit(f"FAIL: target {target} not > prompt_tokens {ptok} (boundary not in GENERATED region)")
if not (ptok < target <= len(pg)):
    raise SystemExit(f"FAIL: target {target} not in (len(P)={ptok}, len(P+G)={len(pg)}] -- generate more G")
# R2 length: pick the smallest multiple s.t. target < len <= target+block, so R2
# hits cached blocks through `target` and re-prefills a <=block tail in one step.
r2_len = min(len(pg), target + block)
if not (target < r2_len <= target + block):
    raise SystemExit(f"FAIL: cannot size R2 tail (r2_len={r2_len}, target={target}, block={block})")
r2 = pg[:r2_len]
json.dump(pg, open(pgout, "w"))
json.dump(r2, open(r2out, "w"))
print(f"[build] len(P)={ptok} len(G)={len(g)} len(P+G)={len(pg)} len(R2)={r2_len} target={target}")
PY
(( $? == 0 )) || { echo "FAIL: PG/R2 build (see above)"; exit 5; }

# ---- R2 (cache-ON, h_on): keep cache warm (NO reset). Hits incl TARGET.
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$RUNDIR/metrics_before_R2.txt" 2>&1 || true
echo "[R2] cache-ON: prompt=[P+G[:G2]] (HITS cached blocks incl $TARGET_BOUNDARY)"
send_ids "$FR13_APC_AB_R2_REQ" "$R2_IDS_FILE" 0 1 0 "$RUNDIR/R2.json" "" | tee "$RUNDIR/R2_http.txt"
RC2=${PIPESTATUS[0]}
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$RUNDIR/metrics_after_R2.txt" 2>&1 || true
if (( RC2 != 0 )); then echo "FAIL: R2 errored (rc=$RC2)"; head -c 800 "$RUNDIR/R2.json"; echo; exit 5; fi

# ---- R3 (cache-OFF, h_off): RESET (force-preempt, clears mamba_state_idx),
# then fresh prefill the FULL [P+G]. Assert hits-delta==0 (cross-request MISS).
curl -fsS -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" >/dev/null 2>&1
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$RUNDIR/metrics_before_R3.txt" 2>&1 || true
echo "[R3] cache-OFF: reset, fresh prefill FULL [P+G] (chunk completing [2048,3072) -> h_off@3072)"
send_ids "$FR13_APC_AB_R3_REQ" "$PG_IDS_FILE" 0 1 0 "$RUNDIR/R3.json" "" | tee "$RUNDIR/R3_http.txt"
RC3=${PIPESTATUS[0]}
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$RUNDIR/metrics_after_R3.txt" 2>&1 || true
if (( RC3 != 0 )); then echo "FAIL: R3 errored (rc=$RC3)"; head -c 800 "$RUNDIR/R3.json"; echo; exit 5; fi

# ---- cache-hit asserts: R2 hits incremented + prompt_tokens_cached >= TARGET;
# R3 hits-delta == 0 (cross-request miss => cache-OFF incumbent).
.venv/bin/python - "$RUNDIR/metrics_before_R2.txt" "$RUNDIR/metrics_after_R2.txt" \
  "$RUNDIR/metrics_before_R3.txt" "$RUNDIR/metrics_after_R3.txt" "$TARGET_BOUNDARY" <<'PY'
import sys, re
from pathlib import Path
def sums(path):
    s = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^(vllm:[a-zA-Z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", line)
        if not m:
            continue
        try:
            s[m.group(1)] = s.get(m.group(1), 0.0) + float(m.group(3))
        except ValueError:
            pass
    return s
b2, a2, b3, a3, target = sums(sys.argv[1]), sums(sys.argv[2]), sums(sys.argv[3]), sums(sys.argv[4]), int(sys.argv[5])
def hits(s):
    return s.get("vllm:prefix_cache_hits_total", s.get("vllm:gpu_prefix_cache_hits_total", 0.0))
r2_hits_delta = hits(a2) - hits(b2)
r2_cached = a2.get("vllm:prompt_tokens_cached_total", 0.0)
r3_hits_delta = hits(a3) - hits(b3)
print(f"[asserts] R2 hits_delta={r2_hits_delta} prompt_tokens_cached_total(after R2)={r2_cached}")
print(f"[asserts] R3 hits_delta={r3_hits_delta}")
ok = True
if not (r2_hits_delta > 0):
    print(f"FAIL: R2 prefix_cache_hits did NOT increment (delta={r2_hits_delta}) -> R2 did not HIT the generated block"); ok = False
if not (r2_cached >= target):
    print(f"FAIL: prompt_tokens_cached_total {r2_cached} < target {target} -> R2 did not cache through the boundary"); ok = False
if not (abs(r3_hits_delta) < 1e-9):
    print(f"FAIL: R3 hits_delta != 0 ({r3_hits_delta}) -> R3 is NOT a clean cross-request MISS (not cache-OFF incumbent)"); ok = False
if ok:
    print("CACHE-ASSERTS PASS: R2 hit incl boundary, R3 clean miss")
    sys.exit(0)
sys.exit(8)
PY
CACHE_RC=$?
if (( CACHE_RC != 0 )); then echo "GATE FAIL: cache-hit asserts (rc=$CACHE_RC)"; GATE_FAIL=1; else GATE_FAIL=0; fi

# ---- REDUCE: SOUND GENREGION verdict over the dumped h_on/h_off JSONL.
AB_JSONL="$RUNDIR/logs/fr13_apc_cache_ab.jsonl"
[[ -s "$AB_JSONL" ]] || AB_JSONL="$RUNDIR/fr13_apc_cache_ab.jsonl"
echo "[reduce] $AB_JSONL --prompt-tokens $PROMPT_TOKENS --target-boundary $TARGET_BOUNDARY"
.venv/bin/python scripts/fr13_apc_cache_ab_reduce.py "$AB_JSONL" \
  --prompt-tokens "$PROMPT_TOKENS" --target-boundary "$TARGET_BOUNDARY" \
  --on-req "$FR13_APC_AB_R2_REQ" --off-req "$FR13_APC_AB_R3_REQ" \
  --out "$RUNDIR/fr13_apc_cache_ab_verdict.json" | tee "$RUNDIR/reduce_stdout.txt" || true

# Keep the legacy GATE-C metric-sum check too (engagement sanity).
# Assert all three prefix-cache counters are > 0 in the post-R3 /metrics.
cp -f "$RUNDIR/metrics_after_R3.txt" "$RUNDIR/metrics_after_req2.txt" 2>/dev/null || true
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
# Legacy engagement sanity only -- do NOT clobber the GENREGION cache-asserts
# GATE_FAIL (the binding gate is the reducer verdict + the R2-hit/R3-miss
# asserts above). Record it separately.
if (( GATEC_RC != 0 )); then echo "[engage] GATE-C metric-sum rc=$GATEC_RC (FAIL)"; ENGAGE_FAIL=1; else ENGAGE_FAIL=0; fi

# ---- LOSSLESS verdict: the SOUND GENREGION reducer (scripts/fr13_apc_cache_ab_reduce.py)
# already ran above and wrote $RUNDIR/fr13_apc_cache_ab_verdict.json. Surface its
# label here. The byte-stream A/B over completion text is NOT the instrument for
# this gate (the binding axis is per-element h_on vs h_off at the generated-region
# boundary 3072 with per-channel argmax match).
echo "[lossless] GENREGION reducer verdict (h_on R2 vs h_off R3 @ $TARGET_BOUNDARY):"
LOSSLESS_RC=1
if [[ -s "$RUNDIR/fr13_apc_cache_ab_verdict.json" ]]; then
  .venv/bin/python - "$RUNDIR/fr13_apc_cache_ab_verdict.json" <<'PY'
import sys, json
d = json.load(open(sys.argv[1]))
v = d.get("verdict")
print(f"  verdict={v} n_matched_at_target={d.get('n_matched_at_target')} "
      f"fp_max_abs={d.get('OVERALL_ssm_fp_max_abs')} "
      f"all_argmax_match={d.get('all_argmax_match')} guards={d.get('guards')}")
# rc: 0 LOSSLESS, 2 VACUOUS, 3 ARGMAX_FLIP escalate, 1 NOT_LOSSLESS
sys.exit({"LOSSLESS": 0, "VACUOUS": 2,
          "ARGMAX_FLIP_WITHIN_ULP_ESCALATE": 3}.get(v, 1))
PY
  LOSSLESS_RC=$?
else
  echo "  (no verdict json produced -- reducer found no records?)"
fi
case "$LOSSLESS_RC" in
  0) echo "[lossless] LOSSLESS";;
  2) echo "[lossless] VACUOUS (gate did not measure the target boundary -- check R2 hit/R3 miss + G length)";;
  3) echo "[lossless] ARGMAX-FLIP within ULP -> ESCALATE to user (parked floor vs real defect)";;
  *) echo "[lossless] NOT-LOSSLESS";;
esac

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
echo "=== FR13_APC_CACHE_AB_GENREGION SUMMARY ==="
echo "GATE-A (boot)              : PASS"
echo "GATE-B (apc engaged)       : PASS"
echo "CACHE-ASSERTS (R2 hit/R3 miss) : $([[ "${GATE_FAIL:-1}" == "0" ]] && echo PASS || echo FAIL)"
echo "GATE-D (no crash)          : $([[ "${GATED_FAIL:-1}" == "0" ]] && echo PASS || echo FAIL)"
case "${LOSSLESS_RC:-1}" in
  0) LOSSLESS_LABEL=LOSSLESS;;
  2) LOSSLESS_LABEL=VACUOUS;;
  3) LOSSLESS_LABEL=ARGMAX_FLIP_ESCALATE;;
  *) LOSSLESS_LABEL=NOT_LOSSLESS;;
esac
echo "LOSSLESS (h_on vs h_off @ $TARGET_BOUNDARY) : $LOSSLESS_LABEL"
# OVERALL PASS requires: cache-asserts pass, no crash, and the reducer verdict
# is LOSSLESS. VACUOUS / ARGMAX_FLIP / NOT_LOSSLESS => not a clean PASS.
OVERALL=0
[[ "${GATE_FAIL:-1}" == "0" && "${GATED_FAIL:-1}" == "0" && "${LOSSLESS_RC:-1}" == "0" ]] || OVERALL=1
echo "OVERALL: $([[ "$OVERALL" == "0" ]] && echo PASS || echo FAIL)  (rundir=$RUNDIR)"
# teardown runs via the EXIT trap
exit $OVERALL
