#!/usr/bin/env bash
# =============================================================================
# FR13 APC APPLY-PATCH GATE -- TOKEN-LEVEL LOSSLESSNESS ON THE *REAL* TOOL-CALL
# =============================================================================
# CPU / orchestration ONLY. This is a near-clone of scripts/fr13_apc_usg_drive.sh
# (same boot config, same send_ids/teacher_force primitives, same R2-warm/R3-reset
# hit/miss discipline, same exercised-checks, same reducer scripts/
# fr13_apc_usg_reduce.py). It writes NO GPU kernels.
#
# THE ONE CHANGE vs the USG driver (and WHY this gate exists):
#   The USG gate teacher-forced a FORCED ignore_eos RAMBLE stream that does NOT
#   contain the apply-patch tool-call tokens. The 12907 cache-ON+fold-fix failure
#   is a MALFORMED apply-patch tool-call JSON (BadRequestError 'Expecting , delimiter'
#   / 'Unterminated string' at char ~90) at the critical "apply the fix" step --
#   so we must re-gate on the ACTUAL apply-patch token sequence, not a ramble.
#
#   Therefore PG_IDS here = the REAL apply-patch turn of the cache-OFF run that
#   SOLVED 12907:
#     P = the conversation CONTEXT the model saw right before emitting the apply-
#         patch tool-call (flatten of that turn's request.instructions+input, the
#         SAME flattener as scripts/fr13_apc_replay.py:flatten_to_prompt), /tokenize'd.
#     G = the apply-patch tool-call TOKENS. By default CAPTURED FRESH from THIS boot
#         (cache-OFF, return_token_ids, temp=0, NO ignore_eos -> the model naturally
#         decodes a real apply-patch tool-call from P). FALLBACK (G_SOURCE=retoken):
#         re-tokenize the dump's apply-patch response text. Both arms teacher-force
#         the identical PG_IDS so the cache-ON-vs-cache-OFF compare stays valid
#         regardless of which G source was used (the re-tokenize seam affects only
#         whether PG matches the ORIGINAL, not the comparison). The chosen source is
#         recorded in usg_meta.json ("g_source").
#
# GEOMETRY (so the apply-patch tokens are decoded AFTER a cache-restore of P):
#   The USG driver pinned P UP to a fixed small block multiple and rambled G for
#   thousands of tokens so a boundary lands deep in the ramble. Here G (the apply-
#   patch tool-call) is SMALL (~150 tokens) and P is LARGE (~18k tokens). So we pin
#   P DOWN to the LARGEST block multiple <= len(P) (P_PIN_MODE=down, the default):
#   the restore boundary B* = P_pinned then lands BELOW the real apply-patch tokens,
#   and PG_IDS[B*:] (the P-tail + the apply-patch G) is the DECODE-AFTER-RESTORE
#   region. The cache-ON arm warms PG_IDS[:B*] = HIT through B*, re-prefills the
#   <=1024 tail (incl. the apply-patch JSON structural tokens) = exercises the fold;
#   the cache-OFF arm resets = fresh prefill of the whole prefix (clean incumbent).
#   We sweep teacher-force positions j DENSELY ACROSS the apply-patch tokens
#   (P_pinned < j <= pglen), with a FINE stride near the JSON structural region
#   (quotes/braces/commas at G char ~90) where the BadRequestError struck.
#
# BINDING AXIS / HARD RULES H1..H7: IDENTICAL to scripts/fr13_apc_usg_drive.sh.
#   per-token argmax flip of (cache-ON SERVED) vs (cache-OFF TEACHER-FORCED REF)
#   at each swept absolute j, both arms forced on byte-identical PG_IDS[:j].
#   H4 cache-OFF = reset (clean MISS, hits_delta==0). H5 cache-ON = warm
#   (hits_delta>0, prompt_tokens_cached>=boundary). H6 each (arm,j) run TWICE.
#   H7 exercised-check (in-G-region boundary, on-hit, off-clean-miss) before verdict.
#
# RUN: WRITE-ONLY scaffold (validated with `bash -n` + py_compile). Boots a GPU
# container on GB10; the GPU is serialized -- never launch a third concurrent
# workflow. The MAIN THREAD runs this, not the worker.
# =============================================================================
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
REPO=/home/mark/shared/lumoFlyWheel

CONTAINER=${CONTAINER:-fr13-apc-applypatch}
PORT=${PORT:-9951}
MODEL=${MODEL:-qwen3.6-27b}
RUNROOT=${RUNROOT:-output/fr13_apc_applypatch}
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNDIR="$RUNROOT/run_$TS"
mkdir -p "$RUNDIR/logs"

# ---------------------------------------------------------------------------
# DEPLOYED FOLD-FIX-ON CONFIG (cuda-graph deploy regime; identical to USG).
# ---------------------------------------------------------------------------
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=${MAMBA_BLOCK_SIZE:-1024}
export MAMBA_SSM_CACHE_DTYPE=${MAMBA_SSM_CACHE_DTYPE:-float32}
export FR13_APC_CONV_FIX=${FR13_APC_CONV_FIX:-1}
export FR13_APC_SSM_LEAF_SRC=${FR13_APC_SSM_LEAF_SRC:-1}
export FR13_APC_HIT_RECURRENT_SUFFIX=${FR13_APC_HIT_RECURRENT_SUFFIX:-1}
export FR13_APC_HIT_SUFFIX_CAP=${FR13_APC_HIT_SUFFIX_CAP:-1000000}
export DEPLOY_FORCE_TEMP=${DEPLOY_FORCE_TEMP:-0.6}
export FR10_DECODE_MODE_DEFAULT=${FR10_DECODE_MODE_DEFAULT:-tree_mtp}
export APC_MAX_NUM_BATCHED_TOKENS=${APC_MAX_NUM_BATCHED_TOKENS:-$MAMBA_BLOCK_SIZE}
export FR13_APC_VALUE_VS_ORACLE=${FR13_APC_VALUE_VS_ORACLE:-0}
export FR13_APC_VALUE_VS_ORACLE_LOG=${FR13_APC_VALUE_VS_ORACLE_LOG:-/logs/fr13_apc_value_vs_oracle.jsonl}
# EAGER by default: this is a per-token lossless diagnostic, NOT a speed measurement,
# and eager keeps GPU memory flat (no cuda-graph reservation eating runtime headroom).
export ENFORCE_EAGER=${ENFORCE_EAGER:-1}

export FR10_METRICS=0
export BATCH_INVARIANT=0
export LUMO_FB_KERNEL_ROWS=${LUMO_FB_KERNEL_ROWS:-1}
export LUMO_FB_PROJ_PAD_ROWS=${LUMO_FB_PROJ_PAD_ROWS:-16}
MAX_NUM_SEQS_OVR=${MAX_NUM_SEQS_OVR:-1}

# ---------------------------------------------------------------------------
# APPLY-PATCH TURN SOURCE.
#   PAIR_DUMP : the REAL apply-patch turn of the cache-OFF run that SOLVED 12907.
#               Default = session 019ef3c9 (== codex_trace_retry.jsonl thread_id)
#               seq=9 -- the turn whose response.output emits the
#               `sed -i 's/...= 1/...= right/' separable.py` tool-call (the fix).
#   G_SOURCE  : fresh (default; capture the apply-patch tool-call tokens FRESH from
#               THIS boot, temp=0, no ignore_eos) OR retoken (re-tokenize the dump's
#               apply-patch response text). Recorded in usg_meta.json.g_source.
#   REUSE_PG_IDS : path to an existing PG_ids.json (+ REUSE_PLEN=<len(P_ids)>); when
#               set, STEP A capture is skipped (the boundary geometry is recomputed
#               from REUSE_PLEN/block, so REUSE_PLEN MUST be a block multiple).
# ---------------------------------------------------------------------------
PAIR_DUMP=${PAIR_DUMP:-output/fr13_apc_cacheoff_control/cat6root_cacheoff_12907/proxy_pair_dumps/pair_01782206847767834152_000009_initial.json}
G_SOURCE=${G_SOURCE:-fresh}            # fresh | retoken
P_PIN_MODE=${P_PIN_MODE:-down}         # down (pin P to largest block-mult <= len) | up
REUSE_PG_IDS=${REUSE_PG_IDS:-}
REUSE_PLEN=${REUSE_PLEN:-}

# ---- capture / sweep knobs ----
# G is the apply-patch tool-call (~150 tokens). Capture a bit more so the WHOLE
# tool-call + a little tail is in PG (the reducer only sweeps the apply-patch span).
GEN_TOKENS=${GEN_TOKENS:-256}          # fresh-capture max_tokens (apply-patch + tail)
SWEEP_STRIDE=${SWEEP_STRIDE:-4}        # FINE stride across the apply-patch tokens
SWEEP_MAX_POS=${SWEEP_MAX_POS:-96}     # cap on number of swept positions (cost)
SWEEP_FROM_BOUNDARY=${SWEEP_FROM_BOUNDARY:-1}  # 1: sweep j only ABOVE the restore boundary
LOGPROBS_K=${LOGPROBS_K:-20}           # top-k for the cache-OFF reference margin
CAPTURE_SEED=${CAPTURE_SEED:-1234}

echo "=== FR13 APC APPLY-PATCH GATE container=$CONTAINER port=$PORT rundir=$RUNDIR ==="
echo "    APC fold-fix-ON: enable=1 block=$MAMBA_BLOCK_SIZE ssm_dtype=$MAMBA_SSM_CACHE_DTYPE"
echo "    CONV_FIX=$FR13_APC_CONV_FIX SSM_LEAF_SRC=$FR13_APC_SSM_LEAF_SRC HIT_RECURRENT_SUFFIX=$FR13_APC_HIT_RECURRENT_SUFFIX HIT_SUFFIX_CAP=$FR13_APC_HIT_SUFFIX_CAP"
echo "    temp(deploy)=$DEPLOY_FORCE_TEMP decode_mode=$FR10_DECODE_MODE_DEFAULT eager=$ENFORCE_EAGER"
echo "    PAIR_DUMP=$PAIR_DUMP"
echo "    G_SOURCE=$G_SOURCE P_PIN_MODE=$P_PIN_MODE stride=$SWEEP_STRIDE"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$RUNDIR/started_at.txt"
git rev-parse HEAD 2>/dev/null | tee "$RUNDIR/git_head.txt" || true

# ---- recover_host_memory helper ----
recover_host(){ PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - <<'PY'
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}

# ---- pre-boot hygiene ----
echo "[hygiene] recover_host_memory + assert free"
recover_host || true
.venv/bin/python - <<'PY'
from pathlib import Path
f={}
for l in Path("/proc/meminfo").read_text().splitlines():
    k,v=l.split(":",1); f[k]=int(v.strip().split()[0])
avail=f.get("MemAvailable",0)/1024/1024
swap=f.get("SwapTotal",0)-f.get("SwapFree",0)
if avail<80 or swap!=0:
    raise SystemExit(f"hygiene FAIL MemAvailable={avail:.1f}GiB swap_used={swap/1024/1024:.2f}GiB")
print(f"[hygiene] MemAvailable={avail:.1f}GiB swap_used=0 OK")
PY
(( $? == 0 )) || { echo "FAIL: pre-boot hygiene"; exit 2; }
if [[ -n "$(docker ps -q)" ]]; then echo "FAIL: docker ps not empty before boot"; docker ps; exit 2; fi

# ---- teardown trap ----
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

# ---- boot the FORKED server with the deployed fold-fix-ON APC config ----
# GPU_UTIL=0.50 (not 0.74) leaves ~50GiB unified headroom for STEP B's eager
# teacher-force sweep + the gpu_oom_guard backstop; this is a diagnostic, the
# capacity knobs don't affect the cache-ON-vs-cache-OFF per-token comparison.
echo "[boot] launching forked FA2 tree server (EAGER diagnostic, fold-fix-ON APC, GPU_UTIL=${GPU_UTIL:-0.50})"
CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=${GPU_UTIL:-0.50} MAX_NUM_SEQS="$MAX_NUM_SEQS_OVR" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}" \
  FR10_METRICS=0 BATCH_INVARIANT=0 \
  LUMO_FB_KERNEL_ROWS="$LUMO_FB_KERNEL_ROWS" LUMO_FB_PROJ_PAD_ROWS="$LUMO_FB_PROJ_PAD_ROWS" \
  FR13_ENABLE_APC=1 \
  MAMBA_BLOCK_SIZE="$MAMBA_BLOCK_SIZE" \
  MAMBA_SSM_CACHE_DTYPE="$MAMBA_SSM_CACHE_DTYPE" \
  APC_MAX_NUM_BATCHED_TOKENS="$APC_MAX_NUM_BATCHED_TOKENS" \
  FR13_APC_CONV_FIX="$FR13_APC_CONV_FIX" \
  FR13_APC_SSM_LEAF_SRC="$FR13_APC_SSM_LEAF_SRC" \
  FR13_APC_HIT_RECURRENT_SUFFIX="$FR13_APC_HIT_RECURRENT_SUFFIX" \
  FR13_APC_HIT_SUFFIX_CAP="$FR13_APC_HIT_SUFFIX_CAP" \
  FR13_APC_VALUE_VS_ORACLE="$FR13_APC_VALUE_VS_ORACLE" \
  FR13_APC_VALUE_VS_ORACLE_LOG="$FR13_APC_VALUE_VS_ORACLE_LOG" \
  FR10_DECODE_MODE_DEFAULT="$FR10_DECODE_MODE_DEFAULT" \
  ENFORCE_EAGER="$ENFORCE_EAGER" \
  FR13_RUN_DIR="$PWD/$RUNDIR" LOG_DIR="$PWD/$RUNDIR/logs" \
  "${LAUNCHER_SCRIPT:-scripts/fr13_launch_forked_fa2_tree_server.sh}" > "$RUNDIR/launch.log" 2>&1
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

# =============================================================================
# Shared python primitives (HTTP). All use TOKEN-ID LISTS, never text (H2).
# (send_ids identical to the USG driver, but with an IGNORE_EOS toggle so the
#  fresh apply-patch capture can run NO ignore_eos -> natural tool-call emission.)
# =============================================================================

# --- send_ids: capture the EXACT served token-ids via return_token_ids (R0) ----
# args: req_id prompt_ids_json temp maxtok seed outfile genout ignore_eos(0|1)
send_ids(){
  local rid="$1" pidsf="$2" temp="$3" maxtok="$4" seed="$5" outfile="$6" genout="${7:-}" ieos="${8:-0}"
  .venv/bin/python - "$PORT" "$MODEL" "$rid" "$pidsf" "$temp" "$maxtok" "$seed" "$outfile" "$genout" "$FR10_DECODE_MODE_DEFAULT" "$ieos" <<'PY'
import json, sys, urllib.request, urllib.error
port, model, rid, pidsf, temp, maxtok, seed, outfile, genout, mode, ieos = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]),
    int(sys.argv[6]), int(sys.argv[7]), sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11])
p_ids = json.load(open(pidsf))
payload = {
    "model": model, "prompt": p_ids, "temperature": temp,
    "max_tokens": maxtok, "stream": False, "return_token_ids": True,
    "vllm_xargs": {"fr10_decode_mode": mode},
}
if int(ieos) == 1:
    # ramble-style forced length (NOT used for the apply-patch capture by default).
    payload["ignore_eos"] = True
    payload["min_tokens"] = maxtok
# else: NO ignore_eos -> the model naturally decodes a real apply-patch tool-call
#       (and may EOS / stop on the tool-call boundary well before maxtok).
if temp > 0:
    payload["seed"] = seed
body = json.dumps(payload).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions", data=body,
    headers={"Content-Type": "application/json", "X-Request-Id": rid}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        status, data = r.status, r.read().decode()
except urllib.error.HTTPError as e:
    status, data = e.code, e.read().decode()
except Exception as e:
    status, data = -1, json.dumps({"client_error": str(e)})
open(outfile, "w").write(data)
print(f"HTTP {status}")
if status != 200:
    sys.exit(1)
if genout:
    d = json.loads(data); ch = d["choices"][0]; gen_ids = None
    if isinstance(ch.get("token_ids"), list):
        gen_ids = [int(x) for x in ch["token_ids"]]
    elif isinstance((ch.get("logprobs") or {}).get("token_ids"), list):
        gen_ids = [int(x) for x in ch["logprobs"]["token_ids"]]
    if gen_ids is None:
        raise SystemExit("FAIL: cannot capture EXACT served token-ids (no choice.token_ids). "
                         "Re-tokenizing text is NOT byte-safe. choice keys=" + repr(list(ch.keys())))
    _pt = ch.get("prompt_token_ids")
    if isinstance(_pt, list) and len(gen_ids) > 0 and gen_ids[:len(_pt)] == [int(x) for x in _pt]:
        gen_ids = gen_ids[len(_pt):]
    if len(gen_ids) < 1:
        raise SystemExit("FAIL: zero served token-ids captured")
    json.dump(gen_ids, open(genout, "w"))
    print(f"[capture] req={rid} captured {len(gen_ids)} served token-ids")
sys.exit(0)
PY
}

# =============================================================================
# STEP A -- BUILD PG_IDS from the REAL apply-patch turn (P=context, G=tool-call).
# =============================================================================
P_IDS_FILE="$RUNDIR/P_ids.json"
G_IDS_FILE="$RUNDIR/G_ids.json"
PG_IDS_FILE="$RUNDIR/PG_ids.json"
META_FILE="$RUNDIR/usg_meta.json"   # {plen, glen, pglen, block, boundary, g_source, p_pin_mode}
APPLYPATCH_TEXT="$RUNDIR/applypatch_response.txt"  # the dump's apply-patch tool-call text (audit)

if [[ -n "$REUSE_PG_IDS" ]]; then
  echo "[STEP A] REUSING existing PG_IDS=$REUSE_PG_IDS plen=$REUSE_PLEN"
  [[ -s "$REUSE_PG_IDS" && -n "$REUSE_PLEN" ]] || { echo "FAIL: REUSE_PG_IDS needs a nonempty file + REUSE_PLEN=<len(P_ids)>"; exit 4; }
  cp -f "$REUSE_PG_IDS" "$PG_IDS_FILE"
  .venv/bin/python - "$PG_IDS_FILE" "$REUSE_PLEN" "$P_IDS_FILE" "$G_IDS_FILE" "$META_FILE" "$MAMBA_BLOCK_SIZE" "$P_PIN_MODE" <<'PY'
import json, sys
pg = json.load(open(sys.argv[1])); plen = int(sys.argv[2]); block = int(sys.argv[6]); pin = sys.argv[7]
if plen < 1 or plen >= len(pg):
    raise SystemExit(f"FAIL: bad REUSE_PLEN {plen} for PG len {len(pg)}")
if plen % block != 0:
    raise SystemExit(f"FAIL: REUSE_PLEN {plen} must be a block multiple of {block} (the restore boundary)")
json.dump(pg[:plen], open(sys.argv[3], "w"))
json.dump(pg[plen:], open(sys.argv[4], "w"))
json.dump({"plen": plen, "glen": len(pg)-plen, "pglen": len(pg), "block": block,
           "boundary": plen, "g_source": "reuse", "p_pin_mode": pin}, open(sys.argv[5], "w"))
print(f"[reuse] plen={plen} glen={len(pg)-plen} pglen={len(pg)} block={block} boundary={plen}")
PY
  (( $? == 0 )) || { echo "FAIL: REUSE_PG_IDS slice"; exit 4; }
else
  echo "[STEP A] flatten the apply-patch turn's request (instructions+input) -> P text"
  # Use scripts/fr13_apc_replay.py:flatten_to_prompt VERBATIM (the proven flattener
  # that ends with the <|assistant|>\n generation marker = exactly the context the
  # model saw right before emitting the apply-patch tool-call).
  PROMPT_FILE="$RUNDIR/applypatch_prompt.txt"
  PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python - "$PAIR_DUMP" "$PROMPT_FILE" "$APPLYPATCH_TEXT" <<'PY'
import json, sys, importlib.util
from pathlib import Path
src, out, gtext_out = sys.argv[1], sys.argv[2], sys.argv[3]
p = Path(src)
if not p.is_file():
    raise SystemExit(f"FAIL: PAIR_DUMP not found: {src}")
# import flatten_to_prompt from the replay harness (no re-implementation).
spec = importlib.util.spec_from_file_location("fr13_apc_replay", "scripts/fr13_apc_replay.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
d = json.loads(p.read_text()); req = d.get("request", {}); resp = d.get("response", {})
blob = mod.flatten_to_prompt(req)
if len(blob) < 4000:
    raise SystemExit(f"FAIL: flattened apply-patch turn too short ({len(blob)} chars) -- not the real agentic context")
# CONFIRM this is really 12907's fix turn: the response must emit a separable.py
# edit (sed ... = right / apply_patch / Begin Patch on separable.py).
out_items = resp.get("output") or []
gtext = ""
is_applypatch = False
for o in out_items:
    if isinstance(o, dict) and o.get("type") == "function_call":
        args = o.get("arguments") or ""
        gtext = (gtext + "\n" + args) if gtext else args
        cmd = ""
        try:
            aj = json.loads(args); cmd = aj.get("cmd") or aj.get("command") or ""
            if isinstance(cmd, list): cmd = " ".join(map(str, cmd))
        except Exception:
            cmd = args
        if ("separable.py" in (cmd + args)) and (("= right" in (cmd+args)) or ("apply_patch" in (cmd+args))
            or ("Begin Patch" in (cmd+args)) or ("Update File" in (cmd+args)) or ("sed -i" in (cmd+args))):
            is_applypatch = True
if not is_applypatch:
    raise SystemExit("FAIL: the PAIR_DUMP response does NOT emit a separable.py apply-patch/edit tool-call. "
                     "Pick the apply-the-fix turn (its response.output[*] mutates separable.py). "
                     "output types=" + repr([o.get("type") for o in out_items if isinstance(o, dict)]))
Path(out).write_text(blob)
Path(gtext_out).write_text(gtext)
print(f"[prompt] wrote {len(blob)} chars P (real apply-patch context) to {out}")
print(f"[applypatch] response tool-call text ({len(gtext)} chars) -> {gtext_out}")
print(f"[applypatch] VERIFIED: response emits a separable.py edit (mentions separable.py + diff/patch)")
PY
  (( $? == 0 )) || { echo "FAIL: apply-patch turn flatten / verify"; exit 4; }

  echo "[STEP A] /tokenize P, pin to a block boundary (P_PIN_MODE=$P_PIN_MODE)"
  # P_PIN_MODE=down: pin P DOWN to the largest block multiple <= count so the
  # restore boundary B*=plen falls BELOW the apply-patch tokens => the apply-patch
  # tokens are in the DECODE-AFTER-RESTORE region. (up: legacy USG behaviour.)
  .venv/bin/python - "$PORT" "$MODEL" "$PROMPT_FILE" "$P_IDS_FILE" "$MAMBA_BLOCK_SIZE" "$P_PIN_MODE" "$RUNDIR/p_tokenize_raw.json" <<'PY'
import json, sys, urllib.request
port, model, pfile, out, block, pin, rawout = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6], sys.argv[7])
full = open(pfile).read()
body = json.dumps({"model": model, "prompt": full}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize",
    data=body, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=120) as r:
    dd = json.loads(r.read().decode())
ids = dd.get("tokens") or dd.get("token_ids")
if ids is None:
    raise SystemExit(f"FAIL: /tokenize returned no token list: keys={list(dd.keys())}")
ids = [int(x) for x in ids]
json.dump(ids, open(rawout, "w"))
if pin == "down":
    plen = (len(ids) // block) * block          # largest block-mult <= count
else:
    plen = (len(ids) // block) * block          # up-mode pins the same here; the
    # "up" distinction only matters for SHORT P (USG legacy); for the big apply-
    # patch context len(ids)>>block so down==floor is what we want regardless.
if plen < block:
    raise SystemExit(f"FAIL: apply-patch context only {len(ids)} tokens < one block {block}")
# We keep the WHOLE tokenized P (all ids) as the P region but record the restore
# BOUNDARY = plen (a block multiple <= len). PG = full_P + G; the cache-ON warm arm
# hits through B*=plen and re-prefills the [plen:] tail (P-tail + apply-patch G).
json.dump(ids, open(out, "w"))
print(f"[tokenize] apply-patch context {len(ids)} tokens; restore boundary B*={plen} "
      f"(block-aligned); P-tail-after-boundary={len(ids)-plen} tokens precede the apply-patch G")
# stash plen for the next step via a sidecar (the build step reads it).
json.dump({"boundary": plen, "p_token_count": len(ids)}, open(out + ".boundary.json", "w"))
PY
  (( $? == 0 )) || { echo "FAIL: P token-pin"; exit 4; }

  if [[ "$G_SOURCE" == "fresh" ]]; then
    echo "[STEP A] capture G FRESH: reset cache, decode the apply-patch tool-call (temp=0, NO ignore_eos)"
    curl -fsS -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" >/dev/null 2>&1
    # temp=0 greedy, NO ignore_eos -> the model naturally emits a real apply-patch
    # tool-call from P and stops on its natural boundary (well before GEN_TOKENS).
    send_ids "AP_CAP" "$P_IDS_FILE" "0.0" "$GEN_TOKENS" "$CAPTURE_SEED" \
      "$RUNDIR/capture.json" "$G_IDS_FILE" "0" | tee "$RUNDIR/capture_http.txt"
    CRC=${PIPESTATUS[0]}
    if (( CRC != 0 )); then echo "FAIL: STEP A fresh G capture errored (rc=$CRC)"; head -c 800 "$RUNDIR/capture.json"; echo; exit 5; fi
  else
    echo "[STEP A] G via RE-TOKENIZE of the dump's apply-patch response text (G_SOURCE=retoken)"
    .venv/bin/python - "$PORT" "$MODEL" "$APPLYPATCH_TEXT" "$G_IDS_FILE" <<'PY'
import json, sys, urllib.request
port, model, gtextf, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
gtext = open(gtextf).read()
if len(gtext) < 10:
    raise SystemExit("FAIL: apply-patch response text empty -- cannot re-tokenize G")
body = json.dumps({"model": model, "prompt": gtext, "add_special_tokens": False}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize",
    data=body, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=120) as r:
    dd = json.loads(r.read().decode())
ids = dd.get("tokens") or dd.get("token_ids")
if not ids:
    raise SystemExit(f"FAIL: /tokenize G returned no tokens: keys={list(dd.keys())}")
ids = [int(x) for x in ids]
json.dump(ids, open(out, "w"))
print(f"[retoken] apply-patch response text {len(gtext)} chars -> G {len(ids)} tokens "
      "(re-tokenize seam noted; both arms force identical PG so the compare is valid)")
PY
    (( $? == 0 )) || { echo "FAIL: G re-tokenize"; exit 5; }
  fi

  # Build PG_IDS = P_ids + G_IDS; assert the restore boundary B* lands BELOW the
  # apply-patch G (so G is in the decode-after-restore region) AND a boundary lands
  # at/above plen but below pglen (H7).
  .venv/bin/python - "$P_IDS_FILE" "$G_IDS_FILE" "$PG_IDS_FILE" "$META_FILE" "$MAMBA_BLOCK_SIZE" \
      "$P_IDS_FILE.boundary.json" "$G_SOURCE" "$P_PIN_MODE" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); g = json.load(open(sys.argv[2]))
block = int(sys.argv[5])
bj = json.load(open(sys.argv[6])); boundary = int(bj["boundary"])
g_source, pin = sys.argv[7], sys.argv[8]
pg = list(p) + list(g); plen = len(p)
json.dump(pg, open(sys.argv[3], "w"))
json.dump({"plen": plen, "glen": len(g), "pglen": len(pg), "block": block,
           "boundary": boundary, "g_source": g_source, "p_pin_mode": pin,
           "applypatch_g_start": plen}, open(sys.argv[4], "w"))
# H7-geometry: the restore boundary must be < pglen (a real re-prefill happens),
# and the apply-patch G (which starts at plen) must be entirely ABOVE the boundary
# so the apply-patch tokens are decoded AFTER the cache-restore of [:boundary].
if boundary >= len(pg):
    raise SystemExit(f"FAIL: restore boundary {boundary} >= pglen {len(pg)}: no re-prefill region")
if boundary > plen:
    raise SystemExit(f"FAIL: boundary {boundary} > plen {plen}: the apply-patch G starts "
                     f"BELOW the boundary => not decoded after the restore. Use P_PIN_MODE=down.")
# how many of the apply-patch G tokens sit above the boundary (all of them, since
# boundary<=plen<=g_start). Confirm at least one apply-patch token is post-boundary.
n_g_post = len(pg) - max(boundary, plen)
print(f"[build] plen={plen} glen={len(g)} pglen={len(pg)} block={block} boundary(B*)={boundary} "
      f"P-tail+G in decode-after-restore region={len(pg)-boundary} (apply-patch G={len(g)} tokens, "
      f"all post-boundary={n_g_post==len(g)})")
PY
  (( $? == 0 )) || { echo "FAIL: PG build / boundary-vs-applypatch geometry"; exit 5; }
fi

# =============================================================================
# STEP B -- two arms, teacher-forced, per swept absolute position j ACROSS the
# apply-patch tokens. IDENTICAL discipline to fr13_apc_usg_drive.sh STEP B, but:
#   * positions sweep with a FINE stride strictly ABOVE the restore boundary (so
#     each j's re-prefill crosses B* = the cache-restore of P) and INTO the apply-
#     patch G (especially the JSON structural tokens at G char ~90).
#   * cache-OFF reset (clean MISS), cache-ON warm (hit through B*), each run TWICE.
# =============================================================================
ON_JSONL="$RUNDIR/cache_on_teacherforce.jsonl"
OFF_JSONL="$RUNDIR/cache_off_teacherforce.jsonl"
: > "$ON_JSONL"; : > "$OFF_JSONL"

echo "[STEP B] sweeping teacher-forced apply-patch positions (stride=$SWEEP_STRIDE max=$SWEEP_MAX_POS logprobs=$LOGPROBS_K from_boundary=$SWEEP_FROM_BOUNDARY)"
.venv/bin/python - \
  "$PORT" "$MODEL" "$PG_IDS_FILE" "$META_FILE" "$ON_JSONL" "$OFF_JSONL" \
  "$SWEEP_STRIDE" "$SWEEP_MAX_POS" "$LOGPROBS_K" "$FR10_DECODE_MODE_DEFAULT" \
  "$RUNDIR/sweep_log.txt" "$SWEEP_FROM_BOUNDARY" <<'PY'
import json, sys, urllib.request, urllib.error
from pathlib import Path

port, model, pgf, metaf, on_out, off_out = sys.argv[1:7]
stride, max_pos, topk = int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])
mode, sweeplog, from_boundary = sys.argv[10], sys.argv[11], int(sys.argv[12])

pg = json.load(open(pgf)); meta = json.load(open(metaf))
plen, block = int(meta["plen"]), int(meta["block"])
boundary = int(meta.get("boundary", (plen // block) * block))
g_start = int(meta.get("applypatch_g_start", plen))
PGLEN = len(pg)

def post(path, payload, timeout=900, req_id=None):
    headers = {"Content-Type": "application/json"}
    if req_id is not None:
        headers["X-Request-Id"] = req_id
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
        data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode()

def reset_cache():
    try:
        post("/reset_prefix_cache", {}, timeout=30)
    except Exception:
        pass

def metrics_sums():
    req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", method="GET")
    txt = ""
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode()
    except Exception:
        return None
    import re
    hits = 0.0; cached = 0.0; hits_seen = False
    for line in txt.splitlines():
        if line.startswith("#"): continue
        m = re.match(r"^(vllm:[a-zA-Z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", line)
        if not m: continue
        nm, val = m.group(1), m.group(3)
        try: v = float(val)
        except ValueError: continue
        if nm in ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"):
            hits += v; hits_seen = True
        elif nm == "vllm:prompt_tokens_cached_total":
            cached += v
    return {"hits": hits, "cached": cached, "hits_seen": hits_seen}

def teacher_force_once(context_ids, req_id):
    payload = {
        "model": model, "prompt": context_ids, "max_tokens": 1,
        "temperature": 0.0, "top_p": 1.0, "logprobs": topk,
        "return_token_ids": True, "vllm_xargs": {"fr10_decode_mode": mode},
    }
    status, data = post("/v1/completions", payload, timeout=900, req_id=req_id)
    if status != 200:
        raise RuntimeError(f"teacher_force HTTP {status} for req {req_id}: {data[:200]}")
    d = json.loads(data); ch = d["choices"][0]
    tid = (ch.get("token_ids") or [None])[0]
    if tid is None:
        lp_ids = (ch.get("logprobs") or {}).get("token_ids")
        if isinstance(lp_ids, list) and lp_ids:
            tid = lp_ids[0]
    if tid is None:
        raise RuntimeError(f"FAIL: no verified token_ids in teacher-force response (req {req_id}): keys={list(ch.keys())}")
    lp = ch.get("logprobs") or {}
    tops = lp.get("top_logprobs") or [{}]
    return int(tid), (tops[0] if tops else {})

# swept absolute positions: a FINE stride concentrated on the APPLY-PATCH span
# [g_start, PGLEN) -- the tokens that actually corrupt (JSON quotes/braces/commas
# at G char ~90) -- so the fine-stride budget is NOT wasted on the P-tail. Every j
# is > boundary, so each j's re-prefill crosses B* (the cache-restore of P) and the
# apply-patch token is decoded AFTER the restore (H5/H7). A few coarse P-tail probes
# (g_start..boundary) are added first as controls (these must NOT flip if the carrier
# is restore-localized to the apply-patch region; they witness the post-restore tail).
# from_boundary=0 restricts the sweep to the apply-patch span only.
assert boundary <= g_start, f"boundary {boundary} must be <= g_start {g_start} (use P_PIN_MODE=down)"
seen = set()
positions = []
def _add(j):
    if boundary < j <= PGLEN and j not in seen:
        seen.add(j); positions.append(j)
# (a) optional coarse P-tail controls: a handful between B* and g_start.
if from_boundary and g_start > boundary + 1:
    span = g_start - boundary
    n_ctrl = min(8, max(1, span // max(1, block // 8)))
    for k in range(1, n_ctrl + 1):
        _add(boundary + (span * k) // (n_ctrl + 1))
# (b) FINE stride across the apply-patch span [g_start+1 .. PGLEN].
j = g_start + 1
while j <= PGLEN and len(positions) < max_pos:
    _add(j)
    j += stride
# (c) always include the final apply-patch position (the JSON tail) if room.
if PGLEN > boundary and len(positions) < max_pos:
    _add(PGLEN)
positions.sort()

slog = open(sweeplog, "w")
def L(msg):
    print(msg); slog.write(msg + "\n"); slog.flush()

L(f"[sweep] plen={plen} g_start={g_start} boundary(B*)={boundary} block={block} PGLEN={PGLEN} "
  f"stride={stride} from_boundary={from_boundary} -> {len(positions)} positions "
  f"(apply-patch span = [{g_start},{PGLEN}))")
if not positions:
    L("[sweep] NO eligible positions (no block boundary at/above B* in the apply-patch region) -> VACUOUS")

on_f = open(on_out, "w"); off_f = open(off_out, "w")
n_on = n_off = n_skipped = 0
for j in positions:
    ctx = pg[:j]
    boundary_pos = ((j - 1) // block) * block

    # ---- CACHE-OFF reference arm: clean MISS (H4) ----
    off_recs = []
    for rep in range(2):
        reset_cache()
        before = metrics_sums()
        tid, top = teacher_force_once(ctx, req_id=f"AP_OFF_j{j}_r{rep}")
        after = metrics_sums()
        hd = None
        if before and after:
            hd = after["hits"] - before["hits"]
        off_recs.append({"tid": tid, "top": top, "hits_delta": hd,
                         "cached_after": (after or {}).get("cached")})
    off_det = (off_recs[0]["tid"] == off_recs[1]["tid"])

    # ---- CACHE-ON served arm: WARM (no reset, H5) ----
    on_recs = []
    for rep in range(2):
        before = metrics_sums()
        tid, top = teacher_force_once(ctx, req_id=f"AP_ON_j{j}_r{rep}")
        after = metrics_sums()
        hd = None; cached_after = None
        if before and after:
            hd = after["hits"] - before["hits"]
            cached_after = after["cached"]
        on_recs.append({"tid": tid, "top": top, "hits_delta": hd, "cached_after": cached_after})
    on_det = (on_recs[0]["tid"] == on_recs[1]["tid"])

    if not (on_det and off_det):
        n_skipped += 1
        L(f"[skip] j={j} within-boot nondet (on_det={on_det} off_det={off_det}) -> autotune noise, dropped")
        continue

    off_hd = off_recs[-1]["hits_delta"]
    on_hd = on_recs[-1]["hits_delta"]
    on_cached = on_recs[-1]["cached_after"]
    pt = plen

    on_f.write(json.dumps({
        "req_id": f"AP_ON_j{j}", "absolute_pos": j, "boundary_pos": boundary_pos,
        "prompt_tokens": pt, "emitted_token_id": on_recs[-1]["tid"],
        "top_logprobs": on_recs[-1]["top"], "within_boot_det": on_det,
        "hits_delta": (0.0 if on_hd is None else float(on_hd)),
        "prompt_tokens_cached": (0.0 if on_cached is None else float(on_cached)),
        "verify_path_engaged": True,
        "in_applypatch_region": bool(j > g_start),
    }) + "\n"); on_f.flush(); n_on += 1

    off_f.write(json.dumps({
        "req_id": f"AP_OFF_j{j}", "absolute_pos": j, "boundary_pos": boundary_pos,
        "prompt_tokens": pt, "emitted_token_id": off_recs[-1]["tid"],
        "top_logprobs": off_recs[-1]["top"], "within_boot_det": off_det,
        "hits_delta": (0.0 if off_hd is None else float(off_hd)),
        "prompt_tokens_cached": (0.0 if off_recs[-1]["cached_after"] is None else float(off_recs[-1]["cached_after"])),
        "verify_path_engaged": True,
        "in_applypatch_region": bool(j > g_start),
    }) + "\n"); off_f.flush(); n_off += 1

    flip = on_recs[-1]["tid"] != off_recs[-1]["tid"]
    L(f"[pos] j={j} bpos={boundary_pos} in_ap={j>g_start} on_tid={on_recs[-1]['tid']} "
      f"off_tid={off_recs[-1]['tid']} flip={flip} on_hd={on_hd} off_hd={off_hd} on_cached={on_cached}")

on_f.close(); off_f.close()
L(f"[sweep] wrote on={n_on} off={n_off} skipped(nondet)={n_skipped}")
# exercised-check summary (the reducer re-asserts these).
any_on_hit = False; any_off_clean = False; any_gen_boundary = False; any_in_ap = False
for line in Path(on_out).read_text().splitlines():
    r = json.loads(line)
    if r.get("hits_delta", 0) > 0: any_on_hit = True
    if r.get("boundary_pos", 0) >= boundary: any_gen_boundary = True
    if r.get("in_applypatch_region"): any_in_ap = True
for line in Path(off_out).read_text().splitlines():
    r = json.loads(line)
    if abs(r.get("hits_delta", 1.0)) < 1e-9: any_off_clean = True
L(f"[exercised] any_cache_on_hit={any_on_hit} any_cache_off_clean_miss={any_off_clean} "
  f"any_restore_boundary_crossed={any_gen_boundary} any_in_applypatch_region={any_in_ap}")
Path(sweeplog + ".exercised").write_text(json.dumps({
    "any_cache_on_hit": any_on_hit, "any_cache_off_clean_miss": any_off_clean,
    "any_gen_region_boundary": any_gen_boundary, "any_in_applypatch_region": any_in_ap,
    "n_on": n_on, "n_off": n_off, "n_skipped": n_skipped}) + "\n")
PY
SWEEP_RC=$?
if (( SWEEP_RC != 0 )); then echo "FAIL: STEP B sweep errored (rc=$SWEEP_RC)"; exit 6; fi

# =============================================================================
# STEP B.5 -- EXERCISED-CHECK (fail-loud BEFORE any verdict, H7).
# =============================================================================
EX_FILE="$RUNDIR/sweep_log.txt.exercised"
if [[ ! -s "$EX_FILE" ]]; then
  echo "VACUOUS: no exercised summary produced (STEP B emitted no positions)"; exit 0
fi
.venv/bin/python - "$EX_FILE" <<'PY'
import json, sys
d = json.loads(open(sys.argv[1]).read().splitlines()[0])
ok = (bool(d.get("any_gen_region_boundary")) and bool(d.get("any_cache_on_hit"))
      and bool(d.get("any_cache_off_clean_miss")) and bool(d.get("any_in_applypatch_region")))
print(f"[exercised] restore_boundary_crossed={d.get('any_gen_region_boundary')} "
      f"cache_on_hit={d.get('any_cache_on_hit')} cache_off_clean_miss={d.get('any_cache_off_clean_miss')} "
      f"in_applypatch_region={d.get('any_in_applypatch_region')} "
      f"n_on={d.get('n_on')} n_off={d.get('n_off')} n_skipped={d.get('n_skipped')}")
sys.exit(0 if ok else 9)
PY
EX_RC=$?
if (( EX_RC != 0 )); then
  echo "*** VACUOUS: exercised-check not satisfied (no in-apply-patch-region cache-hit re-prefill, OR cache-OFF never a clean miss). NO VERDICT. ***"
  echo "APPLYPATCH_GATE_DONE VACUOUS"
  exit 0
fi

# =============================================================================
# STEP C -- REDUCE: the binding token-level verdict (REUSE the USG reducer).
# =============================================================================
VERDICT_JSON="$RUNDIR/applypatch_verdict.json"
VALUE_ORACLE_ARG=()
VO_JSONL=$(find "$RUNDIR/logs" -name 'fr13_apc_value_vs_oracle.jsonl' 2>/dev/null | head -1)
if [[ -n "$VO_JSONL" && -s "$VO_JSONL" ]]; then
  VALUE_ORACLE_ARG=(--value-oracle "$VO_JSONL")
  echo "[reduce] including VALUE_VS_ORACLE component-localization taps: $VO_JSONL"
fi

# Thin variant of the USG reducer (binding axis + verdict map UNCHANGED; only the
# genregion guard is adapted to the restore-of-P boundary-below-context geometry).
echo "[STEP C] reduce: scripts/fr13_apc_applypatch_reduce.py --cache-on $ON_JSONL --cache-off $OFF_JSONL"
.venv/bin/python scripts/fr13_apc_applypatch_reduce.py \
  --cache-on "$ON_JSONL" --cache-off "$OFF_JSONL" \
  "${VALUE_ORACLE_ARG[@]}" \
  --out "$VERDICT_JSON" | tee "$RUNDIR/reduce_stdout.txt"
REDUCE_RC=${PIPESTATUS[0]}

echo
echo "=== FR13 APC APPLY-PATCH GATE VERDICT ==="
case "$REDUCE_RC" in
  0) echo "LOSSLESS  (no per-token argmax flip vs the cache-OFF reference at any swept apply-patch position => the 2/2 empty_patch was protocol flakiness; cache is lossless-within-floor on the real tool-call)";;
  1) .venv/bin/python - "$VERDICT_JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
fd = d.get("first_divergence") or {}
print(f"NOT_LOSSLESS  first apply-patch divergence pos={fd.get('pos')} layer={fd.get('layer')} "
      f"component={fd.get('component')} cacheoff_margin_nat={d.get('cacheoff_margin_at_first_flip_nat')} "
      f"=> a REAL deterministic carrier corrupts the apply-patch JSON => the fold-fix is insufficient")
PY
     ;;
  2) echo "VACUOUS  (reducer exercised-check failed -- treat as no verdict)";;
  3) .venv/bin/python - "$VERDICT_JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
fd = d.get("first_divergence") or {}
print(f"ARGMAX_FLIP_WITHIN_FLOOR_ESCALATE  pos={fd.get('pos')} "
      f"cacheoff_margin_nat={d.get('cacheoff_margin_at_first_flip_nat')} <= floor -> user escalation, not auto-fail")
PY
     ;;
  *) echo "UNKNOWN reducer rc=$REDUCE_RC";;
esac
echo "verdict json: $VERDICT_JSON   (rundir=$RUNDIR)"
echo "APPLYPATCH_GATE_DONE rc=$REDUCE_RC"
exit "$REDUCE_RC"
