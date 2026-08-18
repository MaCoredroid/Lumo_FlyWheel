#!/bin/bash
# FR14 lane 5A: NVFP4-verifier-head generation probe + hidden-state capture.
#
# Boots the RadixArk aggressive-NVFP4 checkpoint with the NVFP4 lm_head live
# and fail-closed (scripts/fr14_patch_nvfp4_lmhead.py, FR14_REQUIRE_NVFP4_LMHEAD=1),
# adds the lane-5A capture patch, and drives SWE-flavoured prompts through it in
# BOTH sampling regimes.  Two outputs:
#
#   1. Verbatim generation traces, for the degeneration eyeball.  Deliberately
#      long max_tokens: a repetition loop needs room to appear, and a probe that
#      stops at 256 tokens cannot see one.
#   2. Real pre-lm_head hidden states + the device kernel's own argmax, for the
#      offline logit characterisation (nvfp4_lmhead_characterization.py --phase
#      logits).
#
# Mirrors boot_smoke_radixark.sh's preflights (GPU-free, checkpoint complete,
# KV surgery applied, unified-memory gate) because those are the four things
# that have actually refused a boot on this box.  Tears the container down
# unconditionally on exit -- GPU discipline: zero containers left.
#
# Usage: bash fr14_lane5a_generation_probe.sh
set -u

REPO=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
RES_DIR=$REPO/results/fr14_nvfp4_port_20260816
SCRIPTS_DIR=$REPO/scripts
SUFFIX=${SUFFIX:-}
OUT=/home/mark/shared/tmp-scratch/fr14_lane5a${SUFFIX}
MODEL_DIR=/home/mark/shared/models/qwen3.8-27b-nvfp4-radixark
CTR_MODEL=/models/qwen3.8-27b-nvfp4-radixark
IMAGE=vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
PORT=8207
NAME=fr14_lane5a_probe
SERVED=qwen3.8-27b-nvfp4-radixark-lane5a
GPU_UTIL=0.70
CAP_ROWS=${CAP_ROWS:-8192}
# Parser flags default to the PRODUCTION serve line's
# (fr13_launch_forked_fa2_tree_server.sh:7168), so a tool call this probe
# rejects is a tool call the real serve would also reject.
SERVE_EXTRA=${SERVE_EXTRA:---enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3}
PROMPT_SET=${PROMPT_SET:-}

mkdir -p "$OUT"
LOG=$OUT/probe.log
CLOG=$OUT/container.log
: > "$LOG"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
cleanup() { docker rm -f $NAME >/dev/null 2>&1; }
trap cleanup EXIT

# ---------------------------------------------------------------- preflights
log "preflight: GPU must be free"
if [ -n "$(docker ps -q)" ]; then
  log "FAIL: containers still running: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
  log "FAIL: compute processes still on the GPU"; exit 1
fi

log "preflight: checkpoint + KV surgery"
python3 - "$MODEL_DIR" <<'PY' || exit 1
import json, sys
qc = json.load(open(sys.argv[1] + "/config.json")).get("quantization_config", {})
bad = [k for k in ("kv_cache_scheme", "kv_cache_quant_algo") if k in qc]
if bad:
    raise SystemExit(f"KV surgery not applied -- config.json still declares {bad}")
assert qc.get("quant_algo") == "MIXED_PRECISION", qc.get("quant_algo")
print("[probe] KV surgery verified; quant_algo=MIXED_PRECISION")
PY

log "preflight: unified-memory gate (GB10: the GPU pool IS host RAM)"
sync
sudo -n sysctl vm.drop_caches=3 >/dev/null 2>&1 || log "  (drop_caches skipped -- no sudoers rule)"
free -h | tee -a "$LOG"
GPU_UTIL="$GPU_UTIL" python3 - <<'PY' || exit 1
import os
from pathlib import Path
f = {}
for line in Path("/proc/meminfo").read_text().splitlines():
    k, v = line.split(":", 1)
    f[k] = int(v.strip().split()[0])
total = f["MemTotal"] / 1024 / 1024
free = f["MemFree"] / 1024 / 1024
util = float(os.environ["GPU_UTIL"])
need = util * total
if free < need:
    raise SystemExit(
        f"unified-memory preflight refused: engine demands {need:.2f}GiB "
        f"(= {util} x MemTotal {total:.2f}GiB) but MemFree is {free:.2f}GiB. "
        "MemAvailable's reclaimable page cache does NOT count on GB10."
    )
print(f"[probe] unified-memory preflight OK: MemFree={free:.2f}GiB >= {need:.2f}GiB")
PY

# ---------------------------------------------------------------- boot
log "booting $NAME (NVFP4 lm_head, fail-closed; lane5A capture on, max_rows=$CAP_ROWS)"
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --gpus all --network host \
  -v /home/mark/shared/models:/models \
  -v "$RES_DIR":/ovl:ro \
  -v "$SCRIPTS_DIR":/ovl_scripts:ro \
  -v "$OUT":/cap \
  -e FR14_REQUIRE_NVFP4_LMHEAD=1 \
  -e FR14_LANE5A_CAPTURE=/cap/hidden.f32 \
  -e FR14_LANE5A_CAPTURE_MAX_ROWS=$CAP_ROWS \
  --entrypoint bash $IMAGE -c "
    set -e
    python3 /ovl_scripts/fr14_patch_nvfp4_lmhead.py
    python3 /ovl/fr14_lane5a_capture_patch.py
    exec vllm serve $CTR_MODEL \
      --served-model-name $SERVED \
      --host 127.0.0.1 --port $PORT \
      --max-model-len 32768 --max-num-seqs 4 \
      --gpu-memory-utilization $GPU_UTIL \
      --enforce-eager $SERVE_EXTRA
  " >> "$LOG" 2>&1

BOOT_OK=0
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { BOOT_OK=1; break; }
  docker ps -q -f name=$NAME | grep -q . || break
  sleep 10
done
docker logs $NAME > "$CLOG" 2>&1
if [ $BOOT_OK -ne 1 ]; then
  log "BOOT FAILED -- see $CLOG"
  grep -nE "FR14_LMHEAD|FR14_LANE5A|Traceback|Error" "$CLOG" | tail -40 | tee -a "$LOG"
  exit 1
fi
log "health OK; head routing:"
grep -E "FR14_LMHEAD" "$CLOG" | tee -a "$LOG"

# ---------------------------------------------------------------- prompts
# The bodies live in a JSON file so the traces are reproducible from the tree.
python3 "$RES_DIR/fr14_lane5a_prompts.py" $PROMPT_SET > "$OUT/prompts.json"

log "driving prompts"
PORT=$PORT SERVED=$SERVED OUT=$OUT python3 - <<'PY' 2>&1 | tee -a "$LOG"
import json, os, time, urllib.request

PORT = os.environ["PORT"]; SERVED = os.environ["SERVED"]; OUT = os.environ["OUT"]
prompts = json.load(open(OUT + "/prompts.json"))
results = []
for p in prompts:
    body = {
        "model": SERVED,
        "messages": p["messages"],
        "max_tokens": p["max_tokens"],
        "seed": 20260818,
    }
    body.update(p["sampling"])
    if "tools" in p:
        body["tools"] = p["tools"]
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.loads(r.read().decode())
        err = None
    except Exception as e:
        out, err = None, f"{type(e).__name__}: {e}"
    dt = time.time() - t0
    print(f"  [{p['id']}] {p['regime']:8s} {dt:7.1f}s "
          f"{'ERR ' + err if err else str(out['usage']['completion_tokens']) + ' tok'}")
    results.append({"prompt": p, "response": out, "error": err, "wall_s": dt})
json.dump(results, open(OUT + "/generations.json", "w"), indent=1)
PY

docker logs $NAME > "$CLOG" 2>&1
cp -f "$OUT/hidden.f32.meta.json" "$RES_DIR/lane5a_capture_meta${SUFFIX}.json" 2>/dev/null
log "capture meta:"; cat "$OUT/hidden.f32.meta.json" 2>/dev/null | tee -a "$LOG"
cleanup
log "teardown complete; containers now: $(docker ps -aq | wc -l)"
log "generations -> $OUT/generations.json"
log "hidden       -> $OUT/hidden.f32 (+ .top.bin, .meta.json)"
