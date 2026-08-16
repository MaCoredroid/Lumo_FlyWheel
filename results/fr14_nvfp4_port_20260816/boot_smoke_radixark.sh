#!/bin/bash
# FR14 stock-boot smoke: first-ever RadixArk aggressive-NVFP4 serve on this box.
#
# Mirrors boot_smoke_nvfp4.sh (the proven unsloth idiom) with four deltas:
#   1. model dir / container name / served name are the RadixArk arm's;
#   2. the FR14 lm_head patch is applied INSIDE the container before `vllm serve`
#      (scripts/fr14_patch_nvfp4_lmhead.py -- see its docstring for the four gaps it
#      closes), with FR14_REQUIRE_NVFP4_LMHEAD=1 so a head that does not resolve
#      to ModelOptNvFp4LinearMethod kills the boot instead of serving silently;
#   3. the FR14 unified-memory preflight from
#      scripts/fr13_launch_forked_fa2_tree_server.sh:5750-5805 -- on GB10 the
#      "GPU" pool IS host RAM, so the engine demands
#      gpu_memory_utilization x MemTotal measured against genuinely FREE pages;
#      MemAvailable's reclaimable page cache does NOT count, which is exactly
#      what refused the first NVFP4 boot attempt;
#   4. the KV surgery is asserted, not assumed.
#
# The GPU must already be free. This script does not kill anything.
set -u
OUT=/home/mark/shared/tmp-scratch/nvfp4_port
RES_DIR=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816/results/fr14_nvfp4_port_20260816
# The lm_head patcher moved into scripts/ when it was promoted to the
# production boot flow; mount that instead so the smoke and the launcher
# exercise the SAME file rather than two copies that can drift.
SCRIPTS_DIR=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816/scripts
MODEL_DIR=/home/mark/shared/models/qwen3.8-27b-nvfp4-radixark
CTR_MODEL=/models/qwen3.8-27b-nvfp4-radixark
IMAGE=vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
PORT=8201
NAME=fr14_radixark_smoke
SERVED=qwen3.8-27b-nvfp4-radixark-smoke
GPU_UTIL=0.70
RES=$OUT/boot_smoke_radixark_result.json
LOG=$OUT/boot_smoke_radixark.log
CLOG=$OUT/boot_smoke_radixark_container.log
: > "$LOG"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

fail() { log "FAIL: $2"; echo "{\"verdict\":\"$1\",\"detail\":\"$2\"}" > "$RES"; exit 1; }

# ---------------------------------------------------------------- preflight 1
log "preflight: GPU must be free"
if [ -n "$(docker ps -q)" ]; then fail GPU_BUSY "containers still running: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
  fail GPU_BUSY "compute processes still on the GPU"
fi
log "GPU is free"

# ---------------------------------------------------------------- preflight 2
log "preflight: checkpoint completeness"
[ -f "$MODEL_DIR/model-00001-of-00003.safetensors" ] || fail CHECKPOINT_MISSING "shard 1 absent"
if find "$MODEL_DIR/.cache" -name '*.incomplete' 2>/dev/null | grep -q .; then
  fail CHECKPOINT_INCOMPLETE "*.incomplete files present"
fi
python3 - "$MODEL_DIR" <<'PY' || exit 1
import json, sys
M = sys.argv[1]
qc = json.load(open(M + "/config.json")).get("quantization_config", {})
bad = [k for k in ("kv_cache_scheme", "kv_cache_quant_algo") if k in qc]
if bad:
    raise SystemExit(
        "FR14 smoke aborted: KV surgery not applied -- config.json still "
        f"declares {bad}. Run results/fr14_nvfp4_port_20260816/"
        "radixark_kv_surgery.py first, or the engine will resolve "
        "cache_dtype=fp8_e4m3 and refuse TREE_ATTN."
    )
assert qc.get("quant_algo") == "MIXED_PRECISION", qc.get("quant_algo")
print("[smoke] KV surgery verified applied; quant_algo=MIXED_PRECISION")
PY

# ---------------------------------------------------------------- preflight 3
# FR14 page-cache drop + unified-memory gate. `sudo -n` is deliberately
# non-interactive so an unattended run never blocks on a password prompt; if the
# sudoers rule is absent the drop is skipped and the MemFree gate below still
# fires, loudly, before ~8 minutes of engine boot are spent.
log "preflight: dropping page cache and gating on MemFree"
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
avail = f.get("MemAvailable", 0) / 1024 / 1024
util = float(os.environ["GPU_UTIL"])
need = util * total
if free < need:
    raise SystemExit(
        f"FR14 smoke aborted: unified-memory preflight. Engine will demand "
        f"gpu_memory_utilization={util} x MemTotal={total:.2f}GiB = "
        f"{need:.2f}GiB, but only MemFree={free:.2f}GiB is free "
        f"(MemAvailable={avail:.2f}GiB includes reclaimable page cache, which "
        "the GB10 unified-memory allocator will NOT reclaim for you)."
    )
print(f"[smoke] unified-memory preflight OK: MemFree={free:.2f}GiB >= {need:.2f}GiB")
PY

# ---------------------------------------------------------------- boot
log "booting $NAME (eager, gpu-mem $GPU_UTIL, max-len 32768, NVFP4 lm_head patch, fail-closed)"
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --gpus all --network host \
  -v /models:/models \
  -v "$RES_DIR":/ovl:ro \
  -v "$SCRIPTS_DIR":/ovl_scripts:ro \
  -e FR14_REQUIRE_NVFP4_LMHEAD=1 \
  --entrypoint bash $IMAGE -c "
    set -e
    python3 /ovl_scripts/fr14_patch_nvfp4_lmhead.py
    exec vllm serve $CTR_MODEL \
      --served-model-name $SERVED \
      --host 127.0.0.1 --port $PORT \
      --max-model-len 32768 --max-num-seqs 4 \
      --gpu-memory-utilization $GPU_UTIL \
      --enforce-eager
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
  python3 - "$RES" "$CLOG" <<'PY'
import json, re, sys
log = open(sys.argv[2], errors="replace").read()
out = {"verdict": "BOOT_FAILED_OR_TIMEOUT", "container_log": sys.argv[2]}
# Surface the parts that decide the next move.
out["lmhead_route_lines"] = re.findall(r".*FR14_LMHEAD.*", log)[:6]
tb = log.rfind("Traceback (most recent call last)")
if tb != -1:
    out["traceback_tail"] = log[tb:tb + 4000]
for pat in ("no parameter named", "KeyError", "AssertionError", "out of memory",
            "is not valid for this configuration", "FR14_REQUIRE_NVFP4_LMHEAD"):
    if pat in log:
        out.setdefault("signals", []).append(pat)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "traceback_tail"}, indent=1))
PY
  docker rm -f $NAME >/dev/null 2>&1
  exit 1
fi
log "health OK, sending smoke requests"

R1=$(curl -s -m 420 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "{
  \"model\":\"$SERVED\",
  \"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: FR14 RadixArk NVFP4 alive. Then state 17*23.\"}],
  \"max_tokens\":512,\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20}")
R2=$(curl -s -m 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "{
  \"model\":\"$SERVED\",
  \"messages\":[{\"role\":\"user\",\"content\":\"Write a python function that reverses a linked list, then explain its invariant in two sentences.\"}],
  \"max_tokens\":4096,\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20}")
echo "$R1" > $OUT/smoke_radixark_r1.json
echo "$R2" > $OUT/smoke_radixark_r2.json
docker logs $NAME > "$CLOG" 2>&1
docker rm -f $NAME >/dev/null 2>&1
log "teardown complete"

python3 - "$RES" "$CLOG" <<'PY'
import json, re, sys
out = {"verdict": "UNKNOWN"}
O = "/home/mark/shared/tmp-scratch/nvfp4_port"
log = open(sys.argv[2], errors="replace").read()
# The fail-closed banner is the proof the head really went through NVFP4.
out["lmhead_route_lines"] = re.findall(r".*FR14_LMHEAD.*", log)[:6]
out["lmhead_nvfp4_confirmed"] = any(
    "quant_method=ModelOptNvFp4LinearMethod" in ln for ln in out["lmhead_route_lines"]
)
m = re.search(r"KV cache size: ([\d,]+) tokens", log)
if m:
    out["kv_cache_tokens"] = m.group(1)
out["kv_cache_dtype_fp8_in_log"] = "fp8_e4m3" in log
try:
    r1 = json.load(open(O + "/smoke_radixark_r1.json"))
    r2 = json.load(open(O + "/smoke_radixark_r2.json"))
    c1 = r1["choices"][0]["message"]
    c2 = r2["choices"][0]["message"]
    t1 = (c1.get("content") or "") + (c1.get("reasoning_content") or "")
    t2 = (c2.get("content") or "") + (c2.get("reasoning_content") or "")
    out["r1_len"], out["r2_len"] = len(t1), len(t2)
    out["r1_tail"] = t1[-300:]
    out["r2_head"] = t2[:300]
    out["r1_has_391"] = "391" in t1
    words = t2.split()
    out["r2_degenerate"] = len(words) > 40 and len(set(words)) < len(words) * 0.25
    out["usage_r1"] = r1.get("usage")
    out["usage_r2"] = r2.get("usage")
    ok = (
        out["r1_len"] > 0
        and out["r2_len"] > 0
        and not out["r2_degenerate"]
        and out["lmhead_nvfp4_confirmed"]
    )
    out["verdict"] = "PASS" if ok else "SUSPECT_OUTPUT"
except Exception as e:
    out["verdict"] = "SMOKE_REQUEST_FAILED"
    out["error"] = f"{type(e).__name__}: {e}"
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
PY
log "verdict written to $RES"
