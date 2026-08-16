#!/usr/bin/env bash
# FR14 arm B boot-refusal probe: the fixed32 stack on the RadixArk checkpoint,
# killed as soon as the two things this train changed have proven themselves.
#
# It MIRRORS THE RUNNER exactly -- same campaign driver, same sequence file,
# same B=1 / CONC=1 / ROOT=1 / empty-selector shape as the stock B1 SWE arm --
# rather than being a bespoke boot script, because the failure modes worth
# finding are the ones the real serve line produces. The only differences are
# TAG/RUNROOT and that it is killed instead of run to completion.
#
# WHAT IT IS LOOKING FOR, in order:
#   1. FR14_LMHEAD_NVFP4 / FR14_LMHEAD_QUANT_ROUTE  -- the lm_head loader patch
#      routed the head through ModelOptNvFp4LinearMethod under the BAKED
#      fail-closed enforcement (the mode this train changed; the previous boot
#      smoke exercised the os.environ-gated one).
#   2. [FR13_DRAFT_VOCAB] shim built                -- the K64 row slice walked
#      a quantized head without dying.
#   3. [FR14_DVK_DEQUANT] phase1 ...                -- THE NEW RISK. First time
#      the chunked NVFP4->BF16 dequant runs on a device. The banner carries the
#      realised byte count, which must be 671088640, plus the widths.
#   4. kv_cache_dtype=auto and no fp8_e4m3          -- the KV surgery holding
#      under the fixed32 serve line, not just under the smoke's.
#
# Usage:  bash radixark_boot_probe.sh [timeout_s]
set -uo pipefail

REPO=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
cd "$REPO"

TIMEOUT_S=${1:-1200}
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT=output/fr14_b1_probe_$TS
RES=results/fr14_nvfp4_port_20260816
VERDICT="$RES/radixark_boot_probe_$TS.json"
SNAP="$RES/radixark_boot_probe_$TS.log"
mkdir -p "$RUNROOT"

echo "[probe] runroot=$RUNROOT timeout=${TIMEOUT_S}s"

# Same idiom as the stock B1 arm; TAG/RUNROOT are the only deliberate deltas.
BSIZE=1 CONC=1 TAG=b1radixprobe RUNROOT="$RUNROOT" WALL=5400 \
  SUBSET=config/fr13_fixed32/subset_b4_four.json \
  SEQUENCE_FILE=scripts/fr13_fixed32_floor_timers_seq.sh \
  FR13_DRAFT_VOCAB_ROOT=1 \
  TMPDIR=/home/mark/shared/tmp-scratch \
  bash scripts/fr13_b4_campaign_driver.sh > "$RUNROOT/driver.log" 2>&1 &
DRIVER_PID=$!
echo "[probe] driver pid=$DRIVER_PID"

CONTAINER=fr13-bigdenom-tail6_fixed32_b1radixprobe
DEADLINE=$(( $(date +%s) + TIMEOUT_S ))
REASON=timeout
while (( $(date +%s) < DEADLINE )); do
  if ! kill -0 "$DRIVER_PID" 2>/dev/null; then REASON=driver_exited; break; fi
  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    LOG=$(docker logs "$CONTAINER" 2>&1 || true)
    if grep -q "FR14_DVK_DEQUANT] phase1" <<<"$LOG"; then REASON=dvk_banner; break; fi
    if grep -qE "FR14_REQUIRE_NVFP4_LMHEAD=1 \(baked|FR14 DVK dequant" <<<"$LOG"; then
      REASON=fr14_refusal; break
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
      REASON=container_exited; break
    fi
  fi
  sleep 10
done
echo "[probe] stop reason=$REASON"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  docker logs "$CONTAINER" > "$SNAP" 2>&1 || true
else
  : > "$SNAP"
fi

kill -TERM "$DRIVER_PID" 2>/dev/null
sleep 5
pkill -TERM -f "fr13_bigdenom_swe_serve_variant.sh .*b1radixprobe" 2>/dev/null
sleep 5
docker rm -f "$CONTAINER" >/dev/null 2>&1
kill -KILL "$DRIVER_PID" 2>/dev/null

TMPDIR=/home/mark/shared/tmp-scratch python3 - "$SNAP" "$REASON" "$RUNROOT" "$VERDICT" <<'PY'
import json
import re
import sys
from pathlib import Path

snapshot, reason, runroot, out = sys.argv[1:5]
log = Path(snapshot).read_text(errors="replace")


def grab(pattern, limit=4):
    return [line.strip() for line in log.splitlines() if pattern in line][:limit]


dvk = grab("FR14_DVK_DEQUANT")
bytes_ok = None
widths_ok = None
method_ok = None
if dvk:
    m = re.search(r"bytes=(\d+)", dvk[0])
    bytes_ok = (int(m.group(1)) == 671_088_640) if m else False
    widths_ok = (
        "logical_widths=[65536]" in dvk[0]
        and "output_size_per_partition=65536" in dvk[0]
    )
    method_ok = "quant_method=UnquantizedEmbeddingMethod" in dvk[0]

route = grab("FR14_LMHEAD_QUANT_ROUTE")
report = {
    "schema": "fr14.radixark_boot_probe.v1",
    "stop_reason": reason,
    "runroot": runroot,
    "lmhead_nvfp4_lines": grab("FR14_LMHEAD_NVFP4"),
    "lmhead_route_lines": route,
    "lmhead_routed_nvfp4": any(
        "ModelOptNvFp4LinearMethod" in line for line in route
    ),
    "dvk_shim_lines": grab("[FR13_DRAFT_VOCAB]"),
    "dvk_root_engaged_lines": grab("[FR13_DRAFT_VOCAB_ROOT] engaged"),
    "dvk_dequant_lines": dvk,
    "dvk_dequant_bytes_is_671088640": bytes_ok,
    "dvk_dequant_widths_are_k64": widths_ok,
    "dvk_dequant_method_is_unquantized": method_ok,
    "kv_cache_dtype_lines": grab("kv_cache_dtype", limit=3),
    "fp8_e4m3_present": "fp8_e4m3" in log,
    "gpu_kv_cache_lines": grab("GPU KV cache size", limit=2),
    "tracebacks": grab("Traceback (most recent call last)", limit=3),
    "fr14_refusals": [
        line.strip()
        for line in log.splitlines()
        if "FR14 DVK" in line or "FR14_REQUIRE_NVFP4_LMHEAD=1 (baked" in line
    ][:5],
    "log_lines": len(log.splitlines()),
}
report["verdict"] = (
    "PASS"
    if (
        report["lmhead_routed_nvfp4"]
        and bytes_ok
        and widths_ok
        and method_ok
        and not report["fp8_e4m3_present"]
        and not report["fr14_refusals"]
    )
    else "FAIL"
)
Path(out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
print(json.dumps(report, indent=1))
PY

echo "[probe] verdict -> $VERDICT"
echo "[probe] log     -> $SNAP"
