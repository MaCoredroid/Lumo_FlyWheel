#!/usr/bin/env bash
# Attribution-only fixed32 profile on canonical real SWE-Verified B1 traffic.
# Nsight terminates the wrapped server when the bounded capture ends, so this
# script must never be used as acceptance evidence.
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
REPO=$(cd "$REPO" && pwd)
cd "$REPO"

STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
TAG=${TAG:-b1_nsys_f32_${STAMP}}
RUNROOT=${RUNROOT:-output/fr13_fixed32_b1_nsys_${STAMP}}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
ARM=tail6_fixed32_${TAG}
REPORT="$RUNROOT/$ARM/logs/fr13_fixed32_b1_real_swe.nsys-rep"
REDUCED="$RUNROOT/$ARM/logs/fr13_fixed32_b1_nsys_attribution.json"

if [[ -n "$(docker ps -q)" ]]; then
  echo "FAIL: GPU campaign containers are already running" >&2
  exit 2
fi

cleanup() {
  docker ps -q --filter name=fr13-bigdenom \
    | xargs -r docker rm -f >/dev/null 2>&1 || true
}
trap cleanup EXIT

export LUMO_SWE_AUTOCOMMIT=0
export FR13_FIXED32_ATTRIBUTION_ONLY=1
export FR13_FIXED32_NVTX_PROFILE=1
export LUMO_NSYS_WRAP_VLLM=1
export LUMO_NSYS_BIN=${LUMO_NSYS_BIN:-/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys}
export LUMO_NSYS_TRACE=${LUMO_NSYS_TRACE:-cuda,cuda-sw,nvtx}
export LUMO_NSYS_DELAY_S=${LUMO_NSYS_DELAY_S:-1200}
export LUMO_NSYS_DURATION_S=${LUMO_NSYS_DURATION_S:-300}
export LUMO_NSYS_FLUSH_MS=${LUMO_NSYS_FLUSH_MS:-100}
export LUMO_NSYS_CONFIG_DIRECTIVES="${LUMO_NSYS_CONFIG_DIRECTIVES:-CuptiUseRawGpuTimestamps=false}"
export LUMO_NSYS_OUTPUT=/logs/fr13_fixed32_b1_real_swe

OUTPUT_ROOT=$(realpath -m "$REPO/output")
RUNROOT_ABS=$(realpath -m "$RUNROOT")
case "$RUNROOT_ABS" in
  "$OUTPUT_ROOT"/*) ;;
  *)
    echo "FAIL: raw profiler artifacts must remain below ignored output/" >&2
    exit 2
    ;;
esac
if [[ -e "$RUNROOT_ABS" ]]; then
  echo "FAIL: profiler RUNROOT must be new (stale evidence is forbidden)" >&2
  exit 2
fi
git check-ignore -q "$RUNROOT_ABS" || {
  echo "FAIL: raw profiler RUNROOT is not ignored by Git" >&2
  exit 2
}
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] || {
  echo "FAIL: canonical exact4 SWE-Verified subset hash drift" >&2
  exit 2
}
[[ "$LUMO_NSYS_BIN" == "/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys" ]] || {
  echo "FAIL: fixed32 attribution requires the pinned Nsight Systems binary" >&2
  exit 2
}
[[ -x "$LUMO_NSYS_BIN" ]] || {
  echo "FAIL: Nsight Systems executable is unavailable" >&2
  exit 2
}
[[ "$LUMO_NSYS_TRACE" == "cuda,cuda-sw,nvtx" ]] || {
  echo "FAIL: fixed32 attribution requires cuda,cuda-sw,nvtx tracing" >&2
  exit 2
}
for _positive_nsys_value in \
  "$LUMO_NSYS_DELAY_S" "$LUMO_NSYS_DURATION_S" "$LUMO_NSYS_FLUSH_MS"; do
  [[ "$_positive_nsys_value" =~ ^[1-9][0-9]*$ ]] || {
    echo "FAIL: Nsight delay, duration, and flush interval must be positive integers" >&2
    exit 2
  }
done
unset _positive_nsys_value
[[ "$LUMO_NSYS_DELAY_S" == "1200" && "$LUMO_NSYS_DURATION_S" == "300" ]] || {
  echo "FAIL: attribution requires the canonical 1200s delay/300s capture" >&2
  exit 2
}
[[ "$LUMO_NSYS_FLUSH_MS" == "100" ]] || {
  echo "FAIL: attribution requires a 100ms CUDA flush interval" >&2
  exit 2
}
[[ "$LUMO_NSYS_CONFIG_DIRECTIVES" == "CuptiUseRawGpuTimestamps=false" ]] || {
  echo "FAIL: fixed32 attribution Nsight config directive drift" >&2
  exit 2
}
mkdir -m 700 "$RUNROOT_ABS"
set +e
RUNROOT="$RUNROOT" \
TAG="$TAG" \
SUBSET="$SUBSET" \
BSIZE=1 \
CONC=1 \
WALL=0 \
DEPLOY_FORCE_TEMP=0.6 \
SEQUENCE_FILE=scripts/fr13_fixed32_floor_timers_seq.sh \
  bash scripts/fr13_b4_campaign_driver.sh \
  >"$RUNROOT/driver.log" 2>&1
driver_rc=$?
set -e

if [[ ! -s "$REPORT" ]]; then
  echo "FAIL: bounded real-SWE Nsight report was not produced" >&2
  exit 3
fi

if ! find "$RUNROOT/$ARM/swe_out/verified/per_task" -mindepth 1 -maxdepth 1 \
  -type d -name 'astropy__astropy-*' -print -quit 2>/dev/null \
  | grep -q .; then
  echo "FAIL: profiler window has no real SWE-Verified task evidence" >&2
  exit 4
fi

if ! .venv/bin/python scripts/fr13_fixed32_nsys_reduce.py \
  "$REPORT" \
  --output "$REDUCED" \
  --nsys-bin "$LUMO_NSYS_BIN" \
  --subset "$SUBSET" \
  --runtime-manifest-launch "$RUNROOT/runtime_manifest.at_launch.json" \
  --runtime-manifest-end "$RUNROOT/runtime_manifest.at_end.json" \
  --external-manifest-launch "$RUNROOT/external_manifest.at_launch.json" \
  --external-manifest-end "$RUNROOT/external_manifest.at_end.json" \
  --process-identity "$RUNROOT/$ARM/fixed32_process_identity.json" \
  --container-identity "$RUNROOT/$ARM/fixed32_container_identity.json" \
  --runtime-attestation \
    "$RUNROOT/$ARM/logs/fr13_fixed32_runtime_attestation.json" \
  --pretask-zero-traffic "$RUNROOT/$ARM/fixed32_pretask_zero_traffic.json" \
  --proxy-ledger "$RUNROOT/$ARM/logs/fr13_fixed32_proxy_ingress.jsonl" \
  --engine-ledger "$RUNROOT/$ARM/logs/fr13_fixed32_engine_ingress.jsonl" \
  --mode tail6_fixed32 \
  --batch-size 1 \
  --concurrency 1 \
  --driver-rc "$driver_rc" \
  --nsys-delay-s "$LUMO_NSYS_DELAY_S" \
  --nsys-duration-s "$LUMO_NSYS_DURATION_S" \
  --nsys-flush-ms "$LUMO_NSYS_FLUSH_MS" \
  --nsys-trace "$LUMO_NSYS_TRACE" \
  --nsys-config-directives "$LUMO_NSYS_CONFIG_DIRECTIVES" \
  --nsys-discard-environment true; then
  echo "FAIL: privacy-safe Nsight attribution reduction failed" >&2
  exit 5
fi

printf '%s\n' \
  "attribution_only=true" \
  "acceptance_valid=false" \
  "driver_rc=$driver_rc" \
  "report_bytes=$(stat -c %s "$REPORT")" \
  "reduced_sha256=$(sha256sum "$REDUCED" | awk '{print $1}')"
