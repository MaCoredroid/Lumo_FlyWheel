#!/usr/bin/env bash
# FR13 B4 WIDTH-4 NSYS ATTRIBUTION -- DIAGNOSTIC, NOT CITABLE, DEFAULT-OFF.
#
# WHAT THIS IS
#   One pool16 arm (16-task pool, 4 slots) served under Nsight Systems with a
#   STEP-GATED bounded capture taken strictly inside the width-4 depth window.
#   It exists to decompose the 387.6/389.6 ms width-4 step wall into kernels and
#   to price the step-wall levers against a measured table.
#
# WHAT THIS IS NOT
#   Not acceptance evidence.  Not a run class.  Not citable, and it never will
#   be.  CUPTI is attached for the WHOLE arm lifetime (the server is launched
#   under `nsys profile`), so every timing this arm posts -- inside and outside
#   the collection window -- is profiler-perturbed.  The reconciliation target
#   is therefore the SEALED, UNPROFILED split published in
#   results/fr13_b4_refill_citable_20260812/fr13_b4_width4_operating_point.json,
#   never this arm's own numbers.  See the B1 precedent: attribution artifacts
#   carry acceptance_valid=false and "must not be compared as a regression
#   against an unprofiled wall point".
#
# WHY IT DOES NOT REUSE scripts/fr13_fixed32_b1_nsys_profile.sh
#   That script is bound to B1: it hard-pins the exact4 4-task subset
#   (subset_b4_four.json), BSIZE=1/CONC=1, the qrow16 production .so, and a
#   canonical WALL-TIME capture (`--delay 1200 --duration 300`, enforced by an
#   equality check).  Every one of those is wrong for the width-4 operating
#   point, and the wall-time gate is specifically forbidden by
#   width4_window.md §6.  Rather than fork its 1767 lines of lifecycle
#   attestation -- which exists to make a citable-adjacent artifact safe, a
#   property this diagnostic does not claim -- this script mirrors the POOL16
#   GATE's env exactly (scripts/fr13_b4_pool16_refill_gate.sh:199-213) and adds
#   the profiler.  Mirroring the pool16 gate is what makes the capture
#   reconcilable against the sealed width-4 split.
#
# THE STEP GATE
#   The launcher is put into deferred collection via LUMO_NSYS_START_LATER=1
#   (default-off; with it unset the B1 delay/duration prefix is byte-identical).
#   scripts/fr13_b4_width4_nsys_stepgate.py then opens and closes the session
#   against vllm:fr13_decode_forward_gpu_steps_total -- the absolute forward-step
#   counter that indexes the work census -- and refuses to arm until trailing
#   events/step proves the engine is at width, which excludes the ~118-step
#   hydration ramp that a wall-time or depth-based gate would sample.
#
# USAGE (must be detached; the arm runs for hours)
#   setsid nohup bash scripts/fr13_b4_width4_nsys_profile.sh \
#     > /home/mark/shared/b4width4nsys.log 2>&1 < /dev/null &
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$STAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] \
  || { echo "FAIL: STAMP must be a UTC basic timestamp" >&2; exit 2; }

TAG=${TAG:-w4nsys}
RUNROOT=${RUNROOT:-output/fr13_b4_width4_nsys_${STAMP}}
ARM=tail6_fixed32_${TAG}
CONTAINER=fr13-bigdenom-${ARM}
SUBSET=config/fr13_fixed32/subset_b4_sixteen.json
SUBSET_SHA256=47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c
SEQUENCE_FILE=scripts/fr13_fixed32_floor_timers_seq.sh
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}

# Capture geometry, in ABSOLUTE forward steps.
#   START_STEP -- the banked windows all begin at census step 0 and run
#     4152-5860 steps; width 1-2 accounts for only ~141 steps of the shortest.
#     1000 clears hydration by ~7x and still leaves >3000 steps of window even
#     on the narrowest banked arm.
#   CAPTURE_STEPS -- 400 steps ~= 156 s at 389 ms/step.  §6: "a capture of a few
#     hundred consecutive steps sits comfortably inside any of the eight".
START_STEP=${START_STEP:-1000}
CAPTURE_STEPS=${CAPTURE_STEPS:-400}
MIN_EVENTS_PER_STEP=${MIN_EVENTS_PER_STEP:-3.4}

NSYS_SESSION_NAME="fr13-b4w4-${STAMP}-p$$"
NSYS_BIN=/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys
METRICS_URL=${METRICS_URL:-http://127.0.0.1:9950/metrics}

# --------------------------------------------------------------- preflight --
echo "===== FR13 B4 WIDTH-4 NSYS ATTRIBUTION $STAMP (DIAGNOSTIC, NOT CITABLE) ====="

[[ -x "$PYTHON_BIN" ]] || { echo "FAIL: no $PYTHON_BIN" >&2; exit 2; }
[[ -x "$NSYS_BIN" ]] || { echo "FAIL: Nsight Systems unavailable at $NSYS_BIN" >&2; exit 2; }
[[ -f "$SUBSET" && ! -L "$SUBSET" \
   && "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "FAIL: pool16 subset missing or hash drift: $SUBSET" >&2; exit 2; }
[[ -f "$SEQUENCE_FILE" && ! -L "$SEQUENCE_FILE" ]] \
  || { echo "FAIL: missing $SEQUENCE_FILE" >&2; exit 2; }
[[ -f scripts/fr13_b4_width4_nsys_stepgate.py ]] \
  || { echo "FAIL: missing step-gate controller" >&2; exit 2; }

# GPU coordination -- identical to the pool16 gate's rule.
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "FAIL: Docker is not clean -- another campaign may be serving" >&2; exit 2; }
if pgrep -af '[f]r13_bigdenom_swe_serve_variant|[f]r13_b4_campaign_driver' >/dev/null 2>&1; then
  echo "FAIL: an fr13 serve/driver process is already running" >&2
  exit 2
fi

RUNROOT_ABS=$(realpath -m "$RUNROOT")
OUTPUT_ROOT=$(realpath -m "$REPO/output")
case "$RUNROOT_ABS" in
  "$OUTPUT_ROOT"/*) ;;
  *) echo "FAIL: raw profiler artifacts must stay below ignored output/" >&2; exit 2 ;;
esac
[[ ! -e "$RUNROOT_ABS" ]] \
  || { echo "FAIL: RUNROOT must be new (stale evidence is forbidden)" >&2; exit 2; }
git check-ignore -q "$RUNROOT_ABS" \
  || { echo "FAIL: RUNROOT is not ignored by Git" >&2; exit 2; }

# Nsight reports are large; refuse to start without room for one.
avail_gb=$(df -BG --output=avail /home/mark/shared | tail -1 | tr -dc '0-9')
(( avail_gb >= 50 )) \
  || { echo "FAIL: only ${avail_gb}G free on /home/mark/shared; need >=50G" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
CAPTURE_DIR="$RUNROOT_ABS/capture"
mkdir -p "$CAPTURE_DIR"

{
  printf 'schema=fr13.b4_width4_nsys_profile.v1\n'
  printf 'citable=false\n'
  printf 'acceptance_valid=false\n'
  printf 'diagnostic_only=true\n'
  printf 'runroot=%s\n' "$RUNROOT_ABS"
  printf 'stamp=%s\n' "$STAMP"
  printf 'arm=%s\n' "$ARM"
  printf 'container=%s\n' "$CONTAINER"
  printf 'subset=%s\n' "$SUBSET"
  printf 'subset_sha256=%s\n' "$SUBSET_SHA256"
  printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'nsys_session=%s\n' "$NSYS_SESSION_NAME"
  printf 'start_step=%s\n' "$START_STEP"
  printf 'capture_steps=%s\n' "$CAPTURE_STEPS"
  printf 'min_events_per_step=%s\n' "$MIN_EVENTS_PER_STEP"
  printf 'started=%s\n' "$(date -u +%FT%TZ)"
} > "$RUNROOT_ABS/profile_meta.txt"

# ------------------------------------------------------------- nsys wiring --
# GB10 CUPTI workarounds are LOAD-BEARING and are carried verbatim from the B1
# path: without CuptiUseRawGpuTimestamps=false, a periodic flush, and the
# cuda-sw software record path, EVERY per-kernel row is dropped as "incomplete"
# at session stop and the export contains zero kernel tables.
export LUMO_NSYS_WRAP_VLLM=1
export LUMO_NSYS_START_LATER=1
export LUMO_NSYS_BIN="$NSYS_BIN"
export LUMO_NSYS_SESSION_NAME="$NSYS_SESSION_NAME"
export LUMO_NSYS_TRACE=cuda,cuda-sw,nvtx
export LUMO_NSYS_FLUSH_MS=100
export LUMO_NSYS_CONFIG_DIRECTIVES=CuptiUseRawGpuTimestamps=false
export LUMO_NSYS_OUTPUT=/logs/fr13_b4_width4_real_swe
# Not used under --start-later, but the launcher still exports them.
export LUMO_NSYS_DELAY_S=600
export LUMO_NSYS_DURATION_S=150

# NVTX phase ranges (sfwd/cfwd/dfwd/postprocess) + the optional host-tail
# sub-ranges.  The tail sub-ranges are what decompose the `other` bucket, which
# is where the F-window 4-byte D2H lever lives.
export FR13_FIXED32_ATTRIBUTION_ONLY=1
export FR13_FIXED32_NVTX_PROFILE=1
export FR13_HOST_TAIL_NVTX=1
export LUMO_SWE_AUTOCOMMIT=0

echo "[profile] launching pool16 arm under deferred-collection Nsight"
echo "[profile] session=$NSYS_SESSION_NAME runroot=$RUNROOT_ABS"

set +e
env \
  BSIZE=4 \
  CONC=4 \
  WALL=0 \
  TAG="$TAG" \
  RUNROOT="$RUNROOT" \
  SUBSET="$SUBSET" \
  SEQUENCE_FILE="$SEQUENCE_FILE" \
  FR13_B4_TASK_REFILL=1 \
  FR13_DRAFT_VOCAB_ROOT=1 \
  FR13_DRAFT_VOCAB_K=65536 \
  bash scripts/fr13_b4_campaign_driver.sh \
  > "$RUNROOT_ABS/driver.log" 2>&1 &
DRIVER_PID=$!
echo "[profile] driver pid=$DRIVER_PID"
printf 'driver_pid=%s\n' "$DRIVER_PID" >> "$RUNROOT_ABS/profile_meta.txt"

# ------------------------------------------------------ wait for container --
CIDFILE="$RUNROOT_ABS/$ARM/logs/fr13_fixed32_container.cid"
echo "[profile] waiting for container cidfile $CIDFILE"
CONTAINER_ID=""
for (( i=0; i<3600; i++ )); do
  if [[ -f "$CIDFILE" ]]; then
    candidate=$(head -1 "$CIDFILE" 2>/dev/null)
    if [[ "$candidate" =~ ^[0-9a-f]{64}$ ]]; then
      CONTAINER_ID=$candidate
      break
    fi
  fi
  if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
    echo "FAIL: driver exited before the container was created" >&2
    tail -50 "$RUNROOT_ABS/driver.log" >&2
    exit 3
  fi
  sleep 2
done
[[ -n "$CONTAINER_ID" ]] || { echo "FAIL: no container cidfile appeared" >&2; exit 3; }
echo "[profile] container id=$CONTAINER_ID"
printf 'container_id=%s\n' "$CONTAINER_ID" >> "$RUNROOT_ABS/profile_meta.txt"

# ------------------------------------------------------------- step gate ----
echo "[profile] handing off to the step gate"
"$PYTHON_BIN" scripts/fr13_b4_width4_nsys_stepgate.py \
  --container-id "$CONTAINER_ID" \
  --nsys-bin "$NSYS_BIN" \
  --session "$NSYS_SESSION_NAME" \
  --metrics-url "$METRICS_URL" \
  --out-dir "$CAPTURE_DIR" \
  --start-step "$START_STEP" \
  --capture-steps "$CAPTURE_STEPS" \
  --min-events-per-step "$MIN_EVENTS_PER_STEP" \
  2>&1 | tee "$RUNROOT_ABS/stepgate.log"
GATE_RC=${PIPESTATUS[0]}
printf 'stepgate_rc=%s\n' "$GATE_RC" >> "$RUNROOT_ABS/profile_meta.txt"

if (( GATE_RC != 0 )); then
  echo "FAIL: step gate rc=$GATE_RC -- capture is not valid" >&2
  # Do NOT tear the arm down here: evidence-first.  The arm keeps serving and
  # the operator decides.  Report and exit.
fi

# ---------------------------------------------------- let the arm complete --
# The arm is deliberately allowed to run to completion even though the capture
# is already closed.  Completing it produces the work census, the admission
# ledger and the per-task Prometheus brackets, which are what prove -- offline
# and after the fact -- that the captured step range sat inside the width-4
# depth window and carried the batch widths this class claims.  `nsys stop`
# leaves the wrapped server running (verified: session returns to `Launched`),
# so no extra GPU time is bought by this.
echo "[profile] capture closed; waiting for the arm to finish (evidence completion)"
wait "$DRIVER_PID"
DRIVER_RC=$?
set -e
printf 'driver_rc=%s\n' "$DRIVER_RC" >> "$RUNROOT_ABS/profile_meta.txt"
printf 'ended=%s\n' "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/profile_meta.txt"
echo "[profile] driver rc=$DRIVER_RC"

REPORT="$RUNROOT_ABS/$ARM/logs/fr13_b4_width4_real_swe.nsys-rep"
if [[ -s "$REPORT" ]]; then
  echo "[profile] report: $REPORT ($(stat -c %s "$REPORT") bytes)"
  printf 'report=%s\n' "$REPORT" >> "$RUNROOT_ABS/profile_meta.txt"
  printf 'report_bytes=%s\n' "$(stat -c %s "$REPORT")" >> "$RUNROOT_ABS/profile_meta.txt"
else
  echo "WARN: no Nsight report at $REPORT" >&2
fi

echo "[profile] evidence-first teardown check"
if [[ "$(docker ps -aq | wc -l)" -ne 0 ]]; then
  docker ps -a > "$RUNROOT_ABS/docker_ps_after.txt" 2>&1 || true
  for cid in $(docker ps -aq); do
    docker logs --tail 200 "$cid" > "$RUNROOT_ABS/docker_logs_$cid.txt" 2>&1 || true
    docker rm -f "$cid" >/dev/null 2>&1 || true
  done
fi
echo "[profile] docker containers after teardown: $(docker ps -aq | wc -l)"
echo "===== DONE (DIAGNOSTIC, NOT CITABLE) stepgate_rc=$GATE_RC driver_rc=$DRIVER_RC ====="
exit "$GATE_RC"
