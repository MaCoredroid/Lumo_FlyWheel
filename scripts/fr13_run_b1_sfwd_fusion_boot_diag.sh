#!/usr/bin/env bash
# DIAGNOSTIC: composed SFWD conv/post-prep FULL-graph boot screen.
#
# Boots the exact candidate arm of scripts/fr13_run_b1_target_sfwd_exact4_timing.sh
# (Qrow16 production + cooperative M128 target production + SFWD conv/post-prep
# fusion, ENFORCE_EAGER=0, CUDAGRAPH_MODE=FULL_AND_PIECEWISE), waits for the
# engine to reach /health -- which is only reachable once EngineCore init has
# completed profiling AND the final FULL cudagraph capture -- checks the
# capture-time evidence, then tears down. No SWE task, no stock arm, no offload
# proxy: the screen aborts the campaign the moment the boot verdict is known.
#
# Rationale: four consecutive candidate-arm failures (2026-08-05 .. 2026-08-08)
# were init-path invariants, each discovered by paying a ~90-minute pair whose
# ~60-minute stock arm contributes nothing to finding them. This screen costs
# roughly 8 minutes.
#
# This produces NO citable evidence. classification=diagnostic_boot_only.
# It does not measure timing and cannot promote anything.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
DIAG_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"
source scripts/fr13_fixed32_sfwd_fusion_env.sh

case "${FR13_RUN_B1_SFWD_FUSION_BOOT_DIAG:-0}" in
  1) ;;
  0)
    echo "SFWD fusion boot diagnostic is disabled; set FR13_RUN_B1_SFWD_FUSION_BOOT_DIAG=1" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_B1_SFWD_FUSION_BOOT_DIAG must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

: "${RUNROOT:?set RUNROOT to a new path below output/}"
: "${TAG:?set TAG to a unique tag}"
: "${QROW16_FA2_SO:?set QROW16_FA2_SO to the pinned Qrow16 binary}"
: "${CUTLASS_TARGET_SO:?set CUTLASS_TARGET_SO to the pinned cooperative target}"
# The launcher hard-requires the SFWD live pass + source manifest under
# /workspace for FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=1, so the screen takes
# the same passthrough the timing runner does. It deliberately does NOT require
# the standalone gate summary: no byte gate runs here.
: "${SFWD_CONV_POSTPREP_PASS:?set the fresh SFWD live PASS}"
: "${SFWD_CONV_POSTPREP_PASS_SHA256:?set its raw SHA-256}"
: "${SFWD_CONV_POSTPREP_SOURCE_MANIFEST:?set the fresh SFWD source manifest}"
: "${SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256:?set its raw SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
BOOT_TIMEOUT_S=${BOOT_TIMEOUT_S:-1800}
QROW16_PASS=results/fr13_fixed32_qrow16_num_splits0_live_pass_20260731T173608Z/fr13_fa2_qrow16_live_paged_ab.json
QROW16_SHA256=1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86
QROW16_BYTES=299507792
QROW16_PASS_SHA256=36940fd43d11399529d1bfe7e11baa9961907193267f3bb43d41057328737b77
TARGET_SELECTOR=identity_wide256_fullgrid_b1
TARGET_SHA256=d8c6502e7a166e6d2124576a9e36814401d6dbc215516adfffa7ac436f93ba0f
TARGET_BYTES=119704312
TARGET_PASS="$REPO/results/fr13_b1_m128_cooperative_target_sfwd_real_gate_a8a904ed6_20260805/target_combined_pass.json"
TARGET_PASS_SHA256=169704fac7c544600437e7785f5d810c9df8ffaf5f9ce70d96d83b21de46236d
TARGET_QUALIFICATION_SOURCE_COMMIT=a8a904ed6c27a6338d43151038c155ebb76e3656
SUBSET=config/fr13_fixed32/subset_b4_four.json
SOURCE_COMMIT=$(git rev-parse HEAD)
DIAG_SHA256=$(sha256sum "$DIAG_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}
ARM="hydra27_fixed32_sfwd_fusion_bootdiag_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* && ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be a new path below $REPO/output" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
# A dev screen must still boot a coherent tree, but it deliberately does NOT
# require the commit to be pushed: it exists to be run on work in progress.
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "boot diagnostic requires a clean source tree" >&2; exit 2; }

for binding in \
    "$TARGET_PASS:$TARGET_PASS_SHA256" \
    "$SFWD_CONV_POSTPREP_PASS:$SFWD_CONV_POSTPREP_PASS_SHA256" \
    "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST:$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256"; do
  path=${binding%:*}
  digest=${binding##*:}
  [[ "$path" == /* && -f "$path" && ! -L "$path" \
     && "$(stat -c '%h' "$path")" == "1" \
     && "$digest" =~ ^[0-9a-f]{64}$ \
     && "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] \
    || { echo "credential or evidence identity drifted: $path" >&2; exit 2; }
done
unset binding path digest
for binary in "$QROW16_FA2_SO" "$CUTLASS_TARGET_SO"; do
  [[ "$binary" == /* && -f "$binary" && ! -L "$binary" \
     && "$(stat -c '%h' "$binary")" == "1" ]] \
    || { echo "candidate binary must be an absolute regular file: $binary" >&2; exit 2; }
done
unset binary
[[ "$(stat -c '%s' "$QROW16_FA2_SO")" == "$QROW16_BYTES" \
   && "$(sha256sum "$QROW16_FA2_SO" | awk '{print $1}')" == "$QROW16_SHA256" \
   && "$(stat -c '%s' "$CUTLASS_TARGET_SO")" == "$TARGET_BYTES" \
   && "$(sha256sum "$CUTLASS_TARGET_SO" | awk '{print $1}')" == "$TARGET_SHA256" \
   && "$(sha256sum "$QROW16_PASS" | awk '{print $1}')" == "$QROW16_PASS_SHA256" ]] \
  || { echo "Qrow16/target binary or committed evidence identity drifted" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the boot screen" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS/runtime_inputs" "$RUNROOT_ABS/sidecars"
cp -- "$SFWD_CONV_POSTPREP_PASS" "$RUNROOT_ABS/runtime_inputs/sfwd_pass.json"
cp -- "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST" "$RUNROOT_ABS/runtime_inputs/sfwd_source_manifest.json"
chmod 0400 "$RUNROOT_ABS/runtime_inputs/sfwd_pass.json" \
  "$RUNROOT_ABS/runtime_inputs/sfwd_source_manifest.json"
SFWD_PASS_CONTAINER="/workspace/$RUNROOT_REL/runtime_inputs/sfwd_pass.json"
SFWD_MANIFEST_CONTAINER="/workspace/$RUNROOT_REL/runtime_inputs/sfwd_source_manifest.json"
[[ "$(sha256sum "$RUNROOT_ABS/runtime_inputs/sfwd_pass.json" | awk '{print $1}')" == "$SFWD_CONV_POSTPREP_PASS_SHA256" \
   && "$(sha256sum "$RUNROOT_ABS/runtime_inputs/sfwd_source_manifest.json" | awk '{print $1}')" == "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" ]] \
  || { echo "runtime SFWD credential copy drifted" >&2; exit 2; }

cat <<BANNER
=========================================================================
 DIAGNOSTIC ONLY - produces no citable evidence
 classification=diagnostic_boot_only
 screen: composed SFWD conv/post-prep FULL-graph boot-through-capture
 arm=$ARM commit=$SOURCE_COMMIT
 no SWE task, no stock arm, no timing, nothing promotable
=========================================================================
BANNER

RUNLOG="$RUNROOT_ABS/$ARM.runlog"
DOCKER_LOG="$RUNROOT_ABS/boot_docker.log"
SUMMARY="$RUNROOT_ABS/boot_diag_summary.json"
VERDICT=unknown
DETAIL=""
CONTAINER_ID=""
VARIANT_PID=""

teardown() {
  local rc=$?
  trap - EXIT
  if [[ -n "$VARIANT_PID" ]] && kill -0 "$VARIANT_PID" 2>/dev/null; then
    kill -TERM -- "-$VARIANT_PID" 2>/dev/null || kill -TERM "$VARIANT_PID" 2>/dev/null || true
    local waited=0
    while kill -0 "$VARIANT_PID" 2>/dev/null && (( waited < 30 )); do
      sleep 1
      waited=$(( waited + 1 ))
    done
    kill -KILL -- "-$VARIANT_PID" 2>/dev/null || kill -KILL "$VARIANT_PID" 2>/dev/null || true
  fi
  local cid
  for cid in $(docker ps -aq); do
    docker rm -f "$cid" >/dev/null 2>&1 || true
  done
  PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -c \
    "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" \
    >/dev/null 2>&1 || true
  free -g > "$RUNROOT_ABS/free_after_teardown.txt" 2>/dev/null || true
  if [[ "$(docker ps -aq | wc -l)" -ne 0 ]]; then
    echo "DIAG WARNING: Docker state is not clean after teardown" >&2
    (( rc == 0 )) && rc=6
  fi
  "$PYTHON_BIN" - "$SUMMARY" "$VERDICT" "$DETAIL" "$ARM" "$SOURCE_COMMIT" \
    "$DIAG_SHA256" "$CONTAINER_ID" "$rc" <<'PY' || true
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "schema": "fr13.fixed32.sfwd_fusion_boot_diag.v1",
            "classification": "diagnostic_boot_only",
            "citable": False,
            "timing_eligible": False,
            "verdict": sys.argv[2],
            "detail": sys.argv[3],
            "arm": sys.argv[4],
            "source_commit": sys.argv[5],
            "diagnostic_sha256": sys.argv[6],
            "container_id": sys.argv[7],
            "exit_code": int(sys.argv[8]),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="ascii",
)
PY
  echo "DIAG VERDICT: $VERDICT ${DETAIL:+- $DETAIL}"
  echo "DIAG SUMMARY: $SUMMARY"
  exit "$rc"
}
trap teardown EXIT

free -g > "$RUNROOT_ABS/free_before_boot.txt" 2>/dev/null || true

declare -a FR13_FIXED32_SFWD_FUSION_ENV
declare device_kernel target_selector target_so target_pass target_pass_sha
declare target_qualification_source_commit sfwd_fusion sfwd_pass sfwd_pass_sha
declare sfwd_manifest sfwd_manifest_sha sfwd_commit conv_wb_batched
fr13_fixed32_sfwd_fusion_env "$ARM" 1

# The campaign script owns the whole boot path (launcher, cidfile promotion,
# health poll). Run it in its own process group and abort it the instant the
# boot verdict lands, so the screen never pays for a task.
set -m
env "${FR13_FIXED32_SFWD_FUSION_ENV[@]}" \
  bash scripts/fr13_bigdenom_swe_serve_variant.sh \
    "$ARM" hydra27_fixed32 "$SUBSET" \
    > "$RUNLOG" 2>&1 &
VARIANT_PID=$!
set +m

echo "[boot-diag] variant pid=$VARIANT_PID runlog=$RUNLOG"
T0=$(date +%s)
while :; do
  if [[ -f "$RUNLOG" ]] && grep -q '^healthy after ' "$RUNLOG"; then
    VERDICT=boot_ok
    DETAIL=$(grep -m1 '^healthy after ' "$RUNLOG")
    break
  fi
  if [[ -f "$RUNLOG" ]] \
     && grep -qE '^FAIL: (container died before health|health not up|launcher rc=)' "$RUNLOG"; then
    VERDICT=boot_failed
    DETAIL=$(grep -m1 -E '^FAIL: (container died before health|health not up|launcher rc=)' "$RUNLOG")
    break
  fi
  if ! kill -0 "$VARIANT_PID" 2>/dev/null; then
    VERDICT=boot_failed
    DETAIL="variant exited before reaching health"
    break
  fi
  if (( $(date +%s) >= T0 + BOOT_TIMEOUT_S )); then
    VERDICT=boot_timeout
    DETAIL="no boot verdict in ${BOOT_TIMEOUT_S}s"
    break
  fi
  sleep 5
done
echo "[boot-diag] verdict=$VERDICT after $(( $(date +%s) - T0 ))s"

CID_PATH="$ARMDIR/logs/fr13_fixed32_container.cid"
if [[ -f "$CID_PATH" ]]; then
  CONTAINER_ID=$(tr -d '[:space:]' < "$CID_PATH")
fi
if [[ -n "$CONTAINER_ID" ]]; then
  docker logs "$CONTAINER_ID" > "$DOCKER_LOG" 2>&1 || true
else
  : > "$DOCKER_LOG"
fi
echo "[boot-diag] docker log -> $DOCKER_LOG ($(wc -l < "$DOCKER_LOG") lines)"

if [[ "$VERDICT" != "boot_ok" ]]; then
  echo "DIAG FAIL: $DETAIL" >&2
  echo "---- last 40 runlog lines ----" >&2
  tail -40 "$RUNLOG" >&2 || true
  if [[ -s "$DOCKER_LOG" ]]; then
    echo "---- last 60 container log lines ----" >&2
    tail -60 "$DOCKER_LOG" >&2
  fi
  exit 3
fi

for forbidden in "${FR13_FIXED32_SFWD_FUSION_FORBIDDEN[@]}"; do
  if grep -Fq -- "$forbidden" "$DOCKER_LOG"; then
    VERDICT=forbidden_string
    DETAIL="container emitted forbidden fallback: $forbidden"
    echo "DIAG FAIL: $DETAIL" >&2
    exit 4
  fi
done
unset forbidden

SFWD_MARKER='[FR13_SFWD_CONV_POSTPREP] production engaged layer='
SFWD_ENGAGED=$(grep -Fc -- "$SFWD_MARKER" "$DOCKER_LOG" || true)
if [[ "$SFWD_ENGAGED" -ne 48 ]]; then
  VERDICT=sfwd_partial
  DETAIL="SFWD conv/post-prep engaged $SFWD_ENGAGED/48 layers"
  echo "DIAG FAIL: $DETAIL" >&2
  exit 5
fi

for artifact in \
    "$ARMDIR/logs/fr13_fa2_qrow16_production_capture.json" \
    "$ARMDIR/logs/fr13_fixed32_sfwd_conv_postprep.production_pass.json" \
    "$ARMDIR/logs/fr13_fixed32_sfwd_conv_postprep.source_manifest.json" \
    "$ARMDIR/logs/fr13_fixed32_cutlass_streamk.production_pass.json"; do
  [[ -f "$artifact" && ! -L "$artifact" ]] \
    || { VERDICT=missing_evidence; DETAIL="boot evidence is missing: $artifact"; \
         echo "DIAG FAIL: $DETAIL" >&2; exit 5; }
done
unset artifact

"$PYTHON_BIN" - \
  "$ARMDIR/logs/fr13_fa2_qrow16_production_capture.json" \
  "$QROW16_SHA256" <<'PY' \
  || { VERDICT=missing_evidence; DETAIL="qrow16 FULL capture record drifted"; \
       echo "DIAG FAIL: $DETAIL" >&2; exit 5; }
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
required = {
    "schema": "fr13.fixed32.fa2_qrow16_production_capture.v1",
    "status": "ENGAGED",
    "runtime_mode": "FULL",
    "batch_size": 1,
    "layer_count": 16,
    "candidate_so_sha256": sys.argv[2],
    "dispatch": "qrow16 exact geometry; no fallback",
}
if any(record.get(key) != value for key, value in required.items()):
    raise SystemExit("qrow16 FULL capture record drifted: " + repr(record))
PY

VERDICT=pass
DETAIL="boot through FULL capture clean: 48/48 SFWD layers, qrow16 FULL capture engaged"
echo "DIAG PASS: $DETAIL"
exit 0
