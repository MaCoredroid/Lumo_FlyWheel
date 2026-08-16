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
# After health it runs a short smoke phase (FR13_BOOT_DIAG_SMOKE=1, the
# default): a handful of real chat completions through the engine's own fixed32
# ingress. Health proves EngineCore init and the final FULL capture and nothing
# else -- the runtime census/commit/replay validators are reached only by a
# MEASURED forward, and the terminal flush only by a teardown that follows one.
# A screen that stops at health cannot see any of them, which is how the
# 2026-08-08 arm passed its screen and then died at the flush validator five
# minutes into the pair. The smoke costs ~1-2 minutes and is the difference
# between "it booted" and "it can serve".
#
# Those completions carry the campaign's sampling shape (temperature 0.6,
# top_p 0.95, top_k 20, presence_penalty 1.0, min_p 0), which the shared env
# owns and a test binds to the proxy relaunch helper's defaults. The pair gets
# them stamped on by the offload proxy; the screen freezes the driver before
# that proxy exists and talks to the engine directly, so it has to carry them
# itself. It is not optional dressing: fixed32 refuses greedy decoding, and the
# temperature 0.0 the first smoke sent raised "FR13 fixed32 requires sampled
# temp>0" inside the rejection sampler, killed EngineCore before the proposal
# seal, and returned a bare 500. Only max_tokens is deliberately unlike the
# pair's, and only because a screen has no reason to pay for long generations.
#
# That profile also decides what a served turn looks like: thinking mode emits
# into the reasoning channel first, and at max_tokens=96 the turn stops on
# length while still inside it, with content null. A turn counts as served when
# it finished and produced text in EITHER channel -- the campaign's own trace
# records thinking blocks beside text and tool_use, and the proxy calls a
# reasoning-only turn unproductive rather than empty. The screen wants the
# forward executed, not an answer written.
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

# Verdict table (verdict -> exit code):
#   pass                     0  boot clean, and smoke clean when it ran
#   boot_failed              3  no /health, launcher failure, or an early exit
#   boot_timeout             3  no boot verdict inside BOOT_TIMEOUT_S
#   forbidden_string         4  capture-time fallback/preseed string
#   sfwd_partial             5  fewer than 48/48 SFWD layers engaged
#   missing_evidence         5  boot evidence absent or drifted
#   smoke_request_failed     7  the engine could not serve the smoke traffic
#   smoke_validator_tripped  8  a runtime validator or the terminal flush raised
#   (any)                    6  teardown could not clean Docker state
#
# Cause outranks consequence. An engine that dies serving the smoke also leaves
# the terminal flush nothing to flush, so the flush fails too -- and reporting
# THAT as the verdict names the symptom and buries the cause. When the smoke
# served nothing (smoke_responses_ok=0) the verdict stays smoke_request_failed,
# whatever the flush sweep then finds; the flush finding is still published in
# the summary. The 2026-08-08 screen reported rc=8 with smoke_responses_ok=0 for
# what was a plain rc=7.
#
# Every failure path captures the container log BEFORE teardown signals anything.
# An engine-side death writes its stack trace there and nowhere else, and
# teardown removes the container that holds it. That same screen came back with
# "HTTP 500 EngineCore encountered an issue" and a capture still ending at
# /health, which is a screen that found a failure and discarded the evidence
# naming it. Per-step request evidence lands in smoke_requests.jsonl.
#
# The smoke phase is ON by default. Set FR13_BOOT_DIAG_SMOKE=0 to stop at
# health, which is exactly what the screen did before the phase existed.
#
# Every smoke knob is read here and then scrubbed out of the exported
# environment. The launcher forwards each FR13_*/LUMO_*/VLLM_* variable it can
# see into the container as -e (fr13_launch_forked_fa2_tree_server.sh:
# `compgen -v | grep -E '^(FR[0-9]+_|LUMO_|VLLM_)'`, an opt-OUT list), so a knob
# left exported would boot an engine whose environment differs from the timing
# pair's by exactly the variable this screen introduced -- and the screen is
# sound only while the two environments match. Demote, do not delete, the
# selector itself: the script still reads it below.
FR13_BOOT_DIAG_SMOKE=${FR13_BOOT_DIAG_SMOKE:-1}
SMOKE_REQUESTS=${FR13_BOOT_DIAG_SMOKE_REQUESTS:-6}
SMOKE_MAX_TOKENS=${FR13_BOOT_DIAG_SMOKE_MAX_TOKENS:-96}
SMOKE_TEARDOWN_GRACE_S=${FR13_BOOT_DIAG_SMOKE_TEARDOWN_GRACE_S:-180}
export -n FR13_BOOT_DIAG_SMOKE
unset FR13_BOOT_DIAG_SMOKE_REQUESTS FR13_BOOT_DIAG_SMOKE_MAX_TOKENS \
  FR13_BOOT_DIAG_SMOKE_TEARDOWN_GRACE_S
case "$FR13_BOOT_DIAG_SMOKE" in
  0|1) ;;
  *)
    echo "FR13_BOOT_DIAG_SMOKE must be exactly 0 or 1" >&2
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
# The variant publishes the engine on this host port
# (scripts/fr13_bigdenom_swe_serve_variant.sh PORT=9950) and the launcher serves
# it under this name (vllm serve --served-model-name).
SMOKE_PORT=9950
SMOKE_MODEL=qwen3.8-27b-nvfp4-radixark
# One canonical task key for every smoke request, deliberately not all four: the
# engine publishes the SFWD real-event arm the moment every canonical task key
# has authenticated, and a boot screen must not mint gate evidence.
SMOKE_TASK_ID=astropy__astropy-12907
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
# Deliberately the pair's 4-task subset, not a 1-task diagnostic subset: the
# screen never dispatches a task, and a diagnostic subset would flip
# FR13_FIXED32_B1_DIAGNOSTIC and boot a different engine than the arm under
# screen. The runlog banner reading "tasks=4 diagnostic=0" is the correct
# signal that the screen matches the candidate arm.
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
# A screen that serves more than a handful of short completions is a run, not a
# screen. Bound both dimensions so the phase cannot grow into one.
[[ "$SMOKE_REQUESTS" =~ ^[1-8]$ ]] \
  || { echo "FR13_BOOT_DIAG_SMOKE_REQUESTS must be 1..8" >&2; exit 2; }
[[ "$SMOKE_MAX_TOKENS" =~ ^[0-9]+$ && "$SMOKE_MAX_TOKENS" -ge 1 \
   && "$SMOKE_MAX_TOKENS" -le 256 ]] \
  || { echo "FR13_BOOT_DIAG_SMOKE_MAX_TOKENS must be 1..256" >&2; exit 2; }
[[ "$SMOKE_TEARDOWN_GRACE_S" =~ ^[0-9]+$ && "$SMOKE_TEARDOWN_GRACE_S" -ge 30 ]] \
  || { echo "FR13_BOOT_DIAG_SMOKE_TEARDOWN_GRACE_S must be >= 30" >&2; exit 2; }
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
SMOKE_LOG="$RUNROOT_ABS/smoke_phase.log"
SMOKE_RECORD="$RUNROOT_ABS/smoke_summary.json"
# One record per HTTP step the smoke performs, written as it goes. The screen
# has to be able to say WHICH step failed and what the engine answered, from the
# runroot alone, without the container that produced it still existing.
SMOKE_EVIDENCE="$RUNROOT_ABS/smoke_requests.jsonl"
VERDICT=unknown
DETAIL=""
CONTAINER_ID=""
VARIANT_PID=""
VARIANT_FROZEN=0
SMOKE_RAN=0
SMOKE_SENT=0
SMOKE_OK=0
# Set only once the driver has published what its terminal flush needs, so the
# screen never reads a flush verdict off a flush that never got to start.
SMOKE_FLUSH_READY=0
SMOKE_FLUSH_PREREQ_S=120
# Tri-state: the clean flags say "unchecked" until the sweep that owns them has
# actually run, so a summary can never claim a validator was clean by default.
SMOKE_VALIDATORS_CLEAN=unchecked
SMOKE_FLUSH_CLEAN=unchecked
# Whether the EngineCore that the terminal flush addresses was still alive when
# teardown began. A flush that fails against a dead producer is reporting the
# engine's death, not delivering a validator's verdict.
SMOKE_PRODUCER_ALIVE=unchecked
TEARDOWN_GRACE_S=30

# The container log is the only place an engine-side death writes its stack
# trace, and every failure path ends in a teardown that removes the container.
# Safe to call repeatedly: each capture supersedes the last with a superset of
# it, and a capture that comes back empty leaves the previous one in place.
capture_container_log() {
  [[ -n "$CONTAINER_ID" ]] || return 1
  local partial="$DOCKER_LOG.partial"
  if docker logs "$CONTAINER_ID" > "$partial" 2>&1 && [[ -s "$partial" ]]; then
    mv -f "$partial" "$DOCKER_LOG"
    return 0
  fi
  rm -f "$partial"
  return 1
}

# The attested EngineCore PID the driver publishes: the same producer the
# terminal flush addresses, and the one whose death the flush reports.
smoke_producer_pid() {
  [[ -f "$ARMDIR/fixed32_process_identity.json" ]] || return 1
  "$PYTHON_BIN" - "$ARMDIR/fixed32_process_identity.json" <<'PY' 2>/dev/null
import json
import sys
from pathlib import Path

pid = (
    json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    .get("engine_core", {})
    .get("pid")
)
if not isinstance(pid, int) or pid <= 0:
    raise SystemExit(1)
print(pid)
PY
}

teardown() {
  local rc=$?
  trap - EXIT
  # Before ANY signal goes out. Teardown is about to stop the driver and remove
  # every container, and an engine that died mid-smoke wrote its stack trace to
  # a log that only exists until then. The 2026-08-08 screen returned "HTTP 500
  # EngineCore encountered an issue" with the capture still ending at /health,
  # because the request-failure path exited before the post-smoke re-read: the
  # screen found the failure and discarded the only evidence naming it.
  if capture_container_log; then
    echo "[boot-diag] container log captured pre-teardown -> $DOCKER_LOG ($(wc -l < "$DOCKER_LOG") lines)"
  fi
  # Probe the producer before the CONT/TERM, so the answer describes the state
  # the smoke left behind rather than anything teardown itself did to it.
  if (( SMOKE_RAN == 1 )) && [[ -n "$CONTAINER_ID" ]]; then
    local producer_pid
    producer_pid=$(smoke_producer_pid || true)
    if [[ -n "$producer_pid" ]] \
       && docker exec "$CONTAINER_ID" kill -0 "$producer_pid" >/dev/null 2>&1; then
      SMOKE_PRODUCER_ALIVE=true
    else
      SMOKE_PRODUCER_ALIVE=false
      echo "DIAG WARNING: EngineCore PID ${producer_pid:-<unpublished>} is not alive before teardown; the terminal flush has no live producer" >&2
    fi
  fi
  if [[ -n "$VARIANT_PID" ]] && kill -0 "$VARIANT_PID" 2>/dev/null; then
    # The smoke phase freezes the campaign driver. Continue it before the TERM
    # or it can never run its own EXIT trap -- and that trap is what performs
    # the fixed32 terminal flush this screen then reads a verdict from.
    if (( VARIANT_FROZEN == 1 )); then
      kill -CONT -- "-$VARIANT_PID" 2>/dev/null || kill -CONT "$VARIANT_PID" 2>/dev/null || true
      VARIANT_FROZEN=0
    fi
    kill -TERM -- "-$VARIANT_PID" 2>/dev/null || kill -TERM "$VARIANT_PID" 2>/dev/null || true
    local waited=0
    while kill -0 "$VARIANT_PID" 2>/dev/null && (( waited < TEARDOWN_GRACE_S )); do
      sleep 1
      waited=$(( waited + 1 ))
    done
    kill -KILL -- "-$VARIANT_PID" 2>/dev/null || kill -KILL "$VARIANT_PID" 2>/dev/null || true
  fi
  # The driver is gone, so its terminal flush is final and its verdict readable:
  # harness-side failures land in the runlog, engine-side ones in the container
  # log the driver captured on its way out.
  if (( SMOKE_RAN == 1 && SMOKE_FLUSH_READY == 1 )); then
    SMOKE_FLUSH_CLEAN=true
    local fragment
    for fragment in "${FR13_FIXED32_SFWD_FUSION_FLUSH_FORBIDDEN[@]}"; do
      if grep -Fq -- "$fragment" "$RUNLOG" 2>/dev/null \
         || grep -Fq -- "$fragment" "$ARMDIR/docker_full.log" 2>/dev/null \
         || grep -Fq -- "$fragment" "$ARMDIR/fixed32_final_flush.stderr" 2>/dev/null; then
        SMOKE_FLUSH_CLEAN=false
        echo "DIAG FAIL: terminal flush emitted: $fragment" >&2
        # Cause before consequence. A smoke that served nothing leaves an engine
        # the flush cannot address, and the flush then fails for that reason --
        # reporting it as a validator trip names the symptom and buries the
        # cause, which is how the 2026-08-08 screen returned rc=8 with
        # smoke_responses_ok=0 for what was a plain rc=7. The flush finding is
        # still published; it just does not get to own the verdict.
        if (( SMOKE_OK == 0 )); then
          if [[ "$VERDICT" == "smoke_request_failed" ]]; then
            DETAIL="$DETAIL; terminal flush then failed against a dead producer"
          else
            VERDICT=smoke_request_failed
            DETAIL="smoke served 0/$SMOKE_SENT completions; terminal flush then failed: $fragment"
            rc=7
          fi
        else
          VERDICT=smoke_validator_tripped
          DETAIL="terminal flush emitted: $fragment"
          rc=8
        fi
        break
      fi
    done
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
    "$DIAG_SHA256" "$CONTAINER_ID" "$rc" "$FR13_BOOT_DIAG_SMOKE" "$SMOKE_RAN" \
    "$SMOKE_SENT" "$SMOKE_OK" "$SMOKE_VALIDATORS_CLEAN" \
    "$SMOKE_FLUSH_CLEAN" "$SMOKE_PRODUCER_ALIVE" <<'PY' || true
import json
import sys
from pathlib import Path

(
    _,
    summary_path,
    verdict,
    detail,
    arm,
    source_commit,
    diagnostic_sha256,
    container_id,
    exit_code,
    smoke_enabled,
    smoke_ran,
    smoke_requests_sent,
    smoke_responses_ok,
    smoke_validators_clean,
    smoke_flush_clean,
    smoke_producer_alive,
) = sys.argv


def tristate(value):
    """null, never False, for a sweep that did not get to run."""
    return {"true": True, "false": False, "unchecked": None}[value]


Path(summary_path).write_text(
    json.dumps(
        {
            "schema": "fr13.fixed32.sfwd_fusion_boot_diag.v3",
            "classification": "diagnostic_boot_only",
            "citable": False,
            "timing_eligible": False,
            "verdict": verdict,
            "detail": detail,
            "arm": arm,
            "source_commit": source_commit,
            "diagnostic_sha256": diagnostic_sha256,
            "container_id": container_id,
            "exit_code": int(exit_code),
            "smoke_enabled": smoke_enabled == "1",
            "smoke_ran": smoke_ran == "1",
            "smoke_requests_sent": int(smoke_requests_sent),
            "smoke_responses_ok": int(smoke_responses_ok),
            "smoke_validator_strings_clean": tristate(smoke_validators_clean),
            "smoke_flush_strings_clean": tristate(smoke_flush_clean),
            "smoke_producer_alive_at_teardown": tristate(smoke_producer_alive),
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

# Same exported route/committer preamble the timing pair boots with. Skipping
# it is how the first screen died pre-docker: the launcher reads
# FR13_FIXED32_TAW_WALK_CAP as "${NAME:-}" and int("") is "fixed32 integer
# route pin is malformed".
fr13_fixed32_sfwd_fusion_route_preamble

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

if [[ "$VERDICT" == "boot_ok" && "$FR13_BOOT_DIAG_SMOKE" == "1" ]]; then
  # The driver publishes the two prerequisites of its own terminal flush -- the
  # attested EngineCore PID and the engine's generation-zero ready ack -- in the
  # seconds AFTER it prints "healthy after". Freeze before both exist and its
  # teardown has nothing to flush: it reports a missing PID, which reads exactly
  # like a flush failure that never happened. Wait for the evidence first, then
  # freeze; if it never lands the screen still smokes, but it declines to claim
  # a flush verdict it did not earn.
  SMOKE_PREREQ_T0=$(date +%s)
  while (( $(date +%s) < SMOKE_PREREQ_T0 + SMOKE_FLUSH_PREREQ_S )); do
    if [[ -f "$ARMDIR/fixed32_process_identity.json" \
       && -f "$ARMDIR/logs/fr13_fixed32_flush_ack.json" ]]; then
      SMOKE_FLUSH_READY=1
      break
    fi
    kill -0 "$VARIANT_PID" 2>/dev/null || break
    sleep 2
  done
  if (( SMOKE_FLUSH_READY == 1 )); then
    echo "[boot-diag] flush prerequisites published after $(( $(date +%s) - SMOKE_PREREQ_T0 ))s"
  else
    echo "DIAG WARNING: no flush prerequisites in ${SMOKE_FLUSH_PREREQ_S}s; the terminal flush will not be screened" >&2
  fi

  # Freeze the campaign driver. Left running it walks on to launch the offload
  # proxy and open its OWN fixed32 ingress campaign, and the engine admits
  # exactly one -- whichever of the two lost that race would die on a 409
  # mid-smoke, taking the container with it. Frozen, the screen owns the
  # campaign and the driver still holds everything its EXIT-trap flush needs.
  # The container is dockerd's child, so nothing about it freezes with the
  # driver.
  if kill -STOP -- "-$VARIANT_PID" 2>/dev/null || kill -STOP "$VARIANT_PID" 2>/dev/null; then
    VARIANT_FROZEN=1
    # The frozen driver still has to run that flush after the CONT, and the
    # flush is the last thing this screen reads a verdict from, so give it room.
    TEARDOWN_GRACE_S=$SMOKE_TEARDOWN_GRACE_S
    echo "[boot-diag] campaign driver frozen for the smoke phase"
  else
    VERDICT=smoke_request_failed
    DETAIL="could not freeze the campaign driver before the smoke phase"
    echo "DIAG FAIL: $DETAIL" >&2
    exit 7
  fi
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

if [[ "$FR13_BOOT_DIAG_SMOKE" == "1" ]]; then
  # The engine reads its ingress secret from a read-only bind mount, so the
  # container's own record names the host file the screen must authenticate
  # with. No copy is taken: the driver owns that file's lifetime and deletes it.
  SMOKE_SECRET=$(docker inspect --format \
    '{{range .Mounts}}{{if eq .Destination "/run/fr13_fixed32_ingress_secret.host"}}{{.Source}}{{end}}{{end}}' \
    "$CONTAINER_ID" 2>/dev/null || true)
  [[ "$SMOKE_SECRET" == /* && -f "$SMOKE_SECRET" && ! -L "$SMOKE_SECRET" ]] \
    || { VERDICT=smoke_request_failed; \
         DETAIL="engine ingress secret mount is unreadable: ${SMOKE_SECRET:-<absent>}"; \
         echo "DIAG FAIL: $DETAIL" >&2; exit 7; }

  SMOKE_RAN=1
  echo "[boot-diag] smoke: $SMOKE_REQUESTS chat completions, max_tokens=$SMOKE_MAX_TOKENS"
  # The driver logs one line per dispatched request and one per accepted
  # response, so both summary counts are read back off evidence rather than
  # assumed from the intended count -- a driver that dies on request 3 reports 3.
  PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - \
    "$SMOKE_SECRET" "$SUBSET" "$SMOKE_TASK_ID" "$SMOKE_PORT" "$SMOKE_MODEL" \
    "$SMOKE_REQUESTS" "$SMOKE_MAX_TOKENS" "$SMOKE_RECORD" "$SMOKE_EVIDENCE" \
    "${FR13_FIXED32_SFWD_FUSION_SMOKE_SAMPLING[@]}" \
    > "$SMOKE_LOG" 2>&1 <<'PY' \
    || { SMOKE_SENT=$(grep -c '^smoke request ' "$SMOKE_LOG" || true); \
         SMOKE_OK=$(grep -c '^smoke response ok ' "$SMOKE_LOG" || true); \
         VERDICT=smoke_request_failed; \
         DETAIL="smoke traffic failed: $(tail -1 "$SMOKE_LOG")"; \
         echo "DIAG FAIL: $DETAIL" >&2; tail -20 "$SMOKE_LOG" >&2; \
         echo "---- per-step smoke evidence: $SMOKE_EVIDENCE ----" >&2; \
         capture_container_log \
           && { echo "---- last 60 container log lines, post-smoke ----" >&2; \
                tail -60 "$DOCKER_LOG" >&2; }; \
         exit 7; }
import json
import re
import secrets
import sys
import time
from pathlib import Path

import requests

from lumo_flywheel_serving.inference_proxy import (
    fixed32_canonical_task_set_sha256,
    fixed32_task_key_id,
    load_fixed32_ingress_secrets,
)

(
    _,
    secret_path,
    subset_path,
    task_id,
    port_text,
    model,
    count_text,
    max_tokens_text,
    record_path,
    evidence_path,
    *sampling_argv,
) = sys.argv
port = int(port_text)
count = int(count_text)
max_tokens = int(max_tokens_text)

# The campaign's sampling shape, passed in from the shared env. The pair gets
# these stamped on by the offload proxy; the screen bypasses the proxy, so it
# has to carry them itself or it is not screening the pair's engine at all.
SAMPLING_TYPES = {
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "presence_penalty": float,
    "min_p": float,
}
sampling = {}
for entry in sampling_argv:
    key, sep, value = entry.partition("=")
    if not sep or key not in SAMPLING_TYPES or key in sampling:
        raise SystemExit(f"smoke sampling entry is not usable: {entry!r}")
    sampling[key] = SAMPLING_TYPES[key](value)
if sorted(sampling) != sorted(SAMPLING_TYPES):
    raise SystemExit(f"smoke sampling shape is incomplete: {sampling!r}")
# fixed32 refuses greedy decoding: temp 0 makes the rejection sampler raise
# "FR13 fixed32 requires sampled temp>0" before the proposal seal and take
# EngineCore down with it. Refuse here, where the message survives, rather than
# discovering it as a bare 500 from a dead engine.
if not sampling["temperature"] > 0:
    raise SystemExit(
        "fixed32 requires sampled decoding: smoke temperature must be > 0, "
        f"got {sampling['temperature']!r}"
    )


def record(event, **fields):
    """Append one evidence record, flushed, so a step that dies still leaves it."""
    with open(evidence_path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema": "fr13.fixed32.sfwd_fusion_boot_diag_smoke_step.v1",
                    "event": event,
                    "monotonic": round(time.monotonic(), 3),
                    **fields,
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()


def observe(event, response, started, **fields):
    """Record an HTTP outcome. The body head is the diagnostic: an engine that
    died mid-forward answers 500 with the reason, and the container log that
    holds the stack trace does not outlive teardown."""
    record(
        event,
        status_code=response.status_code,
        elapsed_s=round(time.monotonic() - started, 3),
        body_head=response.text[:2000],
        **fields,
    )
    return response

subset = json.loads(Path(subset_path).read_text(encoding="utf-8"))
task_ids = tuple(subset["instance_ids"])
if task_id not in task_ids:
    raise SystemExit(f"smoke task {task_id!r} is not in the screened subset")

secret = load_fixed32_ingress_secrets(secret_path)
base = f"http://127.0.0.1:{port}"
control = {"Authorization": f"Bearer {secret.engine_bearer}"}

# The engine admits inference only inside an open ingress campaign, and admits
# exactly one. The campaign driver is frozen, so this is that one campaign.
_started = time.monotonic()
begin = observe(
    "ingress_begin",
    requests.post(
        base + "/fr13/fixed32/ingress/begin",
        headers=control,
        json={
            "schema": "fr13-fixed32-ingress-begin-v1",
            "canonical_task_count": len(task_ids),
            "canonical_task_set_sha256": fixed32_canonical_task_set_sha256(task_ids),
        },
        timeout=60,
    ),
    _started,
)
if begin.status_code != 200:
    raise SystemExit(
        f"ingress begin failed: HTTP {begin.status_code} {begin.text[:2000]}"
    )

# Ordinary chat traffic. The engine always proposes through the tree drafter, so
# every decode step here runs spec decode and replays the FULL graph -- which is
# the only thing that reaches the validators this phase exists to exercise.
PROMPTS = (
    "Summarise what a Python context manager is, in three sentences.",
    "Write a short docstring for a function that reverses a list in place.",
    "Explain the difference between a tuple and a list in Python.",
    "Describe briefly what unittest.mock.patch does.",
)
# A turn the engine finished. "length" is the normal one here: the screen caps
# generation at max_tokens deliberately, so a completed smoke turn stops on it.
FINISHED = ("stop", "length", "tool_calls")
task_key_id = fixed32_task_key_id(task_id)
served = 0
reasoning_only = 0
for index in range(count):
    wire_id = "fr13-chat-" + secrets.token_hex(16)
    print(f"smoke request {index} max_tokens={max_tokens}", flush=True)
    record(
        "chat_dispatch",
        index=index,
        wire_id=wire_id,
        max_tokens=max_tokens,
        sampling=sampling,
    )
    started = time.monotonic()
    response = observe(
        "chat_response",
        requests.post(
            base + "/v1/chat/completions",
            headers={
                **control,
                "Content-Type": "application/json",
                "X-Fr13-Task-Key-ID": task_key_id,
                "X-Request-ID": wire_id,
            },
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": PROMPTS[index % len(PROMPTS)]}
                ],
                "max_tokens": max_tokens,
                "stream": False,
                **sampling,
            },
            timeout=600,
        ),
        started,
        index=index,
        wire_id=wire_id,
    )
    if response.status_code != 200:
        raise SystemExit(
            f"chat completion {index} failed: HTTP {response.status_code} "
            f"{response.text[:2000]}"
        )
    payload = response.json()
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SystemExit(f"chat completion {index} returned no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")
    if finish_reason not in FINISHED:
        raise SystemExit(
            f"chat completion {index} did not finish: finish_reason="
            f"{finish_reason!r}"
        )
    # The qwen thinking profile puts the turn in the reasoning channel first, and
    # at max_tokens=96 it never gets out of it: content stays null and the turn
    # stops on length. That is a served turn, not an empty response, and it is
    # how the campaign accounts for one -- its own agent trace records thinking
    # blocks beside text and tool_use, and the proxy calls a turn that produced
    # only reasoning a "reasoning-only dead turn": unproductive, still a turn.
    # The screen wants the forward, not the answer. vLLM spells the channel
    # reasoning on this build and reasoning_content on others; accept either.
    text = {
        key: message.get(key)
        for key in ("content", "reasoning_content", "reasoning")
        if isinstance(message.get(key), str) and message.get(key)
    }
    usage = payload.get("usage") or {}
    record(
        "chat_served",
        index=index,
        wire_id=wire_id,
        finish_reason=finish_reason,
        completion_tokens=usage.get("completion_tokens"),
        channels={key: len(value) for key, value in text.items()},
    )
    if not text:
        raise SystemExit(
            f"chat completion {index} produced no text in content, "
            f"reasoning_content or reasoning: {message!r}"[:400]
        )
    if "content" not in text:
        reasoning_only += 1
    served += 1
    print(
        f"smoke response ok {index} finish={finish_reason} "
        f"chars={sum(len(value) for value in text.values())} "
        f"channels={','.join(sorted(text))}",
        flush=True,
    )

# The middleware books completion only after the response is fully sent, so a
# finalize that races the last one legitimately sees 409 active_requests.
ledger = None
for attempt in range(6):
    started = time.monotonic()
    finalize = observe(
        "ingress_finalize",
        requests.post(
            base + "/fr13/fixed32/ingress/finalize",
            headers=control,
            json={"schema": "fr13-fixed32-ingress-finalize-v1"},
            timeout=60,
        ),
        started,
        attempt=attempt,
    )
    if finalize.status_code == 200:
        ledger = finalize.json()
        break
    if finalize.status_code != 409:
        raise SystemExit(
            f"ingress finalize failed: HTTP {finalize.status_code} "
            f"{finalize.text[:2000]}"
        )
    time.sleep(2)
if ledger is None:
    raise SystemExit("ingress finalize never cleared its active requests")

# Positive proof the drafter actually ran: served text alone cannot distinguish
# a spec-decode forward from a plain one.
started = time.monotonic()
metrics = requests.get(base + "/metrics", timeout=60)
record(
    "metrics_scrape",
    status_code=metrics.status_code,
    elapsed_s=round(time.monotonic() - started, 3),
)
if metrics.status_code != 200:
    raise SystemExit(f"metrics scrape failed: HTTP {metrics.status_code}")
spec = {}
for name in (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
):
    match = re.search(
        "^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$",
        metrics.text,
        re.MULTILINE,
    )
    spec[name] = float(match.group(1)) if match is not None else 0.0
record(
    "spec_decode_counters",
    **{key.replace(":", "_"): value for key, value in spec.items()},
)
if min(spec.values()) <= 0:
    raise SystemExit(f"spec decode never engaged during the smoke: {spec!r}")

started = time.monotonic()
health = observe("post_smoke_health", requests.get(base + "/health", timeout=30), started)
if health.status_code != 200:
    raise SystemExit(
        f"engine is not healthy after the smoke: HTTP {health.status_code}"
    )

Path(record_path).write_text(
    json.dumps(
        {
            "schema": "fr13.fixed32.sfwd_fusion_boot_diag_smoke.v1",
            "classification": "diagnostic_boot_only",
            "citable": False,
            "timing_eligible": False,
            "task_id": task_id,
            "task_key_id": task_key_id,
            "model": model,
            "max_tokens": max_tokens,
            "sampling": sampling,
            "requests_sent": count,
            "responses_ok": served,
            "responses_reasoning_only": reasoning_only,
            "accepted_engine_requests": ledger.get("accepted_engine_requests"),
            "completed_engine_requests": ledger.get("completed_engine_requests"),
            "spec_decode": spec,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="ascii",
)
print(
    f"smoke served {served}/{count} completions"
    f" ({reasoning_only} reasoning-only)"
    f" drafts={spec['vllm:spec_decode_num_drafts_total']}"
    f" draft_tokens={spec['vllm:spec_decode_num_draft_tokens_total']}",
    flush=True,
)
PY
  SMOKE_SENT=$(grep -c '^smoke request ' "$SMOKE_LOG" || true)
  SMOKE_OK=$(grep -c '^smoke response ok ' "$SMOKE_LOG" || true)
  echo "[boot-diag] $(tail -1 "$SMOKE_LOG")"

  # Re-read the container log: everything the measured forwards raised is in it
  # now, and nothing of it existed at the capture-time sweep above.
  docker logs "$CONTAINER_ID" > "$DOCKER_LOG" 2>&1 || true
  echo "[boot-diag] docker log after smoke -> $DOCKER_LOG ($(wc -l < "$DOCKER_LOG") lines)"
  SMOKE_VALIDATORS_CLEAN=true
  for forbidden in \
      "${FR13_FIXED32_SFWD_FUSION_FORBIDDEN[@]}" \
      "${FR13_FIXED32_SFWD_FUSION_CENSUS_FORBIDDEN[@]}"; do
    if grep -Fq -- "$forbidden" "$DOCKER_LOG"; then
      SMOKE_VALIDATORS_CLEAN=false
      VERDICT=smoke_validator_tripped
      DETAIL="measured forward emitted: $forbidden"
      echo "DIAG FAIL: $DETAIL" >&2
      grep -n -F -m 3 -- "$forbidden" "$DOCKER_LOG" >&2 || true
      exit 8
    fi
  done
  unset forbidden
fi

VERDICT=pass
if (( SMOKE_RAN == 1 )); then
  DETAIL="boot and smoke clean: 48/48 SFWD layers, qrow16 FULL capture engaged, $SMOKE_OK/$SMOKE_SENT completions served with no validator string"
else
  DETAIL="boot through FULL capture clean: 48/48 SFWD layers, qrow16 FULL capture engaged"
fi
echo "DIAG PASS: $DETAIL"
exit 0
