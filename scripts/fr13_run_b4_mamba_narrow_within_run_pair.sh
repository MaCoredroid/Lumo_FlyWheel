#!/usr/bin/env bash
# WITHIN-RUN confirmatory pair for FR13_MAMBA_SPEC_BLOCKS_CDIV: exact4 real
# SWE-Verified B4, STOCK CUTLASS on BOTH arms, the lever OFF on one and ON on
# the other. Same wall, same binaries, same subset, same credentials.
#
# WHY THIS EXISTS, AND WHY IT IS NOT fr13_run_b4_cutlass_persistent_m128_timing.sh
# That runner's arms are stock-vs-CANDIDATE: it varies the CUTLASS selector and
# forwards one inherited FR13_MAMBA_SPEC_BLOCKS_CDIV to BOTH arms, by design, so
# the pair isolates the CUTLASS candidate. That makes it structurally unable to
# answer "what does the lever itself do", and the 2026-08-09 evidence for the
# lever's primary claim (APC recovery) was therefore CROSS-RUN -- this pair vs
# the 20260809T064230Z pair -- and so confounded with agent-trajectory variance.
# This driver varies the lever and NOTHING else.
#
# CLASSIFICATION: diagnostic_within_run_lever_pair. It is NOT the qualified
# CUTLASS timing pair and NOT the formal statistical hardware-floor acceptance
# gate. It carries no CUTLASS attestation because both arms are stock. What it
# does carry, and what the citable pair cannot, is a MECHANICAL only-arm-delta
# proof: the two arms' container_env.txt are diffed and the run FAILS unless
# they differ in exactly one variable, FR13_MAMBA_SPEC_BLOCKS_CDIV.
#
# PRIMARY READOUT (in order):
#   1. APC hit rate, campaign window, per arm -- the claim under test
#      (prefix_cache_hits_total / prefix_cache_queries_total, after minus before)
#   2. Maximum concurrency + Running:1 kv floor + kv peak from each boot log
#   3. per-request step TPS parity and events/step, from deploy_speed_fullwall
#
# The env block below is COPIED VERBATIM from that runner's run_arm() with
# production=0 hardwired (selector stock, every candidate pass json empty,
# FORKED_FA2_SO=the exact-safe stock FA2), so both arms boot the same engine the
# stock reference arm of the citable pair boots.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${STOCK_FA2_SO:?set STOCK_FA2_SO to the exact-safe stock FA2 binary}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
FIXED32_MODE=hydra27_fixed32
TIMING_KIND=$FIXED32_MODE
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
DRAFT_VOCAB_ROOT=1
DRAFT_VOCAB_K=65536
NEEDS_ALLOW=
DRAFT_VOCAB_BLOCKS_HOST=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
QUALIFICATION_PROFILE=k64_root
B4_KV_CACHE_MEMORY_BYTES=49392123904
EXPECTED_TOK_PER_DRAFT=31
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
MANDATORY_WEIGHT_BYTES=32666638208
MANDATORY_WEIGHT_FLOOR_MS=119.658015414

RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
SOURCE_COMMIT=$(git rev-parse HEAD)

OFF_ARM="${FIXED32_MODE}_stock_mambacdiv0_${TAG}"
ON_ARM="${FIXED32_MODE}_stock_mambacdiv1_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* && ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be a new path below $REPO/output" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "repository virtual-environment Python is unavailable" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "within-run lever pair requires a clean source tree" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "exact4 subset identity drifted" >&2; exit 2; }
[[ "$(sha256sum "$DRAFT_VOCAB_BLOCKS_HOST" | awk '{print $1}')" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
  || { echo "pinned root-64K block map drifted" >&2; exit 2; }
[[ -f "$STOCK_FA2_SO" && ! -L "$STOCK_FA2_SO" \
   && "$(stat -c '%s' "$STOCK_FA2_SO")" == "$STOCK_FA2_BYTES" \
   && "$(sha256sum "$STOCK_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "STOCK_FA2_SO is not the exact-safe stock FA2 binary" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the pair" >&2; exit 2; }

# The same exported route preamble the citable runner builds before it launches
# an arm. Skipping it is how the first attempt died pre-docker: the launcher
# reads FR13_FIXED32_TAW_WALK_CAP as "${NAME:-}" and int("") raises "fixed32
# integer route pin is malformed". The floor-contract assertion below is the
# runner's, kept so this driver cannot boot a different fixed-work floor.
export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT"
export FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K"
export FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER"
export FR13_NEEDS_ALLOW="$NEEDS_ALLOW"
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" ]] \
  || { echo "canonical B4 qualification floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
printf 'classification=diagnostic_within_run_lever_pair\nciteable_cutlass_timing=0\nformal_floor_acceptance_eligible=0\nonly_arm_delta=FR13_MAMBA_SPEC_BLOCKS_CDIV_0_to_1\nlever=FR13_MAMBA_SPEC_BLOCKS_CDIV\nboth_arms_cutlass=stock\nfixed32_mode=%s\nbatch_size=4\nconcurrency=4\nkv_cache_memory_bytes=%s\nenforce_eager=0\ncudagraph_mode=FULL_AND_PIECEWISE\nqualification_profile=%s\ndraft_vocab_root=%s\ndraft_vocab_k=%s\nsubset=%s\nsubset_sha256=%s\nstock_fa2_sha256=%s\nsource_commit=%s\nrunner_sha256=%s\noff_arm=%s\non_arm=%s\nrunroot=%s\nstarted=%s\n' \
  "$FIXED32_MODE" "$B4_KV_CACHE_MEMORY_BYTES" "$QUALIFICATION_PROFILE" \
  "$DRAFT_VOCAB_ROOT" "$DRAFT_VOCAB_K" "$SUBSET" "$SUBSET_SHA256" \
  "$STOCK_FA2_SHA256" "$SOURCE_COMMIT" "$RUNNER_SHA256" \
  "$OFF_ARM" "$ON_ARM" "$RUNROOT_ABS" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

run_arm() {
  local arm=$1
  local cdiv=$2
  echo "===== $arm: exact4 B4 STOCK CUTLASS, FR13_MAMBA_SPEC_BLOCKS_CDIV=$cdiv ====="
  printf 'arm=%s mamba_spec_blocks_cdiv=%s started=%s\n' \
    "$arm" "$cdiv" "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
  if env \
      RUNROOT="$RUNROOT_ABS" \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S=5400 \
      KV_CACHE_MEMORY_BYTES="$B4_KV_CACHE_MEMORY_BYTES" \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT" \
      FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K" \
      FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER" \
      FR13_NEEDS_ALLOW="$NEEDS_ALLOW" \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
      FR13_SCAN_ALIGN=0 FR13_NPAD_INVARIANT=0 \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json" \
      FR13_FIXED32_CUTLASS_WAVE=stock \
      FR13_FIXED32_CUTLASS_WAVE_SO= \
      FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
      FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_JSON= \
      FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_SHA256= \
      FR13_FIXED32_CUTLASS_STAGE2_TAIL23_LIVE_PASS_JSON= \
      FR13_FIXED32_CUTLASS_STAGE2_TAIL23_LIVE_PASS_SHA256= \
      FR13_FIXED32_CUTLASS_STAGE2_HYDRA27_LIVE_PASS_JSON= \
      FR13_FIXED32_CUTLASS_STAGE2_HYDRA27_LIVE_PASS_SHA256= \
      FR13_FIXED32_CUTLASS_TWOM_TAIL23_LIVE_PASS_JSON= \
      FR13_FIXED32_CUTLASS_TWOM_TAIL23_LIVE_PASS_SHA256= \
      FR13_FIXED32_CUTLASS_TWOM_HYDRA27_LIVE_PASS_JSON= \
      FR13_FIXED32_CUTLASS_TWOM_HYDRA27_LIVE_PASS_SHA256= \
      FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_SOURCE_COMMIT= \
      FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE="$QUALIFICATION_PROFILE" \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON= \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_INSTANCE_ID= \
      FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
      FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
      FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
      FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$STOCK_FA2_SO" \
      FR13_MAMBA_SPEC_BLOCKS_CDIV="$cdiv" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh \
        "$arm" "$TIMING_KIND" "$SUBSET" \
        > "$RUNROOT_ABS/$arm.runlog" 2>&1; then
    :
  else
    local serve_rc=$?
    printf 'arm=%s serve_rc=%s ended=%s\n' \
      "$arm" "$serve_rc" "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
    return "$serve_rc"
  fi
  printf 'arm=%s serve_rc=0 ended=%s\n' "$arm" "$(date -u +%FT%TZ)" \
    >> "$RUNROOT_ABS/launcher_meta.txt"

  local container_env="$RUNROOT_ABS/$arm/container_env.txt"
  [[ -f "$container_env" && ! -L "$container_env" ]] \
    || { echo "$arm lacks a regular container environment artifact" >&2; return 4; }
  [[ "$(grep -Fxc "FR13_MAMBA_SPEC_BLOCKS_CDIV=$cdiv" "$container_env")" -eq 1 ]] \
    || { echo "$arm did not record the lever it was asked to run" >&2; return 4; }
  [[ "$(grep -Fxc "FR13_FIXED32_MODE=$FIXED32_MODE" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_DRAFT_VOCAB_ROOT=$DRAFT_VOCAB_ROOT" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_DRAFT_VOCAB_K=$DRAFT_VOCAB_K" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_DRAFT_VOCAB_BLOCKS=$DRAFT_VOCAB_BLOCKS_CONTAINER" "$container_env")" -eq 1 ]] \
    || { echo "$arm did not run the pinned B4 profile" >&2; return 4; }
  ! grep -RIn 'FR13_MAMBA_SPEC_BLOCKS_CDIV incoherent' "$RUNROOT_ABS/$arm" >/dev/null 2>&1 \
    || { echo "$arm half-applied the lever pair (anchor drift)" >&2; return 4; }

  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$arm" \
    --out-root "$RUNROOT_ABS/$arm/swe_out" \
    --expected-tok-per-draft "$EXPECTED_TOK_PER_DRAFT" \
    --batch-size 4 \
    --work-census "$RUNROOT_ABS/$arm/logs/fr13_fixed32_work_census.jsonl" \
    --out "$RUNROOT_ABS/$arm/deploy_speed_fullwall.json" \
    > "$RUNROOT_ABS/$arm/deploy_speed_fullwall.log" 2>&1 \
    || { echo "$arm deploy-speed reduction failed" >&2;
         tail -20 "$RUNROOT_ABS/$arm/deploy_speed_fullwall.log" >&2; return 5; }
}

run_arm "$OFF_ARM" 0
run_arm "$ON_ARM" 1

# ---- the only-arm-delta proof ---------------------------------------------
# The whole point of this driver. Any second differing variable makes the APC
# delta attributable to something other than the lever, so it is fatal.
"$PYTHON_BIN" - "$RUNROOT_ABS/$OFF_ARM/container_env.txt" \
                "$RUNROOT_ABS/$ON_ARM/container_env.txt" \
                "$RUNROOT_ABS/only_arm_delta.json" <<'PY' \
  || { echo "only-arm-delta proof FAILED" >&2; exit 6; }
import json
import sys
from pathlib import Path

off_path, on_path, out_path = sys.argv[1:4]


def env(path):
    d = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            d[key] = value
    return d


off, on = env(off_path), env(on_path)
keys = sorted(set(off) | set(on))
# Per-arm paths carry the arm name by construction; they are identity, not
# configuration, and are excluded by name rather than by pattern-matching values.
NAME_BEARING = {
    "FR13_SFWD_GPU_TIMER_JSON",
    "FR13_DFWD_GPU_TIMER_JSON",
    "FR13_CFWD_GPU_TIMER_JSON",
}
diff = {
    k: {"off": off.get(k), "on": on.get(k)}
    for k in keys
    if off.get(k) != on.get(k) and k not in NAME_BEARING
}
result = {
    "schema": "fr13.mamba_narrow.within_run_only_arm_delta.v1",
    "differing_keys": sorted(diff),
    "diff": diff,
    "name_bearing_excluded": sorted(NAME_BEARING),
    "n_keys_compared": len(keys),
}
Path(out_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
expected = {"FR13_MAMBA_SPEC_BLOCKS_CDIV"}
if set(diff) != expected:
    raise SystemExit(
        "only-arm-delta violated: arms differ in "
        f"{sorted(set(diff))}, expected exactly {sorted(expected)}"
    )
if (diff["FR13_MAMBA_SPEC_BLOCKS_CDIV"]["off"], diff["FR13_MAMBA_SPEC_BLOCKS_CDIV"]["on"]) != ("0", "1"):
    raise SystemExit("the lever did not read 0 on the OFF arm and 1 on the ON arm")
print("only-arm-delta OK: the arms differ in exactly FR13_MAMBA_SPEC_BLOCKS_CDIV (0 -> 1)")
PY

"$PYTHON_BIN" scripts/fr13_mamba_narrow_within_run_reduce.py \
  --runroot "$RUNROOT_ABS" \
  --off-arm "$OFF_ARM" \
  --on-arm "$ON_ARM" \
  --out "$RUNROOT_ABS/within_run_summary.json"

printf 'within_run_summary=%s completed=%s\n' \
  "$RUNROOT_ABS/within_run_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
echo "within-run lever pair complete -> $RUNROOT_ABS/within_run_summary.json"
