#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the canonical FA2 shared object}"
: "${FR13_DRAFT_HEAD_M1_SO:?set FR13_DRAFT_HEAD_M1_SO to the candidate SO}"
: "${FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION:?set the pinned build attestation}"
: "${FR13_DRAFT_HEAD_M1_EXPECTED_SOURCE_COMMIT:?pin the exact source commit}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
INSTANCE_ID=astropy__astropy-12907
CANONICAL_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
CANONICAL_FA2_SIZE=299183936
CANDIDATE_SOURCE=csrc/fr13_bf16_gemvx_m1.cu
PATCHER=scripts/fr10_phase4_patch_vllm_tree_gdn.py
VALIDATOR=scripts/fr13_draft_head_m1_validate.py
RUNROOT_ABS=$(realpath -m "$RUNROOT")

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT" == output/* && "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must be repo-relative below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | cut -d' ' -f1)" == "$SUBSET_SHA256" ]] \
  || { echo "canonical real-B1 subset SHA-256 drifted" >&2; exit 2; }
for path in \
  "$FORKED_FA2_SO" \
  "$FR13_DRAFT_HEAD_M1_SO" \
  "$FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION"; do
  [[ -f "$path" && ! -L "$path" && "$path" == /* ]] \
    || { echo "binary must be an absolute regular non-symlink: $path" >&2; exit 2; }
done
[[ "$(stat -c %s "$FORKED_FA2_SO")" == "$CANONICAL_FA2_SIZE" \
   && "$(sha256sum "$FORKED_FA2_SO" | cut -d' ' -f1)" \
      == "$CANONICAL_FA2_SHA256" ]] \
  || { echo "real-B1 M1 gate requires the canonical FA2 binary identity" >&2; exit 2; }

ARM="hydra27_fixed32_${TAG}"
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | cut -d' ' -f1)
CANDIDATE_SOURCE_SHA256=$(sha256sum "$CANDIDATE_SOURCE" | cut -d' ' -f1)
PATCHER_SHA256=$(sha256sum "$PATCHER" | cut -d' ' -f1)
CANDIDATE_SO_SHA256=$(sha256sum "$FR13_DRAFT_HEAD_M1_SO" | cut -d' ' -f1)
CANDIDATE_SO_SIZE=$(stat -c %s "$FR13_DRAFT_HEAD_M1_SO")
BUILD_ATTESTATION_SHA256=$(
  sha256sum "$FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION" | cut -d' ' -f1
)
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ \
   && "$FR13_DRAFT_HEAD_M1_EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ \
   && "$SOURCE_COMMIT" == "$FR13_DRAFT_HEAD_M1_EXPECTED_SOURCE_COMMIT" \
   && "$CANDIDATE_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$PATCHER_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$BUILD_ATTESTATION_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$CANDIDATE_SO_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$CANDIDATE_SO_SIZE" -gt 0 ]]
[[ -z "$(git status --porcelain=v1)" ]] \
  || { echo "real-B1 M1 gate requires a clean tracked checkout" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "real-B1 M1 gate requires no existing Docker containers" >&2; exit 2; }

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=0
export FR13_DRAFT_VOCAB_K=0
export FR13_DRAFT_VOCAB_BLOCKS=
export FR13_MANDATORY_WEIGHT_BYTES=42025179008
export FR13_WEIGHT_FLOOR_MS=153.938384645
export FR13_WEIGHT_FLOOR_SCOPE="five full-vocabulary drafter-head reads"
export FR13_FLOOR_ORDER=TH

source scripts/fr13_canonical_env.sh
# canonical_env normally selects a 64K draft subset; this candidate reads the
# actual full-vocabulary head on every one of the five draft-head calls.
export FR13_DRAFT_VOCAB_ROOT=0
export FR13_DRAFT_VOCAB_K=0
export FR13_DRAFT_VOCAB_BLOCKS=
export FR13_MANDATORY_WEIGHT_BYTES=42025179008
export FR13_WEIGHT_FLOOR_MS=153.938384645
export FR13_WEIGHT_FLOOR_SCOPE="five full-vocabulary drafter-head reads"
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh
export FR13_DRAFT_VOCAB_ROOT=0
export FR13_DRAFT_VOCAB_K=0
export FR13_DRAFT_VOCAB_BLOCKS=
export FR13_MANDATORY_WEIGHT_BYTES=42025179008
export FR13_WEIGHT_FLOOR_MS=153.938384645
export FR13_WEIGHT_FLOOR_SCOPE="five full-vocabulary drafter-head reads"

mkdir -p "$RUNROOT_ABS"
"$PYTHON_BIN" "$VALIDATOR" validate-build \
  --build-attestation "$FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION" \
  --expected-build-attestation-sha256 "$BUILD_ATTESTATION_SHA256" \
  --candidate-source "$CANDIDATE_SOURCE" \
  --expected-candidate-source-sha256 "$CANDIDATE_SOURCE_SHA256" \
  --candidate-so "$FR13_DRAFT_HEAD_M1_SO" \
  --expected-candidate-so-sha256 "$CANDIDATE_SO_SHA256" \
  > "$RUNROOT_ABS/build_validation.json"
printf 'classification=real_swe_verified_b1_kernel_byte_diagnostic\ndiagnostic_only=1\nperformance_measurement=0\nprobe_eligible=0\nfloor_acceptance_eligible=0\nlauncher_pid=%s\nrunroot=%s\narm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\ncandidate_source_sha256=%s\npatcher_sha256=%s\nbuild_attestation_sha256=%s\ncandidate_so_sha256=%s\ncandidate_so_bytes=%s\nfa2_sha256=%s\nstarted=%s\n' \
  "$$" "$RUNROOT_ABS" "$ARM" "$SOURCE_COMMIT" "$RUNNER_SHA256" \
  "$SUBSET_SHA256" "$CANDIDATE_SOURCE_SHA256" "$PATCHER_SHA256" \
  "$BUILD_ATTESTATION_SHA256" \
  "$CANDIDATE_SO_SHA256" "$CANDIDATE_SO_SIZE" \
  "$CANONICAL_FA2_SHA256" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

if OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
  FR13_FIXED32_B1_DIAGNOSTIC=1 \
  FR13_DEVICE_MULTIDRAFT=1 \
  FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
  FR13_SFWD_GPU_TIMER=1 \
  FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT/sidecars/$ARM.json" \
  FR13_DFWD_GPU_TIMER=1 \
  FR13_DFWD_GPU_TIMER_JSON="/workspace/$RUNROOT/sidecars/${ARM}_dfwd.json" \
  FR13_CFWD_GPU_TIMER=1 \
  FR13_CFWD_GPU_TIMER_JSON="/workspace/$RUNROOT/sidecars/${ARM}_cfwd.json" \
  FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
  FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
  FR13_DRAFT_HEAD_PAD_ROWS=0 \
  FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
  FR13_DRAFT_HEAD_M32_LIVE_AB=0 \
  FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
  FR13_DRAFT_HEAD_M32_TIMING_ARM=0 \
  FR13_DRAFT_HEAD_M1_LIVE_AB=1 \
  FR13_DRAFT_HEAD_M1_PRODUCTION=0 \
  FR13_DRAFT_HEAD_M1_TIMING_ARM=0 \
  FR13_DRAFT_HEAD_M1_INSTANCE_ID="$INSTANCE_ID" \
  FR13_DRAFT_HEAD_M1_LIVE_JSON=/logs/fr13_draft_head_m1.live.json \
  FR13_DRAFT_HEAD_M1_SO="$FR13_DRAFT_HEAD_M1_SO" \
  FR13_DRAFT_HEAD_M1_SO_SHA256="$CANDIDATE_SO_SHA256" \
  FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION="$FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION" \
  FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION_SHA256="$BUILD_ATTESTATION_SHA256" \
  FR13_FA2_QROW16_LIVE_PAGED_AB=0 \
  FR13_FA2_QROW16_PRODUCTION=0 \
  FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 \
  FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
  FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
  FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
  FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
  FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
  FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
  FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
  FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
  FR13_FIXED32_CUTLASS_WAVE=stock \
  FR13_FIXED32_CUTLASS_WAVE_SO= \
  FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
  FR13_FIXED32_ATTRIBUTION_ONLY=0 \
  FR13_NEEDS_ALLOW="FR13_DRAFT_VOCAB_K=0" \
  FORKED_FA2_SO="$FORKED_FA2_SO" \
  FR13_FA2_QROW16_SO_SHA256="$CANONICAL_FA2_SHA256" \
  bash scripts/fr13_bigdenom_swe_serve_variant.sh \
    "$ARM" hydra27_fixed32 "$SUBSET" \
    > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi

printf 'serve_rc=%s ended=%s\n' "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json"
cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
  "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  || { echo "runtime/source manifest changed during the B1 gate" >&2; exit 14; }
cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
  "$RUNROOT_ABS/external_manifest.at_end.json" \
  || { echo "external manifest changed during the B1 gate" >&2; exit 14; }
[[ "$(sha256sum "$RUNNER_PATH" | cut -d' ' -f1)" == "$RUNNER_SHA256" \
   && "$(sha256sum "$SUBSET" | cut -d' ' -f1)" == "$SUBSET_SHA256" \
   && "$(sha256sum "$CANDIDATE_SOURCE" | cut -d' ' -f1)" == "$CANDIDATE_SOURCE_SHA256" \
   && "$(sha256sum "$PATCHER" | cut -d' ' -f1)" == "$PATCHER_SHA256" \
   && "$(sha256sum "$FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION" | cut -d' ' -f1)" == "$BUILD_ATTESTATION_SHA256" \
   && "$(sha256sum "$FR13_DRAFT_HEAD_M1_SO" | cut -d' ' -f1)" == "$CANDIDATE_SO_SHA256" \
   && "$(sha256sum "$FORKED_FA2_SO" | cut -d' ' -f1)" == "$CANONICAL_FA2_SHA256" ]] \
  || { echo "B1 gate input identity changed during execution" >&2; exit 14; }
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" \
   && "$SOURCE_COMMIT" == "$FR13_DRAFT_HEAD_M1_EXPECTED_SOURCE_COMMIT" \
   && -z "$(git status --porcelain=v1)" ]] \
  || { echo "B1 gate source checkout changed during execution" >&2; exit 14; }
(( serve_rc == 0 )) || exit "$serve_rc"

LIVE="$RUNROOT_ABS/$ARM/logs/fr13_draft_head_m1.live.json"
FINAL_FLUSH="$RUNROOT_ABS/$ARM/fixed32_final_flush.json"
[[ -f "$LIVE" && ! -L "$LIVE" \
   && -f "$FINAL_FLUSH" && ! -L "$FINAL_FLUSH" ]] \
  || { echo "draft-head M1 live or final-flush evidence is missing" >&2; exit 4; }
FLUSH_GENERATION=$(
  "$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["ack"]["generation"])' \
    "$FINAL_FLUSH"
)
BOUNDARY="$RUNROOT_ABS/$ARM/logs/fr13_fixed32_boundary_snapshot.${FLUSH_GENERATION}.json"
TRAFFIC_AUDIT="$RUNROOT_ABS/$ARM/fixed32_chat_traffic_audit.json"
[[ -f "$BOUNDARY" && ! -L "$BOUNDARY" \
   && -f "$TRAFFIC_AUDIT" && ! -L "$TRAFFIC_AUDIT" ]] \
  || { echo "draft-head M1 boundary or traffic evidence is missing" >&2; exit 4; }

"$PYTHON_BIN" "$VALIDATOR" validate-live \
  --live-result "$LIVE" \
  --expected-live-sha256 "$(sha256sum "$LIVE" | cut -d' ' -f1)" \
  --final-flush "$FINAL_FLUSH" \
  --boundary-snapshot "$BOUNDARY" \
  --chat-traffic-audit "$TRAFFIC_AUDIT" \
  --candidate-source "$CANDIDATE_SOURCE" \
  --expected-candidate-source-sha256 "$CANDIDATE_SOURCE_SHA256" \
  --patcher "$PATCHER" \
  --expected-patcher-sha256 "$PATCHER_SHA256" \
  --build-attestation "$FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION" \
  --expected-build-attestation-sha256 "$BUILD_ATTESTATION_SHA256" \
  --candidate-so "$FR13_DRAFT_HEAD_M1_SO" \
  --expected-candidate-so-sha256 "$CANDIDATE_SO_SHA256" \
  > "$RUNROOT_ABS/$ARM/draft_head_m1_live_validation.json"
printf 'live_validation_sha256=%s\n' \
  "$(sha256sum "$RUNROOT_ABS/$ARM/draft_head_m1_live_validation.json" | cut -d' ' -f1)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
