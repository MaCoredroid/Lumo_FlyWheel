#!/usr/bin/env bash
# One real SWE-Verified K64/root1 B1 byte gate for FP8 quant regcache.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path below output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned stock FA2 binary}"
: "${FP8_QUANT_SO:?set FP8_QUANT_SO to the pinned regcache runtime binary}"
: "${FP8_QUANT_SO_SHA256:?set FP8_QUANT_SO_SHA256 to its raw SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}
ARM="hydra27_fixed32_k64_root_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"
TASK_ID=astropy__astropy-12907
PATCH_SOURCE=scripts/fr13_patch_fp8_quant_fixed32.py
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
for required in "$FORKED_FA2_SO" "$FP8_QUANT_SO"; do
  [[ "$required" == /* && -f "$required" && ! -L "$required" ]] \
    || { echo "required binary is not an absolute regular file: $required" >&2; exit 2; }
done
unset required
[[ "$FP8_QUANT_SO_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$FP8_QUANT_SO" | awk '{print $1}')" == "$FP8_QUANT_SO_SHA256" ]] \
  || { echo "FP8 quant binary identity drifted" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" ]] \
  || { echo "canonical B1 task or K64 block map drifted" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the live gate" >&2; exit 2; }

"$PYTHON_BIN" scripts/fr13_fp8_quant_regcache_runtime.py verify-binary \
  "$FP8_QUANT_SO" --expected-sha256 "$FP8_QUANT_SO_SHA256" >/dev/null

export RUNROOT="$RUNROOT_REL"
export FR13_B1_WORKLOAD_PROFILE=k64_root
export FR13_GATE_QROW16=0
export FR13_GATE_TAW_NATIVE=0
export FR13_GATE_DRAFT_HEAD_PAD=0
export FR13_GATE_DRAFT_HEAD_M32=0
export FR13_GATE_DRAFT_HEAD_FP8=0
export FR13_GATE_BM8=0
export FR13_GATE_GDN_BV=0
export FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0
export FR13_FA2_QROW16_PRODUCTION=0
export FR13_DFWD_UNIFIED_BM8_LIVE_AB=0
export FR13_DFWD_UNIFIED_BM8_PRODUCTION=0
export FR13_FIXED32_GDN_PATH_BV_CANDIDATE=
export FR13_FIXED32_GDN_PATH_BV_PRODUCTION=
export FR13_FIXED32_BATCH_GDN_BYTE_AB=0
export FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0
export FR13_FIXED32_BATCH_GDN_BV_CANDIDATE=
export FR13_FIXED32_BATCH_GDN_PRODUCTION=0
export FR13_FIXED32_BATCH_GDN_BV_PRODUCTION=
export FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0
export FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0
export FR13_FIXED32_CUTLASS_WAVE=stock
export FR13_FIXED32_CUTLASS_WAVE_SO=
export FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0
export FR13_FIXED32_B1_FP8_QUANT_REGCACHE=byte_ab
export FR13_FIXED32_B1_FP8_QUANT_REGCACHE_SO="$FP8_QUANT_SO"
export FR13_FIXED32_B1_FP8_QUANT_REGCACHE_SO_SHA256="$FP8_QUANT_SO_SHA256"
export FR13_FIXED32_B1_FP8_QUANT_REGCACHE_PASS_JSON=
export FR13_FIXED32_B1_FP8_QUANT_REGCACHE_PASS_SHA256=
export FR13_FIXED32_ATTRIBUTION_ONLY=0
export FR10_METRICS=0
export ENFORCE_EAGER=1

bash scripts/fr13_run_b1_kernel_live_gate.sh

RECORDS="$ARMDIR/logs/fr13_fixed32_b1_fp8_quant_regcache.byte_ab.jsonl"
BINARY_ATTESTATION="$ARMDIR/logs/fr13_fixed32_b1_fp8_quant_regcache.binary.json"
TASK_ARM="$ARMDIR/swe_out/verified/per_task/$TASK_ID/fixed32_cutlass_streamk_real_task_arm.json"
DIAGNOSTIC="$ARMDIR/fixed32_b1_diagnostic.json"
CONTAINER_ENV="$ARMDIR/container_env.txt"
TERMINAL="$ARMDIR/fixed32_final_flush_skipped.json"
TRAFFIC="$ARMDIR/fixed32_chat_traffic_audit_skipped.json"
LIVE_RESULT="$ARMDIR/fp8_quant_regcache_k64_root_b1_live_pass.json"
PASS_SIDECAR="$ARMDIR/fp8_quant_regcache.production_pass.json"

"$PYTHON_BIN" scripts/fr13_fp8_quant_regcache_pass.py qualify \
  --records "$RECORDS" \
  --binary-attestation "$BINARY_ATTESTATION" \
  --task-arm "$TASK_ARM" \
  --diagnostic "$DIAGNOSTIC" \
  --container-env "$CONTAINER_ENV" \
  --terminal "$TERMINAL" \
  --traffic "$TRAFFIC" \
  --runtime-manifest-launch "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
  --runtime-manifest-end "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  --external-manifest-launch "$RUNROOT_ABS/external_manifest.at_launch.json" \
  --external-manifest-end "$RUNROOT_ABS/external_manifest.at_end.json" \
  --candidate-so "$FP8_QUANT_SO" \
  --expected-candidate-sha256 "$FP8_QUANT_SO_SHA256" \
  --patch-source "$PATCH_SOURCE" \
  --source-commit "$SOURCE_COMMIT" \
  --out-live "$LIVE_RESULT" \
  --out-sidecar "$PASS_SIDECAR" \
  > "$ARMDIR/fp8_quant_regcache_qualification.json"

PASS_SHA256=$(sha256sum "$PASS_SIDECAR" | awk '{print $1}')
"$PYTHON_BIN" scripts/fr13_fp8_quant_regcache_pass.py verify \
  --sidecar "$PASS_SIDECAR" \
  --expected-sidecar-sha256 "$PASS_SHA256" \
  --candidate-so "$FP8_QUANT_SO" \
  --expected-candidate-sha256 "$FP8_QUANT_SO_SHA256" \
  --patch-source "$PATCH_SOURCE" >/dev/null

printf 'classification=one_real_swe_verified_k64_root_b1_fp8_quant_byte_diagnostic\nselector=byte_ab\nreference_returned=1\ntiming_eligible=0\nfloor_acceptance_eligible=0\nsource=%s\nrunner_sha256=%s\ncandidate_sha256=%s\nlive_result_sha256=%s\nproduction_pass_sha256=%s\ncompleted=%s\n' \
  "$SOURCE_COMMIT" "$RUNNER_SHA256" "$FP8_QUANT_SO_SHA256" \
  "$(sha256sum "$LIVE_RESULT" | awk '{print $1}')" "$PASS_SHA256" \
  "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"

printf 'live_result=%s\nproduction_pass=%s\nproduction_pass_sha256=%s\n' \
  "$LIVE_RESULT" "$PASS_SIDECAR" "$PASS_SHA256"
