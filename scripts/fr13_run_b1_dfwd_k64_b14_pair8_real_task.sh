#!/usr/bin/env bash
# Real SWE-Verified B1 diagnostic serving the fixed32 K64 pair8 draft head.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

case "${FR13_RUN_B1_DFWD_K64_PAIR8_REAL_TASK:-0}" in
  1) ;;
  0)
    echo "pair8 real-task diagnostic is disabled; set FR13_RUN_B1_DFWD_K64_PAIR8_REAL_TASK=1" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_B1_DFWD_K64_PAIR8_REAL_TASK must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
FIXED32_MODE=hydra27_fixed32
TASK_ID=astropy__astropy-12907
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
PAIR8_SOURCE=csrc/fr13_bf16_gemvx_k64_b14_warp4_pair8.cu
PAIR8_SOURCE_SHA256=01fd0ec526afcfa3e3a4a42ec95b95dfa04d566fa8431eee84d72d7eada9ed3f
PAIR8_ARTIFACT=results/fr13_fixed32_dfwd_k64_b14_warp4_pair8_sm121a_20260805
PAIR8_SO=$PAIR8_ARTIFACT/fr13_bf16_k64_b14_warp4_pair8.abi3.so
PAIR8_SO_SHA256=dcd9301f447d1ef4a81240027af3a008558dfa080b8660ac9dba7b2c5800ec7f
PAIR8_SO_BYTES=161056
PAIR8_BUILD=$PAIR8_ARTIFACT/build_attestation.json
PAIR8_BUILD_SHA256=3ca9dda783db5fb8a0cabd05ef9671f6a6d2063c8cc979d4c4694d9530dc848f
PAIR8_MANIFEST=$PAIR8_ARTIFACT/manifest.json
PAIR8_MANIFEST_SHA256=577e780472bbc0af53217f5789bb38145bee22b7e9fc962ecc1a1d58f46b4e3e
PAIR8_SELECTOR=scripts/fr13_dfwd_k64_b14_pair8_selector.py
PAIR8_SELECTOR_SHA256=85c451bed87421c750ed0a24ec35b6c44193bdbeec0ebb5952c7686abcc9e0f5
FORKED_FA2_SO=${FORKED_FA2_SO:-$REPO/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so}
FORKED_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
FORKED_FA2_BYTES=299183936
MANDATORY_WEIGHT_BYTES=32666638208
MANDATORY_WEIGHT_FLOOR_MS=119.658015414
ONE_SIDED_U95_CAP_MS=137.6067177261
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
SOURCE_COMMIT=$(git rev-parse --verify HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}
ARM="hydra27_fixed32_k64_pair8_b1_real_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* \
   && ! -e "$RUNROOT_ABS" \
   && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be a new path below $REPO/output" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ \
   && -z "$(git status --porcelain=v1 --untracked-files=no)" \
   && "$(git rev-parse '@{upstream}')" == "$SOURCE_COMMIT" ]] \
  || { echo "pair8 real task requires a clean source commit pushed to upstream" >&2; exit 2; }

for binding in \
  "$SUBSET:$SUBSET_SHA256" \
  "$BLOCK_MAP:$BLOCK_MAP_SHA256" \
  "$PAIR8_SOURCE:$PAIR8_SOURCE_SHA256" \
  "$PAIR8_SO:$PAIR8_SO_SHA256" \
  "$PAIR8_BUILD:$PAIR8_BUILD_SHA256" \
  "$PAIR8_MANIFEST:$PAIR8_MANIFEST_SHA256" \
  "$PAIR8_SELECTOR:$PAIR8_SELECTOR_SHA256"; do
  path=${binding%%:*}
  expected=${binding#*:}
  [[ -f "$path" && ! -L "$path" \
     && "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] \
    || { echo "pair8 real-task input identity drifted: $path" >&2; exit 2; }
done
unset binding path expected
[[ "$(stat -c '%s' "$PAIR8_SO")" == "$PAIR8_SO_BYTES" ]] \
  || { echo "pair8 candidate binary size drifted" >&2; exit 2; }
[[ "$FORKED_FA2_SO" == /* \
   && -f "$FORKED_FA2_SO" \
   && ! -L "$FORKED_FA2_SO" \
   && "$(stat -c '%s' "$FORKED_FA2_SO")" == "$FORKED_FA2_BYTES" \
   && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" == "$FORKED_FA2_SHA256" ]] \
  || { echo "FORKED_FA2_SO is not the pinned stock FA2 binary" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the real task" >&2; exit 2; }

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER"
export FR13_NEEDS_ALLOW=
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_DRAFT_VOCAB_ROOT" == "1" \
   && "$FR13_DRAFT_VOCAB_K" == "65536" \
   && "$FR13_DRAFT_VOCAB_BLOCKS" == "$BLOCK_MAP_CONTAINER" \
   && "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" \
   && "$LUMO_SWE_AUTOCOMMIT" == "0" ]] \
  || { echo "K64 ROOT=1 B1 floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS/sidecars"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"
printf 'classification=one_real_swe_verified_b1_hydra27_fixed32_k64_pair8_diagnostic\nacceptance_valid=0\ntiming_eligible=0\nfloor_acceptance_eligible=0\ncandidate_served=1\ntarget_authority_unchanged=1\nmode=%s\ntask_count=1\ntask_id=%s\nbatch_size=1\nconcurrency=1\nphysical_rows=32\nlogical_active_nodes=27\ndraft_vocab_root=1\ndraft_vocab_k=65536\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nblock_map_sha256=%s\npair8_source_sha256=%s\npair8_so_sha256=%s\npair8_so_bytes=%s\npair8_build_sha256=%s\npair8_manifest_sha256=%s\npair8_selector_sha256=%s\nfa2_sha256=%s\nstarted=%s\n' \
  "$FIXED32_MODE" "$TASK_ID" "$MANDATORY_WEIGHT_BYTES" \
  "$MANDATORY_WEIGHT_FLOOR_MS" "$ONE_SIDED_U95_CAP_MS" \
  "$SOURCE_COMMIT" "$RUNNER_SHA256" "$SUBSET_SHA256" "$BLOCK_MAP_SHA256" \
  "$PAIR8_SOURCE_SHA256" "$PAIR8_SO_SHA256" "$PAIR8_SO_BYTES" \
  "$PAIR8_BUILD_SHA256" "$PAIR8_MANIFEST_SHA256" "$PAIR8_SELECTOR_SHA256" \
  "$FORKED_FA2_SHA256" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

MANIFEST_FINALIZED=0
finalize_manifests() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json" || return $?
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json" || return $?
  cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || { echo "runtime/source manifest changed during pair8 real task" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during pair8 real task" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
    || { echo "pair8 real-task runner changed during execution" >&2; return 14; }
  MANIFEST_FINALIZED=1
}
runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    if finalize_manifests; then :; else
      local manifest_rc=$?
      (( rc == 0 )) && rc=$manifest_rc
    fi
  fi
  exit "$rc"
}
trap runner_exit EXIT

if env \
    RUNROOT="$RUNROOT_ABS" \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
    LUMO_SWE_AUTOCOMMIT=0 GPU_UTIL=0.70 \
    FR13_FIXED32_B1_DIAGNOSTIC=1 FR13_B1_DIAGNOSTIC_TASK_PROFILE=astropy12907 \
    FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
    FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER" FR13_NEEDS_ALLOW= \
    FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES" \
    FR13_WEIGHT_FLOOR_MS="$MANDATORY_WEIGHT_FLOOR_MS" \
    FR10_METRICS=1 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
    FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
    FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${ARM}.json" \
    FR13_DFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${ARM}_dfwd.json" \
    FR13_CFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${ARM}_cfwd.json" \
    FR13_DEVICE_MULTIDRAFT=1 \
    FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
    FR13_DRAFT_HEAD_B14_WARP4_PAIR8=1 \
    FR13_DRAFT_HEAD_B14_WARP4_PAIR8_SOURCE_COMMIT="$SOURCE_COMMIT" \
    FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_DRAFT_HEAD_M32_TIMING_ARM=0 \
    FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB=0 \
    FR13_DRAFT_HEAD_M1_R64_U8_QUALITY_GATE=0 \
    FR13_DRAFT_HEAD_M1_R64_U8_TAW_QUALITY_GATE=0 \
    FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION=0 \
    FR13_DRAFT_HEAD_M4_R64_U8_LIVE_AB=0 \
    FR13_DRAFT_HEAD_M4_R64_U8_QUALITY_GATE=0 \
    FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION=0 \
    FR13_DRAFT_HEAD_FP8=0 FR13_DRAFT_HEAD_FP8_STATIC_IO=0 \
    FR13_DRAFT_HEAD_FP8_ARM= FR13_DFWD_K64_TOP3=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_SO= \
    FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FA2_QROW32_LIVE_PAGED_AB=0 FR13_FA2_QROW32_PRODUCTION=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
    FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0 \
    FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=0 \
    FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0 FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0 \
    FR13_CFWD_PACKED_WALK_NODE_TRUST_BYTE_AB=0 \
    FR13_CFWD_PACKED_WALK_NODE_TRUST_PRODUCTION=0 \
    FR13_CFWD_PACKED_WALK_ACTIVE_DEPTH_BYTE_AB=0 \
    FR13_FIXED32_COMMITTER_LAYER_BATCH_QUALIFICATION=0 \
    FR13_FIXED32_COMMITTER_LAYER_BATCH=0 \
    FR13_FIXED32_COMMITTER_METADATA_FUSION=0 \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 \
    FORKED_FA2_SO="$FORKED_FA2_SO" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" "$FIXED32_MODE" "$SUBSET" \
      > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi
printf 'serve_rc=%s ended=%s\n' "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
(( serve_rc == 0 )) || exit "$serve_rc"

[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after the pair8 real task" >&2; exit 2; }
CONTAINER_ENV="$ARMDIR/container_env.txt"
RUNTIME_LOG="$ARMDIR/docker_after_tasks.log"
DIAGNOSTIC="$ARMDIR/fixed32_b1_diagnostic.json"
for evidence in "$CONTAINER_ENV" "$RUNTIME_LOG" "$DIAGNOSTIC"; do
  [[ -f "$evidence" && ! -L "$evidence" ]] \
    || { echo "pair8 real-task evidence is missing: $evidence" >&2; exit 4; }
done
unset evidence
for expected in \
  'FR13_FIXED32_MODE=hydra27_fixed32' \
  'FR13_FIXED32_B1_DIAGNOSTIC=1' \
  'FR13_DRAFT_VOCAB_ROOT=1' \
  'FR13_DRAFT_VOCAB_K=65536' \
  'MAX_NUM_SEQS=1' \
  'SWE_CONCURRENCY=1' \
  'FR13_DRAFT_HEAD_B14_WARP4_PAIR8=1' \
  "FR13_DRAFT_HEAD_B14_WARP4_PAIR8_SOURCE_COMMIT=$SOURCE_COMMIT"; do
  [[ "$(grep -Fxc "$expected" "$CONTAINER_ENV")" -eq 1 ]] \
    || { echo "container lacks exact pair8 B1 pin: $expected" >&2; exit 4; }
done
unset expected
grep -F '[FR13_DRAFT_HEAD_B14_WARP4_PAIR8] ready batch=1' "$RUNTIME_LOG" >/dev/null \
  || { echo "pair8 readiness marker is missing" >&2; exit 4; }
grep -F '[FR13_DRAFT_HEAD_B14_WARP4_PAIR8] engaged batch=1 candidate_served=1' "$RUNTIME_LOG" >/dev/null \
  || { echo "pair8 candidate-served marker is missing" >&2; exit 4; }
grep -F '[FR13_DRAFT_HEAD_B14_WARP4_PAIR8] graph captured_calls=4' "$RUNTIME_LOG" >/dev/null \
  || { echo "pair8 full-graph marker is missing" >&2; exit 4; }

"$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
  --arm "$ARM" \
  --out-root "$ARMDIR/swe_out" \
  --expected-tok-per-draft 31 \
  --batch-size 1 \
  --out "$ARMDIR/deploy_speed_fullwall.json"
"$PYTHON_BIN" - \
  "$DIAGNOSTIC" "$ARMDIR/deploy_speed_fullwall.json" "$RUNTIME_LOG" \
  "$SOURCE_COMMIT" "$PAIR8_SO_SHA256" "$ARMDIR/pair8_real_task_summary.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

diagnostic_path, speed_path, runtime_log_path = map(Path, sys.argv[1:4])
source_commit, candidate_sha256, output_name = sys.argv[4:]
diagnostic = json.loads(diagnostic_path.read_text(encoding="ascii"))
speed = json.loads(speed_path.read_text(encoding="ascii"))
if (
    diagnostic.get("schema") != "fr13-fixed32-b1-diagnostic-v1"
    or diagnostic.get("task_ids") != ["astropy__astropy-12907"]
    or diagnostic.get("timing_eligible") is not False
    or diagnostic.get("floor_acceptance_eligible") is not False
    or speed.get("schema") != "fr13.measure.deploy_speed.v1"
    or speed.get("n_tasks") != 1
    or speed.get("task_instance_ids") != ["astropy__astropy-12907"]
    or speed.get("batch_size") != 1
    or speed.get("draft_vocab_root") != 1
    or speed.get("draft_vocab_k") != 65536
    or speed.get("engagement", {}).get("engaged") is not True
):
    raise SystemExit("pair8 real-task measurement binding drifted")
payload = {
    "schema": "fr13.fixed32.dfwd_k64_b14_pair8_real_task.v1",
    "status": "PASS",
    "suite": "SWE-Verified",
    "task_ids": speed["task_instance_ids"],
    "batch_size": 1,
    "physical_rows": 32,
    "active_nodes": 27,
    "draft_vocab_root": 1,
    "draft_vocab_k": 65536,
    "candidate_served": True,
    "proposal_only": True,
    "target_authority_unchanged": True,
    "acceptance_valid": False,
    "timing_eligible": False,
    "floor_acceptance_eligible": False,
    "source_commit": source_commit,
    "candidate_so_sha256": candidate_sha256,
    "runtime_log_sha256": hashlib.sha256(runtime_log_path.read_bytes()).hexdigest(),
    "measurement_sha256": hashlib.sha256(speed_path.read_bytes()).hexdigest(),
    "measurement": {
        key: speed.get(key)
        for key in (
            "step_wall_ms",
            "measured_tps_fullstep_wall",
            "s_per_fwd_gpu_per_forward",
            "drafter_gpu_ms_per_step",
            "committer_gpu_ms_per_step",
            "overhead_other_ms_per_event",
            "accept_per_event",
            "committed_per_event",
            "floor_ratio",
        )
    },
}
Path(output_name).write_text(
    json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n",
    encoding="ascii",
)
print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
PY

finalize_manifests
printf 'summary=%s completed=%s\n' \
  "$ARMDIR/pair8_real_task_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
trap - EXIT
