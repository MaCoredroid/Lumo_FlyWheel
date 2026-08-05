#!/usr/bin/env bash
# Default-off real SWE-Verified exact4 B4 candidate-served DFWD/TAW gate.
set -euo pipefail

case "${FR13_RUN_B4_DFWD_M4_U8_LIVE_GATE:-0}" in
  1) ;;
  0)
    echo "B4 DFWD M4 U8 live gate is disabled" >&2
    echo "set FR13_RUN_B4_DFWD_M4_U8_LIVE_GATE=1 to run it" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_B4_DFWD_M4_U8_LIVE_GATE must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
PATCHER=$(realpath "$SCRIPT_DIR/fr10_phase4_patch_vllm_tree_gdn.py")
GATE=$(realpath "$SCRIPT_DIR/fr13_dfwd_k64_m4_r64_u8_gate.py")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
FORKED_FA2_SO=${FORKED_FA2_SO:-$REPO/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so}
DFWD_M4_U8_SO=${DFWD_M4_U8_SO:-/home/mark/shared/fr13_dfwd_m4_u8_linked_build_bb8a4a8a2_20260805/canonical-primary-bin/fr13_bf16_k64_m4_r64_u8.abi3.so}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SOURCE=csrc/fr13_bf16_gemvx_k64_m4_shuffle_r64_u8.cu
BUILD_ATTESTATION=results/fr13_fixed32_dfwd_k64_m4_r64_u8_linked_build_20260805/build_attestation.json
BLOCKS=scripts/fr13_dvk_subset_blocks.json
TASK_IDS=astropy__astropy-12907,astropy__astropy-13033,astropy__astropy-13236,astropy__astropy-13398
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
SOURCE_SHA256=a52361be1c9052a46509cc230ea320c4beb6d15f261327edc835d8da3ae00d9e
BUILD_SHA256=b31ba7fb24fce81b0dceb97d77134f21107511e97538be15cb778c6ac4da5926
BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
SO_SHA256=6cb24782495ff1c1457ebbf9cbcfcd6ca7b372378d3b435f80054688432a365f
FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
B4_KV_CACHE_MEMORY_BYTES=42949672960
SOURCE_COMMIT=$(git rev-parse HEAD)
TAW_SOURCE_SHA256=$(sha256sum scripts/fr13_device_multidraft_kernel.py | awk '{print $1}')
RUNNER_SHA256=$(sha256sum "$RUNNER" | awk '{print $1}')
PATCHER_SHA256=$(sha256sum "$PATCHER" | awk '{print $1}')
GATE_SHA256=$(sha256sum "$GATE" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
ARM="hydra27_fixed32_dfwd_k64_m4_r64_u8_gate_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* && ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be a new path below $REPO/output" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for binary in "$FORKED_FA2_SO" "$DFWD_M4_U8_SO"; do
  [[ "$binary" == /* && -f "$binary" && ! -L "$binary" ]] \
    || { echo "required binary is not an absolute regular file: $binary" >&2; exit 2; }
done
unset binary
[[ "$(stat -c '%s' "$FORKED_FA2_SO")" == "299183936" \
   && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" == "$FA2_SHA256" \
   && "$(stat -c '%s' "$DFWD_M4_U8_SO")" == "134320" \
   && "$(sha256sum "$DFWD_M4_U8_SO" | awk '{print $1}')" == "$SO_SHA256" \
   && "$(sha256sum "$SOURCE" | awk '{print $1}')" == "$SOURCE_SHA256" \
   && "$(sha256sum "$BUILD_ATTESTATION" | awk '{print $1}')" == "$BUILD_SHA256" \
   && "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$BLOCKS" | awk '{print $1}')" == "$BLOCKS_SHA256" ]] \
  || { echo "DFWD M4 U8 qualification input identity drifted" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(git rev-parse '@{upstream}')" == "$SOURCE_COMMIT" ]] \
  || { echo "source commit must be pushed before the gate" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the gate" >&2; exit 2; }

export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh
unset -f run_variant

mkdir -p "$RUNROOT_ABS"
printf 'classification=real_swe_verified_exact4_b4_dfwd_k64_m4_r64_u8_taw_quality_gate\nacceptance_valid=0\ntiming_eligible=0\nproduction_enabled=0\ncandidate_served=1\ntaw_source_sha256=%s\ntask_ids=%s\nsubset_sha256=%s\nbatch_size=4\nconcurrency=4\nphysical_rows_per_request=32\nphysical_rows_total=128\nlogical_drafts_per_request=27\ndraft_vocab_k=65536\ndraft_vocab_root=1\ncandidate_so_sha256=%s\ncandidate_so_bytes=134320\ncandidate_source_sha256=%s\nbuild_attestation_sha256=%s\npatcher_sha256=%s\nrunner_sha256=%s\ngate_sha256=%s\nstock_fa2_sha256=%s\nsource_commit=%s\nlauncher_pid=%s\nstarted=%s\n' \
  "$TAW_SOURCE_SHA256" "$TASK_IDS" "$SUBSET_SHA256" "$SO_SHA256" "$SOURCE_SHA256" \
  "$BUILD_SHA256" "$PATCHER_SHA256" "$RUNNER_SHA256" "$GATE_SHA256" \
  "$FA2_SHA256" "$SOURCE_COMMIT" "$$" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

if env \
    RUNROOT="$RUNROOT_ABS" \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S= \
    KV_CACHE_MEMORY_BYTES="$B4_KV_CACHE_MEMORY_BYTES" \
    LUMO_SWE_AUTOCOMMIT=0 FR13_FIXED32_B1_DIAGNOSTIC=0 \
    FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_DRAFT_VOCAB_K=65536 FR13_DRAFT_VOCAB_ROOT=1 \
    FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json \
    FR13_DEVICE_MULTIDRAFT=1 \
    FR13_DRAFT_HEAD_M4_R64_U8_LIVE_AB=1 \
    FR13_DRAFT_HEAD_M4_R64_U8_QUALITY_GATE=1 \
    FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION=0 \
    FR13_DRAFT_HEAD_M4_R64_U8_SO="$DFWD_M4_U8_SO" \
    FR13_DRAFT_HEAD_M4_R64_U8_SO_SHA256="$SO_SHA256" \
    FR13_DRAFT_HEAD_M4_R64_U8_SOURCE_SHA256="$SOURCE_SHA256" \
    FR13_DRAFT_HEAD_M4_R64_U8_BUILD_ATTESTATION_SHA256="$BUILD_SHA256" \
    FR13_DRAFT_HEAD_M4_R64_U8_PATCH_SOURCE_SHA256="$PATCHER_SHA256" \
    FR13_DRAFT_HEAD_M4_R64_U8_RUNNER_SHA256="$RUNNER_SHA256" \
    FR13_DRAFT_HEAD_M4_R64_U8_SUBSET_SHA256="$SUBSET_SHA256" \
    FR13_DRAFT_HEAD_M4_R64_U8_VOCAB_BLOCKS_SHA256="$BLOCKS_SHA256" \
    FR13_DRAFT_HEAD_M4_R64_U8_FA2_SHA256="$FA2_SHA256" \
    FR13_DRAFT_HEAD_M4_R64_U8_TAW_SOURCE_SHA256="$TAW_SOURCE_SHA256" \
    FR13_DRAFT_HEAD_M4_R64_U8_SOURCE_COMMIT="$SOURCE_COMMIT" \
    FR13_DRAFT_HEAD_M4_R64_U8_TASK_IDS="$TASK_IDS" \
    FR13_DRAFT_HEAD_M4_R64_U8_LIVE_JSON=/logs/fr13_dfwd_k64_m4_r64_u8.live.json \
    FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB=0 \
    FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION=0 \
    FR13_DRAFT_HEAD_M1_R64_U8_SO= \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_FP8=0 FR13_DRAFT_HEAD_FP8_STATIC_IO=0 \
    FR13_DFWD_K64_TOP3=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=1 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
    FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_SO= \
    FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FA2_QROW32_LIVE_PAGED_AB=0 FR13_FA2_QROW32_LIVE_PAGED_AB_ARM= \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 \
    FORKED_FA2_SO="$FORKED_FA2_SO" \
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
  || { echo "runtime/source manifest changed during gate" >&2; exit 14; }
cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
  "$RUNROOT_ABS/external_manifest.at_end.json" \
  || { echo "external manifest changed during gate" >&2; exit 14; }
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "$RUNNER_SHA256" \
   && "$(sha256sum "$PATCHER" | awk '{print $1}')" == "$PATCHER_SHA256" \
   && "$(sha256sum "$GATE" | awk '{print $1}')" == "$GATE_SHA256" \
   && "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" \
   && "$(git rev-parse '@{upstream}')" == "$SOURCE_COMMIT" \
   && -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "frozen pushed source changed during gate" >&2; exit 14; }
(( serve_rc == 0 )) || exit "$serve_rc"

LIVE_RESULT="$ARMDIR/logs/fr13_dfwd_k64_m4_r64_u8.live.json"
FINAL_FLUSH="$ARMDIR/fixed32_final_flush.json"
TRAFFIC_AUDIT="$ARMDIR/fixed32_chat_traffic_audit.json"
FLUSH_GENERATION=$("$PYTHON_BIN" - "$FINAL_FLUSH" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
generation = payload.get("ack", {}).get("generation")
if type(generation) is not int or generation < 1:
    raise SystemExit("final flush lacks a valid generation")
print(generation)
PY
)
BOUNDARY="$ARMDIR/logs/fr13_fixed32_boundary_snapshot.${FLUSH_GENERATION}.json"
GATE_RESULT="$ARMDIR/dfwd_k64_m4_r64_u8_real_b4_gate.json"
for artifact in "$LIVE_RESULT" "$FINAL_FLUSH" "$TRAFFIC_AUDIT" "$BOUNDARY"; do
  [[ -f "$artifact" && ! -L "$artifact" ]] \
    || { echo "gate artifact is missing or unsafe: $artifact" >&2; exit 4; }
done

"$PYTHON_BIN" "$GATE" \
  --live-result "$LIVE_RESULT" \
  --candidate-so "$DFWD_M4_U8_SO" \
  --candidate-source "$SOURCE" \
  --build-attestation "$BUILD_ATTESTATION" \
  --patch-source "$PATCHER" \
  --runner "$RUNNER" \
  --subset "$SUBSET" \
  --vocab-blocks "$BLOCKS" \
  --fa2-so "$FORKED_FA2_SO" \
  --taw-source scripts/fr13_device_multidraft_kernel.py \
  --expected-source-commit "$SOURCE_COMMIT" \
  --final-flush "$FINAL_FLUSH" \
  --boundary-snapshot "$BOUNDARY" \
  --chat-traffic-audit "$TRAFFIC_AUDIT" \
  --repo "$REPO" \
  --out "$GATE_RESULT" \
  > "$ARMDIR/dfwd_k64_m4_r64_u8_gate_reduction.json"

printf 'gate_result=%s\ngate_result_sha256=%s\nstatus=PASS\n' \
  "$GATE_RESULT" "$(sha256sum "$GATE_RESULT" | awk '{print $1}')" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
