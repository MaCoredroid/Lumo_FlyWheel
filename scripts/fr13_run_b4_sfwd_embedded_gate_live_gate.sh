#!/usr/bin/env bash
# Authenticated SWE-Verified exact4 B4 byte gate for the 40-CTA SFWD schedule.
# The candidate is shadow-only; incumbent tensors and stock attention are served.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned stock FA2 shared object}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
BLOCKS=scripts/fr13_dvk_subset_blocks.json
BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
B4_KV_CACHE_MEMORY_BYTES=49392123904
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
ARM="hydra27_fixed32_k64_sfwd_embedded_gate_b4_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"
MANIFEST_LAUNCH="$RUNROOT_ABS/sfwd_embedded_gate_source_manifest.at_launch.json"
MANIFEST_END="$RUNROOT_ABS/sfwd_embedded_gate_source_manifest.at_end.json"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ "$FORKED_FA2_SO" == /* && -f "$FORKED_FA2_SO" && ! -L "$FORKED_FA2_SO" \
   && "$(stat -c '%s' "$FORKED_FA2_SO")" == "$STOCK_FA2_BYTES" \
   && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "FORKED_FA2_SO is not the pinned stock reference" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$BLOCKS" | awk '{print $1}')" == "$BLOCKS_SHA256" ]] \
  || { echo "exact4 subset or K64 block-map identity drifted" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" \
   && "$SOURCE_COMMIT" == "$(git rev-parse '@{upstream}')" ]] \
  || { echo "source commit must be clean and pushed" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the gate" >&2; exit 2; }

export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh
unset -f run_variant

mkdir -p "$RUNROOT_ABS"
"$PYTHON_BIN" scripts/fr13_sfwd_conv_postprep_gate.py source-manifest \
  --repo "$REPO" --source-commit "$SOURCE_COMMIT" --output "$MANIFEST_LAUNCH"
MANIFEST_SHA256=$(sha256sum "$MANIFEST_LAUNCH" | awk '{print $1}')
MANIFEST_CONTAINER="/workspace/${MANIFEST_LAUNCH#"$REPO/"}"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$REPO" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$REPO" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

if env \
    RUNROOT="$RUNROOT_ABS" \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S= \
    KV_CACHE_MEMORY_BYTES="$B4_KV_CACHE_MEMORY_BYTES" LUMO_SWE_AUTOCOMMIT=0 \
    FR13_FIXED32_B1_DIAGNOSTIC=0 FR10_METRICS=0 \
    ENFORCE_EAGER=1 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
    FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json \
    FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 FR13_TREE_RUNROW_INIT=1 \
    FR13_TREE_CONV_FUSED=1 FR13_CONV_WB_BATCHED=1 \
    FR13_FIXED32_CONV_SOURCE_BATCH=0 \
    FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0 \
    FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=1 \
    FR13_FIXED32_SFWD_EMBED_GATE_CTA=1 \
    FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_JSON= \
    FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_SHA256= \
    FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_PATH="$MANIFEST_CONTAINER" \
    FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256="$MANIFEST_SHA256" \
    FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_COMMIT="$SOURCE_COMMIT" \
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_TIMING_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
    FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB=0 \
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
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FA2_QROW16_SO_SHA256= FR13_FA2_QROW16_LIVE_PASS_SHA256= \
    FR13_FA2_QROW32_LIVE_PAGED_AB=0 FR13_FA2_QROW32_LIVE_PAGED_AB_ARM= \
    FR13_FA2_QROW32_B1_LIVE_AB_ARM= FR13_FA2_QROW32_B1_PRODUCTION_ARM= \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 FORKED_FA2_SO="$FORKED_FA2_SO" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" hydra27_fixed32 "$SUBSET" \
      > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi

"$PYTHON_BIN" scripts/fr13_sfwd_conv_postprep_gate.py source-manifest \
  --repo "$REPO" --source-commit "$SOURCE_COMMIT" --output "$MANIFEST_END"
cmp -s "$MANIFEST_LAUNCH" "$MANIFEST_END" \
  || { echo "source manifest changed during embedded B4 gate" >&2; exit 14; }
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$REPO" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$REPO" --output "$RUNROOT_ABS/external_manifest.at_end.json"
cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
  "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  || { echo "runtime manifest changed during embedded B4 gate" >&2; exit 14; }
cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
  "$RUNROOT_ABS/external_manifest.at_end.json" \
  || { echo "external manifest changed during embedded B4 gate" >&2; exit 14; }
[[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
  || { echo "embedded B4 runner changed during execution" >&2; exit 14; }
(( serve_rc == 0 )) || exit "$serve_rc"

"$PYTHON_BIN" scripts/fr13_sfwd_conv_postprep_gate.py validate-embedded \
  --repo "$REPO" --arm-dir "$ARMDIR" --source-commit "$SOURCE_COMMIT" \
  --batch-size 4 --manifest-launch "$MANIFEST_LAUNCH" \
  --manifest-end "$MANIFEST_END" \
  --output "$ARMDIR/sfwd_embedded_gate_k64_root_exact4_b4_gate.json"

printf 'gate=%s\ngate_sha256=%s\nsource_commit=%s\nsource_manifest_sha256=%s\n' \
  "$ARMDIR/sfwd_embedded_gate_k64_root_exact4_b4_gate.json" \
  "$(sha256sum "$ARMDIR/sfwd_embedded_gate_k64_root_exact4_b4_gate.json" | awk '{print $1}')" \
  "$SOURCE_COMMIT" "$MANIFEST_SHA256"
