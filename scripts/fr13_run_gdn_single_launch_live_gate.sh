#!/usr/bin/env bash
# Shared implementation for the three fixed-scope ordered-GDN real-task gates.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned stock FA2 shared object}"
: "${FR13_GDN_GATE_MODE:?wrapper must bake FR13_GDN_GATE_MODE}"
: "${FR13_GDN_GATE_BATCH:?wrapper must bake FR13_GDN_GATE_BATCH}"
: "${FR13_GDN_GATE_ENTRYPOINT:?wrapper must bake FR13_GDN_GATE_ENTRYPOINT}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
COMMON_RUNNER=scripts/fr13_run_gdn_single_launch_live_gate.sh
REDUCER=scripts/fr13_gdn_single_launch_gate.py
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
KV_CACHE_MEMORY_BYTES=
if [[ "$FR13_GDN_GATE_BATCH" == "4" ]]; then
  KV_CACHE_MEMORY_BYTES=42949672960
fi
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNROOT_ABS=$(realpath -m "$RUNROOT")
FR13_GDN_GATE_CANDIDATE=${FR13_GDN_GATE_CANDIDATE:-single_launch}
case "$FR13_GDN_GATE_CANDIDATE" in
  single_launch)
    GATE_CANDIDATE_SLUG=single_launch
    GATE_CANDIDATE_ID=fixed32_gdn_single_launch_tree_v2
    ;;
  gqa_group3)
    GATE_CANDIDATE_SLUG=gqa_group3
    GATE_CANDIDATE_ID=fixed32_gdn_single_launch_gqa_group3_v1
    ;;
  *)
    echo "FR13_GDN_GATE_CANDIDATE must be single_launch or gqa_group3" >&2
    exit 2
    ;;
esac

case "$FR13_GDN_GATE_MODE:$FR13_GDN_GATE_BATCH:$FR13_GDN_GATE_ENTRYPOINT:$FR13_GDN_GATE_CANDIDATE" in
  hydra27_fixed32:1:scripts/fr13_run_b1_gdn_single_launch_live_gate.sh:single_launch)
    LOGICAL_SLUG=hydra27
    SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
    SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
    B1_DIAGNOSTIC=1
    ;;
  hydra27_fixed32:1:scripts/fr13_run_b1_gdn_gqa_group3_live_gate.sh:gqa_group3)
    LOGICAL_SLUG=hydra27
    SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
    SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
    B1_DIAGNOSTIC=1
    ;;
  tail6_fixed32:4:scripts/fr13_run_b4_tail23_gdn_single_launch_live_gate.sh:single_launch|tail6_fixed32:4:scripts/fr13_run_b4_tail23_gdn_single_launch_live_gate.sh:gqa_group3)
    LOGICAL_SLUG=tail23
    SUBSET=config/fr13_fixed32/subset_b4_four.json
    SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
    B1_DIAGNOSTIC=0
    ;;
  hydra27_fixed32:4:scripts/fr13_run_b4_hydra27_gdn_single_launch_live_gate.sh:single_launch|hydra27_fixed32:4:scripts/fr13_run_b4_hydra27_gdn_single_launch_live_gate.sh:gqa_group3)
    LOGICAL_SLUG=hydra27
    SUBSET=config/fr13_fixed32/subset_b4_four.json
    SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
    B1_DIAGNOSTIC=0
    ;;
  *)
    echo "ordered-GDN wrapper mode/batch/entrypoint identity is invalid" >&2
    exit 2
    ;;
esac

BATCH=$FR13_GDN_GATE_BATCH
ARM="${FR13_GDN_GATE_MODE}_${LOGICAL_SLUG}_gdn_${GATE_CANDIDATE_SLUG}_b${BATCH}_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"
ENTRYPOINT_PATH="$REPO/$FR13_GDN_GATE_ENTRYPOINT"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ "$FORKED_FA2_SO" == /* && -f "$FORKED_FA2_SO" && ! -L "$FORKED_FA2_SO" ]] \
  || { echo "FORKED_FA2_SO must be an absolute regular non-symlink file" >&2; exit 2; }
[[ "$(stat -c '%s' "$FORKED_FA2_SO")" == "$STOCK_FA2_BYTES" \
   && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "FORKED_FA2_SO is not the exact-safe stock reference" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical real-task subset identity drifted" >&2; exit 2; }
[[ "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" ]] \
  || { echo "pinned K64 block map identity drifted" >&2; exit 2; }
[[ -f "$ENTRYPOINT_PATH" && ! -L "$ENTRYPOINT_PATH" ]] \
  || { echo "ordered-GDN entrypoint is missing" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the gate" >&2; exit 2; }

ENTRYPOINT_SHA256=$(sha256sum "$ENTRYPOINT_PATH" | awk '{print $1}')
COMMON_RUNNER_SHA256=$(sha256sum "$COMMON_RUNNER" | awk '{print $1}')
REDUCER_SHA256=$(sha256sum "$REDUCER" | awk '{print $1}')

export BSIZE=$BATCH
export CONC=$BATCH
export WALL=0
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant

[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "32666638208" \
   && "$FR13_WEIGHT_FLOOR_MS" == "119.658015414" \
   && "$FR13_DRAFT_VOCAB_K" == "65536" \
   && "$FR13_DRAFT_VOCAB_ROOT" == "1" \
   && "$LUMO_SWE_AUTOCOMMIT" == "0" ]] \
  || { echo "fixed K64/root1 floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
printf 'classification=real_swe_verified_gdn_ordered_graph_byte_diagnostic\ncandidate_selector=%s\ncandidate=%s\nacceptance_valid=0\ntiming_eligible=0\nfloor_acceptance_eligible=0\nproduction_enabled=0\nreference_always_served=1\nmode=%s\nlogical_topology=%s\nexpected_batch=%s\ntask_count=%s\nconcurrency=%s\ndraft_vocab_k=65536\ndraft_vocab_root=1\nphysical_rows=32\nreference_launches_per_request_layer=2\ncandidate_launches_per_request_layer=1\nsubset_sha256=%s\nsource_commit=%s\nentrypoint=%s\nentrypoint_sha256=%s\ncommon_runner_sha256=%s\nreducer_sha256=%s\nstarted=%s\n' \
  "$FR13_GDN_GATE_CANDIDATE" "$GATE_CANDIDATE_ID" \
  "$FR13_GDN_GATE_MODE" "$LOGICAL_SLUG" "$BATCH" "$BATCH" "$BATCH" \
  "$SUBSET_SHA256" "$SOURCE_COMMIT" "$FR13_GDN_GATE_ENTRYPOINT" \
  "$ENTRYPOINT_SHA256" "$COMMON_RUNNER_SHA256" "$REDUCER_SHA256" \
  "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

if env \
    RUNROOT="$RUNROOT_ABS" \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR="$BATCH" SWE_CONCURRENCY="$BATCH" AGENT_WALL_S= \
    KV_CACHE_MEMORY_BYTES="$KV_CACHE_MEMORY_BYTES" \
    LUMO_SWE_AUTOCOMMIT=0 \
    FR13_FIXED32_B1_DIAGNOSTIC="$B1_DIAGNOSTIC" \
    FR10_METRICS=1 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
    FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 \
    FR13_SCAN_ALIGN=0 FR13_NPAD_INVARIANT=0 \
    FR13_DRAFT_VOCAB_K=65536 FR13_DRAFT_VOCAB_ROOT=1 \
    FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json \
    FR13_DEVICE_MULTIDRAFT=1 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE="$FR13_GDN_GATE_CANDIDATE" \
    FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH="$BATCH" \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=0 \
    FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH= \
    FR13_FIXED32_GDN_GQA_GROUP3_PASS_JSON= \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
    FR13_FIXED32_CUTLASS_WAVE=stock \
    FR13_FIXED32_CUTLASS_WAVE_SO= \
    FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 \
    FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_DRAFT_HEAD_PAD_ROWS=0 \
    FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 \
    FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 \
    FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FA2_QROW32_LIVE_PAGED_AB=0 \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 \
    FORKED_FA2_SO="$FORKED_FA2_SO" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" "$FR13_GDN_GATE_MODE" "$SUBSET" \
      > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi

printf 'serve_rc=%s ended=%s\n' "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json"
cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
  "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  || { echo "runtime/source manifest changed during GDN diagnostic" >&2; exit 14; }
cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
  "$RUNROOT_ABS/external_manifest.at_end.json" \
  || { echo "external manifest changed during GDN diagnostic" >&2; exit 14; }
[[ "$(sha256sum "$ENTRYPOINT_PATH" | awk '{print $1}')" == "$ENTRYPOINT_SHA256" \
   && "$(sha256sum "$COMMON_RUNNER" | awk '{print $1}')" == "$COMMON_RUNNER_SHA256" \
   && "$(sha256sum "$REDUCER" | awk '{print $1}')" == "$REDUCER_SHA256" ]] \
  || { echo "GDN runner/reducer source changed during execution" >&2; exit 14; }
(( serve_rc == 0 )) || exit "$serve_rc"

CREDENTIAL="$ARMDIR/${LOGICAL_SLUG}_gdn_${GATE_CANDIDATE_SLUG}_b${BATCH}_credential.json"
"$PYTHON_BIN" "$REDUCER" \
  --arm "$ARMDIR" \
  --subset "$SUBSET" \
  --mode "$FR13_GDN_GATE_MODE" \
  --batch "$BATCH" \
  --candidate "$FR13_GDN_GATE_CANDIDATE" \
  --live-pass "$ARMDIR/logs/fr13_fixed32_gdn_path_bv.live_pass.json" \
  --runtime-launch "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
  --runtime-end "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  --external-launch "$RUNROOT_ABS/external_manifest.at_launch.json" \
  --external-end "$RUNROOT_ABS/external_manifest.at_end.json" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$CREDENTIAL" \
  > "$ARMDIR/gdn_single_launch_reduction.json"
