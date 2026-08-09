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
QROW32_B1_FA2_SHA256=a9d8a6887b8b27b3a83af60bba7945eb66caff174ba710c2ee2aea92b8e7081a
QROW32_B1_FA2_BYTES=300154616
QROW32_B1_FA2_HEAD=29210221863736a08f71a866459e368ad1ac4a95
QROW32_B1_SOURCE_CLOSURE_SHA256=22b8c2016443a151bf50f62166f7cc3b9ce45137138d948b76fdfded74c395ff
DFWD_K64_TOP3_SHA256=c0ed75cafdd926eceafcf28671869d54f37addb51bfef5a37c0b07c34f5420ff
DFWD_K64_TOP3_BYTES=159288
DFWD_K64_TOP3_SOURCE_SHA256=38ea96d955355bf172f534174aa2d91e6db23170144b1c84c9474016a6c05e72
COMBINED_GRAPH_GATE=${FR13_GDN_QROW32_DFWD_TOP3_COMBINED_GATE:-0}
case "$COMBINED_GRAPH_GATE" in
  0|1) ;;
  *) echo "FR13_GDN_QROW32_DFWD_TOP3_COMBINED_GATE must be exactly 0 or 1" >&2; exit 2 ;;
esac
QROW32_GATE_ARM=
if [[ "$COMBINED_GRAPH_GATE" == "1" ]]; then
  QROW32_GATE_ARM=${FR13_GATE_QROW32_ARM:-nosplit}
  case "$QROW32_GATE_ARM" in
    nosplit|split2) ;;
    *) echo "FR13_GATE_QROW32_ARM must be nosplit or split2" >&2; exit 2 ;;
  esac
elif [[ -n "${FR13_GATE_QROW32_ARM:-}" ]]; then
  echo "FR13_GATE_QROW32_ARM requires the combined graph gate" >&2
  exit 2
fi
KV_CACHE_MEMORY_BYTES=
if [[ "$FR13_GDN_GATE_BATCH" == "4" ]]; then
  KV_CACHE_MEMORY_BYTES=49392123904
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

if [[ "$COMBINED_GRAPH_GATE" == "1" ]]; then
  [[ "$FR13_GDN_GATE_MODE:$FR13_GDN_GATE_BATCH:$FR13_GDN_GATE_ENTRYPOINT:$FR13_GDN_GATE_CANDIDATE" \
      == "hydra27_fixed32:1:scripts/fr13_run_b1_gdn_gqa_group3_live_gate.sh:gqa_group3" ]] || {
    echo "combined graph gate is restricted to the canonical Hydra27 B1 GQA3 entrypoint" >&2
    exit 2
  }
  : "${QROW32_B1_FA2_SOURCE:?combined graph gate requires QROW32_B1_FA2_SOURCE}"
  : "${FR13_GATE_DFWD_TOP3_SO:?combined graph gate requires FR13_GATE_DFWD_TOP3_SO}"
  : "${FR13_GATE_DFWD_TOP3_BUILD_ATTESTATION:?combined graph gate requires FR13_GATE_DFWD_TOP3_BUILD_ATTESTATION}"
fi

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
if [[ "$COMBINED_GRAPH_GATE" == "1" ]]; then
  [[ "$(stat -c '%s' "$FORKED_FA2_SO")" == "$QROW32_B1_FA2_BYTES" \
     && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" == "$QROW32_B1_FA2_SHA256" ]] \
    || { echo "FORKED_FA2_SO is not the pinned combined Qrow32 binary" >&2; exit 2; }
  [[ "$QROW32_B1_FA2_SOURCE" == /* \
     && -d "$QROW32_B1_FA2_SOURCE" \
     && ! -L "$QROW32_B1_FA2_SOURCE" ]] \
    || { echo "QROW32_B1_FA2_SOURCE must be an absolute non-symlink directory" >&2; exit 2; }
  [[ "$FR13_GATE_DFWD_TOP3_SO" == /* \
     && -f "$FR13_GATE_DFWD_TOP3_SO" \
     && ! -L "$FR13_GATE_DFWD_TOP3_SO" \
     && "$(stat -c '%s' "$FR13_GATE_DFWD_TOP3_SO")" == "$DFWD_K64_TOP3_BYTES" \
     && "$(sha256sum "$FR13_GATE_DFWD_TOP3_SO" | awk '{print $1}')" == "$DFWD_K64_TOP3_SHA256" ]] \
    || { echo "DFWD K64 top3 binary identity drifted" >&2; exit 2; }
  [[ "$FR13_GATE_DFWD_TOP3_BUILD_ATTESTATION" == /* \
     && -f "$FR13_GATE_DFWD_TOP3_BUILD_ATTESTATION" \
     && ! -L "$FR13_GATE_DFWD_TOP3_BUILD_ATTESTATION" ]] \
    || { echo "DFWD K64 top3 build attestation must be an absolute regular file" >&2; exit 2; }
else
  [[ "$(stat -c '%s' "$FORKED_FA2_SO")" == "$STOCK_FA2_BYTES" \
     && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
    || { echo "FORKED_FA2_SO is not the exact-safe stock reference" >&2; exit 2; }
fi
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical real-task subset identity drifted" >&2; exit 2; }
[[ "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" ]] \
  || { echo "pinned K64 block map identity drifted" >&2; exit 2; }
[[ -f "$ENTRYPOINT_PATH" && ! -L "$ENTRYPOINT_PATH" ]] \
  || { echo "ordered-GDN entrypoint is missing" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
if [[ "$COMBINED_GRAPH_GATE" == "1" ]]; then
  [[ "$(git rev-parse '@{upstream}')" == "$SOURCE_COMMIT" ]] \
    || { echo "combined graph gate source commit must be pushed to its upstream" >&2; exit 2; }
  "$PYTHON_BIN" scripts/fr13_qrow32_b1_pass_sidecar.py validate-source \
    --source-root "$QROW32_B1_FA2_SOURCE" >/dev/null
  "$PYTHON_BIN" scripts/fr13_b1_composed_stack_gate.py validate-dfwd-build \
    --candidate-so "$FR13_GATE_DFWD_TOP3_SO" \
    --build-attestation "$FR13_GATE_DFWD_TOP3_BUILD_ATTESTATION" \
    --repo "$REPO" \
    --source-commit "$SOURCE_COMMIT" >/dev/null
fi
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
printf 'classification=real_swe_verified_gdn_ordered_graph_byte_diagnostic\ncandidate_selector=%s\ncandidate=%s\nacceptance_valid=0\ntiming_eligible=0\nfloor_acceptance_eligible=0\nproduction_enabled=0\nreference_always_served=1\nmode=%s\nlogical_topology=%s\nexpected_batch=%s\ntask_count=%s\nconcurrency=%s\ndraft_vocab_k=65536\ndraft_vocab_root=1\nphysical_rows=32\nreference_launches_per_request_layer=2\ncandidate_launches_per_request_layer=1\ncombined_qrow32_gqa3_dfwd_top3=%s\nqrow32_live_arm=%s\nqrow32_so_sha256=%s\ndfwd_top3_so_sha256=%s\ndfwd_top3_source_sha256=%s\nsubset_sha256=%s\nsource_commit=%s\nentrypoint=%s\nentrypoint_sha256=%s\ncommon_runner_sha256=%s\nreducer_sha256=%s\nstarted=%s\n' \
  "$FR13_GDN_GATE_CANDIDATE" "$GATE_CANDIDATE_ID" \
  "$FR13_GDN_GATE_MODE" "$LOGICAL_SLUG" "$BATCH" "$BATCH" "$BATCH" \
  "$COMBINED_GRAPH_GATE" "$QROW32_GATE_ARM" "$QROW32_B1_FA2_SHA256" \
  "$DFWD_K64_TOP3_SHA256" "$DFWD_K64_TOP3_SOURCE_SHA256" \
  "$SUBSET_SHA256" "$SOURCE_COMMIT" "$FR13_GDN_GATE_ENTRYPOINT" \
  "$ENTRYPOINT_SHA256" "$COMMON_RUNNER_SHA256" "$REDUCER_SHA256" \
  "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

QROW32_LIVE_ARM=
QROW32_LIVE_INSTANCE=
QROW32_SO_SHA_ENV=
QROW32_SO_BYTES_ENV=
QROW32_FA2_HEAD_ENV=
QROW32_SOURCE_CLOSURE_ENV=
QROW32_SOURCE_COMMIT_ENV=
QROW32_PATCH_SOURCE_SHA_ENV=
QROW32_LIVE_FILE=fr13_fa2_qrow32_b1_live_paged_ab.json
QROW32_VERIFICATION_FILE=qrow32_live_verification.json
DFWD_TOP3_SHA_ENV=
if [[ "$COMBINED_GRAPH_GATE" == "1" ]]; then
  QROW32_LIVE_ARM=$QROW32_GATE_ARM
  QROW32_LIVE_INSTANCE=astropy__astropy-12907
  QROW32_SO_SHA_ENV=$QROW32_B1_FA2_SHA256
  QROW32_SO_BYTES_ENV=$QROW32_B1_FA2_BYTES
  QROW32_FA2_HEAD_ENV=$QROW32_B1_FA2_HEAD
  QROW32_SOURCE_CLOSURE_ENV=$QROW32_B1_SOURCE_CLOSURE_SHA256
  QROW32_SOURCE_COMMIT_ENV=$SOURCE_COMMIT
  QROW32_PATCH_SOURCE_SHA_ENV=$(sha256sum scripts/fr13_patch_fa2_tree_bias.py | awk '{print $1}')
  QROW32_LIVE_FILE="fr13_fa2_qrow32_b1_${QROW32_LIVE_ARM}_live_paged_ab.json"
  QROW32_VERIFICATION_FILE="qrow32_${QROW32_LIVE_ARM}_live_verification.json"
  DFWD_TOP3_SHA_ENV=$DFWD_K64_TOP3_SHA256
fi

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
    FR13_DFWD_K64_TOP3="$COMBINED_GRAPH_GATE" \
    FR13_DFWD_K64_TOP3_SO="${FR13_GATE_DFWD_TOP3_SO:-}" \
    FR13_DFWD_K64_TOP3_SHA256="$DFWD_TOP3_SHA_ENV" \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 \
    FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_DRAFT_HEAD_PAD_ROWS=0 \
    FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 \
    FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 \
    FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FA2_QROW32_LIVE_PAGED_AB=0 \
    FR13_FA2_QROW32_B1_LIVE_AB_ARM="$QROW32_LIVE_ARM" \
    FR13_FA2_QROW32_B1_LIVE_AB_INSTANCE_ID="$QROW32_LIVE_INSTANCE" \
    FR13_FA2_QROW32_B1_LIVE_AB_JSON="/logs/$QROW32_LIVE_FILE" \
    FR13_FA2_QROW32_B1_PRODUCTION_ARM= \
    FR13_FA2_QROW32_B1_SO_SHA256="$QROW32_SO_SHA_ENV" \
    FR13_FA2_QROW32_B1_SO_SIZE="$QROW32_SO_BYTES_ENV" \
    FR13_FA2_QROW32_B1_FA2_HEAD="$QROW32_FA2_HEAD_ENV" \
    FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256="$QROW32_SOURCE_CLOSURE_ENV" \
    FR13_FA2_QROW32_B1_SOURCE_COMMIT="$QROW32_SOURCE_COMMIT_ENV" \
    FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256="$QROW32_PATCH_SOURCE_SHA_ENV" \
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

if [[ "$COMBINED_GRAPH_GATE" == "1" ]]; then
  "$PYTHON_BIN" scripts/fr13_b1_composed_stack_gate.py issue-graph-gate \
    --repo "$REPO" \
    --source-commit "$SOURCE_COMMIT" \
    --arm "$ARMDIR" \
    --qrow-arm "$QROW32_LIVE_ARM" \
    --qrow-live "$ARMDIR/logs/$QROW32_LIVE_FILE" \
    --qrow-candidate-so "$FORKED_FA2_SO" \
    --candidate-so "$FR13_GATE_DFWD_TOP3_SO" \
    --build-attestation "$FR13_GATE_DFWD_TOP3_BUILD_ATTESTATION" \
    --gdn-credential "$CREDENTIAL" \
    --runtime-launch "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    --runtime-end "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    --external-launch "$RUNROOT_ABS/external_manifest.at_launch.json" \
    --external-end "$RUNROOT_ABS/external_manifest.at_end.json" \
    --qrow-output "$ARMDIR/$QROW32_VERIFICATION_FILE" \
    --dfwd-output "$ARMDIR/dfwd_k64_top3_credential.json"
fi
