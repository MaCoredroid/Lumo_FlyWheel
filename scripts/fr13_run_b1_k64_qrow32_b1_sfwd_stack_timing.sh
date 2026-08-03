#!/usr/bin/env bash
# PASS-gated real SWE-Verified exact4 timing for Hydra27 qrow32 split2.
# Despite the retained historical filename, this isolates tree attention in FULL graph mode.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

case "${FR13_RUN_QROW32_SPLIT2_TIMING:-0}" in
  1) ;;
  0)
    echo "qrow32 split2 timing is disabled; set FR13_RUN_QROW32_SPLIT2_TIMING=1" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_QROW32_SPLIT2_TIMING must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${QROW32_B1_FA2_SO:?set QROW32_B1_FA2_SO to the pinned combined binary}"
: "${QROW32_B1_FA2_SOURCE:?set QROW32_B1_FA2_SOURCE to the pinned FA2 source closure}"
: "${QROW32_B1_PASS:?set QROW32_B1_PASS to the qrow32 split2 real-task live PASS}"
: "${QROW32_B1_PASS_SHA256:?set QROW32_B1_PASS_SHA256 to its raw SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
CANDIDATE_SHA256=5eec90f317cf6126cd57ab7f77b392ae6a1430d28210dcb31756abe788ef3467
CANDIDATE_BYTES=300140712
FA2_HEAD=29210221863736a08f71a866459e368ad1ac4a95
SOURCE_CLOSURE_SHA256=c10888e721335ff99f93dabdfea7d8a524fbd7e21e8aee3f425f50af06bf5d84
BASELINE=$REPO/results/fr13_fixed32_qrow16_prod_exact4_b1_20260731T182827Z/hydra_valid/deploy_speed_qrow16_prod_exact4_b1_20260731T182827Z.json
BASELINE_SHA256=0350e791bc825083bfc3635e11c875617fa1d3823eba5f93ebd7f392c50f18d0
MANDATORY_WEIGHT_BYTES=32666638208
MANDATORY_WEIGHT_FLOOR_MS=119.658015414
ONE_SIDED_U95_CAP_MS=137.6067177261
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
SOURCE_COMMIT=$(git rev-parse HEAD)
PATCH_SOURCE_SHA256=$(sha256sum scripts/fr13_patch_fa2_tree_bias.py | awk '{print $1}')
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}
ARM="hydra27_fixed32_k64_qrow32_split2_exact4_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for required in "$QROW32_B1_FA2_SO" "$QROW32_B1_PASS" "$BASELINE"; do
  [[ "$required" == /* && -f "$required" && ! -L "$required" ]] \
    || { echo "required input must be an absolute regular file: $required" >&2; exit 2; }
done
unset required
[[ "$QROW32_B1_FA2_SOURCE" == /* \
   && -d "$QROW32_B1_FA2_SOURCE" \
   && ! -L "$QROW32_B1_FA2_SOURCE" ]] \
  || { echo "QROW32_B1_FA2_SOURCE must be an absolute non-symlink directory" >&2; exit 2; }
[[ "$(stat -c '%s' "$QROW32_B1_FA2_SO")" == "$CANDIDATE_BYTES" \
   && "$(sha256sum "$QROW32_B1_FA2_SO" | awk '{print $1}')" == "$CANDIDATE_SHA256" \
   && "$(sha256sum "$QROW32_B1_PASS" | awk '{print $1}')" == "$QROW32_B1_PASS_SHA256" \
   && "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" \
   && "$(sha256sum "$BASELINE" | awk '{print $1}')" == "$BASELINE_SHA256" ]] \
  || { echo "qrow32 exact4 timing prerequisite identity drifted" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }

"$PYTHON_BIN" scripts/fr13_qrow32_b1_pass_sidecar.py validate-source \
  --source-root "$QROW32_B1_FA2_SOURCE" >/dev/null
"$PYTHON_BIN" - \
  "$QROW32_B1_PASS" "$CANDIDATE_SHA256" "$SOURCE_COMMIT" \
  "$PATCH_SOURCE_SHA256" "$QROW32_B1_FA2_SO" <<'PY'
import sys
from pathlib import Path

from scripts import fr13_qrow32_b1_pass_sidecar as qrow

payload, _ = qrow.load_json(Path(sys.argv[1]))
qrow.validate_live_result(
    payload,
    candidate_sha256=sys.argv[2],
    arm=qrow.ARM,
    source_commit=sys.argv[3],
    patch_source_sha256=sys.argv[4],
)
qrow.validate_candidate(Path(sys.argv[5]), sys.argv[2])
qrow.validate_patch_source(
    Path("scripts/fr13_patch_fa2_tree_bias.py"),
    expected_source_commit=sys.argv[3],
)
PY
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER"
export FR13_FLOOR_ORDER=HT
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_DRAFT_VOCAB_ROOT" == "1" \
   && "$FR13_DRAFT_VOCAB_K" == "65536" \
   && "$FR13_DRAFT_VOCAB_BLOCKS" == "$BLOCK_MAP_CONTAINER" \
   && "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" ]] \
  || { echo "ROOT=1 K64 hardware-floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS/sidecars"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"
printf 'classification=real_swe_verified_exact4_qrow32_split2\ntask_count=4\nbatch_size=1\nconcurrency=1\ntiming_eligible=1\nformal_floor_acceptance_eligible=0\ntopology=hydra27_fixed32\nphysical_rows=32\nlogical_drafts=27\nvalid_mask=0x7abdffff\ndraft_vocab_root=1\ndraft_vocab_k=65536\nqrow32_split2_production=1\nruntime=FULL_graph_exact_geometry\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nexact16_rule=only_after_exact4_u95_clears_cap\narm=%s\nsource=%s\npatch_source_sha256=%s\nrunner_sha256=%s\nsubset_sha256=%s\ncandidate_so_sha256=%s\ncandidate_so_bytes=%s\nfa2_head=%s\nfa2_source_closure_sha256=%s\npass_sha256=%s\nqrow16_historical_baseline_sha256=%s\nstarted=%s\n' \
  "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$ARM" "$SOURCE_COMMIT" \
  "$PATCH_SOURCE_SHA256" "$RUNNER_SHA256" "$SUBSET_SHA256" \
  "$CANDIDATE_SHA256" "$CANDIDATE_BYTES" "$FA2_HEAD" \
  "$SOURCE_CLOSURE_SHA256" "$QROW32_B1_PASS_SHA256" "$BASELINE_SHA256" \
  "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"

MANIFEST_FINALIZED=0
finalize_manifests() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json"
  cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || { echo "runtime/source manifest changed during timing" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
    || { echo "timing runner changed during execution" >&2; return 14; }
  MANIFEST_FINALIZED=1
}
runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    finalize_manifests || { local mrc=$?; (( rc == 0 )) && rc=$mrc; }
  fi
  exit "$rc"
}
trap runner_exit EXIT

if env \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
    FR13_FIXED32_B1_DIAGNOSTIC=0 \
    FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
    FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER" \
    FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES" \
    FR13_WEIGHT_FLOOR_MS="$MANDATORY_WEIGHT_FLOOR_MS" \
    FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=0 FR13_CFWD_GPU_TIMER=0 \
    FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${ARM}.json" \
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
    FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0 \
    FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=0 \
    FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0 FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0 \
    FR13_FIXED32_CONV_SOURCE_BATCH=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FA2_QROW32_LIVE_PAGED_AB=0 \
    FR13_FA2_QROW32_B1_LIVE_AB_ARM= \
    FR13_FA2_QROW32_B1_SO_SHA256="$CANDIDATE_SHA256" \
    FR13_FA2_QROW32_B1_SO_SIZE="$CANDIDATE_BYTES" \
    FR13_FA2_QROW32_B1_FA2_HEAD="$FA2_HEAD" \
    FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256="$SOURCE_CLOSURE_SHA256" \
    FR13_FA2_QROW32_B1_SOURCE_COMMIT="$SOURCE_COMMIT" \
    FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256="$PATCH_SOURCE_SHA256" \
    FR13_FA2_QROW32_B1_PRODUCTION_ARM=split2 \
    FR13_FA2_QROW32_B1_LIVE_PASS_JSON="$QROW32_B1_PASS" \
    FR13_FA2_QROW32_B1_LIVE_PASS_SHA256="$QROW32_B1_PASS_SHA256" \
    FR13_FA2_QROW32_B1_EXACT4_TASK_IDS=astropy__astropy-12907,astropy__astropy-13033,astropy__astropy-13236,astropy__astropy-13398 \
    FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256="$SUBSET_SHA256" \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 \
    FORKED_FA2_SO="$QROW32_B1_FA2_SO" RUNROOT="$RUNROOT_ABS" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" hydra27_fixed32 "$SUBSET" \
      > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi
printf 'serve_rc=%s ended=%s\n' "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
(( serve_rc == 0 )) || exit "$serve_rc"

CONTAINER_ENV="$ARMDIR/container_env.txt"
for expected in \
  'FR13_FIXED32_MODE=hydra27_fixed32' \
  'FR13_DRAFT_VOCAB_ROOT=1' \
  'FR13_DRAFT_VOCAB_K=65536' \
  'MAX_NUM_SEQS=1' \
  'SWE_CONCURRENCY=1' \
  'ENFORCE_EAGER=0' \
  'CUDAGRAPH_MODE=FULL_AND_PIECEWISE' \
  'FR13_FA2_QROW32_B1_LIVE_AB_ARM=' \
  'FR13_FA2_QROW32_B1_PRODUCTION_ARM=split2' \
  'FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0' \
  'FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0' \
  'FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0' \
  "FR13_FA2_QROW32_B1_SO_SHA256=$CANDIDATE_SHA256" \
  "FR13_FA2_QROW32_B1_SO_SIZE=$CANDIDATE_BYTES" \
  "FR13_FA2_QROW32_B1_FA2_HEAD=$FA2_HEAD" \
  "FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256=$SOURCE_CLOSURE_SHA256" \
  "FR13_FA2_QROW32_B1_SOURCE_COMMIT=$SOURCE_COMMIT" \
  "FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256=$PATCH_SOURCE_SHA256"; do
  [[ "$(grep -Fxc "$expected" "$CONTAINER_ENV")" -eq 1 ]] \
    || { echo "container lacks exact qrow32 timing pin: $expected" >&2; exit 4; }
done
unset expected

MEASURE="$ARMDIR/deploy_speed_fullwall.json"
"$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
  --arm "$ARM" \
  --out-root "$ARMDIR/swe_out" \
  --expected-tok-per-draft 31 \
  --batch-size 1 \
  --out "$MEASURE"

SIDECAR="$ARMDIR/logs/fr13_fa2_qrow32_b1_production_pass.json"
ENGAGEMENT="$ARMDIR/logs/fr13_fa2_qrow32_b1_production_engagement.json"
HEALTH="$ARMDIR/health.json"
TRAFFIC_AUDIT="$ARMDIR/fixed32_chat_traffic_audit.json"
for artifact in "$MEASURE" "$SIDECAR" "$ENGAGEMENT" "$HEALTH" "$TRAFFIC_AUDIT"; do
  [[ -f "$artifact" && ! -L "$artifact" ]] \
    || { echo "exact4 timing artifact is missing or unsafe: $artifact" >&2; exit 4; }
done
unset artifact
SIDECAR_SHA256=$(sha256sum "$SIDECAR" | awk '{print $1}')
"$PYTHON_BIN" scripts/fr13_qrow32_b1_pass_sidecar.py verify \
  --sidecar "$SIDECAR" \
  --expected-sidecar-sha256 "$SIDECAR_SHA256" \
  --candidate-so "$QROW32_B1_FA2_SO" \
  --expected-candidate-sha256 "$CANDIDATE_SHA256" \
  --arm split2 \
  --patch-source scripts/fr13_patch_fa2_tree_bias.py \
  --expected-source-commit "$SOURCE_COMMIT" >/dev/null
finalize_manifests

"$PYTHON_BIN" scripts/fr13_qrow32_split2_timing.py \
  --subset "$SUBSET" \
  --measure "$MEASURE" \
  --baseline "$BASELINE" \
  --engagement "$ENGAGEMENT" \
  --health "$HEALTH" \
  --traffic-audit "$TRAFFIC_AUDIT" \
  --source-commit "$SOURCE_COMMIT" \
  --patch-source-sha256 "$PATCH_SOURCE_SHA256" \
  --pass-sha256 "$QROW32_B1_PASS_SHA256" \
  --pass-sidecar-sha256 "$SIDECAR_SHA256" \
  --runner-sha256 "$RUNNER_SHA256" \
  --block-map-sha256 "$BLOCK_MAP_SHA256" \
  --floor-ms "$MANDATORY_WEIGHT_FLOOR_MS" \
  --cap-ms "$ONE_SIDED_U95_CAP_MS" \
  --arm "$ARM" \
  --out "$RUNROOT_ABS/timing_summary.json"
printf 'summary=%s ended=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
