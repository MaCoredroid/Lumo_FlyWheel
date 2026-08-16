#!/usr/bin/env bash
# Exact4 B1 pair: K64 qrow16+SFWD reference vs source-v7 all-parent full stack.
# This is a real SWE-Verified timing screen, not the formal exact16 U95 gate.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${MODE:?set MODE to tail6_fixed32 or hydra27_fixed32}"
: "${QROW16_FA2_SO:?set QROW16_FA2_SO to the pinned qrow16 binary}"
: "${TAW_B1_CREDENTIAL:?set TAW_B1_CREDENTIAL to the mode-bound B1 credential}"
: "${TAW_B1_CREDENTIAL_SHA256:?set its raw SHA-256}"
: "${TAW_B1_LIVE_BUNDLE:?set TAW_B1_LIVE_BUNDLE to the credentialed B1 replay}"
: "${TAW_B1_LIVE_BUNDLE_SHA256:?set its raw SHA-256}"
: "${TAW_REVIEWED_B4_PASS:?set it to the corrected B4 source-v7 bundle}"
: "${TAW_REVIEWED_B4_PASS_SHA256:?set its raw SHA-256}"
: "${TAW_REVIEWED_B4_VERDICT:?set it to the corrected B4 exact4 verdict}"
: "${TAW_REVIEWED_B4_VERDICT_SHA256:?set its raw SHA-256}"
: "${TAW_MERGE_BINDING:?set it to the B1/B4 merge binding}"
: "${TAW_MERGE_BINDING_SHA256:?set its raw SHA-256}"
: "${TAW_PRODUCTION_PASS:?set TAW_PRODUCTION_PASS to the merged source-v7 bundle}"
: "${TAW_PRODUCTION_PASS_SHA256:?set its raw SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
TAW_SOURCE=scripts/fr13_device_multidraft_kernel.py
TAW_SOURCE_SCHEMA=fr13-fixed32-taw-all-parent-v7
TAW_SOURCE_CONTRACT_SHA256=484babd7a883c81c7317ef23862940143c248dcbc1b66c9d4ac6775ff5a0fa93
QROW16_SHA256=1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86
QROW16_BYTES=299507792
QROW16_PASS=$REPO/results/fr13_fixed32_qrow16_num_splits0_live_pass_20260731T173608Z/fr13_fa2_qrow16_live_paged_ab.json
QROW16_PASS_SHA256=36940fd43d11399529d1bfe7e11baa9961907193267f3bb43d41057328737b77
SFWD_PASS=$REPO/results/fr13_fixed32_sfwd_b1_real_task_byte_pass_20260801/run_evidence/fr13_fixed32_sfwd_state_fusion.live_pass.json
SFWD_PASS_SHA256=7ccfaf5cc907909b0646b752b94027e250b234a3b98bf461de61e6ae70f31782
MANDATORY_WEIGHT_BYTES=25210209416
MANDATORY_WEIGHT_FLOOR_MS=92.345089436
ONE_SIDED_U95_CAP_MS=106.1968528514
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}

case "$MODE" in
  tail6_fixed32)
    LOGICAL_TOPOLOGY=Tail23
    LOGICAL_DRAFTS=23
    VALID_MASK=0x7a9ce7ff
    ;;
  hydra27_fixed32)
    LOGICAL_TOPOLOGY=Hydra27
    LOGICAL_DRAFTS=27
    VALID_MASK=0x7abdffff
    ;;
  *)
    echo "MODE must be tail6_fixed32 or hydra27_fixed32" >&2
    exit 2
    ;;
esac
STOCK_ARM="${MODE}_k64_qrow16_sfwd_reference_${TAG}"
CANDIDATE_ARM="${MODE}_k64_qrow16_sfwd_taw_source_v7_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for input in \
  "$QROW16_FA2_SO" "$QROW16_PASS" "$SFWD_PASS" \
  "$TAW_B1_CREDENTIAL" "$TAW_B1_LIVE_BUNDLE" \
  "$TAW_REVIEWED_B4_PASS" "$TAW_REVIEWED_B4_VERDICT" \
  "$TAW_MERGE_BINDING" "$TAW_PRODUCTION_PASS"; do
  [[ "$input" == /* && -f "$input" && ! -L "$input" ]] \
    || { echo "timing input must be an absolute regular file: $input" >&2; exit 2; }
done
unset input
[[ "$(stat -c '%s' "$QROW16_FA2_SO")" == "$QROW16_BYTES" \
   && "$(sha256sum "$QROW16_FA2_SO" | awk '{print $1}')" == "$QROW16_SHA256" ]] \
  || { echo "QROW16_FA2_SO is not the pinned candidate" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" \
   && "$(sha256sum "$QROW16_PASS" | awk '{print $1}')" == "$QROW16_PASS_SHA256" \
   && "$(sha256sum "$SFWD_PASS" | awk '{print $1}')" == "$SFWD_PASS_SHA256" ]] \
  || { echo "tracked K64 stack prerequisite identity drifted" >&2; exit 2; }
[[ "$TAW_B1_CREDENTIAL_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$TAW_B1_CREDENTIAL" | awk '{print $1}')" == "$TAW_B1_CREDENTIAL_SHA256" \
   && "$TAW_B1_LIVE_BUNDLE_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$TAW_B1_LIVE_BUNDLE" | awk '{print $1}')" == "$TAW_B1_LIVE_BUNDLE_SHA256" \
   && "$TAW_REVIEWED_B4_PASS_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$TAW_REVIEWED_B4_PASS" | awk '{print $1}')" == "$TAW_REVIEWED_B4_PASS_SHA256" \
   && "$TAW_REVIEWED_B4_VERDICT_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$TAW_REVIEWED_B4_VERDICT" | awk '{print $1}')" == "$TAW_REVIEWED_B4_VERDICT_SHA256" \
   && "$TAW_MERGE_BINDING_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$TAW_MERGE_BINDING" | awk '{print $1}')" == "$TAW_MERGE_BINDING_SHA256" \
   && "$TAW_PRODUCTION_PASS_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$TAW_PRODUCTION_PASS" | awk '{print $1}')" == "$TAW_PRODUCTION_PASS_SHA256" ]] \
  || { echo "TAW credential or production bundle identity mismatch" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }

"$PYTHON_BIN" - "$QROW16_PASS" "$QROW16_SHA256" <<'PY'
import sys
from pathlib import Path

from scripts import fr13_qrow16_pass_sidecar as qrow


payload, _ = qrow.load_json(Path(sys.argv[1]))
qrow.validate_live_result(payload, candidate_sha256=sys.argv[2])
PY
"$PYTHON_BIN" scripts/fr13_sfwd_state_fusion_pass.py validate \
  --live-result "$SFWD_PASS" \
  --expected-live-sha256 "$SFWD_PASS_SHA256" \
  --kernel-source src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  >/dev/null
"$PYTHON_BIN" scripts/fr13_taw_b1_credential.py validate-production \
  --mode "$MODE" \
  --source "$TAW_SOURCE" \
  --credential "$TAW_B1_CREDENTIAL" \
  --b1-live-bundle "$TAW_B1_LIVE_BUNDLE" \
  --b4-production-pass "$TAW_REVIEWED_B4_PASS" \
  --b4-gate-verdict "$TAW_REVIEWED_B4_VERDICT" \
  --merge-binding "$TAW_MERGE_BINDING" \
  --production-pass "$TAW_PRODUCTION_PASS" \
  >/dev/null
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before paired timing" >&2; exit 2; }

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
  || { echo "K64 ROOT=1 B1 hardware-floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS/sidecars"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"
printf 'classification=real_swe_verified_exact4_k64_b1_fullstack_pair\ntiming_eligible=1\nformal_floor_acceptance_eligible=0\nonly_arm_delta=source_v7_all_parent_committer_production_0_to_1\nmode=%s\nlogical_topology=%s\nlogical_drafts=%s\nvalid_mask=%s\nphysical_drafts=31\nphysical_rows_root_inclusive=32\ntask_count=4\nbatch_size=1\nconcurrency=1\ndraft_vocab_root=1\ndraft_vocab_k=65536\ndraft_vocab_blocks=%s\ndraft_vocab_blocks_sha256=%s\nqrow16_production=1\nsfwd_state_fusion_production=1\nsource_contract_schema=%s\nsource_contract_sha256=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nstock_arm=%s\ncandidate_arm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nqrow16_sha256=%s\nqrow16_pass_sha256=%s\nsfwd_pass_sha256=%s\ntaw_b1_credential_sha256=%s\ntaw_b1_live_bundle_sha256=%s\ntaw_reviewed_b4_pass_sha256=%s\ntaw_reviewed_b4_verdict_sha256=%s\ntaw_merge_binding_sha256=%s\ntaw_production_pass_sha256=%s\nstarted=%s\n' \
  "$MODE" "$LOGICAL_TOPOLOGY" "$LOGICAL_DRAFTS" "$VALID_MASK" \
  "$BLOCK_MAP_CONTAINER" "$BLOCK_MAP_SHA256" \
  "$TAW_SOURCE_SCHEMA" "$TAW_SOURCE_CONTRACT_SHA256" \
  "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$STOCK_ARM" "$CANDIDATE_ARM" \
  "$SOURCE_COMMIT" "$RUNNER_SHA256" "$SUBSET_SHA256" \
  "$QROW16_SHA256" "$QROW16_PASS_SHA256" "$SFWD_PASS_SHA256" \
  "$TAW_B1_CREDENTIAL_SHA256" "$TAW_B1_LIVE_BUNDLE_SHA256" \
  "$TAW_REVIEWED_B4_PASS_SHA256" "$TAW_REVIEWED_B4_VERDICT_SHA256" \
  "$TAW_MERGE_BINDING_SHA256" "$TAW_PRODUCTION_PASS_SHA256" \
  "$(date -u +%FT%TZ)" \
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
    || { echo "runtime/source manifest changed during paired timing" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during paired timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
    || { echo "paired timing runner changed during execution" >&2; return 14; }
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

run_arm() {
  local arm=$1
  local taw_production=$2
  local taw_pass=""
  if [[ "$taw_production" == "1" ]]; then
    taw_pass=$TAW_PRODUCTION_PASS
  fi
  if env \
      RUNROOT="$RUNROOT_ABS" \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
      LUMO_SWE_AUTOCOMMIT=0 \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
      FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER" \
      FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES" \
      FR13_WEIGHT_FLOOR_MS="$MANDATORY_WEIGHT_FLOOR_MS" \
      FR10_METRICS=0 ENFORCE_EAGER=1 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_cfwd.json" \
      FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
      FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=1 \
      FR13_FIXED32_SFWD_STATE_FUSION_LIVE_PASS_JSON="$SFWD_PASS" \
      FR13_FIXED32_SFWD_STATE_FUSION_LIVE_PASS_SHA256="$SFWD_PASS_SHA256" \
      FR13_FIXED32_CONV_SOURCE_BATCH=0 \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 \
      FR13_FA2_QROW16_SO_SHA256="$QROW16_SHA256" \
      FR13_FA2_QROW16_PRODUCTION=1 \
      FR13_FA2_QROW16_LIVE_PASS_JSON="$QROW16_PASS" \
      FR13_FA2_QROW16_LIVE_PASS_SHA256="$QROW16_PASS_SHA256" \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION="$taw_production" \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON="$taw_pass" \
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
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$QROW16_FA2_SO" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh \
        "$arm" "$MODE" "$SUBSET" \
        > "$RUNROOT_ABS/$arm.runlog" 2>&1; then
    :
  else
    local serve_rc=$?
    printf 'arm=%s taw_production=%s serve_rc=%s ended=%s\n' \
      "$arm" "$taw_production" "$serve_rc" "$(date -u +%FT%TZ)" \
      >> "$RUNROOT_ABS/launcher_meta.txt"
    return "$serve_rc"
  fi

  local arm_dir="$RUNROOT_ABS/$arm"
  local container_env="$arm_dir/container_env.txt"
  for expected in \
    "FR13_FIXED32_MODE=$MODE" \
    'FR13_FIXED32_B1_DIAGNOSTIC=0' \
    'FR13_DRAFT_VOCAB_ROOT=1' \
    'FR13_DRAFT_VOCAB_K=65536' \
    "FR13_DRAFT_VOCAB_BLOCKS=$BLOCK_MAP_CONTAINER" \
    'MAX_NUM_SEQS=1' \
    'ENFORCE_EAGER=1' \
    'FR13_FA2_QROW16_PRODUCTION=1' \
    'FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=1' \
    "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=$taw_production"; do
    [[ "$(grep -Fxc "$expected" "$container_env")" -eq 1 ]] \
      || { echo "$arm lacks exact stack pin: $expected" >&2; return 4; }
  done
  unset expected
  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$arm" \
    --out-root "$arm_dir/swe_out" \
    --expected-tok-per-draft 31 \
    --batch-size 1 \
    --out "$arm_dir/deploy_speed_fullwall.json"
  printf 'arm=%s taw_production=%s serve_rc=0 container_env_sha256=%s ended=%s\n' \
    "$arm" "$taw_production" \
    "$(sha256sum "$container_env" | awk '{print $1}')" \
    "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
}

run_arm "$STOCK_ARM" 0
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after the stock arm" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" 1
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after the candidate arm" >&2; exit 2; }

STOCK_TAW_SELECTOR="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_taw_native_precompute_production.arm"
STOCK_TAW_PASS="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_taw_native_precompute.production_pass.json"
CANDIDATE_TAW_SELECTOR="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_taw_native_precompute_production.arm"
CANDIDATE_TAW_PASS="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_taw_native_precompute.production_pass.json"
[[ ! -e "$STOCK_TAW_SELECTOR" && ! -L "$STOCK_TAW_SELECTOR" \
   && ! -e "$STOCK_TAW_PASS" && ! -L "$STOCK_TAW_PASS" ]] \
  || { echo "stock arm emitted TAW production state" >&2; exit 4; }
[[ -f "$CANDIDATE_TAW_SELECTOR" && ! -L "$CANDIDATE_TAW_SELECTOR" \
   && -f "$CANDIDATE_TAW_PASS" && ! -L "$CANDIDATE_TAW_PASS" \
   && "$(<"$CANDIDATE_TAW_SELECTOR")" == "1" ]] \
  || { echo "candidate arm lacks TAW production state" >&2; exit 4; }
cmp -s "$TAW_PRODUCTION_PASS" "$CANDIDATE_TAW_PASS" \
  || { echo "served TAW credential differs from pinned input" >&2; exit 4; }

for arm in "$STOCK_ARM" "$CANDIDATE_ARM"; do
  arm_dir="$RUNROOT_ABS/$arm"
  sfwd_engagement="$arm_dir/logs/fr13_fixed32_sfwd_state_fusion.production_engagement.json"
  qrow_sidecar="$arm_dir/logs/fr13_fa2_qrow16_production_pass.json"
  qrow_engagement="$arm_dir/logs/fr13_fa2_qrow16_production_capture.json"
  "$PYTHON_BIN" scripts/fr13_sfwd_state_fusion_pass.py verify-engagement \
    --engagement "$sfwd_engagement" \
    --expected-live-sha256 "$SFWD_PASS_SHA256" \
    --kernel-source src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
    >/dev/null
  qrow_sidecar_sha256=$(sha256sum "$qrow_sidecar" | awk '{print $1}')
  "$PYTHON_BIN" scripts/fr13_qrow16_pass_sidecar.py verify \
    --sidecar "$qrow_sidecar" \
    --expected-sidecar-sha256 "$qrow_sidecar_sha256" \
    --candidate-so "$QROW16_FA2_SO" \
    --expected-candidate-sha256 "$QROW16_SHA256" \
    >/dev/null
  [[ -f "$qrow_engagement" && ! -L "$qrow_engagement" ]] \
    || { echo "$arm lacks qrow16 eager engagement" >&2; exit 4; }
done
unset arm arm_dir sfwd_engagement qrow_sidecar qrow_engagement qrow_sidecar_sha256

finalize_manifests
"$PYTHON_BIN" scripts/fr13_taw_b1_credential.py reduce-pair \
  --mode "$MODE" \
  --source "$TAW_SOURCE" \
  --subset "$SUBSET" \
  --credential "$TAW_B1_CREDENTIAL" \
  --b1-live-bundle "$TAW_B1_LIVE_BUNDLE" \
  --b4-production-pass "$TAW_REVIEWED_B4_PASS" \
  --b4-gate-verdict "$TAW_REVIEWED_B4_VERDICT" \
  --merge-binding "$TAW_MERGE_BINDING" \
  --production-pass "$TAW_PRODUCTION_PASS" \
  --stock-measure "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  --candidate-measure "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  --stock-health "$RUNROOT_ABS/$STOCK_ARM/health.json" \
  --candidate-health "$RUNROOT_ABS/$CANDIDATE_ARM/health.json" \
  --stock-audit "$RUNROOT_ABS/$STOCK_ARM/fixed32_chat_traffic_audit.json" \
  --candidate-audit "$RUNROOT_ABS/$CANDIDATE_ARM/fixed32_chat_traffic_audit.json" \
  --stock-sfwd-engagement "$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_sfwd_state_fusion.production_engagement.json" \
  --candidate-sfwd-engagement "$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_sfwd_state_fusion.production_engagement.json" \
  --stock-qrow-engagement "$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fa2_qrow16_production_capture.json" \
  --candidate-qrow-engagement "$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fa2_qrow16_production_capture.json" \
  --stock-taw-census "$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_work_census.jsonl" \
  --candidate-taw-census "$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_work_census.jsonl" \
  --source-commit "$SOURCE_COMMIT" \
  --runner-sha256 "$RUNNER_SHA256" \
  --out "$RUNROOT_ABS/timing_summary.json" \
  > "$RUNROOT_ABS/timing_summary.validation.json"

printf 'timing_summary=%s timing_summary_sha256=%s ended=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" \
  "$(sha256sum "$RUNROOT_ABS/timing_summary.json" | awk '{print $1}')" \
  "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
