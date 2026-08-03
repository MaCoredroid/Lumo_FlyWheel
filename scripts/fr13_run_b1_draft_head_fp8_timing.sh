#!/usr/bin/env bash
# Real SWE-Verified exact4 B1 timing pair: stock BF16 vs direct block-FP8 head.
# This is a timing promotion screen, not the formal Tail23/Hydra27 floor gate.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${STOCK_FA2_SO:?set STOCK_FA2_SO to the canonical stock FA2 binary}"
: "${GATE_RESULT_JSON:?set GATE_RESULT_JSON to the passed real-B1 gate}"
: "${GATE_RESULT_SHA256:?set GATE_RESULT_SHA256 to its raw SHA-256}"
: "${GATE_ENGAGEMENT_JSON:?set GATE_ENGAGEMENT_JSON to the gate engagement}"
: "${GATE_ACCEPTANCE_JSON:?set GATE_ACCEPTANCE_JSON to the gate deploy-speed telemetry}"
: "${GATE_FINAL_FLUSH_JSON:?set GATE_FINAL_FLUSH_JSON to the gate final flush}"
: "${GATE_BOUNDARY_SNAPSHOT_JSON:?set GATE_BOUNDARY_SNAPSHOT_JSON to the gate final boundary}"
: "${GATE_CHAT_TRAFFIC_AUDIT_JSON:?set GATE_CHAT_TRAFFIC_AUDIT_JSON to the gate traffic audit}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
DRAFT_VOCAB_BLOCKS_HOST=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
CANDIDATE_SOURCE=scripts/fr10_phase4_patch_vllm_tree_gdn.py
CANDIDATE_SOURCE_SHA256=$(sha256sum "$CANDIDATE_SOURCE" | cut -d' ' -f1)
SOURCE_COMMIT=$(git rev-parse HEAD)
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | cut -d' ' -f1)
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO"/}
STOCK_ARM="hydra27_fixed32_k64_root_head_bf16_${TAG}"
CANDIDATE_ARM="hydra27_fixed32_k64_root_head_fp8_${TAG}"
PROMOTION_CREDENTIAL="$RUNROOT_ABS/gate_promotion_credential.json"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$STOCK_FA2_SO" && ! -L "$STOCK_FA2_SO" \
   && "$STOCK_FA2_SO" == /* ]] \
  || { echo "STOCK_FA2_SO must be an absolute regular non-symlink" >&2; exit 2; }
[[ "$(sha256sum "$STOCK_FA2_SO" | cut -d' ' -f1)" == "$STOCK_FA2_SHA256" ]] \
  || { echo "STOCK_FA2_SO is not the canonical stock reference" >&2; exit 2; }
[[ -f "$SUBSET" && ! -L "$SUBSET" \
   && "$(sha256sum "$SUBSET" | cut -d' ' -f1)" == "$SUBSET_SHA256" ]] \
  || { echo "canonical exact4 subset SHA-256 drifted" >&2; exit 2; }
[[ -f "$DRAFT_VOCAB_BLOCKS_HOST" && ! -L "$DRAFT_VOCAB_BLOCKS_HOST" \
   && "$(sha256sum "$DRAFT_VOCAB_BLOCKS_HOST" | cut -d' ' -f1)" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
  || { echo "canonical K64 vocabulary map drifted" >&2; exit 2; }
[[ "$GATE_RESULT_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "GATE_RESULT_SHA256 must be lowercase SHA-256" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"

# Rebuild the promotion credential from every raw one-task gate artifact before
# either timing arm is allowed to boot.
"$PYTHON_BIN" scripts/fr13_draft_head_fp8_gate.py \
  --engagement "$GATE_ENGAGEMENT_JSON" \
  --acceptance "$GATE_ACCEPTANCE_JSON" \
  --final-flush "$GATE_FINAL_FLUSH_JSON" \
  --boundary-snapshot "$GATE_BOUNDARY_SNAPSHOT_JSON" \
  --chat-traffic-audit "$GATE_CHAT_TRAFFIC_AUDIT_JSON" \
  --candidate-source "$CANDIDATE_SOURCE" \
  --expected-source-sha256 "$CANDIDATE_SOURCE_SHA256" \
  --expected-source-commit "$SOURCE_COMMIT" \
  --repo "$PWD" \
  --gate-result "$GATE_RESULT_JSON" \
  --expected-gate-sha256 "$GATE_RESULT_SHA256" \
  --out "$PROMOTION_CREDENTIAL"
PROMOTION_CREDENTIAL_SHA256=$(sha256sum "$PROMOTION_CREDENTIAL" | cut -d' ' -f1)

printf 'classification=real_swe_verified_exact4_b1_draft_head_fp8_timing_pair\ntiming_eligible=1\nformal_floor_acceptance_eligible=0\nproduction_default_enabled=0\nonly_arm_delta=FR13_DRAFT_HEAD_FP8_0_to_1\nbatch_size=1\nconcurrency=1\nphysical_rows=32\ndraft_vocab_root=1\ndraft_vocab_k=65536\nlauncher_pid=%s\nrunroot=%s\nstock_arm=%s\ncandidate_arm=%s\nsource=%s\ncandidate_source_sha256=%s\nrunner_sha256=%s\nsubset_sha256=%s\ndraft_vocab_blocks_sha256=%s\nstock_fa2_sha256=%s\ngate_result_sha256=%s\npromotion_credential_sha256=%s\nstarted=%s\n' \
  "$$" "$RUNROOT_ABS" "$STOCK_ARM" "$CANDIDATE_ARM" \
  "$SOURCE_COMMIT" "$CANDIDATE_SOURCE_SHA256" "$RUNNER_SHA256" \
  "$SUBSET_SHA256" "$DRAFT_VOCAB_BLOCKS_SHA256" "$STOCK_FA2_SHA256" \
  "$GATE_RESULT_SHA256" "$PROMOTION_CREDENTIAL_SHA256" \
  "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

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
  [[ "$(sha256sum "$RUNNER_PATH" | cut -d' ' -f1)" == "$RUNNER_SHA256" ]] \
    || { echo "FP8 timing runner changed during execution" >&2; return 14; }
  [[ "$(sha256sum "$GATE_RESULT_JSON" | cut -d' ' -f1)" == "$GATE_RESULT_SHA256" ]] \
    || { echo "real-B1 gate result changed during timing" >&2; return 14; }
  MANIFEST_FINALIZED=1
}

runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    finalize_manifests \
      || { local manifest_rc=$?; (( rc == 0 )) && rc=$manifest_rc; }
  fi
  exit "$rc"
}
trap runner_exit EXIT

run_arm() {
  local arm=$1
  local fp8=$2
  local expected_bytes expected_floor fp8_arm
  if [[ "$fp8" == "1" ]]; then
    expected_bytes=30989326208
    expected_floor=113.514015414
    fp8_arm=$arm
  else
    expected_bytes=32666638208
    expected_floor=119.658015414
    fp8_arm=
  fi
  echo "===== $arm: real exact4 B1 FP8=$fp8 ====="
  (
    export BSIZE=1 CONC=1 WALL=0
    export FR13_DRAFT_VOCAB_ROOT=1
    export FR13_DRAFT_VOCAB_K=65536
    export FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER"
    export FR13_DRAFT_HEAD_FP8="$fp8"
    export FR13_NEEDS_ALLOW=
    export FR13_FLOOR_ORDER=HT
    source scripts/fr13_canonical_env.sh
    run_variant() { :; }
    source "$SEQUENCE"
    unset -f run_variant
    [[ "$FR13_MANDATORY_WEIGHT_BYTES" == "$expected_bytes" \
       && "$FR13_WEIGHT_FLOOR_MS" == "$expected_floor" ]] \
      || { echo "arm-specific floor contract drifted" >&2; exit 2; }

    if env \
        RUNROOT="$RUNROOT_ABS" \
        OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
        LUMO_SWE_AUTOCOMMIT=0 \
        FR13_FIXED32_B1_DIAGNOSTIC=0 \
        FR10_METRICS=0 ENFORCE_EAGER=0 \
        CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
        FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
        FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}.json" \
        FR13_DFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_dfwd.json" \
        FR13_CFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${arm}_cfwd.json" \
        FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
        FR13_DRAFT_HEAD_M32_LIVE_AB=0 \
        FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
        FR13_DRAFT_HEAD_M32_TIMING_ARM=0 \
        FR13_DRAFT_HEAD_FP8="$fp8" \
        FR13_DRAFT_HEAD_FP8_ARM="$fp8_arm" \
        FR13_DRAFT_HEAD_FP8_ENGAGEMENT_JSON=/logs/fr13_draft_head_fp8.engagement.json \
        FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
        FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
        FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
        FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
        FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
        FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
        FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
        FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
        FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
        FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
        FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
        FR13_FIXED32_CUTLASS_WAVE=stock \
        FR13_FIXED32_CUTLASS_WAVE_SO= \
        FR13_FIXED32_ATTRIBUTION_ONLY=0 \
        FORKED_FA2_SO="$STOCK_FA2_SO" \
        bash scripts/fr13_bigdenom_swe_serve_variant.sh \
          "$arm" hydra27_fixed32 "$SUBSET" \
          > "$RUNROOT_ABS/$arm.runlog" 2>&1; then
      :
    else
      local serve_rc=$?
      printf 'arm=%s fp8=%s serve_rc=%s ended=%s\n' \
        "$arm" "$fp8" "$serve_rc" "$(date -u +%FT%TZ)" \
        >> "$RUNROOT_ABS/launcher_meta.txt"
      exit "$serve_rc"
    fi
    "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
      --arm "$arm" \
      --out-root "$RUNROOT_ABS/$arm/swe_out" \
      --expected-tok-per-draft 31 \
      --batch-size 1 \
      --out "$RUNROOT_ABS/$arm/deploy_speed_fullwall.json"
  )
  printf 'arm=%s fp8=%s serve_rc=0 ended=%s\n' \
    "$arm" "$fp8" "$(date -u +%FT%TZ)" \
    >> "$RUNROOT_ABS/launcher_meta.txt"
}

run_arm "$STOCK_ARM" 0
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after stock arm" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" 1
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after candidate arm" >&2; exit 2; }

STOCK_ENGAGEMENT="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_draft_head_fp8.engagement.json"
CANDIDATE_ENGAGEMENT="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_draft_head_fp8.engagement.json"
[[ ! -e "$STOCK_ENGAGEMENT" && ! -L "$STOCK_ENGAGEMENT" ]] \
  || { echo "stock arm emitted FP8 engagement" >&2; exit 4; }
[[ -f "$CANDIDATE_ENGAGEMENT" && ! -L "$CANDIDATE_ENGAGEMENT" ]] \
  || { echo "candidate FP8 engagement is missing" >&2; exit 4; }

finalize_manifests
"$PYTHON_BIN" scripts/fr13_draft_head_fp8_timing.py \
  --subset "$SUBSET" \
  --stock "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  --candidate "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  --candidate-engagement "$CANDIDATE_ENGAGEMENT" \
  --promotion-credential "$PROMOTION_CREDENTIAL" \
  --expected-source-commit "$SOURCE_COMMIT" \
  --expected-source-sha256 "$CANDIDATE_SOURCE_SHA256" \
  --stock-arm "$STOCK_ARM" \
  --candidate-arm "$CANDIDATE_ARM" \
  --stock-fa2-sha256 "$STOCK_FA2_SHA256" \
  --out "$RUNROOT_ABS/timing_summary.json"

printf 'timing_summary=%s completed=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
