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
: "${QROW16_FA2_SO:?set QROW16_FA2_SO to the pinned Qrow16 production binary}"
: "${GATE_RESULT_JSON:?set GATE_RESULT_JSON to the passed real-B1 gate}"
: "${GATE_RESULT_SHA256:?set GATE_RESULT_SHA256 to its raw SHA-256}"
: "${GATE_ENGAGEMENT_JSON:?set GATE_ENGAGEMENT_JSON to the gate engagement}"
: "${GATE_ACCEPTANCE_JSON:?set GATE_ACCEPTANCE_JSON to the gate deploy-speed telemetry}"
: "${GATE_FINAL_FLUSH_JSON:?set GATE_FINAL_FLUSH_JSON to the gate final flush}"
: "${GATE_BOUNDARY_SNAPSHOT_JSON:?set GATE_BOUNDARY_SNAPSHOT_JSON to the gate final boundary}"
: "${GATE_CHAT_TRAFFIC_AUDIT_JSON:?set GATE_CHAT_TRAFFIC_AUDIT_JSON to the gate traffic audit}"
: "${GATE_QROW16_SIDECAR_JSON:?set GATE_QROW16_SIDECAR_JSON to the gate Qrow16 sidecar}"
: "${GATE_QROW16_CAPTURE_JSON:?set GATE_QROW16_CAPTURE_JSON to the gate Qrow16 capture}"

FR13_DRAFT_HEAD_FP8_STATIC_IO=${FR13_DRAFT_HEAD_FP8_STATIC_IO:-0}
case "$FR13_DRAFT_HEAD_FP8_STATIC_IO" in
  0|1) ;;
  *) echo "FR13_DRAFT_HEAD_FP8_STATIC_IO must be 0 or 1" >&2; exit 2 ;;
esac

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
DRAFT_VOCAB_BLOCKS_HOST=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
QROW16_FA2_SHA256=1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86
QROW16_FA2_BYTES=299507792
QROW16_LIVE_PASS=$REPO/results/fr13_fixed32_qrow16_num_splits0_live_pass_20260731T173608Z/fr13_fa2_qrow16_live_paged_ab.json
QROW16_LIVE_PASS_SHA256=36940fd43d11399529d1bfe7e11baa9961907193267f3bb43d41057328737b77
CANDIDATE_SOURCE=scripts/fr10_phase4_patch_vllm_tree_gdn.py
CANDIDATE_SOURCE_SHA256=$(sha256sum "$CANDIDATE_SOURCE" | cut -d' ' -f1)
SOURCE_COMMIT=$(git rev-parse HEAD)
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | cut -d' ' -f1)
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO"/}
STOCK_ARM="hydra27_fixed32_k64_root_head_bf16_${TAG}"
if [[ "$FR13_DRAFT_HEAD_FP8_STATIC_IO" == "1" ]]; then
  CANDIDATE_ARM="hydra27_fixed32_k64_root_head_fp8_static_io_${TAG}"
  ONLY_ARM_DELTA=FR13_DRAFT_HEAD_FP8_0_to_1_and_STATIC_IO_0_to_1
else
  CANDIDATE_ARM="hydra27_fixed32_k64_root_head_fp8_${TAG}"
  ONLY_ARM_DELTA=FR13_DRAFT_HEAD_FP8_0_to_1
fi
PROMOTION_CREDENTIAL="$RUNROOT_ABS/gate_promotion_credential.json"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$QROW16_FA2_SO" && ! -L "$QROW16_FA2_SO" \
   && "$QROW16_FA2_SO" == /* \
   && "$(stat -c '%s' "$QROW16_FA2_SO")" == "$QROW16_FA2_BYTES" \
   && "$(sha256sum "$QROW16_FA2_SO" | cut -d' ' -f1)" == "$QROW16_FA2_SHA256" ]] \
  || { echo "QROW16_FA2_SO is not the pinned production binary" >&2; exit 2; }
[[ -f "$QROW16_LIVE_PASS" && ! -L "$QROW16_LIVE_PASS" \
   && "$(sha256sum "$QROW16_LIVE_PASS" | cut -d' ' -f1)" \
      == "$QROW16_LIVE_PASS_SHA256" ]] \
  || { echo "pinned Qrow16 live PASS identity drifted" >&2; exit 2; }
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

"$PYTHON_BIN" - "$QROW16_LIVE_PASS" "$QROW16_FA2_SHA256" <<'PY'
import sys
from pathlib import Path

from scripts import fr13_qrow16_pass_sidecar as qrow16

payload, _ = qrow16.load_json(Path(sys.argv[1]))
qrow16.validate_live_result(payload, candidate_sha256=sys.argv[2])
PY

mkdir -p "$RUNROOT_ABS"

# Rebuild the promotion credential from every raw one-task gate artifact before
# either timing arm is allowed to boot.
"$PYTHON_BIN" scripts/fr13_draft_head_fp8_gate.py \
  --engagement "$GATE_ENGAGEMENT_JSON" \
  --acceptance "$GATE_ACCEPTANCE_JSON" \
  --final-flush "$GATE_FINAL_FLUSH_JSON" \
  --boundary-snapshot "$GATE_BOUNDARY_SNAPSHOT_JSON" \
  --chat-traffic-audit "$GATE_CHAT_TRAFFIC_AUDIT_JSON" \
  --qrow16-sidecar "$GATE_QROW16_SIDECAR_JSON" \
  --qrow16-capture "$GATE_QROW16_CAPTURE_JSON" \
  --qrow16-so "$QROW16_FA2_SO" \
  --candidate-source "$CANDIDATE_SOURCE" \
  --expected-source-sha256 "$CANDIDATE_SOURCE_SHA256" \
  --expected-source-commit "$SOURCE_COMMIT" \
  --expected-static-io "$FR13_DRAFT_HEAD_FP8_STATIC_IO" \
  --repo "$PWD" \
  --gate-result "$GATE_RESULT_JSON" \
  --expected-gate-sha256 "$GATE_RESULT_SHA256" \
  --out "$PROMOTION_CREDENTIAL"
PROMOTION_CREDENTIAL_SHA256=$(sha256sum "$PROMOTION_CREDENTIAL" | cut -d' ' -f1)

printf 'classification=real_swe_verified_exact4_b1_draft_head_fp8_timing_pair\ntiming_eligible=1\nformal_floor_acceptance_eligible=0\nproduction_default_enabled=0\nonly_arm_delta=%s\ndraft_head_fp8_static_io=%s\nbatch_size=1\nconcurrency=1\nphysical_rows=32\ndraft_vocab_root=1\ndraft_vocab_k=65536\nqrow16_production=1\nlauncher_pid=%s\nrunroot=%s\nstock_arm=%s\ncandidate_arm=%s\nsource=%s\ncandidate_source_sha256=%s\nrunner_sha256=%s\nsubset_sha256=%s\ndraft_vocab_blocks_sha256=%s\nqrow16_fa2_sha256=%s\nqrow16_live_pass_sha256=%s\ngate_result_sha256=%s\npromotion_credential_sha256=%s\nstarted=%s\n' \
  "$ONLY_ARM_DELTA" "$FR13_DRAFT_HEAD_FP8_STATIC_IO" \
  "$$" "$RUNROOT_ABS" "$STOCK_ARM" "$CANDIDATE_ARM" \
  "$SOURCE_COMMIT" "$CANDIDATE_SOURCE_SHA256" "$RUNNER_SHA256" \
  "$SUBSET_SHA256" "$DRAFT_VOCAB_BLOCKS_SHA256" "$QROW16_FA2_SHA256" \
  "$QROW16_LIVE_PASS_SHA256" "$GATE_RESULT_SHA256" "$PROMOTION_CREDENTIAL_SHA256" \
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
  local expected_bytes expected_floor fp8_arm static_io
  if [[ "$fp8" == "1" ]]; then
    expected_bytes=30989326208
    expected_floor=113.514015414
    fp8_arm=$arm
    static_io=$FR13_DRAFT_HEAD_FP8_STATIC_IO
  else
    expected_bytes=27977022848
    expected_floor=102.479937172
    fp8_arm=
    static_io=0
  fi
  echo "===== $arm: real exact4 B1 FP8=$fp8 ====="
  (
    export BSIZE=1 CONC=1 WALL=0
    export FR13_DRAFT_VOCAB_ROOT=1
    export FR13_DRAFT_VOCAB_K=65536
    export FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER"
    export FR13_DRAFT_HEAD_FP8="$fp8"
    export FR13_DRAFT_HEAD_FP8_STATIC_IO="$static_io"
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
        FR13_DRAFT_HEAD_FP8_STATIC_IO="$static_io" \
        FR13_DRAFT_HEAD_FP8_ARM="$fp8_arm" \
        FR13_DRAFT_HEAD_FP8_ENGAGEMENT_JSON=/logs/fr13_draft_head_fp8.engagement.json \
        FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
        FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
        FR13_FA2_QROW16_LIVE_PAGED_AB=0 \
        FR13_FA2_QROW16_SO_SHA256="$QROW16_FA2_SHA256" \
        FR13_FA2_QROW16_PRODUCTION=1 \
        FR13_FA2_QROW16_LIVE_PASS_JSON="$QROW16_LIVE_PASS" \
        FR13_FA2_QROW16_LIVE_PASS_SHA256="$QROW16_LIVE_PASS_SHA256" \
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
        FORKED_FA2_SO="$QROW16_FA2_SO" \
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
    local container_env="$RUNROOT_ABS/$arm/container_env.txt"
    local qrow16_sidecar="$RUNROOT_ABS/$arm/logs/fr13_fa2_qrow16_production_pass.json"
    local qrow16_capture="$RUNROOT_ABS/$arm/logs/fr13_fa2_qrow16_production_capture.json"
    [[ -f "$container_env" && ! -L "$container_env" \
       && "$(grep -Fxc "FR13_DRAFT_HEAD_FP8_STATIC_IO=$static_io" "$container_env")" -eq 1 \
       && "$(grep -Fxc 'FR13_FA2_QROW16_LIVE_PAGED_AB=0' "$container_env")" -eq 1 \
       && "$(grep -Fxc 'FR13_FA2_QROW16_PRODUCTION=1' "$container_env")" -eq 1 \
       && "$(grep -Fxc "FR13_FA2_QROW16_SO_SHA256=$QROW16_FA2_SHA256" "$container_env")" -eq 1 \
       && -f "$qrow16_sidecar" && ! -L "$qrow16_sidecar" \
       && -f "$qrow16_capture" && ! -L "$qrow16_capture" ]] \
      || { echo "$arm lacks pinned Qrow16 production evidence" >&2; exit 4; }
    local qrow16_sidecar_sha256
    qrow16_sidecar_sha256=$(sha256sum "$qrow16_sidecar" | cut -d' ' -f1)
    "$PYTHON_BIN" scripts/fr13_qrow16_pass_sidecar.py verify \
      --sidecar "$qrow16_sidecar" \
      --expected-sidecar-sha256 "$qrow16_sidecar_sha256" \
      --candidate-so "$QROW16_FA2_SO" \
      --expected-candidate-sha256 "$QROW16_FA2_SHA256" \
      >/dev/null
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
STOCK_QROW16_SIDECAR="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fa2_qrow16_production_pass.json"
CANDIDATE_QROW16_SIDECAR="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fa2_qrow16_production_pass.json"
STOCK_QROW16_CAPTURE="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fa2_qrow16_production_capture.json"
CANDIDATE_QROW16_CAPTURE="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fa2_qrow16_production_capture.json"
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
  --expected-static-io "$FR13_DRAFT_HEAD_FP8_STATIC_IO" \
  --stock-arm "$STOCK_ARM" \
  --candidate-arm "$CANDIDATE_ARM" \
  --stock-qrow16-sidecar "$STOCK_QROW16_SIDECAR" \
  --candidate-qrow16-sidecar "$CANDIDATE_QROW16_SIDECAR" \
  --stock-qrow16-capture "$STOCK_QROW16_CAPTURE" \
  --candidate-qrow16-capture "$CANDIDATE_QROW16_CAPTURE" \
  --qrow16-so "$QROW16_FA2_SO" \
  --qrow16-fa2-sha256 "$QROW16_FA2_SHA256" \
  --out "$RUNROOT_ABS/timing_summary.json"

printf 'timing_summary=%s completed=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
