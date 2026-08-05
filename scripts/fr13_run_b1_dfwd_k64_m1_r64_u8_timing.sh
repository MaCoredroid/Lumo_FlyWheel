#!/usr/bin/env bash
# Real SWE-Verified B1 timing pair for stock vs credentialed K64/root1 U8.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path below output/}"
: "${TAG:?set TAG to a unique tag}"
: "${STOCK_FA2_SO:?set STOCK_FA2_SO to the pinned stock FA2 binary}"
: "${DFWD_U8_SO:?set DFWD_U8_SO to the pinned U8 shared object}"
: "${LIVE_PASS_JSON:?set LIVE_PASS_JSON to the exact real-B1 U8 PASS}"
: "${LIVE_PASS_SHA256:?set LIVE_PASS_SHA256 to its raw SHA-256}"
: "${LIVE_FINAL_FLUSH_JSON:?set LIVE_FINAL_FLUSH_JSON}"
: "${LIVE_BOUNDARY_SNAPSHOT_JSON:?set LIVE_BOUNDARY_SNAPSHOT_JSON}"
: "${LIVE_CHAT_TRAFFIC_AUDIT_JSON:?set LIVE_CHAT_TRAFFIC_AUDIT_JSON}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
TASK_SET=${TASK_SET:-exact4}
case "$TASK_SET" in
  exact4)
    SUBSET=config/fr13_fixed32/subset_b4_four.json
    SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
    EXPECTED_TASKS=4
    ;;
  exact16)
    SUBSET=config/fr13_fixed32/subset_b4_sixteen.json
    SUBSET_SHA256=47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c
    EXPECTED_TASKS=16
    ;;
  *)
    echo "TASK_SET must be exactly exact4 or exact16" >&2
    exit 2
    ;;
esac

QUALIFICATION_SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
QUALIFICATION_SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
QUALIFICATION_RUNNER=scripts/fr13_run_b1_dfwd_k64_m1_r64_u8_live_gate.sh
PATCH_SOURCE=scripts/fr10_phase4_patch_vllm_tree_gdn.py
CANDIDATE_SOURCE=csrc/fr13_bf16_gemvx_k64_m1_shuffle_r64_u8.cu
BUILD_ATTESTATION=results/fr13_fixed32_dfwd_k64_m1_r64_u8_linked_build_20260805/build_attestation.json
VOCAB_BLOCKS=scripts/fr13_dvk_subset_blocks.json
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
DFWD_U8_SHA256=8b27df4f3c6a5a0574261ee984159582a87615c3e6d83f2a267f4fa46a3e421e
CANDIDATE_SOURCE_SHA256=af0044edd84ff58d353a816f6887894d05a62b221e0efa5af933c2c59676b01b
BUILD_ATTESTATION_SHA256=e7ec95d1fff3b665373ad7b3a14f7e3fad346cf77a5f2f992a90a689e5672c8f
VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
SOURCE_COMMIT=$(git rev-parse HEAD)
PATCH_SOURCE_SHA256=$(sha256sum "$PATCH_SOURCE" | cut -d' ' -f1)
QUALIFICATION_RUNNER_SHA256=$(sha256sum "$QUALIFICATION_RUNNER" | cut -d' ' -f1)
RUNNER_SHA256=$(sha256sum "$RUNNER" | cut -d' ' -f1)
RUNROOT_ABS=$(realpath -m "$RUNROOT")
STOCK_ARM="hydra27_fixed32_u8_stock_${TASK_SET}_${TAG}"
CANDIDATE_ARM="hydra27_fixed32_u8_candidate_${TASK_SET}_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* \
   && ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be a new path below $REPO/output" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for path in \
  "$STOCK_FA2_SO" "$DFWD_U8_SO" "$LIVE_PASS_JSON" \
  "$LIVE_FINAL_FLUSH_JSON" "$LIVE_BOUNDARY_SNAPSHOT_JSON" \
  "$LIVE_CHAT_TRAFFIC_AUDIT_JSON"; do
  [[ "$path" == /* && -f "$path" && ! -L "$path" ]] \
    || { echo "required evidence must be an absolute regular file: $path" >&2; exit 2; }
done
unset path
[[ "$(sha256sum "$STOCK_FA2_SO" | cut -d' ' -f1)" == "$STOCK_FA2_SHA256" \
   && "$(sha256sum "$DFWD_U8_SO" | cut -d' ' -f1)" == "$DFWD_U8_SHA256" \
   && "$(stat -c '%s' "$DFWD_U8_SO")" == "117904" \
   && "$(sha256sum "$CANDIDATE_SOURCE" | cut -d' ' -f1)" == "$CANDIDATE_SOURCE_SHA256" \
   && "$(sha256sum "$BUILD_ATTESTATION" | cut -d' ' -f1)" == "$BUILD_ATTESTATION_SHA256" \
   && "$(sha256sum "$VOCAB_BLOCKS" | cut -d' ' -f1)" == "$VOCAB_BLOCKS_SHA256" \
   && "$(sha256sum "$QUALIFICATION_SUBSET" | cut -d' ' -f1)" == "$QUALIFICATION_SUBSET_SHA256" \
   && "$(sha256sum "$SUBSET" | cut -d' ' -f1)" == "$SUBSET_SHA256" \
   && "$LIVE_PASS_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$LIVE_PASS_JSON" | cut -d' ' -f1)" == "$LIVE_PASS_SHA256" ]] \
  || { echo "U8 timing input identity drifted" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
PREFLIGHT_CREDENTIAL="$RUNROOT_ABS/qualification_credential.preflight.json"
"$PYTHON_BIN" scripts/fr13_dfwd_k64_m1_r64_u8_production_credential.py issue \
  --live-result "$LIVE_PASS_JSON" \
  --final-flush "$LIVE_FINAL_FLUSH_JSON" \
  --boundary-snapshot "$LIVE_BOUNDARY_SNAPSHOT_JSON" \
  --chat-traffic-audit "$LIVE_CHAT_TRAFFIC_AUDIT_JSON" \
  --repo "$REPO" \
  --candidate-so "$DFWD_U8_SO" \
  --candidate-source "$CANDIDATE_SOURCE" \
  --build-attestation "$BUILD_ATTESTATION" \
  --patch-source "$PATCH_SOURCE" \
  --qualification-runner "$QUALIFICATION_RUNNER" \
  --subset "$QUALIFICATION_SUBSET" \
  --vocab-blocks "$VOCAB_BLOCKS" \
  --fa2-so "$STOCK_FA2_SO" \
  --expected-source-commit "$SOURCE_COMMIT" \
  --out "$PREFLIGHT_CREDENTIAL" >/dev/null
PREFLIGHT_CREDENTIAL_SHA256=$(sha256sum "$PREFLIGHT_CREDENTIAL" | cut -d' ' -f1)

printf 'classification=real_swe_verified_b1_u8_timing_pair\ntask_set=%s\nexpected_tasks=%s\nproduction_default_enabled=0\nperformance_claim=0\nfloor_acceptance_eligible=0\nsource_commit=%s\nrunner_sha256=%s\nsubset_sha256=%s\nlive_pass_sha256=%s\npreflight_credential_sha256=%s\nstock_arm=%s\ncandidate_arm=%s\nstarted=%s\n' \
  "$TASK_SET" "$EXPECTED_TASKS" "$SOURCE_COMMIT" "$RUNNER_SHA256" \
  "$SUBSET_SHA256" "$LIVE_PASS_SHA256" "$PREFLIGHT_CREDENTIAL_SHA256" \
  "$STOCK_ARM" "$CANDIDATE_ARM" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$REPO" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$REPO" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

run_arm() {
  local arm=$1
  local production=$2
  local live_json=""
  local live_sha=""
  local final_flush=""
  local boundary=""
  local traffic=""
  local candidate_so=""
  local candidate_so_sha=""
  local candidate_source_sha=""
  local build_sha=""
  local patch_sha=""
  local qualification_runner_sha=""
  local qualification_subset_sha=""
  local vocab_sha=""
  local fa2_sha=""
  local qualification_commit=""
  local qualification_instance=""
  if [[ "$production" == "1" ]]; then
    live_json=$LIVE_PASS_JSON
    live_sha=$LIVE_PASS_SHA256
    final_flush=$LIVE_FINAL_FLUSH_JSON
    boundary=$LIVE_BOUNDARY_SNAPSHOT_JSON
    traffic=$LIVE_CHAT_TRAFFIC_AUDIT_JSON
    candidate_so=$DFWD_U8_SO
    candidate_so_sha=$DFWD_U8_SHA256
    candidate_source_sha=$CANDIDATE_SOURCE_SHA256
    build_sha=$BUILD_ATTESTATION_SHA256
    patch_sha=$PATCH_SOURCE_SHA256
    qualification_runner_sha=$QUALIFICATION_RUNNER_SHA256
    qualification_subset_sha=$QUALIFICATION_SUBSET_SHA256
    vocab_sha=$VOCAB_BLOCKS_SHA256
    fa2_sha=$STOCK_FA2_SHA256
    qualification_commit=$SOURCE_COMMIT
    qualification_instance=astropy__astropy-12907
  fi
  echo "===== $arm: B1 task_set=$TASK_SET U8 production=$production ====="
  env \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
    FR13_FIXED32_B1_DIAGNOSTIC=0 \
    FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
    FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
    FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB=0 \
    FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION="$production" \
    FR13_DRAFT_HEAD_M1_R64_U8_SO="$candidate_so" \
    FR13_DRAFT_HEAD_M1_R64_U8_SO_SHA256="$candidate_so_sha" \
    FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_SHA256="$candidate_source_sha" \
    FR13_DRAFT_HEAD_M1_R64_U8_BUILD_ATTESTATION_SHA256="$build_sha" \
    FR13_DRAFT_HEAD_M1_R64_U8_PATCH_SOURCE_SHA256="$patch_sha" \
    FR13_DRAFT_HEAD_M1_R64_U8_RUNNER_SHA256="$qualification_runner_sha" \
    FR13_DRAFT_HEAD_M1_R64_U8_SUBSET_SHA256="$qualification_subset_sha" \
    FR13_DRAFT_HEAD_M1_R64_U8_VOCAB_BLOCKS_SHA256="$vocab_sha" \
    FR13_DRAFT_HEAD_M1_R64_U8_FA2_SHA256="$fa2_sha" \
    FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_COMMIT="$qualification_commit" \
    FR13_DRAFT_HEAD_M1_R64_U8_INSTANCE_ID="$qualification_instance" \
    FR13_DRAFT_HEAD_M1_R64_U8_LIVE_PASS_JSON="$live_json" \
    FR13_DRAFT_HEAD_M1_R64_U8_LIVE_PASS_SHA256="$live_sha" \
    FR13_DRAFT_HEAD_M1_R64_U8_LIVE_FINAL_FLUSH_JSON="$final_flush" \
    FR13_DRAFT_HEAD_M1_R64_U8_LIVE_BOUNDARY_SNAPSHOT_JSON="$boundary" \
    FR13_DRAFT_HEAD_M1_R64_U8_LIVE_CHAT_TRAFFIC_AUDIT_JSON="$traffic" \
    FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION_ENGAGEMENT_JSON=/logs/fr13_dfwd_k64_m1_r64_u8.production_engagement.json \
    FR13_DRAFT_HEAD_FP8=0 FR13_DFWD_K64_TOP3=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_SO= \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 FORKED_FA2_SO="$STOCK_FA2_SO" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$arm" hydra27_fixed32 "$SUBSET" \
      > "$RUNROOT_ABS/$arm.runlog" 2>&1
  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$arm" --out-root "$RUNROOT_ABS/$arm/swe_out" \
    --expected-tok-per-draft 31 --batch-size 1 \
    --out "$RUNROOT_ABS/$arm/deploy_speed_fullwall.json"
  printf 'arm=%s production=%s ended=%s\n' \
    "$arm" "$production" "$(date -u +%FT%TZ)" \
    >> "$RUNROOT_ABS/launcher_meta.txt"
}

run_arm "$STOCK_ARM" 0
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after stock" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" 1

STOCK_ENGAGEMENT="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_dfwd_k64_m1_r64_u8.production_engagement.json"
CANDIDATE_ENGAGEMENT="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_dfwd_k64_m1_r64_u8.production_engagement.json"
CANDIDATE_CREDENTIAL="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_dfwd_k64_m1_r64_u8.production_credential.json"
[[ ! -e "$STOCK_ENGAGEMENT" && ! -L "$STOCK_ENGAGEMENT" ]] \
  || { echo "stock arm emitted U8 engagement" >&2; exit 4; }
[[ -f "$CANDIDATE_CREDENTIAL" && ! -L "$CANDIDATE_CREDENTIAL" ]] \
  || { echo "candidate credential is missing" >&2; exit 4; }
CANDIDATE_CREDENTIAL_SHA256=$(sha256sum "$CANDIDATE_CREDENTIAL" | cut -d' ' -f1)
"$PYTHON_BIN" scripts/fr13_dfwd_k64_m1_r64_u8_production_credential.py engagement \
  --engagement "$CANDIDATE_ENGAGEMENT" \
  --expected-credential-sha256 "$CANDIDATE_CREDENTIAL_SHA256" \
  --expected-source-commit "$SOURCE_COMMIT" \
  > "$RUNROOT_ABS/engagement_validation.json"

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$REPO" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$REPO" --output "$RUNROOT_ABS/external_manifest.at_end.json"
cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
  "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  || { echo "runtime manifest changed during timing" >&2; exit 14; }
cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
  "$RUNROOT_ABS/external_manifest.at_end.json" \
  || { echo "external manifest changed during timing" >&2; exit 14; }
[[ "$(sha256sum "$RUNNER" | cut -d' ' -f1)" == "$RUNNER_SHA256" ]] \
  || { echo "timing runner changed during execution" >&2; exit 14; }

"$PYTHON_BIN" - \
  "$SUBSET" \
  "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/engagement_validation.json" \
  "$RUNROOT_ABS/timing_summary.json" \
  "$TASK_SET" "$EXPECTED_TASKS" "$SOURCE_COMMIT" \
  "$SUBSET_SHA256" "$LIVE_PASS_SHA256" "$CANDIDATE_CREDENTIAL_SHA256" <<'PY'
import json
import sys
from pathlib import Path

subset_path, stock_path, candidate_path, engagement_path, out_path = map(
    Path, sys.argv[1:6]
)
task_set, expected_tasks, source_commit = sys.argv[6:9]
subset_sha256, live_pass_sha256, credential_sha256 = sys.argv[9:12]
expected_ids = json.loads(subset_path.read_text())["instance_ids"]
stock = json.loads(stock_path.read_text())
candidate = json.loads(candidate_path.read_text())
engagement = json.loads(engagement_path.read_text())
if (
    len(expected_ids) != int(expected_tasks)
    or stock.get("schema") != "fr13.measure.deploy_speed.v1"
    or candidate.get("schema") != "fr13.measure.deploy_speed.v1"
    or stock.get("batch_size") != 1
    or candidate.get("batch_size") != 1
    or stock.get("task_instance_ids") != expected_ids
    or candidate.get("task_instance_ids") != expected_ids
    or stock.get("n_tasks") != len(expected_ids)
    or candidate.get("n_tasks") != len(expected_ids)
    or engagement.get("status") != "PASS"
    or engagement.get("candidate_served") is not True
):
    raise SystemExit("U8 fixed-task timing evidence drifted")

keys = (
    "accept_per_event",
    "committer_gpu_ms_per_step",
    "derived_tps_fullstep_gpu",
    "drafter_gpu_ms_per_step",
    "measured_tps_fullstep_wall",
    "s_per_fwd_gpu",
    "step_wall_ms",
)
payload = {
    "schema": "fr13.fixed32.dfwd_k64_m1_r64_u8_timing.v1",
    "status": "MEASURED",
    "task_set": task_set,
    "task_ids": expected_ids,
    "batch_size": 1,
    "source_commit": source_commit,
    "subset_sha256": subset_sha256,
    "live_pass_sha256": live_pass_sha256,
    "production_credential_sha256": credential_sha256,
    "selector": "fr13_bf16_k64_m1_r64_u8_direct",
    "stock": {key: stock.get(key) for key in keys},
    "candidate": {key: candidate.get(key) for key in keys},
    "graph_lifecycle_validated": True,
    "production_default_enabled": False,
    "performance_claim": False,
    "floor_acceptance_eligible": False,
}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

printf 'completed=%s\n' "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
echo "U8 B1 timing completed: $RUNROOT_ABS/timing_summary.json"
