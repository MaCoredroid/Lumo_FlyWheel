#!/usr/bin/env bash
# Exact4 real SWE-Verified K64 B1 timing: stock vs gated coefficient staging.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path below output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned FA2 shared object}"
: "${LIVE_PASS_JSON:?set LIVE_PASS_JSON to the completed real-task byte PASS}"
: "${LIVE_PASS_SHA256:?set LIVE_PASS_SHA256 to its raw SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
TOPOLOGY=${TOPOLOGY:-hydra27_fixed32}
case "$TOPOLOGY" in
  tail6_fixed32|hydra27_fixed32) ;;
  *) echo "TOPOLOGY must be tail6_fixed32 or hydra27_fixed32" >&2; exit 2 ;;
esac
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
TASK_ID=astropy__astropy-12907
KERNEL_SOURCE=src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
MANDATORY_WEIGHT_BYTES=32666638208
MANDATORY_WEIGHT_FLOOR_MS=119.658015414
ONE_SIDED_U95_CAP_MS=137.6067177261
SOURCE_COMMIT=$(git rev-parse HEAD)
SOURCE_SHA256=$(sha256sum "$KERNEL_SOURCE" | awk '{print $1}')
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
FA2_SHA256=$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
STOCK_ARM="${TOPOLOGY}_k64_gdn_coeff_stock_${TAG}"
CANDIDATE_ARM="${TOPOLOGY}_k64_gdn_coeff_candidate_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable" >&2; exit 2; }
for required in "$FORKED_FA2_SO" "$LIVE_PASS_JSON"; do
  [[ "$required" == /* && -f "$required" && ! -L "$required" ]] \
    || { echo "required input must be an absolute regular file: $required" >&2; exit 2; }
done
[[ "$LIVE_PASS_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$LIVE_PASS_JSON" | awk '{print $1}')" == "$LIVE_PASS_SHA256" ]] \
  || { echo "live PASS SHA-256 is invalid or mismatched" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" ]] \
  || { echo "canonical exact4/K64 inputs drifted" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }

"$PYTHON_BIN" scripts/fr13_gdn_level0_coeff_pass.py \
  --live-result "$LIVE_PASS_JSON" \
  --kernel-source "$KERNEL_SOURCE" \
  --expected-task-id "$TASK_ID" \
  --expected-mode "$TOPOLOGY" \
  --expected-live-sha256 "$LIVE_PASS_SHA256" \
  >/dev/null
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" ]] \
  || { echo "ROOT=1 K64 hardware-floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"
printf 'classification=real_swe_verified_exact4_k64_b1_gdn_coefficient_timing\ntiming_eligible=1\nfloor_acceptance_eligible=0\nonly_arm_delta=fixed32_gdn_level0_coeff\ntopology=%s\nbatch_size=1\nconcurrency=1\nphysical_rows=32\ndraft_vocab_root=1\ndraft_vocab_k=65536\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nsource=%s\nsource_sha256=%s\nrunner_sha256=%s\nsubset_sha256=%s\nblock_map_sha256=%s\nfa2_sha256=%s\nlive_pass_sha256=%s\nstarted=%s\n' \
  "$TOPOLOGY" "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$SOURCE_COMMIT" "$SOURCE_SHA256" \
  "$RUNNER_SHA256" "$SUBSET_SHA256" "$BLOCK_MAP_SHA256" "$FA2_SHA256" \
  "$LIVE_PASS_SHA256" "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"

MANIFEST_FINALIZED=0
finalize_manifests() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json"
  cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json"
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json"
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]]
  MANIFEST_FINALIZED=1
}
trap 'rc=$?; trap - EXIT; (( MANIFEST_FINALIZED == 1 )) || finalize_manifests || rc=$?; exit "$rc"' EXIT

run_arm() {
  local arm=$1
  local production=$2
  local pass_json=""
  local pass_sha=""
  local pass_task=""
  if [[ "$production" == "1" ]]; then
    pass_json=$LIVE_PASS_JSON
    pass_sha=$LIVE_PASS_SHA256
    pass_task=$TASK_ID
  fi
  if env \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
      FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json \
      FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES" \
      FR13_WEIGHT_FLOOR_MS="$MANDATORY_WEIGHT_FLOOR_MS" \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json" \
      FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 \
      FR13_FIXED32_GDN_LEVEL0_COEFF_BYTE_AB=0 \
      FR13_FIXED32_GDN_LEVEL0_COEFF="$production" \
      FR13_FIXED32_GDN_LEVEL0_COEFF_PASS_JSON="$pass_json" \
      FR13_FIXED32_GDN_LEVEL0_COEFF_PASS_SHA256="$pass_sha" \
      FR13_FIXED32_GDN_LEVEL0_COEFF_PASS_TASK_ID="$pass_task" \
      FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
      FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
      FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
      FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
      FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
      FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
      FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
      FORKED_FA2_SO="$FORKED_FA2_SO" RUNROOT="$RUNROOT_ABS" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh \
        "$arm" "$TOPOLOGY" "$SUBSET" \
        > "$RUNROOT_ABS/$arm.runlog" 2>&1; then
    :
  else
    local serve_rc=$?
    printf 'arm=%s serve_rc=%s ended=%s\n' \
      "$arm" "$serve_rc" "$(date -u +%FT%TZ)" \
      >> "$RUNROOT_ABS/launcher_meta.txt"
    return "$serve_rc"
  fi
  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$arm" --out-root "$RUNROOT_ABS/$arm/swe_out" \
    --expected-tok-per-draft 31 --batch-size 1 \
    --out "$RUNROOT_ABS/$arm/deploy_speed_fullwall.json"
  printf 'arm=%s serve_rc=0 ended=%s\n' \
    "$arm" "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
}

run_arm "$STOCK_ARM" 0
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after the stock arm" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" 1

STOCK_ENGAGEMENT="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_gdn_level0_coeff.production_engagement.json"
CANDIDATE_ENGAGEMENT="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_gdn_level0_coeff.production_engagement.json"
[[ ! -e "$STOCK_ENGAGEMENT" && ! -L "$STOCK_ENGAGEMENT" ]] \
  || { echo "stock arm emitted coefficient production engagement" >&2; exit 4; }
[[ -f "$CANDIDATE_ENGAGEMENT" && ! -L "$CANDIDATE_ENGAGEMENT" ]] \
  || { echo "candidate arm lacks coefficient production engagement" >&2; exit 4; }
finalize_manifests

"$PYTHON_BIN" - \
  "$SUBSET" \
  "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  "$CANDIDATE_ENGAGEMENT" "$RUNROOT_ABS/timing_summary.json" \
  "$TOPOLOGY" "$SOURCE_SHA256" "$LIVE_PASS_SHA256" \
  "$MANDATORY_WEIGHT_FLOOR_MS" "$ONE_SIDED_U95_CAP_MS" <<'PY'
import json
import math
import sys
from pathlib import Path

subset_path, stock_path, candidate_path, engagement_path, out_path = map(
    Path, sys.argv[1:6]
)
topology, source_sha, pass_sha = sys.argv[6:9]
floor_ms, cap_ms = map(float, sys.argv[9:11])
task_ids = sorted(json.loads(subset_path.read_text(encoding="ascii"))["instance_ids"])
stock = json.loads(stock_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
engagement = json.loads(engagement_path.read_text(encoding="ascii"))


def positive(record, key):
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{key} is absent from full-wall evidence")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"{key} is not finite and positive")
    return value


def phase_record(record, label):
    if (
        record.get("schema") != "fr13.measure.deploy_speed.v1"
        or record.get("regime") != "deployment"
        or record.get("instrument") != "OFF"
        or record.get("batch_size") != 1
        or record.get("n_tasks") != 4
        or sorted(record.get("task_instance_ids", [])) != task_ids
    ):
        raise SystemExit(f"{label} is not exact4 real SWE-Verified B1 evidence")
    wall = positive(record, "step_wall_ms")
    sfwd = positive(record, "s_per_fwd_gpu") * 1000.0
    dfwd = positive(record, "drafter_gpu_ms_per_step")
    cfwd = positive(record, "committer_gpu_ms_per_step")
    other = wall - sfwd - dfwd - cfwd
    if other < -1e-6:
        raise SystemExit(f"{label} phase timers exceed full wall")
    return {
        "step_wall_ms": wall,
        "measured_tps_fullstep_wall": positive(
            record, "measured_tps_fullstep_wall"
        ),
        "accepted_drafts_per_event": positive(record, "accept_per_event"),
        "sfwd_gpu_ms_per_step": sfwd,
        "dfwd_gpu_ms_per_step": dfwd,
        "cfwd_gpu_ms_per_step": cfwd,
        "other_wall_ms_per_step": max(other, 0.0),
        "wall_steps_measured": positive(record, "wall_steps_measured"),
        "step_wall_to_mandatory_weight_floor_ratio": wall / floor_ms,
        "within_1p15x_weight_floor": wall <= cap_ms,
    }


if (
    engagement.get("schema")
    != "fr13.fixed32.gdn_level0_coeff.production_engagement.v1"
    or engagement.get("status") != "ENGAGED"
    or engagement.get("candidate") != "fixed32_gdn_level0_coeff_v1"
    or engagement.get("source_sha256") != source_sha
    or engagement.get("production_pass_sha256") != pass_sha
    or engagement.get("mode") != topology
    or engagement.get("batch_size") != 1
    or engagement.get("records") != 48
    or engagement.get("physical_rows") != 32
    or engagement.get("path_lengths") != [5, 7]
    or engagement.get("launches_per_layer") != 2
    or engagement.get("fallback") != 0
):
    raise SystemExit("candidate production engagement is invalid")

stock_phase = phase_record(stock, "stock")
candidate_phase = phase_record(candidate, "candidate")
summary = {
    "schema": "fr13.fixed32.gdn_level0_coeff.k64_b1_timing_pair.v1",
    "status": "complete",
    "run_classification": "real_swe_verified_exact4_k64_b1_timing",
    "task_ids": task_ids,
    "topology": topology,
    "batch_size": 1,
    "concurrency": 1,
    "physical_rows": 32,
    "draft_vocab_root": 1,
    "draft_vocab_k": 65536,
    "mandatory_weight_floor_ms": floor_ms,
    "one_sided_u95_cap_ms": cap_ms,
    "stock": stock_phase,
    "candidate": candidate_phase,
    "candidate_to_stock_full_wall_tps_ratio": (
        candidate_phase["measured_tps_fullstep_wall"]
        / stock_phase["measured_tps_fullstep_wall"]
    ),
    "stock_to_candidate_step_wall_ratio": (
        stock_phase["step_wall_ms"] / candidate_phase["step_wall_ms"]
    ),
    "formal_floor_acceptance_eligible": False,
    "formal_floor_acceptance_reason": (
        "paired exact4 timing candidate; canonical two-arm statistical floor "
        "acceptance was not run"
    ),
}
out_path.write_text(
    json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
PY

printf 'timing_summary=%s completed=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
