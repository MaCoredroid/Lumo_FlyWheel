#!/usr/bin/env bash
# Real SWE-Verified exact4 K64 B1 timing: Hydra27 qrow16 + SFWD state fusion.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${QROW16_FA2_SO:?set QROW16_FA2_SO to the pinned qrow16 binary}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
QROW16_SHA256=1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86
QROW16_BYTES=299507792
QROW16_PASS=$REPO/results/fr13_fixed32_qrow16_num_splits0_live_pass_20260731T173608Z/fr13_fa2_qrow16_live_paged_ab.json
QROW16_PASS_SHA256=36940fd43d11399529d1bfe7e11baa9961907193267f3bb43d41057328737b77
SFWD_PASS=$REPO/results/fr13_fixed32_sfwd_b1_real_task_byte_pass_20260801/run_evidence/fr13_fixed32_sfwd_state_fusion.live_pass.json
SFWD_PASS_SHA256=7ccfaf5cc907909b0646b752b94027e250b234a3b98bf461de61e6ae70f31782
BASELINE=$REPO/results/fr13_fixed32_qrow16_prod_exact4_b1_20260731T182827Z/hydra_valid/deploy_speed_qrow16_prod_exact4_b1_20260731T182827Z.json
BASELINE_SHA256=0350e791bc825083bfc3635e11c875617fa1d3823eba5f93ebd7f392c50f18d0
MANDATORY_WEIGHT_BYTES=32666638208
MANDATORY_WEIGHT_FLOOR_MS=119.658015414
ONE_SIDED_U95_CAP_MS=137.6067177261
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
ARM="hydra27_fixed32_k64_qrow16_sfwd_fusion_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for required in "$QROW16_FA2_SO" "$QROW16_PASS" "$SFWD_PASS" "$BASELINE"; do
  [[ "$required" == /* && -f "$required" && ! -L "$required" ]] \
    || { echo "required input must be an absolute regular file: $required" >&2; exit 2; }
done
unset required
[[ "$(stat -c '%s' "$QROW16_FA2_SO")" == "$QROW16_BYTES" \
   && "$(sha256sum "$QROW16_FA2_SO" | awk '{print $1}')" == "$QROW16_SHA256" ]] \
  || { echo "QROW16_FA2_SO is not the pinned candidate" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$QROW16_PASS" | awk '{print $1}')" == "$QROW16_PASS_SHA256" \
   && "$(sha256sum "$SFWD_PASS" | awk '{print $1}')" == "$SFWD_PASS_SHA256" \
   && "$(sha256sum "$BASELINE" | awk '{print $1}')" == "$BASELINE_SHA256" ]] \
  || { echo "tracked timing prerequisite identity drifted" >&2; exit 2; }
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
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_K=65536
export FR13_FLOOR_ORDER=HT
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_DRAFT_VOCAB_ROOT" == "1" \
   && "$FR13_DRAFT_VOCAB_K" == "65536" \
   && "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" ]] \
  || { echo "ROOT=1 K64 hardware-floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS/sidecars"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"
printf 'classification=real_swe_verified_exact4_k64_b1_kernel_stack\ntask_count=4\nbatch_size=1\nconcurrency=1\ntiming_eligible=1\nfloor_acceptance_eligible=0\ntopology=hydra27_fixed32\nphysical_rows=32\nlogical_drafts=27\nvalid_mask=0x7abdffff\ndraft_vocab_root=1\ndraft_vocab_k=65536\nqrow16_production=1\nqrow16_runtime=eager_exact_geometry\nsfwd_state_fusion_production=1\nsfwd_runtime=eager_byte_qualified\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\narm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nqrow16_sha256=%s\nqrow16_pass_sha256=%s\nsfwd_pass_sha256=%s\nbaseline_sha256=%s\nstarted=%s\n' \
  "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$ARM" "$SOURCE_COMMIT" "$RUNNER_SHA256" \
  "$SUBSET_SHA256" "$QROW16_SHA256" "$QROW16_PASS_SHA256" \
  "$SFWD_PASS_SHA256" "$BASELINE_SHA256" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

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
    FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES" \
    FR13_WEIGHT_FLOOR_MS="$MANDATORY_WEIGHT_FLOOR_MS" \
    FR10_METRICS=0 ENFORCE_EAGER=1 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
    FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${ARM}.json" \
    FR13_DFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${ARM}_dfwd.json" \
    FR13_CFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${ARM}_cfwd.json" \
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
    FORKED_FA2_SO="$QROW16_FA2_SO" RUNROOT="$RUNROOT_ABS" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" hydra27_fixed32 "$SUBSET" \
      > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  :
else
  rc=$?
  printf 'serve_rc=%s ended=%s\n' "$rc" "$(date -u +%FT%TZ)" \
    >> "$RUNROOT_ABS/launcher_meta.txt"
  exit "$rc"
fi

CONTAINER_ENV="$RUNROOT_ABS/$ARM/container_env.txt"
for expected in \
    'FR13_FIXED32_MODE=hydra27_fixed32' \
    'FR13_DRAFT_VOCAB_ROOT=1' \
    'FR13_DRAFT_VOCAB_K=65536' \
    'MAX_NUM_SEQS=1' \
    'ENFORCE_EAGER=1' \
    'FR13_FA2_QROW16_PRODUCTION=1' \
    'FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=1'; do
  [[ "$(grep -Fxc "$expected" "$CONTAINER_ENV")" -eq 1 ]] \
    || { echo "container lacks exact pin: $expected" >&2; exit 4; }
done
unset expected
"$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
  --arm "$ARM" \
  --out-root "$RUNROOT_ABS/$ARM/swe_out" \
  --expected-tok-per-draft 31 \
  --batch-size 1 \
  --out "$RUNROOT_ABS/$ARM/deploy_speed_fullwall.json"

SFWD_ENGAGEMENT="$RUNROOT_ABS/$ARM/logs/fr13_fixed32_sfwd_state_fusion.production_engagement.json"
QROW16_SIDECAR="$RUNROOT_ABS/$ARM/logs/fr13_fa2_qrow16_production_pass.json"
QROW16_ENGAGEMENT="$RUNROOT_ABS/$ARM/logs/fr13_fa2_qrow16_production_capture.json"
"$PYTHON_BIN" scripts/fr13_sfwd_state_fusion_pass.py verify-engagement \
  --engagement "$SFWD_ENGAGEMENT" \
  --expected-live-sha256 "$SFWD_PASS_SHA256" \
  --kernel-source src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  >/dev/null
QROW16_SIDECAR_SHA256=$(sha256sum "$QROW16_SIDECAR" | awk '{print $1}')
"$PYTHON_BIN" scripts/fr13_qrow16_pass_sidecar.py verify \
  --sidecar "$QROW16_SIDECAR" \
  --expected-sidecar-sha256 "$QROW16_SIDECAR_SHA256" \
  --candidate-so "$QROW16_FA2_SO" \
  --expected-candidate-sha256 "$QROW16_SHA256" \
  >/dev/null
finalize_manifests

"$PYTHON_BIN" - \
  "$SUBSET" "$RUNROOT_ABS/$ARM/deploy_speed_fullwall.json" "$BASELINE" \
  "$SFWD_ENGAGEMENT" "$QROW16_ENGAGEMENT" "$RUNROOT_ABS/timing_summary.json" \
  "$SOURCE_COMMIT" "$RUNNER_SHA256" "$SUBSET_SHA256" \
  "$QROW16_SHA256" "$QROW16_PASS_SHA256" "$SFWD_PASS_SHA256" \
  "$MANDATORY_WEIGHT_FLOOR_MS" "$ONE_SIDED_U95_CAP_MS" "$ARM" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

subset_path, measure_path, baseline_path = map(Path, sys.argv[1:4])
sfwd_path, qrow_path, out_path = map(Path, sys.argv[4:7])
source, runner_sha, subset_sha = sys.argv[7:10]
qrow_sha, qrow_pass_sha, sfwd_pass_sha = sys.argv[10:13]
floor_ms, cap_ms = map(float, sys.argv[13:15])
arm = sys.argv[15]


def load(path):
    raw = path.read_bytes()
    return json.loads(raw.decode("ascii")), raw


subset, _ = load(subset_path)
measure, measure_raw = load(measure_path)
baseline, baseline_raw = load(baseline_path)
sfwd, sfwd_raw = load(sfwd_path)
qrow, qrow_raw = load(qrow_path)
task_ids = sorted(subset["instance_ids"])
if (
    measure.get("schema") != "fr13.measure.deploy_speed.v1"
    or measure.get("instrument") != "OFF"
    or measure.get("regime") != "deployment"
    or measure.get("arm") != arm
    or measure.get("batch_size") != 1
    or measure.get("n_tasks") != 4
    or sorted(measure.get("task_instance_ids", [])) != task_ids
    or measure.get("draft_vocab_root") != 1
    or measure.get("draft_vocab_k") != 65536
    or measure.get("mandatory_weight_bytes") != 32666638208
):
    raise SystemExit("candidate measure is not exact4 K64 B1")
for key in (
    "step_wall_ms",
    "measured_tps_fullstep_wall",
    "accept_per_event",
    "s_per_fwd_gpu",
    "drafter_gpu_ms_per_step",
    "committer_gpu_ms_per_step",
):
    value = measure.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"candidate measure lacks numeric {key}")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise SystemExit(f"candidate measure lacks positive {key}")
if (
    sfwd.get("schema")
    != "fr13.fixed32.sfwd_state_fusion.production_engagement.v1"
    or sfwd.get("candidate_served") is not True
    or sfwd.get("layer_count") != 48
    or sfwd.get("draft_vocab_root") != 1
    or sfwd.get("draft_vocab_k") != 65536
):
    raise SystemExit("SFWD production engagement is incomplete")
if (
    qrow.get("schema")
    != "fr13.fixed32.fa2_qrow16_eager_production_engagement.v1"
    or qrow.get("status") != "ENGAGED"
    or qrow.get("runtime_mode") != "EAGER"
    or qrow.get("layer_count") != 16
    or qrow.get("sfwd_state_fusion_production") is not True
):
    raise SystemExit("qrow16 eager production engagement is incomplete")

sfwd_ms = float(measure["s_per_fwd_gpu"]) * 1000.0
dfwd_ms = float(measure["drafter_gpu_ms_per_step"])
cfwd_ms = float(measure["committer_gpu_ms_per_step"])
wall_ms = float(measure["step_wall_ms"])
baseline_wall_ms = float(baseline["step_wall_ms"])
summary = {
    "schema": "fr13.fixed32.k64_qrow16_sfwd_stack.exact4_b1_timing.v1",
    "status": "complete",
    "run_classification": "real_swe_verified_exact4_k64_b1_kernel_stack",
    "task_ids": task_ids,
    "task_count": 4,
    "batch_size": 1,
    "concurrency": 1,
    "topology": "hydra27_fixed32",
    "physical_rows": 32,
    "logical_drafts": 27,
    "valid_mask": "0x7abdffff",
    "draft_vocab_root": 1,
    "draft_vocab_k": 65536,
    "source_commit": source,
    "runner_sha256": runner_sha,
    "subset_sha256": subset_sha,
    "qrow16_candidate_sha256": qrow_sha,
    "qrow16_pass_sha256": qrow_pass_sha,
    "sfwd_pass_sha256": sfwd_pass_sha,
    "measure_sha256": hashlib.sha256(measure_raw).hexdigest(),
    "sfwd_engagement_sha256": hashlib.sha256(sfwd_raw).hexdigest(),
    "qrow16_engagement_sha256": hashlib.sha256(qrow_raw).hexdigest(),
    "baseline_measure_sha256": hashlib.sha256(baseline_raw).hexdigest(),
    "timing_eligible": True,
    "floor_acceptance_eligible": False,
    "acceptance_blocker": (
        "single eager candidate arm has no matched two-arm one-sided U95"
    ),
    "full_step_wall_ms": wall_ms,
    "full_wall_tps": measure["measured_tps_fullstep_wall"],
    "accepted_drafts_per_event": measure["accept_per_event"],
    "phase_ms_per_event": {
        "sfwd": sfwd_ms,
        "dfwd": dfwd_ms,
        "cfwd": cfwd_ms,
        "gpu_component_total": sfwd_ms + dfwd_ms + cfwd_ms,
        "other_wall": wall_ms - sfwd_ms - dfwd_ms - cfwd_ms,
    },
    "mandatory_weight_floor_ms": floor_ms,
    "one_sided_u95_cap_ms": cap_ms,
    "wall_over_floor_ratio": wall_ms / floor_ms,
    "wall_gap_to_cap_ms": wall_ms - cap_ms,
    "prior_qrow16_baseline_wall_ms": baseline_wall_ms,
    "wall_delta_vs_prior_qrow16_ms": wall_ms - baseline_wall_ms,
    "wall_ratio_vs_prior_qrow16": wall_ms / baseline_wall_ms,
}
out_path.write_text(
    json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
PY
printf 'serve_rc=0 summary=%s ended=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
