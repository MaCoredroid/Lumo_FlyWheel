#!/usr/bin/env bash
# Exact4 real SWE-Verified B4 full-wall timing pair: stock GDN vs gated batched BV8.
# This is a timing-candidate runner, not the formal two-arm floor acceptance gate.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${STOCK_FA2_SO:?set STOCK_FA2_SO to the exact-safe stock FA2 binary}"
: "${GRAPH_PASS_JSON:?set GRAPH_PASS_JSON to a completed B4 graph-byte PASS}"
: "${GRAPH_PASS_SHA256:?set GRAPH_PASS_SHA256 to its raw SHA-256}"
: "${GRAPH_GATE_VERDICT_JSON:?set GRAPH_GATE_VERDICT_JSON to the completed exact4 gate verdict}"
: "${GRAPH_GATE_VERDICT_SHA256:?set GRAPH_GATE_VERDICT_SHA256 to its raw SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
KERNEL_SOURCE=src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py
GATE_RUNNER=scripts/fr13_run_b4_gdn_wide_live_gate.sh
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
B4_KV_CACHE_MEMORY_BYTES=49392123904
RUNROOT_ABS=$(realpath -m "$RUNROOT")
TIMING_KIND=hydra27_fixed32
STOCK_ARM="hydra27_fixed32_stock_${TAG}"
CANDIDATE_ARM="hydra27_fixed32_gdn_bv8_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$STOCK_FA2_SO" && ! -L "$STOCK_FA2_SO" ]] \
  || { echo "STOCK_FA2_SO must be a regular non-symlink file" >&2; exit 2; }
[[ "$(sha256sum "$STOCK_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "STOCK_FA2_SO is not the exact-safe stock reference" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical exact4 subset SHA-256 drift" >&2; exit 2; }
[[ "$GRAPH_PASS_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "GRAPH_PASS_SHA256 must be lowercase SHA-256" >&2; exit 2; }
[[ "$GRAPH_GATE_VERDICT_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "GRAPH_GATE_VERDICT_SHA256 must be lowercase SHA-256" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
RUNTIME_MANIFEST_SHA256=$("$PYTHON_BIN" - \
  "$RUNROOT_ABS/runtime_manifest.at_launch.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["overall_canonical_sha256"])
PY
)
GATE_RUNNER_SHA256=$(sha256sum "$GATE_RUNNER" | awk '{print $1}')
[[ "$RUNTIME_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$GATE_RUNNER_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "current BV8 runtime closure identity is invalid" >&2; exit 2; }

# This validation binds the old gate to the current full runtime closure and
# precedes every Docker query or launch. The candidate launcher repeats it.
"$PYTHON_BIN" scripts/fr13_b4_gdn_bv8_pass.py validate \
  --live-result "$GRAPH_PASS_JSON" \
  --expected-live-sha256 "$GRAPH_PASS_SHA256" \
  --gate-verdict "$GRAPH_GATE_VERDICT_JSON" \
  --expected-gate-verdict-sha256 "$GRAPH_GATE_VERDICT_SHA256" \
  --kernel-source "$KERNEL_SOURCE" \
  --runtime-manifest "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
  --gate-runner "$GATE_RUNNER"
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=0
export FR13_DRAFT_VOCAB_K=0
export FR13_NEEDS_ALLOW="FR13_DRAFT_VOCAB_K=0"
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_DRAFT_VOCAB_ROOT" == "0" \
   && "$FR13_DRAFT_VOCAB_K" == "0" \
   && "$FR13_MANDATORY_WEIGHT_BYTES" == "42025179008" \
   && "$FR13_WEIGHT_FLOOR_MS" == "153.938384645" ]] \
  || { echo "full-vocabulary fixed32 floor contract did not engage" >&2; exit 2; }

printf 'classification=real_swe_verified_exact4_b4_timing_candidate\ntiming_eligible=0\nfloor_acceptance_eligible=0\nproduction_default_enabled=0\narm=hydra27_fixed32\nlineage=successor_to_legacy_hydra23_not_same_topology\nfixed32_mode=hydra27_fixed32\nphysical_drafts=31\nactive_drafts=27\nvalid_mask=0x7abdffff\ndraft_vocab_k=0\ndraft_vocab_root=0\nmandatory_weight_bytes=42025179008\nweight_floor_ms=153.938384645\nlauncher_pid=%s\nrunroot=%s\nstock_arm=%s\ncandidate_arm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nstock_fa2_sha256=%s\ngraph_pass_sha256=%s\ngraph_gate_verdict_sha256=%s\nruntime_manifest_sha256=%s\ngate_runner_sha256=%s\nfr10_metrics=1\nring_export=1\nflags_inkernel=1\ntree_gdn_geom_override=BV=8\nenforce_eager=0\ncudagraph_mode=FULL_AND_PIECEWISE\nkv_cache_memory_bytes=%s\nstarted=%s\n' \
  "$$" "$RUNROOT_ABS" "$STOCK_ARM" "$CANDIDATE_ARM" \
  "$(git rev-parse HEAD)" "$RUNNER_SHA256" "$SUBSET_SHA256" \
  "$STOCK_FA2_SHA256" "$GRAPH_PASS_SHA256" \
  "$GRAPH_GATE_VERDICT_SHA256" "$RUNTIME_MANIFEST_SHA256" \
  "$GATE_RUNNER_SHA256" "$B4_KV_CACHE_MEMORY_BYTES" \
  "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

MANIFEST_FINALIZED=0
finalize_manifests() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || return $?
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json" \
    || return $?
  cmp -s \
    "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || { echo "runtime/source manifest changed during timing" >&2; return 14; }
  cmp -s \
    "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
    || { echo "B4 batched BV8 timing runner changed during execution" >&2; return 14; }
  MANIFEST_FINALIZED=1
}

runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    if finalize_manifests; then
      :
    else
      local manifest_rc=$?
      (( rc == 0 )) && rc=$manifest_rc
    fi
  fi
  exit "$rc"
}
trap runner_exit EXIT

run_arm() {
  local arm=$1
  local production=$2
  local production_bv=""
  local pass_json=""
  local pass_sha=""
  local verdict_json=""
  local verdict_sha=""
  local runtime_manifest=""
  local gate_runner=""
  if [[ "$production" == "1" ]]; then
    production_bv=8
    pass_json=$GRAPH_PASS_JSON
    pass_sha=$GRAPH_PASS_SHA256
    verdict_json=$GRAPH_GATE_VERDICT_JSON
    verdict_sha=$GRAPH_GATE_VERDICT_SHA256
    runtime_manifest=$RUNROOT_ABS/runtime_manifest.at_launch.json
    gate_runner=$GATE_RUNNER
  fi
  echo "===== $arm: exact4 B4 production=$production ====="
  if env \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S= \
      KV_CACHE_MEMORY_BYTES="$B4_KV_CACHE_MEMORY_BYTES" \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR13_FIXED32_BATCH_GDN_BV8_TIMING=1 \
      FR10_METRICS=1 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
      FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 \
      FR13_SCAN_ALIGN=0 FR13_NPAD_INVARIANT=0 \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json" \
      FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
      FR13_FIXED32_BATCH_GDN_PRODUCTION="$production" \
      FR13_FIXED32_BATCH_GDN_BV_PRODUCTION="$production_bv" \
      FR13_FIXED32_BATCH_GDN_GRAPH_LIVE_PASS_JSON="$pass_json" \
      FR13_FIXED32_BATCH_GDN_GRAPH_LIVE_PASS_SHA256="$pass_sha" \
      FR13_FIXED32_BATCH_GDN_GRAPH_GATE_VERDICT_JSON="$verdict_json" \
      FR13_FIXED32_BATCH_GDN_GRAPH_GATE_VERDICT_SHA256="$verdict_sha" \
      FR13_FIXED32_BATCH_GDN_RUNTIME_MANIFEST_JSON="$runtime_manifest" \
      FR13_FIXED32_BATCH_GDN_GATE_RUNNER="$gate_runner" \
      FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
      FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
      FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_SO= \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 \
      FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_INSTANCE_ID= \
      FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$STOCK_FA2_SO" \
      RUNROOT="$RUNROOT_ABS" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh \
        "$arm" "$TIMING_KIND" "$SUBSET" \
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
    --arm "$arm" \
    --out-root "$RUNROOT_ABS/$arm/swe_out" \
    --expected-tok-per-draft 31 \
    --batch-size 4 \
    --out "$RUNROOT_ABS/$arm/deploy_speed_fullwall.json"
  printf 'arm=%s serve_rc=0 ended=%s\n' \
    "$arm" "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
}

run_arm "$STOCK_ARM" 0
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after the stock reference" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" 1
STOCK_ENGAGEMENT="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_batch_gdn_bv8.production_engagement.json"
CANDIDATE_ENGAGEMENT="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_batch_gdn_bv8.production_engagement.json"
CANDIDATE_CREDENTIAL="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_batch_gdn_byte_ab.pass.json"
[[ ! -e "$STOCK_ENGAGEMENT" && ! -L "$STOCK_ENGAGEMENT" ]] \
  || { echo "stock BV8 arm emitted a batched BV8 production engagement artifact" >&2; exit 4; }
[[ -f "$CANDIDATE_CREDENTIAL" && ! -L "$CANDIDATE_CREDENTIAL" ]] \
  || { echo "candidate lacks its installed regular BV8 credential" >&2; exit 4; }
CANDIDATE_CREDENTIAL_SHA256=$(sha256sum "$CANDIDATE_CREDENTIAL" | awk '{print $1}')
"$PYTHON_BIN" scripts/fr13_b4_gdn_bv8_pass.py engagement \
  --engagement "$CANDIDATE_ENGAGEMENT" \
  --expected-live-sha256 "$GRAPH_PASS_SHA256" \
  --expected-gate-verdict-sha256 "$GRAPH_GATE_VERDICT_SHA256" \
  --expected-production-sidecar-sha256 "$CANDIDATE_CREDENTIAL_SHA256" \
  --expected-runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA256" \
  --expected-gate-runner-sha256 "$GATE_RUNNER_SHA256" \
  --kernel-source "$KERNEL_SOURCE" \
  > "$RUNROOT_ABS/$CANDIDATE_ARM/bv8_engagement_validation.json"
finalize_manifests

"$PYTHON_BIN" - \
  "$SUBSET" \
  "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/timing_summary.json" \
  "$CANDIDATE_ENGAGEMENT" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/bv8_engagement_validation.json" \
  "$GRAPH_PASS_SHA256" "$GRAPH_GATE_VERDICT_SHA256" \
  "$STOCK_FA2_SHA256" "$RUNTIME_MANIFEST_SHA256" \
  "$GATE_RUNNER_SHA256" <<'PY'
import hashlib
import json
import math
import os
import sys
from pathlib import Path


subset_path, stock_path, candidate_path, out_path = map(Path, sys.argv[1:5])
engagement_path, engagement_validation_path = map(Path, sys.argv[5:7])
(
    graph_pass_sha256,
    gate_verdict_sha256,
    stock_fa2_sha256,
    runtime_manifest_sha256,
    gate_runner_sha256,
) = sys.argv[7:12]
task_ids = sorted(json.loads(subset_path.read_text(encoding="ascii"))["instance_ids"])
stock = json.loads(stock_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
engagement_raw = engagement_path.read_bytes()
engagement = json.loads(engagement_raw.decode("ascii"))
engagement_validation = json.loads(
    engagement_validation_path.read_text(encoding="ascii")
)


def finite_positive(record, key):
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{key} is missing from full-wall timing evidence")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"{key} is not finite and positive")
    return value


def validate(record, label):
    if (
        record.get("schema") != "fr13.measure.deploy_speed.v1"
        or record.get("regime") != "deployment"
        or record.get("instrument") != "OFF"
        or record.get("batch_size") != 4
        or record.get("n_tasks") != 4
        or sorted(record.get("task_instance_ids", [])) != task_ids
        or record.get("floor_is_full_step_hardware_floor") is not False
        or record.get("mandatory_weight_bytes") != 42_025_179_008
        or not math.isclose(
            float(record.get("weight_floor_ms", 0.0)),
            153.938384645,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise SystemExit(f"{label} deploy-speed provenance is not exact4 B4")
    for key in (
        "measured_tps_fullstep_wall",
        "step_wall_ms",
        "accept_per_event",
        "committed_per_event",
        "wall_steps_measured",
        "events_per_step",
        "s_per_fwd_gpu",
        "drafter_gpu_ms_per_step",
        "committer_gpu_ms_per_step",
        "floor_ms",
        "floor_ratio",
    ):
        finite_positive(record, key)


validate(stock, "stock")
validate(candidate, "candidate")
if (
    engagement.get("schema")
    != "fr13.fixed32.batch_gdn.bv8.production_engagement.v1"
    or engagement.get("status") != "ENGAGED"
    or engagement.get("mode") != "hydra27_fixed32"
    or engagement.get("batch_size") != 4
    or engagement.get("candidate") != "fixed32_batch_gdn_bv8_v1"
    or engagement.get("reference_bv") != 8
    or engagement.get("candidate_bv") != 8
    or engagement.get("reference_kernel_structure")
    != "per_request_tree_gdn_path"
    or engagement.get("candidate_kernel_structure")
    != "fixed32_batch_tree_gdn_path"
    or engagement.get("reference_physical_launches_per_layer") != 8
    or engagement.get("candidate_physical_launches_per_layer") != 2
    or engagement.get("count_invocation") is not True
    or engagement.get("ring_export") is not True
    or engagement.get("flags_inkernel") is not True
    or engagement.get("scan_align") is not False
    or engagement.get("npad_invariant") is not False
    or engagement.get("layer_count") != 48
    or engagement.get("batched_route_capture_layers_by_batch")
    != {"1": 0, "2": 48, "3": 48, "4": 48}
    or engagement.get("qualified_batch_sizes") != [4]
    or engagement.get("lower_batch_route")
    != "b1_legacy_b2_b3_fixed32_batched_bv8"
    or engagement.get("physical_launches_per_layer_by_batch")
    != {"1": 2, "2": 2, "3": 2, "4": 2}
    or engagement.get("all_b_le_4_launch_invariant") is not True
    or engagement.get("graph_pass_sha256") != graph_pass_sha256
    or engagement.get("gate_verdict_sha256") != gate_verdict_sha256
    or engagement.get("runtime_manifest_sha256") != runtime_manifest_sha256
    or engagement.get("gate_runner_sha256") != gate_runner_sha256
    or engagement.get("task_marker", "").removeprefix("swe_verified:")
    not in task_ids
    or engagement.get("observed_full_graph_replays_at_least") != 1
    or engagement.get("fallback") != 0
    or engagement_validation.get("status") != "ENGAGED"
    or engagement_validation.get("graph_pass_sha256") != graph_pass_sha256
    or engagement_validation.get("graph_id") != engagement.get("graph_id")
    or engagement_validation.get("graph_signature")
    != engagement.get("graph_signature")
    or engagement_validation.get("kernel_source_sha256")
    != engagement.get("kernel_source_sha256")
    or engagement_validation.get("b4_replays_at_least") != 1
    or engagement_validation.get("gate_verdict_sha256")
    != gate_verdict_sha256
    or engagement_validation.get("runtime_manifest_sha256")
    != runtime_manifest_sha256
    or engagement_validation.get("gate_runner_sha256")
    != gate_runner_sha256
    or engagement_validation.get("production_sidecar_sha256")
    != engagement.get("production_sidecar_sha256")
    or engagement_validation.get("lower_batch_batched_capture_layers") != 96
):
    raise SystemExit("candidate lacks exact B4 batched BV8 production engagement")
stock_wall = finite_positive(stock, "step_wall_ms")
candidate_wall = finite_positive(candidate, "step_wall_ms")
stock_tps = finite_positive(stock, "measured_tps_fullstep_wall")
candidate_tps = finite_positive(candidate, "measured_tps_fullstep_wall")
stock_floor = finite_positive(stock, "floor_ms")
candidate_floor = finite_positive(candidate, "floor_ms")
if not math.isclose(stock_floor, candidate_floor, rel_tol=0.0, abs_tol=1e-9):
    raise SystemExit("stock and candidate optimistic floor values differ")
summary = {
    "schema": "fr13.fixed32.b4_gdn_bv8.full_wall_timing_pair.v1",
    "status": "complete",
    "run_classification": "real_swe_verified_exact4_b4_timing",
    "task_count": 4,
    "batch_size": 4,
    "concurrency": 4,
    "arm": "hydra27_fixed32",
    "lineage": "successor_to_legacy_hydra23_not_same_topology",
    "fixed32_mode": "hydra27_fixed32",
    "physical_drafts": 31,
    "active_drafts": 27,
    "valid_mask": "0x7abdffff",
    "task_ids": task_ids,
    "decision_metric": "measured_tps_fullstep_wall",
    "stock_reference": {
        "selector": "hydra27_fixed32_legacy_per_request_bv8",
        "fa2_sha256": stock_fa2_sha256,
        "step_wall_ms": stock_wall,
        "measured_tps_fullstep_wall": stock_tps,
        "accepted_drafts_per_event": float(stock["accept_per_event"]),
        "step_wall_to_optimistic_floor_ratio": float(stock["floor_ratio"]),
    },
    "candidate": {
        "selector": "hydra27_fixed32_batched_gdn_bv8_b4",
        "graph_pass_sha256": graph_pass_sha256,
        "graph_gate_verdict_sha256": gate_verdict_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "gate_runner_sha256": gate_runner_sha256,
        "production_engagement_sha256": hashlib.sha256(engagement_raw).hexdigest(),
        "production_sidecar_sha256": engagement["production_sidecar_sha256"],
        "production_graph_id": int(engagement["graph_id"]),
        "production_graph_signature": engagement["graph_signature"],
        "production_kernel_source_sha256": engagement[
            "kernel_source_sha256"
        ],
        "observed_full_graph_replays_at_least": 1,
        "lower_batch_batched_capture_layers": 96,
        "qualified_batch_sizes": [4],
        "lower_batch_route": "b1_legacy_b2_b3_fixed32_batched_bv8",
        "physical_launches_per_layer_by_batch": {
            "1": 2,
            "2": 2,
            "3": 2,
            "4": 2,
        },
        "all_b_le_4_launch_invariant": True,
        "step_wall_ms": candidate_wall,
        "measured_tps_fullstep_wall": candidate_tps,
        "accepted_drafts_per_event": float(candidate["accept_per_event"]),
        "step_wall_to_optimistic_floor_ratio": float(candidate["floor_ratio"]),
    },
    "optimistic_floor_ms": stock_floor,
    "optimistic_floor_is_full_step_hardware_floor": False,
    "candidate_to_stock_full_wall_tps_ratio": candidate_tps / stock_tps,
    "stock_to_candidate_step_wall_ratio": stock_wall / candidate_wall,
    "formal_floor_acceptance_eligible": False,
    "formal_floor_acceptance_reason": (
        "paired exact4 timing candidate only; the canonical Tail/Hydra floor gate "
        "and its statistical acceptance procedure were not run"
    ),
    "production_default_enabled": False,
}
temporary = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
temporary.write_text(
    json.dumps(summary, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    + "\n",
    encoding="ascii",
)
temporary.replace(out_path)
print(json.dumps(summary, sort_keys=True))
PY

printf 'timing_summary=%s completed=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
