#!/usr/bin/env bash
# Exact4 real SWE-Verified B4 full-wall timing pair: stock GDN vs gated BV64.
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
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
STOCK_ARM="tail6_fixed32_stock_${TAG}"
CANDIDATE_ARM="tail6_fixed32_gdn_bv64_${TAG}"

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

# This is the first operation that can qualify the candidate. It intentionally
# precedes every Docker query or launch, and the candidate launcher repeats it.
"$PYTHON_BIN" scripts/fr13_b4_gdn_bv64_pass.py validate \
  --live-result "$GRAPH_PASS_JSON" \
  --expected-live-sha256 "$GRAPH_PASS_SHA256" \
  --gate-verdict "$GRAPH_GATE_VERDICT_JSON" \
  --expected-gate-verdict-sha256 "$GRAPH_GATE_VERDICT_SHA256" \
  --kernel-source "$KERNEL_SOURCE"
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant

mkdir -p "$RUNROOT_ABS"
printf 'classification=real_swe_verified_exact4_b4_timing_candidate\ntiming_eligible=0\nfloor_acceptance_eligible=0\nproduction_default_enabled=0\nlauncher_pid=%s\nrunroot=%s\nstock_arm=%s\ncandidate_arm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nstock_fa2_sha256=%s\ngraph_pass_sha256=%s\ngraph_gate_verdict_sha256=%s\nstarted=%s\n' \
  "$$" "$RUNROOT_ABS" "$STOCK_ARM" "$CANDIDATE_ARM" \
  "$(git rev-parse HEAD)" "$RUNNER_SHA256" "$SUBSET_SHA256" \
  "$STOCK_FA2_SHA256" "$GRAPH_PASS_SHA256" \
  "$GRAPH_GATE_VERDICT_SHA256" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

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
    || { echo "B4 BV64 timing runner changed during execution" >&2; return 14; }
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
  if [[ "$production" == "1" ]]; then
    production_bv=64
    pass_json=$GRAPH_PASS_JSON
    pass_sha=$GRAPH_PASS_SHA256
    verdict_json=$GRAPH_GATE_VERDICT_JSON
    verdict_sha=$GRAPH_GATE_VERDICT_SHA256
  fi
  echo "===== $arm: exact4 B4 production=$production ====="
  if env \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S= \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
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
        "$arm" tail6_fixed32 "$SUBSET" \
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
finalize_manifests

"$PYTHON_BIN" - \
  "$SUBSET" \
  "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/timing_summary.json" \
  "$GRAPH_PASS_SHA256" "$GRAPH_GATE_VERDICT_SHA256" \
  "$STOCK_FA2_SHA256" <<'PY'
import json
import math
import os
import sys
from pathlib import Path


subset_path, stock_path, candidate_path, out_path = map(Path, sys.argv[1:5])
graph_pass_sha256, gate_verdict_sha256, stock_fa2_sha256 = sys.argv[5:8]
task_ids = sorted(json.loads(subset_path.read_text(encoding="ascii"))["instance_ids"])
stock = json.loads(stock_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))


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
stock_wall = finite_positive(stock, "step_wall_ms")
candidate_wall = finite_positive(candidate, "step_wall_ms")
stock_tps = finite_positive(stock, "measured_tps_fullstep_wall")
candidate_tps = finite_positive(candidate, "measured_tps_fullstep_wall")
stock_floor = finite_positive(stock, "floor_ms")
candidate_floor = finite_positive(candidate, "floor_ms")
if not math.isclose(stock_floor, candidate_floor, rel_tol=0.0, abs_tol=1e-9):
    raise SystemExit("stock and candidate optimistic floor values differ")
summary = {
    "schema": "fr13.fixed32.b4_gdn_bv64.full_wall_timing_pair.v1",
    "status": "complete",
    "run_classification": "real_swe_verified_exact4_b4_timing",
    "task_count": 4,
    "batch_size": 4,
    "concurrency": 4,
    "task_ids": task_ids,
    "decision_metric": "measured_tps_fullstep_wall",
    "stock_reference": {
        "selector": "legacy_per_request_bv8",
        "fa2_sha256": stock_fa2_sha256,
        "step_wall_ms": stock_wall,
        "measured_tps_fullstep_wall": stock_tps,
        "accepted_drafts_per_event": float(stock["accept_per_event"]),
        "step_wall_to_optimistic_floor_ratio": float(stock["floor_ratio"]),
    },
    "candidate": {
        "selector": "fixed32_batched_gdn_bv64_b4",
        "graph_pass_sha256": graph_pass_sha256,
        "graph_gate_verdict_sha256": gate_verdict_sha256,
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
