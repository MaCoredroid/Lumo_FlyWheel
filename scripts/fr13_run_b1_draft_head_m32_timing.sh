#!/usr/bin/env bash
# Exact4 real SWE-Verified B1 full-wall timing pair: stock BF16 head vs M32.
# This is a timing-candidate runner, not the formal Tail/Hydra floor gate.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${STOCK_FA2_SO:?set STOCK_FA2_SO to the exact-safe stock FA2 binary}"
: "${LIVE_PASS_JSON:?set LIVE_PASS_JSON to a completed real-B1 M32 byte PASS}"
: "${LIVE_PASS_SHA256:?set LIVE_PASS_SHA256 to its raw SHA-256}"
: "${LIVE_FINAL_FLUSH_JSON:?set LIVE_FINAL_FLUSH_JSON to the gate final-flush record}"
: "${LIVE_BOUNDARY_SNAPSHOT_JSON:?set LIVE_BOUNDARY_SNAPSHOT_JSON to the gate final boundary snapshot}"
: "${LIVE_CHAT_TRAFFIC_AUDIT_JSON:?set LIVE_CHAT_TRAFFIC_AUDIT_JSON to the gate authenticated traffic audit}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
CANDIDATE_SOURCE=scripts/fr10_phase4_patch_vllm_tree_gdn.py
CANDIDATE_SOURCE_SHA256=$(sha256sum "$CANDIDATE_SOURCE" | cut -d' ' -f1)
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | cut -d' ' -f1)
RUNROOT_ABS=$(realpath -m "$RUNROOT")
STOCK_ARM="hydra27_fixed32_head_stock_${TAG}"
CANDIDATE_ARM="hydra27_fixed32_head_m32_${TAG}"

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
[[ "$STOCK_FA2_SO" == /* ]] \
  || { echo "STOCK_FA2_SO must be an absolute path" >&2; exit 2; }
[[ "$(sha256sum "$STOCK_FA2_SO" | cut -d' ' -f1)" == "$STOCK_FA2_SHA256" ]] \
  || { echo "STOCK_FA2_SO is not the exact-safe stock reference" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | cut -d' ' -f1)" == "$SUBSET_SHA256" ]] \
  || { echo "canonical exact4 subset SHA-256 drift" >&2; exit 2; }
[[ "$LIVE_PASS_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "LIVE_PASS_SHA256 must be lowercase SHA-256" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }

# Credential validation is intentionally the first candidate-qualifying action.
"$PYTHON_BIN" scripts/fr13_draft_head_m32_pass.py validate-live \
  --live-result "$LIVE_PASS_JSON" \
  --expected-live-sha256 "$LIVE_PASS_SHA256" \
  --final-flush "$LIVE_FINAL_FLUSH_JSON" \
  --boundary-snapshot "$LIVE_BOUNDARY_SNAPSHOT_JSON" \
  --chat-traffic-audit "$LIVE_CHAT_TRAFFIC_AUDIT_JSON" \
  --candidate-source "$CANDIDATE_SOURCE" \
  --expected-candidate-source-sha256 "$CANDIDATE_SOURCE_SHA256"
LIVE_FINAL_FLUSH_SHA256=$(sha256sum "$LIVE_FINAL_FLUSH_JSON" | cut -d' ' -f1)
LIVE_BOUNDARY_SNAPSHOT_SHA256=$(
  sha256sum "$LIVE_BOUNDARY_SNAPSHOT_JSON" | cut -d' ' -f1
)
LIVE_CHAT_TRAFFIC_AUDIT_SHA256=$(
  sha256sum "$LIVE_CHAT_TRAFFIC_AUDIT_JSON" | cut -d' ' -f1
)
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_FLOOR_ORDER=HT
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant

mkdir -p "$RUNROOT_ABS"
printf 'classification=real_swe_verified_exact4_b1_draft_head_timing_candidate\ntiming_eligible=1\nfloor_acceptance_eligible=0\nproduction_default_enabled=0\nlauncher_pid=%s\nrunroot=%s\nstock_arm=%s\ncandidate_arm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nstock_fa2_sha256=%s\nlive_pass_sha256=%s\nlive_final_flush_sha256=%s\nlive_boundary_snapshot_sha256=%s\nlive_chat_traffic_audit_sha256=%s\ncandidate_source_sha256=%s\nstarted=%s\n' \
  "$$" "$RUNROOT_ABS" "$STOCK_ARM" "$CANDIDATE_ARM" \
  "$(git rev-parse HEAD)" "$RUNNER_SHA256" "$SUBSET_SHA256" \
  "$STOCK_FA2_SHA256" "$LIVE_PASS_SHA256" \
  "$LIVE_FINAL_FLUSH_SHA256" "$LIVE_BOUNDARY_SNAPSHOT_SHA256" \
  "$LIVE_CHAT_TRAFFIC_AUDIT_SHA256" \
  "$CANDIDATE_SOURCE_SHA256" \
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
    || { echo "draft-head timing runner changed during execution" >&2; return 14; }
  [[ "$(sha256sum "$LIVE_CHAT_TRAFFIC_AUDIT_JSON" | cut -d' ' -f1)" \
        == "$LIVE_CHAT_TRAFFIC_AUDIT_SHA256" ]] \
    || { echo "live authenticated traffic audit changed during timing" >&2; return 14; }
  MANIFEST_FINALIZED=1
}

runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    finalize_manifests || { local manifest_rc=$?; (( rc == 0 )) && rc=$manifest_rc; }
  fi
  exit "$rc"
}
trap runner_exit EXIT

run_arm() {
  local arm=$1
  local production=$2
  local live_json=""
  local live_sha=""
  if [[ "$production" == "1" ]]; then
    live_json=$LIVE_PASS_JSON
    live_sha=$LIVE_PASS_SHA256
  fi
  echo "===== $arm: exact4 B1 draft-head production=$production ====="
  if env \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json" \
      FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_DRAFT_HEAD_M32_LIVE_AB=0 \
      FR13_DRAFT_HEAD_M32_INSTANCE_ID= \
      FR13_DRAFT_HEAD_M32_LIVE_JSON=/logs/fr13_draft_head_m32.live.json \
      FR13_DRAFT_HEAD_M32_PRODUCTION="$production" \
      FR13_DRAFT_HEAD_M32_LIVE_PASS_JSON="$live_json" \
      FR13_DRAFT_HEAD_M32_LIVE_PASS_SHA256="$live_sha" \
      FR13_DRAFT_HEAD_M32_LIVE_FINAL_FLUSH_JSON="$LIVE_FINAL_FLUSH_JSON" \
      FR13_DRAFT_HEAD_M32_LIVE_BOUNDARY_SNAPSHOT_JSON="$LIVE_BOUNDARY_SNAPSHOT_JSON" \
      FR13_DRAFT_HEAD_M32_LIVE_CHAT_TRAFFIC_AUDIT_JSON="$LIVE_CHAT_TRAFFIC_AUDIT_JSON" \
      FR13_DRAFT_HEAD_M32_PRODUCTION_ENGAGEMENT_JSON=/logs/fr13_draft_head_m32.production_engagement.json \
      FR13_DRAFT_HEAD_M32_TIMING_ARM=1 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
      FR13_FA2_QROW32_B1_LIVE_AB_ARM= \
      FR13_FA2_QROW32_B1_PRODUCTION_ARM= \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
      FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
      FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
      FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
      FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_SO= \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$STOCK_FA2_SO" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh \
        "$arm" hydra27_fixed32 "$SUBSET" \
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
    --batch-size 1 \
    --out "$RUNROOT_ABS/$arm/deploy_speed_fullwall.json"
  printf 'arm=%s serve_rc=0 ended=%s\n' \
    "$arm" "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
}

run_arm "$STOCK_ARM" 0
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after the stock arm" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" 1

STOCK_ENGAGEMENT="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_draft_head_m32.production_engagement.json"
CANDIDATE_ENGAGEMENT="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_draft_head_m32.production_engagement.json"
CANDIDATE_SIDECAR="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_draft_head_m32.production_pass.json"
[[ ! -e "$STOCK_ENGAGEMENT" && ! -L "$STOCK_ENGAGEMENT" ]] \
  || { echo "stock arm emitted M32 production engagement" >&2; exit 4; }
[[ -f "$CANDIDATE_SIDECAR" && ! -L "$CANDIDATE_SIDECAR" ]] \
  || { echo "candidate production sidecar is missing" >&2; exit 4; }
CANDIDATE_SIDECAR_SHA256=$(sha256sum "$CANDIDATE_SIDECAR" | cut -d' ' -f1)
CANDIDATE_ENGAGEMENT_VALIDATION="$RUNROOT_ABS/$CANDIDATE_ARM/draft_head_m32_engagement_validation.json"
"$PYTHON_BIN" scripts/fr13_draft_head_m32_pass.py engagement \
  --engagement "$CANDIDATE_ENGAGEMENT" \
  --expected-source-sha256 "$CANDIDATE_SOURCE_SHA256" \
  --expected-sidecar-sha256 "$CANDIDATE_SIDECAR_SHA256" \
  > "$CANDIDATE_ENGAGEMENT_VALIDATION"
finalize_manifests

"$PYTHON_BIN" - \
  "$SUBSET" \
  "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/timing_summary.json" \
  "$CANDIDATE_ENGAGEMENT" "$CANDIDATE_ENGAGEMENT_VALIDATION" \
  "$CANDIDATE_SIDECAR" "$LIVE_PASS_JSON" \
  "$LIVE_FINAL_FLUSH_JSON" "$LIVE_BOUNDARY_SNAPSHOT_JSON" \
  "$LIVE_CHAT_TRAFFIC_AUDIT_JSON" \
  "$LIVE_PASS_SHA256" "$CANDIDATE_SOURCE_SHA256" \
  "$CANDIDATE_SIDECAR_SHA256" "$STOCK_FA2_SHA256" \
  "$LIVE_FINAL_FLUSH_SHA256" "$LIVE_BOUNDARY_SNAPSHOT_SHA256" \
  "$LIVE_CHAT_TRAFFIC_AUDIT_SHA256" \
  "$STOCK_ARM" "$CANDIDATE_ARM" <<'PY'
import hashlib
import json
import math
import os
import sys
from pathlib import Path


subset_path, stock_path, candidate_path, out_path = map(Path, sys.argv[1:5])
engagement_path, engagement_validation_path, sidecar_path = map(
    Path, sys.argv[5:8]
)
live_path, final_flush_path, boundary_path, traffic_audit_path = map(
    Path, sys.argv[8:12]
)
live_sha, source_sha, sidecar_sha, stock_fa2_sha = sys.argv[12:16]
final_flush_sha, boundary_sha, traffic_audit_sha = sys.argv[16:19]
stock_arm, candidate_arm = sys.argv[19:21]
task_ids = sorted(json.loads(subset_path.read_text(encoding="ascii"))["instance_ids"])
MIN_RETAINED_WALL_FRACTION = 0.99
MIN_TASK_COUNTER_STEPS = 64
RAW = {
    "spec_drafts": "vllm:spec_decode_num_drafts_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "fwd_gpu_seconds": "vllm:fr13_decode_forward_gpu_seconds_total",
    "fwd_gpu_steps": "vllm:fr13_decode_forward_gpu_steps_total",
    "fwd_gpu_drafts": "vllm:fr13_decode_forward_gpu_drafts_total",
    "wall_seconds": "vllm:fr13_decode_step_wall_seconds_total",
    "wall_steps": "vllm:fr13_decode_step_wall_steps_total",
    "wall_drafts": "vllm:fr13_decode_step_wall_drafts_total",
    "drafter_gpu_seconds": "vllm:fr13_drafter_gpu_seconds_total",
    "drafter_gpu_spans": "vllm:fr13_drafter_gpu_spans_total",
    "committer_gpu_seconds": "vllm:fr13_committer_gpu_seconds_total",
    "committer_gpu_spans": "vllm:fr13_committer_gpu_spans_total",
}


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), raw


stock, stock_raw = load(stock_path)
candidate, candidate_raw = load(candidate_path)
engagement, engagement_raw = load(engagement_path)
engagement_validation, engagement_validation_raw = load(
    engagement_validation_path
)
sidecar, sidecar_raw = load(sidecar_path)
_live, live_raw = load(live_path)
_final_flush, final_flush_raw = load(final_flush_path)
_boundary, boundary_raw = load(boundary_path)
_traffic_audit, traffic_audit_raw = load(traffic_audit_path)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def finite(record, key):
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{key} is missing from full-wall timing evidence")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"{key} is not finite and positive")
    return value


def nonnegative(record, key, label):
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{label}.{key} is missing or nonnumeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise SystemExit(f"{label}.{key} is not finite and nonnegative")
    return value


def integral(record, key, label):
    value = nonnegative(record, key, label)
    if value != round(value):
        raise SystemExit(f"{label}.{key} is not an integer counter")
    return int(value)


def close(actual, expected, label, *, absolute=1e-9, relative=1e-9):
    if not math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute):
        raise SystemExit(f"{label} consistency check failed: {actual} != {expected}")


def validate(record, raw, label, expected_arm):
    if (
        record.get("schema") != "fr13.measure.deploy_speed.v1"
        or record.get("regime") != "deployment"
        or record.get("instrument") != "OFF"
        or record.get("arm") != expected_arm
        or record.get("batch_size") != 1
        or record.get("n_tasks") != 4
        or sorted(record.get("task_instance_ids", [])) != task_ids
        or record.get("floor_is_full_step_hardware_floor") is not False
        or record.get("floor_reference_scope")
        != "fixed32_mandatory_weight_read_or_row_compute_lower_bound"
    ):
        raise SystemExit(f"{label} deploy-speed provenance is not exact4 B1")
    raw_counters = record.get("raw_counter_delta_aggregate")
    task_records = record.get("per_task")
    if not isinstance(raw_counters, dict) or not isinstance(task_records, list):
        raise SystemExit(f"{label} raw counter evidence is missing")
    if (
        len(task_records) != 4
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("instance_id"), str)
            for row in task_records
        )
        or sorted(row["instance_id"] for row in task_records) != task_ids
    ):
        raise SystemExit(f"{label} per-task counter evidence is not exact4")

    per_task_fields = {
        "drafts": RAW["spec_drafts"],
        "accepted_tokens": RAW["accepted_tokens"],
        "fwd_gpu_seconds": RAW["fwd_gpu_seconds"],
        "fwd_gpu_steps": RAW["fwd_gpu_steps"],
        "fwd_gpu_drafts": RAW["fwd_gpu_drafts"],
        "wall_seconds": RAW["wall_seconds"],
        "wall_steps": RAW["wall_steps"],
        "wall_drafts": RAW["wall_drafts"],
        "drafter_gpu_seconds": RAW["drafter_gpu_seconds"],
        "drafter_gpu_spans": RAW["drafter_gpu_spans"],
        "committer_gpu_seconds": RAW["committer_gpu_seconds"],
        "committer_gpu_spans": RAW["committer_gpu_spans"],
    }
    task_sums = {metric: 0.0 for metric in per_task_fields.values()}
    retained_task_fractions = {}
    for row in task_records:
        if not isinstance(row, dict):
            raise SystemExit(f"{label} has a non-object per-task record")
        task_label = f"{label}:{row['instance_id']}"
        fwd_steps = integral(row, "fwd_gpu_steps", task_label)
        fwd_drafts = integral(row, "fwd_gpu_drafts", task_label)
        wall_steps = integral(row, "wall_steps", task_label)
        wall_drafts = integral(row, "wall_drafts", task_label)
        if (
            fwd_steps < MIN_TASK_COUNTER_STEPS
            or fwd_drafts != fwd_steps
            or wall_drafts != wall_steps
            or wall_steps > fwd_steps
            or wall_steps / fwd_steps < MIN_RETAINED_WALL_FRACTION
        ):
            raise SystemExit(f"{task_label} retained counter window is too small")
        retained_task_fractions[row["instance_id"]] = wall_steps / fwd_steps
        for field, metric in per_task_fields.items():
            task_sums[metric] += nonnegative(row, field, task_label)

    for metric, task_sum in task_sums.items():
        close(
            nonnegative(raw_counters, metric, f"{label}:aggregate"),
            task_sum,
            f"{label} per-task sum {metric}",
            absolute=1e-7,
        )
    agg_fwd_steps = integral(raw_counters, RAW["fwd_gpu_steps"], label)
    agg_fwd_drafts = integral(raw_counters, RAW["fwd_gpu_drafts"], label)
    agg_wall_steps = integral(raw_counters, RAW["wall_steps"], label)
    agg_wall_drafts = integral(raw_counters, RAW["wall_drafts"], label)
    if (
        agg_fwd_steps < 4 * MIN_TASK_COUNTER_STEPS
        or agg_fwd_drafts != agg_fwd_steps
        or agg_wall_drafts != agg_wall_steps
        or agg_wall_steps > agg_fwd_steps
        or agg_wall_steps / agg_fwd_steps < MIN_RETAINED_WALL_FRACTION
    ):
        raise SystemExit(f"{label} aggregate retained counter window is too small")
    values = {
        key: finite(record, key)
        for key in (
            "accept_per_event",
            "committed_per_event",
            "committer_gpu_ms_per_step",
            "compute_floor_ms",
            "drafter_gpu_ms_per_step",
            "events_per_step",
            "floor_ms",
            "floor_ratio",
            "measured_tps_fullstep_wall",
            "rows_per_step",
            "s_per_fwd_gpu",
            "step_wall_ms",
            "wall_s_per_event",
            "wall_steps_measured",
            "weight_floor_ms",
        )
    }
    if (
        record.get("mandatory_weight_bytes") != 27977022848
        or record.get("weight_floor_bandwidth_bytes_per_s") != 273000000000
    ):
        raise SystemExit(f"{label} corrected mandatory-weight floor identity drifted")
    close(values["events_per_step"], 1.0, f"{label} B1 events_per_step")
    close(values["rows_per_step"], 32.0, f"{label} fixed32 rows_per_step")
    close(values["weight_floor_ms"], 102.479937172, f"{label} weight floor")
    close(values["floor_ms"], 102.479937172, f"{label} active floor")
    close(values["compute_floor_ms"], 17.28, f"{label} compute floor")
    close(
        values["committed_per_event"],
        values["accept_per_event"] + 1.0,
        f"{label} committed/accepted",
    )
    close(
        values["step_wall_ms"],
        values["wall_s_per_event"] * 1000.0 * values["events_per_step"],
        f"{label} wall step",
    )
    close(
        values["measured_tps_fullstep_wall"],
        values["committed_per_event"] / values["wall_s_per_event"],
        f"{label} full-wall TPS",
        absolute=1e-8,
    )
    close(
        values["floor_ratio"],
        values["step_wall_ms"] / values["floor_ms"],
        f"{label} floor ratio",
    )
    close(
        values["events_per_step"],
        agg_fwd_drafts / agg_fwd_steps,
        f"{label} events_per_step/raw counters",
    )
    close(
        values["s_per_fwd_gpu"],
        nonnegative(raw_counters, RAW["fwd_gpu_seconds"], label)
        / agg_fwd_drafts,
        f"{label} SFWD/raw counters",
    )
    close(
        values["wall_s_per_event"],
        nonnegative(raw_counters, RAW["wall_seconds"], label)
        / agg_wall_drafts,
        f"{label} wall/raw counters",
    )
    spec_drafts = integral(raw_counters, RAW["spec_drafts"], label)
    if spec_drafts == 0:
        raise SystemExit(f"{label} has no speculative drafts")
    close(
        values["accept_per_event"],
        nonnegative(raw_counters, RAW["accepted_tokens"], label) / spec_drafts,
        f"{label} acceptance/raw counters",
    )
    drafter_spans = integral(raw_counters, RAW["drafter_gpu_spans"], label)
    committer_spans = integral(raw_counters, RAW["committer_gpu_spans"], label)
    if drafter_spans == 0 or committer_spans == 0:
        raise SystemExit(f"{label} component span counters are empty")
    close(
        values["drafter_gpu_ms_per_step"],
        nonnegative(raw_counters, RAW["drafter_gpu_seconds"], label)
        / drafter_spans
        * 1000.0,
        f"{label} DFWD/raw counters",
    )
    close(
        values["committer_gpu_ms_per_step"],
        nonnegative(raw_counters, RAW["committer_gpu_seconds"], label)
        / committer_spans
        * 1000.0,
        f"{label} CFWD/raw counters",
    )
    close(
        values["wall_steps_measured"],
        round(values["wall_steps_measured"]),
        f"{label} wall sample count",
    )
    close(
        values["wall_steps_measured"],
        agg_wall_steps,
        f"{label} wall sample/raw counters",
    )
    return {
        "raw_sha256": digest(raw),
        "measured_tps_fullstep_wall": values["measured_tps_fullstep_wall"],
        "step_wall_ms": values["step_wall_ms"],
        "accept_per_event": values["accept_per_event"],
        "committed_per_event": values["committed_per_event"],
        "wall_steps_measured": int(values["wall_steps_measured"]),
        "pure_decode_steps_measured": agg_fwd_steps,
        "retained_wall_fraction": agg_wall_steps / agg_fwd_steps,
        "retained_task_fractions": retained_task_fractions,
        "minimum_task_counter_steps": MIN_TASK_COUNTER_STEPS,
        "events_per_step": values["events_per_step"],
        "sfwd_gpu_ms_per_event": values["s_per_fwd_gpu"] * 1000.0,
        "dfwd_gpu_ms_per_step": values["drafter_gpu_ms_per_step"],
        "cfwd_gpu_ms_per_step": values["committer_gpu_ms_per_step"],
        "floor_ms": values["floor_ms"],
        "floor_ratio": values["floor_ratio"],
    }


if (
    digest(live_raw) != live_sha
    or digest(final_flush_raw) != final_flush_sha
    or digest(boundary_raw) != boundary_sha
    or digest(traffic_audit_raw) != traffic_audit_sha
    or digest(sidecar_raw) != sidecar_sha
    or sidecar.get("live_result_sha256") != live_sha
    or sidecar.get("qualified_candidate_source_sha256") != source_sha
    or sidecar.get("final_flush_sha256") != final_flush_sha
    or sidecar.get("boundary_snapshot_sha256") != boundary_sha
    or sidecar.get("chat_traffic_audit_sha256") != traffic_audit_sha
    or sidecar.get("qualified_events_sha256") != _live.get("events_sha256")
    or sidecar.get("qualified_flush_generation")
    != _live.get("flush_generation")
    or engagement.get("production_pass_sidecar_sha256") != sidecar_sha
    or engagement.get("candidate_source_sha256") != source_sha
    or engagement.get("observed_measured_replays_at_least") != 1
    or engagement.get("drafter_graph_signature")
    != "d9a4ddece41d146e9949b9f8ff7c2603b8948d157b28ef69244e44469b36150c"
    or engagement_validation != engagement
):
    raise SystemExit("candidate credential or replay engagement evidence drifted")

s = validate(stock, stock_raw, "stock", stock_arm)
c = validate(candidate, candidate_raw, "candidate", candidate_arm)
close(s["floor_ms"], c["floor_ms"], "cross-arm floor identity")
summary = {
    "schema": "fr13.fixed32.draft_head_m32_exact4_b1_timing.v1",
    "status": "complete",
    "classification": "real_swe_verified_exact4_b1_timing_candidate",
    "timing_eligible": True,
    "floor_acceptance_eligible": False,
    "batch_size": 1,
    "concurrency": 1,
    "instance_ids": task_ids,
    "live_pass_sha256": live_sha,
    "candidate_source_sha256": source_sha,
    "production_sidecar_sha256": sidecar_sha,
    "evidence_sha256": {
        "live_final_flush": digest(final_flush_raw),
        "live_boundary_snapshot": digest(boundary_raw),
        "live_chat_traffic_audit": digest(traffic_audit_raw),
        "production_engagement": digest(engagement_raw),
        "production_engagement_validation": digest(engagement_validation_raw),
        "stock_deploy_speed": digest(stock_raw),
        "candidate_deploy_speed": digest(candidate_raw),
        "stock_fa2": stock_fa2_sha,
    },
    "stock": s,
    "candidate": c,
    "delta": {
        "wall_ms": c["step_wall_ms"] - s["step_wall_ms"],
        "wall_percent": (c["step_wall_ms"] / s["step_wall_ms"] - 1.0) * 100.0,
        "dfwd_gpu_ms_per_step": (
            c["dfwd_gpu_ms_per_step"] - s["dfwd_gpu_ms_per_step"]
        ),
        "full_wall_tps": c["measured_tps_fullstep_wall"] - s["measured_tps_fullstep_wall"],
        "accept_per_event": c["accept_per_event"] - s["accept_per_event"],
    },
    "mandatory_weight_floor_ms": 102.479937172,
    "mandatory_weight_bytes": 27977022848,
    "floor_bandwidth_bytes_per_s": 273000000000,
    "acceptance_cap_ms": 117.8519277478,
    "note": "One topology timing pair; not the formal Tail/Hydra U95 gate.",
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
