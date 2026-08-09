#!/usr/bin/env bash
# Real SWE-Verified exact4 B4 proposal-quality timing: stock K64 head vs M32.
# Draft logits and acceptance may differ. Target sampling remains authoritative.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${STOCK_FA2_SO:?set STOCK_FA2_SO to the exact-safe stock FA2 binary}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
FIXED32_MODE=${FR13_DRAFT_HEAD_B4_FIXED32_MODE:-hydra27_fixed32}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
DRAFT_VOCAB_BLOCKS_HOST=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
CANDIDATE_SOURCE=scripts/fr10_phase4_patch_vllm_tree_gdn.py
CANDIDATE_SOURCE_SHA256=$(sha256sum "$CANDIDATE_SOURCE" | awk '{print $1}')
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNROOT_ABS=$(realpath -m "$RUNROOT")
B4_KV_CACHE_MEMORY_BYTES=49392123904
MANDATORY_WEIGHT_BYTES=32666638208
MANDATORY_WEIGHT_FLOOR_MS=119.658015414
ONE_SIDED_U95_CAP_MS=137.6067177261

case "$FIXED32_MODE" in
  tail6_fixed32)
    LOGICAL_TOPOLOGY=Tail23
    ACTIVE_DRAFTS=23
    VALID_MASK=0x7a9ce7ff
    ;;
  hydra27_fixed32)
    LOGICAL_TOPOLOGY=Hydra27
    ACTIVE_DRAFTS=27
    VALID_MASK=0x7abdffff
    ;;
  *)
    echo "FR13_DRAFT_HEAD_B4_FIXED32_MODE must be tail6_fixed32 or hydra27_fixed32" >&2
    exit 2
    ;;
esac

STOCK_ARM="${FIXED32_MODE}_draft_head_stock_b4_k64_root_${TAG}"
CANDIDATE_ARM="${FIXED32_MODE}_draft_head_m32_b4_k64_root_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ "$STOCK_FA2_SO" == /* && -f "$STOCK_FA2_SO" && ! -L "$STOCK_FA2_SO" ]] \
  || { echo "STOCK_FA2_SO must be an absolute regular non-symlink file" >&2; exit 2; }
[[ "$(stat -c '%s' "$STOCK_FA2_SO")" == "$STOCK_FA2_BYTES" \
   && "$(sha256sum "$STOCK_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "STOCK_FA2_SO is not the exact-safe stock reference" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical exact4 subset SHA-256 drift" >&2; exit 2; }
[[ -f "$DRAFT_VOCAB_BLOCKS_HOST" && ! -L "$DRAFT_VOCAB_BLOCKS_HOST" \
   && "$(sha256sum "$DRAFT_VOCAB_BLOCKS_HOST" | awk '{print $1}')" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
  || { echo "pinned root-64K draft-vocabulary block map drifted" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER"
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" ]] \
  || { echo "canonical K64/root1 floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
printf 'classification=real_swe_verified_exact4_b4_k64_root_proposal_quality_timing\ntiming_eligible=1\nformal_floor_acceptance_eligible=0\nproduction_default_enabled=0\nonly_arm_delta=FR13_DRAFT_HEAD_PAD_ROWS_0_to_32\ndraft_logits_may_differ=1\nacceptance_may_differ=1\ntopology=%s\nlogical_topology=%s\nactive_drafts=%s\nvalid_mask=%s\nphysical_drafts=31\nphysical_rows_root_inclusive=32\nbatch_size=4\nconcurrency=4\ndraft_vocab_root=1\ndraft_vocab_k=65536\ndraft_vocab_blocks=%s\ndraft_vocab_blocks_sha256=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nrunroot=%s\nstock_arm=%s\ncandidate_arm=%s\nsource=%s\ncandidate_source_sha256=%s\nrunner_sha256=%s\nsubset_sha256=%s\nstock_fa2_sha256=%s\nenforce_eager=0\ncudagraph_mode=FULL_AND_PIECEWISE\nkv_cache_memory_bytes=%s\nstarted=%s\n' \
  "$FIXED32_MODE" "$LOGICAL_TOPOLOGY" "$ACTIVE_DRAFTS" "$VALID_MASK" \
  "$DRAFT_VOCAB_BLOCKS_CONTAINER" "$DRAFT_VOCAB_BLOCKS_SHA256" \
  "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$RUNROOT_ABS" "$STOCK_ARM" "$CANDIDATE_ARM" \
  "$SOURCE_COMMIT" "$CANDIDATE_SOURCE_SHA256" "$RUNNER_SHA256" \
  "$SUBSET_SHA256" "$STOCK_FA2_SHA256" "$B4_KV_CACHE_MEMORY_BYTES" \
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
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json" || return $?
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json" || return $?
  cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || { echo "runtime/source manifest changed during timing" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
    || { echo "B4 draft-head timing runner changed during execution" >&2; return 14; }
  [[ "$(sha256sum "$CANDIDATE_SOURCE" | awk '{print $1}')" == "$CANDIDATE_SOURCE_SHA256" ]] \
    || { echo "draft-head candidate source changed during timing" >&2; return 14; }
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
  local pad_rows=$2
  echo "===== $arm: exact4 B4 draft-head pad_rows=$pad_rows ====="
  if env \
      RUNROOT="$RUNROOT_ABS" \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S=5400 \
      KV_CACHE_MEMORY_BYTES="$B4_KV_CACHE_MEMORY_BYTES" \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
      FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER" \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
      FR13_SCAN_ALIGN=0 FR13_NPAD_INVARIANT=0 \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json" \
      FR13_DRAFT_HEAD_PAD_ROWS="$pad_rows" \
      FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
      FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_SO= \
      FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_INSTANCE_ID= \
      FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
      FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
      FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
      FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$STOCK_FA2_SO" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh \
        "$arm" "$FIXED32_MODE" "$SUBSET" \
        > "$RUNROOT_ABS/$arm.runlog" 2>&1; then
    :
  else
    local serve_rc=$?
    printf 'arm=%s serve_rc=%s ended=%s\n' \
      "$arm" "$serve_rc" "$(date -u +%FT%TZ)" \
      >> "$RUNROOT_ABS/launcher_meta.txt"
    return "$serve_rc"
  fi

  local container_env="$RUNROOT_ABS/$arm/container_env.txt"
  local traffic_audit="$RUNROOT_ABS/$arm/fixed32_chat_traffic_audit.json"
  local docker_log="$RUNROOT_ABS/$arm/docker_full.log"
  for artifact in "$container_env" "$traffic_audit" "$docker_log"; do
    [[ -f "$artifact" && ! -L "$artifact" ]] \
      || { echo "$arm lacks regular runtime evidence: $artifact" >&2; return 4; }
  done
  for expected in \
    "FR13_FIXED32_MODE=$FIXED32_MODE" \
    "FR13_DRAFT_VOCAB_ROOT=1" \
    "FR13_DRAFT_VOCAB_K=65536" \
    "FR13_DRAFT_VOCAB_BLOCKS=$DRAFT_VOCAB_BLOCKS_CONTAINER" \
    "FR13_DRAFT_HEAD_PAD_ROWS=$pad_rows" \
    "FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0" \
    "FR13_DRAFT_HEAD_M32_LIVE_AB=0" \
    "FR13_DRAFT_HEAD_M32_PRODUCTION=0"; do
    [[ "$(grep -Fxc "$expected" "$container_env")" -eq 1 ]] \
      || { echo "$arm runtime environment lacks exact pin: $expected" >&2; return 4; }
  done

  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$arm" --out-root "$RUNROOT_ABS/$arm/swe_out" \
    --expected-tok-per-draft 31 --batch-size 4 \
    --out "$RUNROOT_ABS/$arm/deploy_speed_fullwall.json"
  printf 'arm=%s pad_rows=%s serve_rc=0 container_env_sha256=%s traffic_audit_sha256=%s ended=%s\n' \
    "$arm" "$pad_rows" \
    "$(sha256sum "$container_env" | awk '{print $1}')" \
    "$(sha256sum "$traffic_audit" | awk '{print $1}')" \
    "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
}

run_arm "$STOCK_ARM" 0
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after the stock reference" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" 32
finalize_manifests

"$PYTHON_BIN" - \
  "$SUBSET" \
  "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$STOCK_ARM" "$RUNROOT_ABS/$CANDIDATE_ARM" \
  "$RUNROOT_ABS/timing_summary.json" \
  "$FIXED32_MODE" "$LOGICAL_TOPOLOGY" "$ACTIVE_DRAFTS" "$VALID_MASK" \
  "$SOURCE_COMMIT" "$CANDIDATE_SOURCE_SHA256" "$RUNNER_SHA256" \
  "$SUBSET_SHA256" "$DRAFT_VOCAB_BLOCKS_SHA256" "$STOCK_FA2_SHA256" \
  "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" <<'PY'
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from fr13_b4_timing_math import phase_breakdown, positive, promotion_verdict
from fr13_floor_gate import (
    build_fixed32_chat_traffic_audit,
    pinned_dataset_record_digests,
    validate_fixed32_run_subset,
)

subset_path, stock_path, candidate_path, stock_root, candidate_root, out_path = map(
    Path, sys.argv[1:7]
)
fixed32_mode, logical_topology = sys.argv[7:9]
active_drafts = int(sys.argv[9])
valid_mask = int(sys.argv[10], 0)
source_commit, candidate_source_sha256, runner_sha256 = sys.argv[11:14]
subset_sha256, block_map_sha256, stock_fa2_sha256 = sys.argv[14:17]
mandatory_weight_bytes = int(sys.argv[17])
mandatory_weight_floor_ms = float(sys.argv[18])
one_sided_u95_cap_ms = float(sys.argv[19])
repo = Path.cwd().resolve()


def regular(path: Path, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"{label} is not a regular file")
    return path.read_bytes()


subset = validate_fixed32_run_subset(subset_path, b1_diagnostic=False)
task_ids = list(subset["task_ids"])
if subset["sha256"] != subset_sha256 or len(task_ids) != 4:
    raise SystemExit("canonical exact4 subset binding drifted")
dataset_record_digests = pinned_dataset_record_digests(str(repo))


def validate_audit(arm_root: Path, label: str) -> dict[str, object]:
    audit_path = arm_root / "fixed32_chat_traffic_audit.json"
    raw = regular(audit_path, f"{label} authenticated traffic audit")
    audit = json.loads(raw.decode("ascii"))
    rebuilt = build_fixed32_chat_traffic_audit(
        arm_root,
        mode=fixed32_mode,
        subset=subset,
        dataset_record_digests=dataset_record_digests,
        concurrency=4,
    )
    if audit != rebuilt:
        raise SystemExit(f"{label} authenticated exact4 traffic audit does not rebuild")
    terminals = [audit["tasks"][task_id]["terminal"] for task_id in task_ids]
    verdicts = [terminal["eval"]["verdict"] for terminal in terminals]
    if any(
        terminal["agent"]
        != {
            "exit_code": 0,
            "timed_out": False,
            "offloaded": True,
            "network_drop": False,
        }
        or terminal["eval"]["verdict"] not in {"resolved", "failed"}
        for terminal in terminals
    ):
        raise SystemExit(f"{label} has a nonterminal exact4 task")
    return {
        "schema": audit["schema"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "pure_decode_forward_steps": audit["complete_stream"][
            "pure_decode_forward_steps"
        ],
        "complete_work_census_events": audit["complete_stream"][
            "complete_work_census_events"
        ],
        "resolved_tasks": verdicts.count("resolved"),
        "failed_tasks": verdicts.count("failed"),
        "all_tasks_terminal": True,
        "all_authenticated_checks_true": all(audit["checks"].values()),
    }


def validate_measure(path: Path, label: str) -> tuple[dict[str, object], str]:
    raw = regular(path, f"{label} deploy-speed evidence")
    record = json.loads(raw.decode("utf-8"))
    if (
        record.get("schema") != "fr13.measure.deploy_speed.v1"
        or record.get("regime") != "deployment"
        or record.get("instrument") != "OFF"
        or record.get("batch_size") != 4
        or record.get("n_tasks") != 4
        or record.get("task_instance_ids") != task_ids
        or record.get("draft_vocab_root") != 1
        or record.get("draft_vocab_k") != 65536
        or record.get("mandatory_weight_bytes") != mandatory_weight_bytes
        or not math.isclose(
            float(record.get("weight_floor_ms", math.nan)),
            mandatory_weight_floor_ms,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or record.get("floor_reference_scope")
        != "fixed32_mandatory_weight_read_or_row_compute_lower_bound"
        or record.get("floor_is_full_step_hardware_floor") is not False
    ):
        raise SystemExit(f"{label} deploy-speed provenance is not exact4 B4 K64/root1")
    for key in (
        "measured_tps_fullstep_wall",
        "step_wall_ms",
        "accept_per_event",
        "committed_per_event",
        "wall_steps_measured",
        "events_per_step",
        "s_per_fwd_gpu",
        "s_per_fwd_gpu_per_forward",
        "wall_s_per_event",
        "drafter_gpu_ms_per_step",
        "committer_gpu_ms_per_step",
        "floor_ms",
        "floor_ratio",
    ):
        positive(record, key)
    per_task = record.get("per_task")
    if (
        not isinstance(per_task, list)
        or len(per_task) != 4
        or [row.get("instance_id") for row in per_task] != task_ids
        or any(float(row.get("drafts", 0.0)) <= 0 for row in per_task)
    ):
        raise SystemExit(f"{label} lacks four nonempty per-task timing windows")
    return record, hashlib.sha256(raw).hexdigest()


EAGER_ENGAGEMENT = re.compile(
    r"\[FR13_DRAFT_HEAD_PAD\] engaged candidate_rows=32 "
    r"source_rows=(?P<source_rows>[1-4]) eager_launch=1"
)
GRAPH_CAPTURE = re.compile(
    r"\[FR13_DRAFT_HEAD_PAD\] captured candidate_rows=32 source_rows=4"
)
FAILURE = re.compile(
    r"(?:\[FR13_DRAFT_HEAD_PAD\].*(?:fallback|error|failed|disabled)|"
    r"\[FR13_DRAFT_VOCAB\] DISABLED|"
    r"FR13 draft-head padding failed its strict runtime contract)",
    re.IGNORECASE,
)


def validate_engagement(arm_root: Path, label: str, expected: bool) -> dict[str, object]:
    log_path = arm_root / "docker_full.log"
    if not log_path.is_file() or log_path.is_symlink():
        raise SystemExit(f"{label} lacks a regular terminal container log")
    marker_rows: list[int] = []
    capture_markers = 0
    failures = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if FAILURE.search(line):
                failures += 1
            if "[FR13_DRAFT_HEAD_PAD] engaged" in line:
                match = EAGER_ENGAGEMENT.search(line)
                if match is None:
                    raise SystemExit(f"{label} emitted a malformed draft-head engagement")
                marker_rows.append(int(match.group("source_rows")))
            if "[FR13_DRAFT_HEAD_PAD] captured" in line:
                if GRAPH_CAPTURE.search(line) is None:
                    raise SystemExit(f"{label} emitted a malformed draft-head graph capture")
                capture_markers += 1
    if failures:
        raise SystemExit(f"{label} emitted draft-head fallback/error markers")
    if expected:
        if not marker_rows:
            raise SystemExit("candidate lacks eager M32 engagement")
        if capture_markers < 1:
            raise SystemExit("candidate lacks source_rows=4 M32 graph capture")
    elif marker_rows or capture_markers:
        raise SystemExit("stock arm emitted M32 engagement or graph capture")
    return {
        "eager_engagement_markers": len(marker_rows),
        "source_rows_observed": sorted(set(marker_rows)),
        "source_rows_4_observed": 4 in marker_rows,
        "b4_graph_capture_markers": capture_markers,
        "b4_graph_capture_observed": capture_markers > 0,
        "fallback_or_error_markers": failures,
    }


stock, stock_measure_sha256 = validate_measure(stock_path, "stock")
candidate, candidate_measure_sha256 = validate_measure(candidate_path, "candidate")
stock_audit = validate_audit(stock_root, "stock")
candidate_audit = validate_audit(candidate_root, "candidate")
stock_engagement = validate_engagement(stock_root, "stock", False)
candidate_engagement = validate_engagement(candidate_root, "candidate", True)
stock_phases = phase_breakdown(stock, "stock")
candidate_phases = phase_breakdown(candidate, "candidate")
stock_wall = positive(stock, "step_wall_ms")
candidate_wall = positive(candidate, "step_wall_ms")
stock_tps = positive(stock, "measured_tps_fullstep_wall")
candidate_tps = positive(candidate, "measured_tps_fullstep_wall")
stock_accept = positive(stock, "accept_per_event")
candidate_accept = positive(candidate, "accept_per_event")
stock_floor = positive(stock, "floor_ms")
candidate_floor = positive(candidate, "floor_ms")
if not math.isclose(stock_floor, candidate_floor, rel_tol=0.0, abs_tol=1e-9):
    raise SystemExit("stock and candidate optimistic floor values differ")


def arm_summary(
    record: dict[str, object],
    measurement_sha256: str,
    audit: dict[str, object],
    engagement: dict[str, object],
    phases: dict[str, float],
    *,
    selector: str,
    pad_rows: int,
) -> dict[str, object]:
    return {
        "selector": selector,
        "draft_head_pad_rows": pad_rows,
        "measurement_sha256": measurement_sha256,
        "authenticated_task_provenance": audit,
        "engagement": engagement,
        "step_wall_ms": float(record["step_wall_ms"]),
        "measured_tps_fullstep_wall": float(record["measured_tps_fullstep_wall"]),
        "accepted_drafts_per_event": float(record["accept_per_event"]),
        "committed_tokens_per_event": float(record["committed_per_event"]),
        "events_per_step": phases["events_per_step"],
        "wall_ms_per_event": phases["wall_ms_per_event"],
        "sfwd_gpu_ms_per_event": phases["sfwd_gpu_ms_per_event"],
        "sfwd_gpu_ms_per_step": phases["sfwd_gpu_ms_per_step"],
        "dfwd_gpu_ms_per_step": phases["dfwd_gpu_ms_per_step"],
        "cfwd_gpu_ms_per_step": phases["cfwd_gpu_ms_per_step"],
        "gpu_component_ms_per_step": phases["gpu_component_ms_per_step"],
        "other_wall_ms_per_step": phases["other_wall_ms_per_step"],
        "step_wall_to_optimistic_floor_ratio": float(record["floor_ratio"]),
    }


summary = {
    "schema": "fr13.fixed32.draft_head_m32.b4_k64_root.proposal_quality.full_wall_timing_pair.v1",
    "status": "complete",
    "run_classification": "real_swe_verified_exact4_b4_k64_root_proposal_quality_timing",
    "task_count": 4,
    "batch_size": 4,
    "concurrency": 4,
    "arm": fixed32_mode,
    "logical_topology": logical_topology,
    "active_drafts": active_drafts,
    "valid_mask": hex(valid_mask),
    "physical_drafts": 31,
    "physical_rows_root_inclusive": 32,
    "task_ids": task_ids,
    "draft_vocab_root": 1,
    "draft_vocab_k": 65536,
    "draft_vocab_blocks_sha256": block_map_sha256,
    "mandatory_weight_bytes": mandatory_weight_bytes,
    "mandatory_weight_floor_ms": mandatory_weight_floor_ms,
    "one_sided_u95_cap_ms": one_sided_u95_cap_ms,
    "optimistic_floor_ms": stock_floor,
    "optimistic_floor_is_full_step_hardware_floor": False,
    "decision_metric": "measured_tps_fullstep_wall",
    "promotion": promotion_verdict(stock_phases, candidate_phases),
    "candidate_scope": "proposal_quality",
    "draft_logits_may_differ": True,
    "acceptance_may_differ": True,
    "served_output_byte_identity_required": False,
    "target_rejection_sampler_remains_authoritative": True,
    "only_arm_delta": "FR13_DRAFT_HEAD_PAD_ROWS_0_to_32",
    "source_commit": source_commit,
    "candidate_source_sha256": candidate_source_sha256,
    "runner_sha256": runner_sha256,
    "stock_fa2_sha256": stock_fa2_sha256,
    "stock_reference": arm_summary(
        stock,
        stock_measure_sha256,
        stock_audit,
        stock_engagement,
        stock_phases,
        selector="stock_k64_root_draft_head",
        pad_rows=0,
    ),
    "candidate": arm_summary(
        candidate,
        candidate_measure_sha256,
        candidate_audit,
        candidate_engagement,
        candidate_phases,
        selector="replicated_hidden_m32_k64_root_draft_head",
        pad_rows=32,
    ),
    "candidate_to_stock_full_wall_tps_ratio": candidate_tps / stock_tps,
    "stock_to_candidate_step_wall_ratio": stock_wall / candidate_wall,
    "candidate_minus_stock_accepted_drafts_per_event": candidate_accept - stock_accept,
    "candidate_to_stock_acceptance_ratio": candidate_accept / stock_accept,
    "stock_to_candidate_dfwd_gpu_speedup": (
        stock_phases["dfwd_gpu_ms_per_step"]
        / candidate_phases["dfwd_gpu_ms_per_step"]
    ),
    "formal_floor_acceptance_eligible": False,
    "formal_floor_acceptance_reason": (
        "proposal-quality paired exact4 timing only; run the canonical statistical "
        "Tail23/Hydra27 floor campaign after selecting the acceptance-speed point"
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
