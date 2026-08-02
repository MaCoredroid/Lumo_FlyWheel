#!/usr/bin/env bash
# One real SWE-Verified K64/root1 B1 SFWD byte diagnostic.
# The fused candidate runs in shadow; every served tensor remains incumbent.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the exact FA2 shared object}"

RUNROOT_ABS=$(realpath -m "$RUNROOT")
SOURCE_COMMIT=$(git rev-parse HEAD)
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
DRAFT_VOCAB_BLOCKS=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
K64_ROOT_WEIGHT_BYTES=32666638208
K64_ROOT_FLOOR_MS=119.658015414
K64_ROOT_CAP_MS=137.6067177261
ARM="hydra27_fixed32_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"
TASK_ID=astropy__astropy-12907

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical one-task B1 subset SHA-256 drift" >&2; exit 2; }
[[ -f "$DRAFT_VOCAB_BLOCKS" && ! -L "$DRAFT_VOCAB_BLOCKS" ]] \
  || { echo "K64 draft-vocab block map must be a regular source file" >&2; exit 2; }
[[ "$(sha256sum "$DRAFT_VOCAB_BLOCKS" | awk '{print $1}')" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
  || { echo "K64 draft-vocab block map SHA-256 drift" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "current source identity is invalid" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }

RUNROOT=${RUNROOT_ABS#"$REPO/"}
export RUNROOT

export FR13_GATE_QROW16=0
export FR13_GATE_TAW_NATIVE=0
export FR13_GATE_DRAFT_HEAD_PAD=0
export FR13_GATE_GDN_BV=0
export FR13_GATE_BM8=0
export FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0
export FR13_FA2_QROW16_PRODUCTION=0
export FR13_DFWD_UNIFIED_BM8_LIVE_AB=0
export FR13_DFWD_UNIFIED_BM8_PRODUCTION=0
export FR13_FIXED32_GDN_PATH_BV_CANDIDATE=
export FR13_FIXED32_GDN_PATH_BV_PRODUCTION=
export FR13_FIXED32_BATCH_GDN_BYTE_AB=0
export FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0
export FR13_FIXED32_BATCH_GDN_PRODUCTION=0
export FR13_FIXED32_BATCH_GDN_BV_CANDIDATE=
export FR13_FIXED32_BATCH_GDN_BV_PRODUCTION=
export FR13_FIXED32_BATCH_GDN_BV8_TIMING=0
export FR13_FIXED32_CUTLASS_WAVE=stock
export FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0
export FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=1
export FR13_CONV_WB_BATCHED=1
export FR13_TREE_CONV_FUSED=1
export ENFORCE_EAGER=1
export FR10_METRICS=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER"
unset FR13_NEEDS_ALLOW
export FR13_MANDATORY_WEIGHT_BYTES="$K64_ROOT_WEIGHT_BYTES"
export FR13_WEIGHT_FLOOR_MS="$K64_ROOT_FLOOR_MS"

bash scripts/fr13_run_b1_kernel_live_gate.sh

printf '%s\n' \
  'classification=one_real_swe_verified_k64_root_b1_sfwd_state_fusion_byte_diagnostic' \
  'task_set=one' \
  'task_count=1' \
  'timing_eligible=false' \
  'floor_acceptance_eligible=false' \
  'reference_returned=true' \
  'production_enabled=false' \
  'physical_rows_per_request=32' \
  'candidate_conv_launches_per_layer=1' \
  'gdn_physical_launches_per_layer=2' \
  'gdn_level_path_programs=1,11' \
  'draft_vocab_root=1' \
  'draft_vocab_k=65536' \
  "draft_vocab_blocks_sha256=$DRAFT_VOCAB_BLOCKS_SHA256" \
  >> "$RUNROOT_ABS/launcher_meta.txt"

.venv/bin/python - \
  "$ARMDIR" "$SOURCE_COMMIT" "$TASK_ID" \
  "$K64_ROOT_WEIGHT_BYTES" "$K64_ROOT_FLOOR_MS" "$K64_ROOT_CAP_MS" \
  "$DRAFT_VOCAB_BLOCKS_SHA256" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

arm_dir = Path(sys.argv[1]).resolve()
source_commit = sys.argv[2]
task_id = sys.argv[3]
weight_bytes = int(sys.argv[4])
floor_ms = float(sys.argv[5])
cap_ms = float(sys.argv[6])
draft_vocab_blocks_sha256 = sys.argv[7]
logs = arm_dir / "logs"
run_root = arm_dir.parent
records_path = logs / "fr13_fixed32_sfwd_state_fusion.byte_ab.jsonl"
pass_path = logs / "fr13_fixed32_sfwd_state_fusion.live_pass.json"
marker_path = logs / "fr13_fixed32_sfwd_state_fusion.real_event.arm"
diagnostic_path = arm_dir / "fixed32_b1_diagnostic.json"
container_env_path = arm_dir / "container_env.txt"
process_identity_path = arm_dir / "fixed32_process_identity.json"
engine_ledger_path = logs / "fr13_fixed32_engine_ingress.jsonl"
docker_after_tasks_path = arm_dir / "docker_after_tasks.log"
runtime_manifest_launch_path = run_root / "runtime_manifest.at_launch.json"
runtime_manifest_end_path = run_root / "runtime_manifest.at_end.json"
external_manifest_launch_path = run_root / "external_manifest.at_launch.json"
external_manifest_end_path = run_root / "external_manifest.at_end.json"
terminal_path = arm_dir / "fixed32_final_flush_skipped.json"
traffic_path = arm_dir / "fixed32_chat_traffic_audit_skipped.json"
output_path = arm_dir / "sfwd_state_fusion_k64_root_b1_gate.json"


def regular(path: Path, *, nonempty: bool = True) -> bytes:
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise SystemExit(f"required gate artifact is not regular: {path}")
    raw = path.read_bytes()
    if nonempty and not raw:
        raise SystemExit(f"required gate artifact is empty: {path}")
    return raw


records_raw = regular(records_path)
pass_raw = regular(pass_path)
marker_info = os.lstat(marker_path)
if (
    not stat.S_ISREG(marker_info.st_mode)
    or stat.S_ISLNK(marker_info.st_mode)
    or marker_info.st_nlink != 1
    or stat.S_IMODE(marker_info.st_mode) != 0o444
):
    raise SystemExit("authenticated real-event marker metadata mismatch")
marker_raw = regular(marker_path)
diagnostic_raw = regular(diagnostic_path)
container_env_raw = regular(container_env_path)
process_identity_raw = regular(process_identity_path)
engine_ledger_raw = regular(engine_ledger_path)
docker_after_tasks_raw = regular(docker_after_tasks_path)
runtime_manifest_launch_raw = regular(runtime_manifest_launch_path)
runtime_manifest_end_raw = regular(runtime_manifest_end_path)
external_manifest_launch_raw = regular(external_manifest_launch_path)
external_manifest_end_raw = regular(external_manifest_end_path)
terminal_raw = regular(terminal_path)
traffic_raw = regular(traffic_path)
records = [json.loads(line) for line in records_raw.decode("ascii").splitlines()]
live_pass = json.loads(pass_raw.decode("ascii"))
diagnostic = json.loads(diagnostic_raw.decode("ascii"))
process_identity = json.loads(process_identity_raw.decode("ascii"))
terminal = json.loads(terminal_raw.decode("ascii"))
traffic = json.loads(traffic_raw.decode("ascii"))
marker = f"swe_verified:{task_id}"
errors = []

if runtime_manifest_launch_raw != runtime_manifest_end_raw:
    errors.append("fixed32 runtime manifest drifted during the real task")
if external_manifest_launch_raw != external_manifest_end_raw:
    errors.append("fixed32 external manifest drifted during the real task")

docker_after_tasks = docker_after_tasks_raw.decode("utf-8", errors="replace")
shim_prefix = "[FR13_DRAFT_VOCAB] shim built K=65536 "
root_prefix = "[FR13_DRAFT_VOCAB_ROOT] engaged K=65536 "
disabled_prefix = "[FR13_DRAFT_VOCAB] DISABLED"
shim_lines = [
    line for line in docker_after_tasks.splitlines() if shim_prefix in line
]
root_lines = [
    line for line in docker_after_tasks.splitlines() if root_prefix in line
]
if len(shim_lines) != 1 or "mode=gather" not in shim_lines[0]:
    errors.append("K64 draft-vocabulary gather shim did not engage exactly once")
if len(root_lines) != 1 or "mode=gather" not in root_lines[0]:
    errors.append("K64 root gather did not engage exactly once")
if disabled_prefix in docker_after_tasks:
    errors.append("draft-vocabulary runtime fallback to full vocabulary engaged")

bracket_paths = list(
    arm_dir.glob(
        f"swe_out/*/per_task/{task_id}/fixed32_task_boundary.json"
    )
)
if len(bracket_paths) != 1:
    errors.append("eager real-task metrics bracket is missing or ambiguous")
    bracket = {}
    bracket_raw = b""
else:
    bracket_raw = regular(bracket_paths[0])
    bracket = json.loads(bracket_raw.decode("utf-8"))

if not records:
    errors.append("SFWD state-fusion byte gate was vacuous")
if any(record.get("schema") != "fr13.fixed32.sfwd_state_fusion.byte_ab.v1" for record in records):
    errors.append("comparison record schema mismatch")
if any(record.get("task_marker") != marker for record in records):
    errors.append("comparison record is not bound to the real SWE task")
if any(record.get("batch") != 1 for record in records):
    errors.append("comparison did not use B1")
if any(record.get("physical_rows_per_request") != 32 for record in records):
    errors.append("comparison did not preserve fixed32 physical rows")
if any(record.get("candidate_conv_launches_per_layer") != 1 for record in records):
    errors.append("candidate conv launch count drifted")
if any(record.get("gdn_level_path_programs") != [1, 11] for record in records):
    errors.append("GDN fixed32 program geometry drifted")
if any(record.get("gdn_physical_launches_per_layer") != 2 for record in records):
    errors.append("GDN physical launch count drifted")
if any(record.get("zero_diff") is not True for record in records):
    errors.append("at least one SFWD state-fusion surface differed")
if any(record.get("reference_always_served") is not True for record in records):
    errors.append("a comparison was not reference-returning")
if any(record.get("production_eligible") is not False for record in records):
    errors.append("a one-task comparison claimed production eligibility")

expected_pass = {
    "schema": "fr13.fixed32.sfwd_state_fusion.live_pass.v1",
    "status": "byte_pass_source_only",
    "run_classification": "one_real_swe_verified_k64_root_b1_byte_diagnostic",
    "candidate": "fixed32_sfwd_state_fusion_rowgroup32_c64_xreuse_v6",
    "task_marker": marker,
    "batch": 1,
    "draft_vocab_k": 65536,
    "draft_vocab_root": 1,
    "draft_vocab_blocks_sha256": draft_vocab_blocks_sha256,
    "layer_count": 48,
    "physical_rows_per_request": 32,
    "candidate_conv_launches_per_layer": 1,
    "gdn_level_path_programs": [1, 11],
    "gdn_physical_launches_per_layer": 2,
    "reference_always_served": True,
    "production_eligible": False,
    "timing_eligible": False,
    "floor_acceptance_eligible": False,
}
for key, expected in expected_pass.items():
    if live_pass.get(key) != expected:
        errors.append(f"live PASS {key} mismatch")
if len(live_pass.get("layer_keys", [])) != 48:
    errors.append("live PASS does not cover all 48 layers")
if marker_raw != (marker + "\n").encode("ascii"):
    errors.append("authenticated real-event marker bytes mismatch")
if diagnostic.get("task_ids") != [task_id]:
    errors.append("B1 diagnostic task binding mismatch")
if diagnostic.get("floor_acceptance_eligible") is not False:
    errors.append("B1 diagnostic claimed floor acceptance eligibility")

expected_terminal = {
    "schema": "fr13-fixed32-eager-kernel-terminal-v1",
    "run_classification": "eager_kernel_byte_diagnostic",
    "acceptance_valid": False,
    "flush_protocol_used": False,
}
if terminal != expected_terminal:
    errors.append("eager terminal no-flush marker mismatch")
if (
    traffic.get("schema")
    != "fr13-fixed32-eager-kernel-traffic-audit-skip-v1"
    or traffic.get("run_classification")
    != "eager_kernel_byte_diagnostic"
    or traffic.get("acceptance_valid") is not False
    or traffic.get("authenticated_engine_ledger_snapshotted") is not True
    or traffic.get("graph_census_audit_used") is not False
):
    errors.append("eager graph-census skip marker mismatch")
if (arm_dir / "fixed32_final_flush.json").exists():
    errors.append("eager diagnostic unexpectedly emitted a graph flush")
if (arm_dir / "fixed32_chat_traffic_audit.json").exists():
    errors.append("eager diagnostic unexpectedly emitted a graph census audit")
if (
    bracket.get("schema")
    != "fr13-fixed32-eager-kernel-diagnostic-task-bracket-v1"
    or bracket.get("run_classification")
    != "eager_kernel_byte_diagnostic"
    or bracket.get("instance_id") != task_id
    or bracket.get("acceptance_valid") is not False
    or bracket.get("flush_protocol_used") is not False
    or not isinstance(bracket.get("pre_metrics"), dict)
    or not isinstance(bracket.get("post_metrics"), dict)
):
    errors.append("eager real-task metrics bracket contract mismatch")

container_env = container_env_raw.decode("ascii").splitlines()
for expected in (
    "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=1",
    "FR13_DRAFT_VOCAB_ROOT=1",
    "FR13_DRAFT_VOCAB_K=65536",
    "FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json",
    "FR13_CONV_WB_BATCHED=1",
    "FR13_TREE_CONV_FUSED=1",
    "FR13_FIXED32_CUTLASS_WAVE=stock",
    "ENFORCE_EAGER=1",
):
    if container_env.count(expected) != 1:
        errors.append(f"container environment mismatch: {expected}")

pid1 = process_identity.get("pid1")
pid1_argv = pid1.get("argv") if isinstance(pid1, dict) else None
if (
    not isinstance(pid1_argv, list)
    or not all(isinstance(argument, str) for argument in pid1_argv)
    or pid1_argv.count("--max-num-seqs") != 1
):
    errors.append("captured PID 1 argv has invalid --max-num-seqs cardinality")
else:
    max_num_seqs_index = pid1_argv.index("--max-num-seqs")
    if (
        max_num_seqs_index + 1 >= len(pid1_argv)
        or pid1_argv[max_num_seqs_index + 1] != "1"
    ):
        errors.append("captured PID 1 argv did not use --max-num-seqs 1")

payload = {
    "schema": "fr13.fixed32.sfwd_state_fusion.k64_root_b1_gate.v1",
    "status": "pass" if not errors else "fail",
    "run_classification": "one_real_swe_verified_k64_root_b1_byte_diagnostic",
    "task_set": "one",
    "task_count": 1,
    "task_ids": [task_id],
    "task_marker": marker,
    "real_task_authenticated": True,
    "reference_returned": True,
    "timing_eligible": False,
    "floor_acceptance_eligible": False,
    "production_enabled": False,
    "candidate": "fixed32_sfwd_state_fusion_rowgroup32_c64_xreuse_v6",
    "batch_size": 1,
    "physical_rows_per_request": 32,
    "candidate_conv_launches_per_layer": 1,
    "gdn_level_path_programs": [1, 11],
    "gdn_physical_launches_per_layer": 2,
    "draft_vocab_root": 1,
    "draft_vocab_k": 65536,
    "draft_vocab_blocks_sha256": draft_vocab_blocks_sha256,
    "mandatory_weight_bytes": weight_bytes,
    "mandatory_weight_floor_ms": floor_ms,
    "one_sided_u95_cap_ms": cap_ms,
    "comparisons": len(records),
    "layer_count": live_pass.get("layer_count"),
    "mismatching_comparisons": sum(record.get("zero_diff") is not True for record in records),
    "source_commit": source_commit,
    "records_sha256": hashlib.sha256(records_raw).hexdigest(),
    "live_pass_sha256": hashlib.sha256(pass_raw).hexdigest(),
    "real_event_marker_sha256": hashlib.sha256(marker_raw).hexdigest(),
    "engine_ingress_ledger_sha256": hashlib.sha256(engine_ledger_raw).hexdigest(),
    "docker_after_tasks_sha256": hashlib.sha256(docker_after_tasks_raw).hexdigest(),
    "runtime_manifest_sha256": hashlib.sha256(
        runtime_manifest_launch_raw
    ).hexdigest(),
    "external_manifest_sha256": hashlib.sha256(
        external_manifest_launch_raw
    ).hexdigest(),
    "eager_task_bracket_sha256": (
        hashlib.sha256(bracket_raw).hexdigest() if bracket_raw else None
    ),
    "terminal_skip_sha256": hashlib.sha256(terminal_raw).hexdigest(),
    "traffic_skip_sha256": hashlib.sha256(traffic_raw).hexdigest(),
    "container_env_sha256": hashlib.sha256(container_env_raw).hexdigest(),
    "process_identity_sha256": hashlib.sha256(process_identity_raw).hexdigest(),
    "errors": errors,
}
output_path.write_text(
    json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
if errors:
    raise SystemExit(4)
PY
