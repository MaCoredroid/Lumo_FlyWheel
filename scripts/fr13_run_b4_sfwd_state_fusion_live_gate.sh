#!/usr/bin/env bash
# Real SWE-Verified exact4 B4 byte gate for fixed32_sfwd_state_fusion_v1.
# The candidate is shadow-only; the incumbent reference remains served.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned FA2 shared object}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
KERNEL_SOURCE=src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py
PATCHER_SOURCE=scripts/fr10_phase4_patch_vllm_tree_gdn.py
RAW_SCHEMA=fr13.fixed32.sfwd_state_fusion.byte_ab.v1
SOURCE_PASS_SCHEMA=fr13.fixed32.sfwd_state_fusion.b4_source_pass.v1
LIVE_SCHEMA=fr13.fixed32.sfwd_state_fusion.exact4_b4_live_gate.v1
CONTAINER_JSONL=/logs/fr13_fixed32_sfwd_state_fusion.byte_ab.jsonl
CONTAINER_SOURCE_PASS=/logs/fr13_fixed32_sfwd_state_fusion.live_pass.json
B4_KV_CACHE_MEMORY_BYTES=42949672960
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
ARM="hydra27_fixed32_sfwd_b4_gate_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ "$FORKED_FA2_SO" == /* && -f "$FORKED_FA2_SO" && ! -L "$FORKED_FA2_SO" ]] \
  || { echo "FORKED_FA2_SO must be an absolute regular non-symlink file" >&2; exit 2; }
[[ "$(stat -c '%s' "$FORKED_FA2_SO")" == "$STOCK_FA2_BYTES" \
   && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "FORKED_FA2_SO is not the exact-safe stock reference" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical exact4 subset SHA-256 drift" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the gate" >&2; exit 2; }

export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=0
export FR13_DRAFT_VOCAB_K=0
export FR13_NEEDS_ALLOW='FR13_DRAFT_VOCAB_K=0'
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant

[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "42025179008" \
   && "$FR13_WEIGHT_FLOOR_MS" == "153.9383846446886" ]] \
  || { echo "canonical B4 full-vocabulary floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
printf 'classification=real_swe_verified_exact4_b4_byte_diagnostic\nacceptance_valid=0\ntiming_eligible=0\nfloor_acceptance_eligible=0\nproduction_enabled=0\ncandidate_shadow_only=1\nreference_always_served=1\ncandidate=fixed32_sfwd_state_fusion_v1\ntask_count=4\nbatch_size=4\nconcurrency=4\nphysical_rows_per_request=32\nphysical_rows_total=128\nlayer_count=48\ndraft_vocab_root=0\ndraft_vocab_k=0\nfr13_needs_allow=FR13_DRAFT_VOCAB_K=0\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\nlauncher_pid=%s\nrunroot=%s\narm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nstock_fa2_sha256=%s\nstock_fa2_bytes=%s\nenforce_eager=1\nkv_cache_memory_bytes=%s\nstarted=%s\n' \
  "$FR13_MANDATORY_WEIGHT_BYTES" "$FR13_WEIGHT_FLOOR_MS" "$$" \
  "$RUNROOT_ABS" "$ARM" "$SOURCE_COMMIT" "$RUNNER_SHA256" \
  "$SUBSET_SHA256" "$STOCK_FA2_SHA256" "$STOCK_FA2_BYTES" \
  "$B4_KV_CACHE_MEMORY_BYTES" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

if env \
    RUNROOT="$RUNROOT_ABS" \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S= \
    KV_CACHE_MEMORY_BYTES="$B4_KV_CACHE_MEMORY_BYTES" \
    FR13_FIXED32_B1_DIAGNOSTIC=0 \
    FR10_METRICS=0 ENFORCE_EAGER=1 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_DRAFT_VOCAB_ROOT=0 FR13_DRAFT_VOCAB_K=0 \
    FR13_NEEDS_ALLOW='FR13_DRAFT_VOCAB_K=0' \
    FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 FR13_TREE_RUNROW_INIT=1 \
    FR13_TREE_CONV_FUSED=1 FR13_CONV_WB_BATCHED=1 \
    FR13_FIXED32_CONV_SOURCE_BATCH=1 \
    FR13_SCAN_ALIGN=0 FR13_NPAD_INVARIANT=0 \
    FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=1 \
    FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_CUTLASS_WAVE=stock \
    FR13_FIXED32_CUTLASS_WAVE_SO= \
    FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
    FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_JSON= \
    FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_SHA256= \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 \
    FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_INSTANCE_ID= \
    FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 \
    FORKED_FA2_SO="$FORKED_FA2_SO" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" hydra27_fixed32 "$SUBSET" \
      > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi

printf 'serve_rc=%s ended=%s\n' "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json"
cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
  "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  || { echo "runtime/source manifest changed during diagnostic" >&2; exit 14; }
cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
  "$RUNROOT_ABS/external_manifest.at_end.json" \
  || { echo "external manifest changed during diagnostic" >&2; exit 14; }
[[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
  || { echo "SFWD B4 gate runner changed during execution" >&2; exit 14; }
(( serve_rc == 0 )) || exit "$serve_rc"

"$PYTHON_BIN" - \
  "$ARMDIR" "$ARMDIR$CONTAINER_JSONL" "$ARMDIR$CONTAINER_SOURCE_PASS" \
  "$ARMDIR/sfwd_state_fusion_b4_byte_gate.json" "$KERNEL_SOURCE" \
  "$PATCHER_SOURCE" "$SOURCE_COMMIT" "$SUBSET_SHA256" "$RAW_SCHEMA" \
  "$SOURCE_PASS_SCHEMA" "$LIVE_SCHEMA" "$RUNNER_SHA256" \
  "$RUNROOT_ABS/runtime_manifest.at_end.json" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, "src")
from lumo_flywheel_serving.inference_proxy import (
    fixed32_canonical_task_set_sha256,
    fixed32_task_key_id,
    verify_fixed32_ingress_ledger,
)

arm = Path(sys.argv[1]).resolve()
jsonl_path = Path(sys.argv[2])
source_pass_path = Path(sys.argv[3])
output_path = Path(sys.argv[4])
kernel_source = Path(sys.argv[5])
patcher_source = Path(sys.argv[6])
source_commit = sys.argv[7]
subset_sha256 = sys.argv[8]
raw_schema = sys.argv[9]
source_pass_schema = sys.argv[10]
live_schema = sys.argv[11]
runner_sha256 = sys.argv[12]
runtime_manifest_path = Path(sys.argv[13])
expected_tasks = [
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
]
expected_markers = [f"swe_verified:{task_id}" for task_id in expected_tasks]
expected_marker_raw = ("\n".join(expected_markers) + "\n").encode("ascii")
logs = arm / "logs"
marker_path = logs / "fr13_fixed32_sfwd_state_fusion.real_event.arm"
ledger_path = logs / "fr13_fixed32_engine_ingress.jsonl"
container_env_path = arm / "container_env.txt"
errors = []

marker_info = os.lstat(marker_path)
if (
    not stat.S_ISREG(marker_info.st_mode)
    or stat.S_ISLNK(marker_info.st_mode)
    or marker_info.st_nlink != 1
    or stat.S_IMODE(marker_info.st_mode) != 0o444
):
    errors.append("SFWD exact4 marker identity or mode is invalid")
marker_raw = marker_path.read_bytes()
if marker_raw != expected_marker_raw:
    errors.append("SFWD marker is not the canonical authenticated exact4 set")

try:
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="ascii").splitlines()
    ]
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"SFWD byte records cannot be read: {error}")
if len(records) != 48:
    errors.append("SFWD byte gate did not record exactly 48 unique layers")
layer_keys = [record.get("layer_key") for record in records]
if len(set(layer_keys)) != 48:
    errors.append("SFWD byte records do not cover 48 unique layer keys")
for index, record in enumerate(records):
    if record.get("schema") != raw_schema:
        errors.append(f"record {index} schema mismatch")
    if record.get("candidate") != "fixed32_sfwd_state_fusion_v1":
        errors.append(f"record {index} candidate mismatch")
    if record.get("task_markers") != expected_markers:
        errors.append(f"record {index} marker set mismatch")
    if record.get("batch") != 4:
        errors.append(f"record {index} is not physical B4")
    if record.get("physical_rows_per_request") != 32:
        errors.append(f"record {index} row geometry mismatch")
    if (
        record.get("status") != "pass"
        or record.get("zero_diff") is not True
        or record.get("first_nonzero") is not None
    ):
        errors.append(f"record {index} did not byte-pass")
    if (
        record.get("candidate_shadow_only") is not True
        or record.get("served_result") != "reference"
        or record.get("reference_always_served") is not True
        or record.get("acceptance_valid") is not False
        or record.get("timing_eligible") is not False
        or record.get("production_eligible") is not False
    ):
        errors.append(f"record {index} classification drift")
    comparisons = record.get("comparisons")
    if (
        not isinstance(comparisons, list)
        or [item.get("name") for item in comparisons]
        != ["conv_out", "commit_source_stage"]
        or any(item.get("byte_equal") is not True for item in comparisons)
        or any(item.get("differing_bytes") != 0 for item in comparisons)
    ):
        errors.append(f"record {index} comparison surface mismatch")

source_pass_raw = source_pass_path.read_bytes()
source_pass = json.loads(source_pass_raw.decode("ascii"))
source_expected = {
    "schema": source_pass_schema,
    "status": "byte_pass_source_only",
    "run_classification": "real_swe_verified_exact4_b4_byte_diagnostic",
    "candidate": "fixed32_sfwd_state_fusion_v1",
    "source_sha256": hashlib.sha256(kernel_source.read_bytes()).hexdigest(),
    "task_count": 4,
    "task_ids": expected_tasks,
    "task_markers": expected_markers,
    "subset_sha256": subset_sha256,
    "real_task_authenticated": True,
    "batch": 4,
    "batch_size": 4,
    "concurrency": 4,
    "layer_count": 48,
    "physical_rows_per_request": 32,
    "physical_rows_total": 128,
    "draft_vocab_root": 0,
    "draft_vocab_k": 0,
    "comparison_records": 48,
    "candidate_shadow_only": True,
    "served_result": "reference",
    "reference_always_served": True,
    "probe_inputs": False,
    "synthetic_inputs": False,
    "acceptance_valid": False,
    "timing_eligible": False,
    "floor_acceptance_eligible": False,
    "production_enabled": False,
    "production_eligible": False,
}
for key, expected in source_expected.items():
    if source_pass.get(key) != expected:
        errors.append(f"source PASS {key} mismatch")
source_pass_layer_keys = source_pass.get("layer_keys")
if (
    not isinstance(source_pass_layer_keys, list)
    or len(source_pass_layer_keys) != 48
    or set(source_pass_layer_keys) != set(layer_keys)
):
    errors.append("source PASS layer keys do not bind the byte records")

ledger_verification = verify_fixed32_ingress_ledger(
    ledger_path, expected_role="engine", require_finalized=True
)
ledger_rows = [
    json.loads(line)
    for line in ledger_path.read_text(encoding="ascii").splitlines()
]
expected_set_sha256 = fixed32_canonical_task_set_sha256(tuple(expected_tasks))
if not any(
    row.get("event") == "campaign_begin"
    and row.get("evidence_sha256") == expected_set_sha256
    for row in ledger_rows
):
    errors.append("engine ledger is not bound to canonical exact4")
for task_id in expected_tasks:
    task_key = fixed32_task_key_id(task_id)
    if not any(
        row.get("event") == "request_accepted"
        and row.get("task_key_id") == task_key
        for row in ledger_rows
    ):
        errors.append(f"authenticated marker has no accepted request: {task_id}")

health = json.loads((arm / "health.json").read_text(encoding="utf-8"))
health_tasks = health.get("tasks")
if (
    not isinstance(health_tasks, list)
    or len(health_tasks) != 4
    or {task.get("instance_id") for task in health_tasks} != set(expected_tasks)
    or health.get("swe_orchestrator_rc") != 0
):
    errors.append("diagnostic did not complete canonical exact4")

container_env_raw = container_env_path.read_bytes()
container_env_lines = container_env_raw.decode("ascii").splitlines()
for expected_env in (
    "FR13_FIXED32_MODE=hydra27_fixed32",
    "FR13_DRAFT_VOCAB_ROOT=0",
    "FR13_DRAFT_VOCAB_K=0",
    "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=1",
    "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0",
    "FR13_CONV_WB_BATCHED=1",
    "FR13_FIXED32_CONV_SOURCE_BATCH=1",
):
    if container_env_lines.count(expected_env) != 1:
        errors.append(f"B4 environment mismatch: {expected_env}")

runtime_manifest_raw = runtime_manifest_path.read_bytes()
payload = {
    "schema": live_schema,
    "status": "pass" if not errors else "fail",
    "run_classification": "real_swe_verified_exact4_b4_byte_diagnostic",
    "candidate": "fixed32_sfwd_state_fusion_v1",
    "task_set": "canonical real SWE-Verified exact4 B4",
    "task_count": 4,
    "task_ids": expected_tasks,
    "task_markers": expected_markers,
    "subset_sha256": subset_sha256,
    "real_task_authenticated": True,
    "real_task_arm_sha256": hashlib.sha256(marker_raw).hexdigest(),
    "engine_ledger_chain_head_sha256": ledger_verification["chain_head_sha256"],
    "batch_size": 4,
    "concurrency": 4,
    "physical_rows_per_request": 32,
    "physical_rows_total": 128,
    "layer_count": len(set(layer_keys)),
    "layer_keys": sorted(set(layer_keys)),
    "comparison_records": len(records),
    "compared_byte_surfaces": ["conv_out", "commit_source_stage"],
    "mismatching_records": sum(record.get("zero_diff") is not True for record in records),
    "differing_bytes": sum(
        item.get("differing_bytes", 0)
        for record in records
        for item in record.get("comparisons", [])
        if isinstance(item, dict)
    ),
    "draft_vocab_root": 0,
    "draft_vocab_k": 0,
    "candidate_shadow_only": True,
    "served_result": "reference",
    "reference_always_served": True,
    "probe_inputs": False,
    "synthetic_inputs": False,
    "acceptance_valid": False,
    "timing_eligible": False,
    "floor_acceptance_eligible": False,
    "production_enabled": False,
    "production_eligible": False,
    "kernel_source_sha256": hashlib.sha256(kernel_source.read_bytes()).hexdigest(),
    "patcher_source_sha256": hashlib.sha256(patcher_source.read_bytes()).hexdigest(),
    "runtime_manifest_sha256": hashlib.sha256(runtime_manifest_raw).hexdigest(),
    "runner_sha256": runner_sha256,
    "source_pass_sha256": hashlib.sha256(source_pass_raw).hexdigest(),
    "source_commit": source_commit,
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

LIVE_RESULT="$ARMDIR/sfwd_state_fusion_b4_byte_gate.json"
LIVE_SHA256=$(sha256sum "$LIVE_RESULT" | awk '{print $1}')
B4_QUALIFICATION="$ARMDIR/sfwd_state_fusion_b4.qualification.json"
"$PYTHON_BIN" scripts/fr13_sfwd_state_fusion_b4_pass.py issue-b4 \
  --live-result "$LIVE_RESULT" --expected-live-sha256 "$LIVE_SHA256" \
  --kernel-source "$KERNEL_SOURCE" --patcher-source "$PATCHER_SOURCE" \
  --expected-source-commit "$SOURCE_COMMIT" --out "$B4_QUALIFICATION"

printf 'live_result=%s\nlive_sha256=%s\nb4_qualification=%s\nb4_qualification_sha256=%s\nproduction_default_enabled=0\nnext_prerequisite=bind_authenticated_B1_PASS_with_B4_qualification\n' \
  "$LIVE_RESULT" "$LIVE_SHA256" "$B4_QUALIFICATION" \
  "$(sha256sum "$B4_QUALIFICATION" | awk '{print $1}')"
