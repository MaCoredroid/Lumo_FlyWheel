#!/usr/bin/env bash
# Real SWE-Verified exact4 B4 byte gate for the persistent CUTLASS M128 kernel.
# The diagnostic always serves the stock result and contributes no timing samples.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned FA2 shared object}"
: "${CUTLASS_B4_SO:?set CUTLASS_B4_SO to the pinned persistent-M128 shared object}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
PATCH_SOURCE=scripts/fr13_patch_cutlass_fixed32_wave.py
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
DIAGNOSTIC_SELECTOR=persistent_b4_m128_byte_ab
RECORD_SCHEMA=fr13.fixed32.cutlass_persistent_b4_m128_byte_ab.v1
LIVE_SCHEMA=fr13.fixed32.cutlass_persistent_b4_m128_live_gate.v1
CONTAINER_JSONL=/logs/fr13_fixed32_cutlass_persistent_b4_m128_byte_ab.jsonl
B4_KV_CACHE_MEMORY_BYTES=42949672960
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
ARM="tail6_fixed32_cutlass_b4_m128_gate_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for input in "$FORKED_FA2_SO" "$CUTLASS_B4_SO"; do
  [[ "$input" == /* && -f "$input" && ! -L "$input" ]] \
    || { echo "gate input must be an absolute regular non-symlink file: $input" >&2; exit 2; }
done
[[ "$(stat -c '%s' "$FORKED_FA2_SO")" == "$STOCK_FA2_BYTES" \
   && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "FORKED_FA2_SO is not the exact-safe stock reference" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical exact4 subset SHA-256 drift" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the gate" >&2; exit 2; }

"$PYTHON_BIN" scripts/fr13_cutlass_wave_binary.py verify \
  "$CUTLASS_B4_SO" --selector "$DIAGNOSTIC_SELECTOR" >/dev/null

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
printf 'classification=real_swe_verified_exact4_b4_byte_diagnostic\ntiming_eligible=0\nfloor_acceptance_eligible=0\nreference_always_served=1\nbatch_size=4\nconcurrency=4\nfixed_rows=128\neager_builder_capacity=128\ndraft_vocab_root=0\ndraft_vocab_k=0\nfr13_needs_allow=FR13_DRAFT_VOCAB_K=0\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=177.0291423413919\nlauncher_pid=%s\nrunroot=%s\narm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nstock_fa2_sha256=%s\nstock_fa2_bytes=%s\ncandidate_selector=persistent_b4_m128\ndiagnostic_selector=%s\nenforce_eager=1\nkv_cache_memory_bytes=%s\nstarted=%s\n' \
  "$FR13_MANDATORY_WEIGHT_BYTES" "$FR13_WEIGHT_FLOOR_MS" "$$" \
  "$RUNROOT_ABS" "$ARM" "$SOURCE_COMMIT" "$RUNNER_SHA256" \
  "$SUBSET_SHA256" "$STOCK_FA2_SHA256" "$STOCK_FA2_BYTES" \
  "$DIAGNOSTIC_SELECTOR" "$B4_KV_CACHE_MEMORY_BYTES" \
  "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"

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
    FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
    FR13_SCAN_ALIGN=0 FR13_NPAD_INVARIANT=0 \
    FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_CUTLASS_WAVE="$DIAGNOSTIC_SELECTOR" \
    FR13_FIXED32_CUTLASS_WAVE_SO="$CUTLASS_B4_SO" \
    FR13_FIXED32_CUTLASS_WAVE_BYTE_AB_JSONL="$CONTAINER_JSONL" \
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
      "$ARM" tail6_fixed32 "$SUBSET" \
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
  || { echo "B4 CUTLASS gate runner changed during execution" >&2; exit 14; }
(( serve_rc == 0 )) || exit "$serve_rc"

"$PYTHON_BIN" - \
  "$ARMDIR" "$ARMDIR$CONTAINER_JSONL" \
  "$ARMDIR/logs/fr13_fixed32_cutlass_streamk_binary.json" \
  "$ARMDIR/cutlass_b4_m128_byte_gate.json" "$PATCH_SOURCE" \
  "$SOURCE_COMMIT" "$SUBSET_SHA256" "$DIAGNOSTIC_SELECTOR" \
  "$RECORD_SCHEMA" "$LIVE_SCHEMA" "$STOCK_FA2_SHA256" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import fr13_cutlass_b4_pass as qualification
import fr13_cutlass_wave_binary as binary
from lumo_flywheel_serving.inference_proxy import (
    fixed32_canonical_task_set_sha256,
    fixed32_task_key_id,
    verify_fixed32_ingress_ledger,
)

arm = Path(sys.argv[1]).resolve()
jsonl_path = Path(sys.argv[2])
binary_path = Path(sys.argv[3])
output_path = Path(sys.argv[4])
patch_source = Path(sys.argv[5])
source_commit = sys.argv[6]
subset_sha256 = sys.argv[7]
diagnostic_selector = sys.argv[8]
record_schema = sys.argv[9]
live_schema = sys.argv[10]
stock_fa2_sha256 = sys.argv[11]
logs = arm / "logs"
marker_path = logs / "fr13_fixed32_cutlass_b4_byte_ab.real_event.arm"
ledger_path = logs / "fr13_fixed32_engine_ingress.jsonl"
container_env_path = arm / "container_env.txt"
expected_tasks = list(qualification.EXPECTED_TASK_IDS)
expected_task_set = set(expected_tasks)
expected_shapes = set(qualification.EXPECTED_PROJECTION_NK)
expected_sha256, expected_size, expected_family = binary.candidate_identity(
    diagnostic_selector
)

marker_info = os.lstat(marker_path)
if (
    not stat.S_ISREG(marker_info.st_mode)
    or stat.S_ISLNK(marker_info.st_mode)
    or marker_info.st_nlink != 1
    or stat.S_IMODE(marker_info.st_mode) != 0o400
):
    raise SystemExit("CUTLASS B4 real-event marker identity or mode is invalid")
marker_raw = marker_path.read_bytes()
try:
    marker_text = marker_raw.decode("ascii")
except UnicodeDecodeError as error:
    raise SystemExit("CUTLASS B4 real-event marker is not ASCII") from error
if not marker_text.endswith("\n") or marker_text.count("\n") != 1:
    raise SystemExit("CUTLASS B4 real-event marker framing is invalid")
task_marker = marker_text.removesuffix("\n")
prefix = "swe_verified:"
if not task_marker.startswith(prefix) or task_marker[len(prefix):] not in expected_task_set:
    raise SystemExit("CUTLASS B4 marker is not bound to canonical exact4")

lines = jsonl_path.read_text(encoding="ascii").splitlines()
if not lines:
    raise SystemExit("CUTLASS B4 byte gate was vacuous")
records = [json.loads(line) for line in lines]
observed_shapes = {(record.get("n"), record.get("k")) for record in records}
invocations = [record.get("invocation") for record in records]
errors = []
if len(records) > 256:
    errors.append("diagnostic exceeded its 256-call bound")
if any(record.get("schema") != record_schema for record in records):
    errors.append("comparison record schema mismatch")
if any(record.get("task_marker") != task_marker for record in records):
    errors.append("comparison record is not bound to the authenticated exact4 marker")
if invocations != list(range(len(records))):
    errors.append("invocations are not contiguous from zero")
if not expected_shapes.issubset(observed_shapes):
    errors.append("not all five real projection shapes were exercised")
if any(record.get("m") != 128 for record in records):
    errors.append("a comparison did not use the exact B4 physical row count")
if any(record.get("bytes") != 2 * record["m"] * record["n"] for record in records):
    errors.append("a comparison reported an invalid BF16 byte count")
if any(record.get("byte_equal") is not True for record in records):
    errors.append("at least one stock/persistent-M128 output differed")
if any(
    not isinstance(record.get("mismatch_count"), int)
    or isinstance(record.get("mismatch_count"), bool)
    or record["mismatch_count"] != 0
    or record.get("first_mismatch") is not None
    for record in records
):
    errors.append("byte mismatch accounting is invalid")

binary_raw = binary_path.read_bytes()
binary_record = json.loads(binary_raw.decode("ascii"))
if binary_record.get("schema") != qualification.ATTESTATION_SCHEMA:
    errors.append("installed binary attestation schema mismatch")
if binary_record.get("selector") != diagnostic_selector:
    errors.append("installed binary selector attestation mismatch")
if binary_record.get("candidate_family") != expected_family:
    errors.append("installed binary candidate-family attestation mismatch")
if binary_record.get("production_enabled") is not False:
    errors.append("diagnostic binary unexpectedly enabled production")
if binary_record.get("installed_mode") != "0555":
    errors.append("installed binary mode attestation mismatch")
for label, expected_path in (
    ("source", str(binary.CONTAINER_SOURCE)),
    ("destination", str(binary.CONTAINER_DESTINATION)),
):
    identity = binary_record.get(label) or {}
    if (
        identity.get("path") != expected_path
        or identity.get("regular") is not True
        or identity.get("symlink") is not False
        or identity.get("sha256") != expected_sha256
        or identity.get("bytes") != expected_size
        or identity.get("candidate_family") != expected_family
    ):
        errors.append(f"installed binary {label} identity mismatch")

patch_sha256 = hashlib.sha256(patch_source.read_bytes()).hexdigest()
if patch_sha256 != qualification.PATCH_SOURCE_SHA256:
    errors.append("CUTLASS patch source SHA-256 mismatch")

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
marker_task_key = fixed32_task_key_id(task_marker[len(prefix):])
if not any(
    row.get("event") == "request_accepted"
    and row.get("task_key_id") == marker_task_key
    for row in ledger_rows
):
    errors.append("real-event marker has no matching accepted request")

health = json.loads((arm / "health.json").read_text(encoding="utf-8"))
health_tasks = health.get("tasks")
if (
    not isinstance(health_tasks, list)
    or len(health_tasks) != 4
    or {task.get("instance_id") for task in health_tasks} != expected_task_set
    or health.get("swe_orchestrator_rc") != 0
):
    errors.append("diagnostic did not complete canonical exact4")

container_env_raw = container_env_path.read_bytes()
container_env_lines = container_env_raw.decode("ascii").splitlines()
for expected_env in ("FR13_DRAFT_VOCAB_ROOT=0", "FR13_DRAFT_VOCAB_K=0"):
    if container_env_lines.count(expected_env) != 1:
        errors.append(f"B4 draft-vocabulary environment mismatch: {expected_env}")

payload = {
    "schema": live_schema,
    "status": "pass" if not errors else "fail",
    "run_classification": "real_swe_verified_exact4_b4_byte_diagnostic",
    "acceptance_valid": False,
    "task_set": "canonical real SWE-Verified exact4 B4",
    "task_count": 4,
    "task_ids": expected_tasks,
    "task_marker": task_marker,
    "subset_sha256": subset_sha256,
    "real_task_arm_sha256": hashlib.sha256(marker_raw).hexdigest(),
    "container_env_sha256": hashlib.sha256(container_env_raw).hexdigest(),
    "engine_ledger_chain_head_sha256": ledger_verification["chain_head_sha256"],
    "draft_vocab_root": qualification.EXPECTED_DRAFT_VOCAB_ROOT,
    "draft_vocab_k": qualification.EXPECTED_DRAFT_VOCAB_K,
    "mandatory_weight_bytes": qualification.EXPECTED_MANDATORY_WEIGHT_BYTES,
    "mandatory_weight_floor_ms": qualification.EXPECTED_MANDATORY_WEIGHT_FLOOR_MS,
    "one_sided_u95_cap_ms": qualification.EXPECTED_SLO_CAP_MS,
    "comparator_timing_eligible": False,
    "batch_size": 4,
    "concurrency": 4,
    "fixed_rows": 128,
    "eager_builder_capacity": 128,
    "candidate": "persistent_b4_m128",
    "diagnostic_selector": diagnostic_selector,
    "served_result": "stock",
    "production_enabled": False,
    "comparisons": len(records),
    "observed_m_values": sorted({record["m"] for record in records}),
    "observed_projection_nk": [list(shape) for shape in sorted(observed_shapes)],
    "mismatching_comparisons": sum(record.get("byte_equal") is not True for record in records),
    "differing_bytes": sum(record.get("mismatch_count", 0) for record in records),
    "candidate_family": expected_family,
    "candidate_sha256": expected_sha256,
    "candidate_bytes": expected_size,
    "stock_fa2_sha256": stock_fa2_sha256,
    "patch_source_sha256": patch_sha256,
    "vllm_base_commit": qualification.VLLM_BASE_COMMIT,
    "patched_dispatch_sha256": qualification.PATCHED_DISPATCH_SHA256,
    "source_commit": source_commit,
    "binary_attestation_sha256": hashlib.sha256(binary_raw).hexdigest(),
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

LIVE_RESULT="$ARMDIR/cutlass_b4_m128_byte_gate.json"
LIVE_SHA256=$(sha256sum "$LIVE_RESULT" | awk '{print $1}')
"$PYTHON_BIN" scripts/fr13_cutlass_b4_pass.py issue \
  --live-result "$LIVE_RESULT" --expected-live-sha256 "$LIVE_SHA256" \
  --candidate-so "$CUTLASS_B4_SO" --patch-source "$PATCH_SOURCE" \
  --expected-source-commit "$SOURCE_COMMIT" \
  --candidate-selector persistent_b4_m128 \
  --out "$ARMDIR/cutlass_b4_m128.production_pass.json"

printf 'live_result=%s\nlive_sha256=%s\nproduction_pass=%s\nproduction_pass_sha256=%s\n' \
  "$LIVE_RESULT" "$LIVE_SHA256" \
  "$ARMDIR/cutlass_b4_m128.production_pass.json" \
  "$(sha256sum "$ARMDIR/cutlass_b4_m128.production_pass.json" | awk '{print $1}')"
