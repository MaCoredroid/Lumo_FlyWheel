#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned FA2 shared object}"
: "${CUTLASS_STREAMK_SO:?set CUTLASS_STREAMK_SO to the pinned Stream-K shared object}"
PATCH_SOURCE=scripts/fr13_patch_cutlass_fixed32_wave.py
SOURCE_COMMIT=$(git rev-parse HEAD)

if [[ -e "$RUNROOT" || -L "$RUNROOT" ]]; then
  echo "CUTLASS Stream-K gate requires a fresh RUNROOT: $RUNROOT" >&2
  exit 2
fi

.venv/bin/python scripts/fr13_cutlass_wave_binary.py verify \
  "$CUTLASS_STREAMK_SO" >/dev/null

export FR13_GATE_QROW16=0
export FR13_GATE_TAW_NATIVE=0
export FR13_GATE_DRAFT_HEAD_PAD=0
export FR13_GATE_GDN_BV=0
export FR13_DFWD_UNIFIED_BM8_LIVE_AB=0
export FR13_DFWD_UNIFIED_BM8_PRODUCTION=0
export ENFORCE_EAGER=1
export FR13_DRAFT_VOCAB_ROOT=0
export FR13_DRAFT_VOCAB_K=0
export FR13_NEEDS_ALLOW='FR13_DRAFT_VOCAB_K=0'
export FR13_MANDATORY_WEIGHT_BYTES=42025179008
export FR13_WEIGHT_FLOOR_MS=153.9383846446886
export FR13_FIXED32_CUTLASS_WAVE=streamk_coop128_byte_ab
export FR13_FIXED32_CUTLASS_WAVE_SO="$CUTLASS_STREAMK_SO"
export FR13_FIXED32_CUTLASS_WAVE_BYTE_AB_JSONL=/logs/fr13_fixed32_cutlass_streamk_byte_ab.jsonl

bash scripts/fr13_run_b1_kernel_live_gate.sh

ARM="hydra27_fixed32_${TAG}"
ARMDIR="$RUNROOT/$ARM"
TASK_ID=astropy__astropy-12907
CUTLASS_ARM_ARTIFACT="$ARMDIR/swe_out/verified/per_task/$TASK_ID/fixed32_cutlass_streamk_real_task_arm.json"
.venv/bin/python - \
  "$ARMDIR/logs/fr13_fixed32_cutlass_streamk_byte_ab.jsonl" \
  "$ARMDIR/logs/fr13_fixed32_cutlass_streamk_binary.json" \
  "$ARMDIR/cutlass_streamk_byte_gate.json" \
  "$PATCH_SOURCE" "$SOURCE_COMMIT" "$CUTLASS_ARM_ARTIFACT" \
  "$ARMDIR/container_env.txt" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import fr13_cutlass_wave_binary as binary
import fr13_cutlass_streamk_pass as qualification

jsonl_path, binary_path, output_path, patch_source = map(Path, sys.argv[1:5])
source_commit = sys.argv[5]
arm_path = Path(sys.argv[6])
container_env_path = Path(sys.argv[7])
lines = jsonl_path.read_text(encoding="utf-8").splitlines()
if not lines:
    raise SystemExit("CUTLASS Stream-K byte gate was vacuous")
records = [json.loads(line) for line in lines]
expected_record_schema = "fr13.fixed32.cutlass_streamk_byte_ab.v2"
expected_task_id = "astropy__astropy-12907"
expected_task_marker = f"swe_verified:{expected_task_id}"
expected_shapes = {
    (34816, 5120),
    (5120, 17408),
    (5120, 6144),
    (16384, 5120),
    (8192, 5120),
}
observed_shapes = {(record["n"], record["k"]) for record in records}
invocations = [record["invocation"] for record in records]
binary_record = json.loads(binary_path.read_text(encoding="ascii"))
if not arm_path.is_file() or arm_path.is_symlink():
    raise SystemExit("CUTLASS real-task arm artifact is missing or symlinked")
arm_raw = arm_path.read_bytes()
arm_record = json.loads(arm_raw.decode("ascii"))
if not container_env_path.is_file() or container_env_path.is_symlink():
    raise SystemExit("CUTLASS container environment artifact is missing or symlinked")
container_env_raw = container_env_path.read_bytes()
container_env_lines = container_env_raw.decode("ascii").splitlines()
errors = []
if len(records) > 256:
    errors.append("diagnostic exceeded its 256-call bound")
if any(record.get("schema") != expected_record_schema for record in records):
    errors.append("comparison record schema mismatch")
if any(record.get("task_marker") != expected_task_marker for record in records):
    errors.append("comparison record is not bound to the real SWE task")
if invocations != list(range(len(records))):
    errors.append("invocations are not contiguous from zero")
if not expected_shapes.issubset(observed_shapes):
    errors.append("not all five real projection shapes were exercised")
if any(record.get("m") != 32 for record in records):
    errors.append("a comparison did not use the fixed32 B1 row count")
if any(record.get("bytes") != 2 * record["m"] * record["n"] for record in records):
    errors.append("a comparison reported an invalid BF16 byte count")
if any(
    not isinstance(record.get("mismatch_count"), int)
    or isinstance(record.get("mismatch_count"), bool)
    or record["mismatch_count"] < 0
    or record["mismatch_count"] > record["bytes"]
    for record in records
):
    errors.append("a comparison reported an invalid differing-byte count")
if any(record.get("byte_equal") is not True for record in records):
    errors.append("at least one stock/Stream-K output differed")
if any(
    record.get("byte_equal") is not (record.get("mismatch_count") == 0)
    for record in records
):
    errors.append("byte equality and differing-byte count disagree")
if any(
    (record.get("first_mismatch") is None) is not (record.get("mismatch_count") == 0)
    or (
        record.get("first_mismatch") is not None
        and (
            not isinstance(record.get("first_mismatch"), int)
            or isinstance(record.get("first_mismatch"), bool)
            or record["first_mismatch"] < 0
            or record["first_mismatch"] >= record["bytes"]
        )
    )
    for record in records
):
    errors.append("first mismatch and differing-byte count disagree")
if binary_record.get("schema") != "fr13.fixed32.cutlass_streamk_binary.v2":
    errors.append("installed binary attestation schema mismatch")
if binary_record.get("selector") != "streamk_coop128_byte_ab":
    errors.append("installed binary selector attestation mismatch")
if binary_record.get("production_enabled") is not False:
    errors.append("binary attestation did not remain production-off")
expected_arm = {
    "schema": "fr13-fixed32-cutlass-streamk-real-task-arm-v1",
    "state": "ended",
    "instance_id": expected_task_id,
    "marker": expected_task_marker,
}
for key, expected in expected_arm.items():
    if arm_record.get(key) != expected:
        errors.append(f"CUTLASS real-task arm {key} mismatch")
for expected_env in (
    "FR13_DRAFT_VOCAB_ROOT=0",
    "FR13_DRAFT_VOCAB_K=0",
):
    if container_env_lines.count(expected_env) != 1:
        errors.append(f"full-vocabulary environment mismatch: {expected_env}")
for prefix in ("FR13_DRAFT_VOCAB_ROOT=", "FR13_DRAFT_VOCAB_K="):
    if sum(line.startswith(prefix) for line in container_env_lines) != 1:
        errors.append(f"full-vocabulary environment is ambiguous: {prefix}")
destination = binary_record.get("destination") or {}
source = binary_record.get("source") or {}
if binary_record.get("installed_mode") != "0555":
    errors.append("installed binary mode attestation mismatch")
for label, identity, expected_path in (
    ("source", source, str(binary.CONTAINER_SOURCE)),
    ("destination", destination, str(binary.CONTAINER_DESTINATION)),
):
    if identity.get("path") != expected_path:
        errors.append(f"installed binary {label} path mismatch")
    if identity.get("regular") is not True or identity.get("symlink") is not False:
        errors.append(f"installed binary {label} file identity mismatch")
    if identity.get("sha256") != binary.CANDIDATE_SHA256:
        errors.append(f"installed binary {label} SHA-256 mismatch")
    if identity.get("bytes") != binary.CANDIDATE_SIZE:
        errors.append(f"installed binary {label} size mismatch")
patch_source_sha256 = hashlib.sha256(patch_source.read_bytes()).hexdigest()
if patch_source_sha256 != qualification.PATCH_SOURCE_SHA256:
    errors.append("Stream-K patch source SHA-256 mismatch")

payload = {
    "schema": "fr13.fixed32.cutlass_streamk_live_gate.v3",
    "status": "pass" if not errors else "fail",
    "run_classification": "one_real_swe_verified_b1_byte_diagnostic",
    "acceptance_valid": False,
    "task_set": "one real SWE-Verified B1 diagnostic task",
    "task_count": 1,
    "task_ids": [expected_task_id],
    "task_marker": expected_task_marker,
    "real_task_arm_sha256": hashlib.sha256(arm_raw).hexdigest(),
    "container_env_sha256": hashlib.sha256(container_env_raw).hexdigest(),
    "draft_vocab_root": 0,
    "draft_vocab_k": 0,
    "mandatory_weight_bytes": 42025179008,
    "mandatory_weight_floor_ms": 153.9383846446886,
    "one_sided_u95_cap_ms": 177.0291423413919,
    "comparator_timing_eligible": False,
    "batch_size": 1,
    "concurrency": 1,
    "fixed_rows": 32,
    "candidate": "streamk_coop128",
    "diagnostic_selector": "streamk_coop128_byte_ab",
    "served_result": "stock",
    "production_enabled": False,
    "comparisons": len(records),
    "observed_m_values": sorted({record["m"] for record in records}),
    "observed_projection_nk": sorted([list(shape) for shape in observed_shapes]),
    "mismatching_comparisons": sum(
        record.get("byte_equal") is not True for record in records
    ),
    "differing_bytes": sum(record.get("mismatch_count", 0) for record in records),
    "candidate_sha256": binary.CANDIDATE_SHA256,
    "candidate_bytes": binary.CANDIDATE_SIZE,
    "patch_source_sha256": patch_source_sha256,
    "vllm_base_commit": qualification.VLLM_BASE_COMMIT,
    "patched_dispatch_sha256": qualification.PATCHED_DISPATCH_SHA256,
    "source_commit": source_commit,
    "binary_attestation_sha256": hashlib.sha256(binary_path.read_bytes()).hexdigest(),
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
