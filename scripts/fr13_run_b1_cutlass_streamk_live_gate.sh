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
GATE_CANDIDATE=${FR13_STREAMK_GATE_CANDIDATE:-streamk_coop128}
case "$GATE_CANDIDATE" in
  streamk_coop128)
    DIAGNOSTIC_SELECTOR=streamk_coop128_byte_ab
    RECORD_SCHEMA=fr13.fixed32.cutlass_streamk_byte_ab.v2
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_streamk_live_gate.v3
    K64_ROOT_LIVE_SCHEMA=
    CONTAINER_JSONL=/logs/fr13_fixed32_cutlass_streamk_byte_ab.jsonl
    K64_ROOT_RESULT_NAME=
    ;;
  streamk_force_wide256)
    DIAGNOSTIC_SELECTOR=streamk_force_wide256_byte_ab
    RECORD_SCHEMA=fr13.fixed32.cutlass_streamk_wide256_byte_ab.v1
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_streamk_wide256_live_gate.v1
    K64_ROOT_LIVE_SCHEMA=fr13.fixed32.cutlass_streamk_wide256_k64_root_live_gate.v1
    CONTAINER_JSONL=/logs/fr13_fixed32_cutlass_streamk_wide256_byte_ab.jsonl
    K64_ROOT_RESULT_NAME=cutlass_streamk_k64_root_byte_gate.json
    ;;
  static_persistent_stocktile)
    DIAGNOSTIC_SELECTOR=static_persistent_stocktile_byte_ab
    RECORD_SCHEMA=fr13.fixed32.cutlass_static_persistent_byte_ab.v1
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_static_persistent_live_gate.v1
    K64_ROOT_LIVE_SCHEMA=fr13.fixed32.cutlass_static_persistent_k64_root_live_gate.v1
    CONTAINER_JSONL=/logs/fr13_fixed32_cutlass_static_persistent_byte_ab.jsonl
    K64_ROOT_RESULT_NAME=cutlass_static_persistent_k64_root_byte_gate.json
    ;;
  divisor_static_stocktile)
    DIAGNOSTIC_SELECTOR=divisor_static_stocktile_byte_ab
    RECORD_SCHEMA=fr13.fixed32.cutlass_divisor_static_byte_ab.v1
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_divisor_static_live_gate.v1
    K64_ROOT_LIVE_SCHEMA=fr13.fixed32.cutlass_divisor_static_k64_root_live_gate.v1
    CONTAINER_JSONL=/logs/fr13_fixed32_cutlass_divisor_static_byte_ab.jsonl
    K64_ROOT_RESULT_NAME=cutlass_divisor_static_k64_root_byte_gate.json
    ;;
  identity_stage2_static)
    DIAGNOSTIC_SELECTOR=identity_stage2_static_byte_ab
    RECORD_SCHEMA=fr13.fixed32.cutlass_identity_stage2_static_byte_ab.v1
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_stage2_live_gate.v1
    K64_ROOT_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_stage2_k64_root_live_gate.v1
    CONTAINER_JSONL=/logs/fr13_fixed32_cutlass_identity_stage2_static_byte_ab.jsonl
    K64_ROOT_RESULT_NAME=cutlass_identity_stage2_k64_root_byte_gate.json
    ;;
  identity_stage2_pingpong_b1)
    DIAGNOSTIC_SELECTOR=identity_stage2_pingpong_b1_byte_ab
    RECORD_SCHEMA=fr13.fixed32.cutlass_identity_stage2_pingpong_b1_byte_ab.v1
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_stage2_pingpong_b1_live_gate.v1
    K64_ROOT_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_stage2_pingpong_b1_k64_root_live_gate.v1
    CONTAINER_JSONL=/logs/fr13_fixed32_cutlass_identity_stage2_pingpong_b1_byte_ab.jsonl
    K64_ROOT_RESULT_NAME=cutlass_identity_stage2_pingpong_b1_k64_root_byte_gate.json
    ;;
  identity_onen_b1)
    DIAGNOSTIC_SELECTOR=identity_onen_b1_byte_ab
    RECORD_SCHEMA=fr13.fixed32.cutlass_identity_onen_b1_byte_ab.v1
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_onen_b1_live_gate.v1
    K64_ROOT_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_onen_b1_k64_root_live_gate.v1
    CONTAINER_JSONL=/logs/fr13_fixed32_cutlass_identity_onen_b1_byte_ab.jsonl
    K64_ROOT_RESULT_NAME=cutlass_identity_onen_b1_k64_root_byte_gate.json
    ;;
  identity_onen_n5120_single_b1)
    DIAGNOSTIC_SELECTOR=identity_onen_n5120_single_b1_byte_ab
    RECORD_SCHEMA=fr13.fixed32.cutlass_identity_onen_n5120_single_b1_byte_ab.v1
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_onen_n5120_single_b1_live_gate.v1
    K64_ROOT_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_onen_n5120_single_b1_k64_root_live_gate.v1
    CONTAINER_JSONL=/logs/fr13_fixed32_cutlass_identity_onen_n5120_single_b1_byte_ab.jsonl
    K64_ROOT_RESULT_NAME=cutlass_identity_onen_n5120_single_b1_k64_root_byte_gate.json
    ;;
  *)
    echo "unsupported Stream-K gate candidate: $GATE_CANDIDATE" >&2
    exit 2
    ;;
esac
QUALIFICATION_PROFILE_EXPLICIT=0
if [[ -v FR13_STREAMK_QUALIFICATION_PROFILE ]]; then
  QUALIFICATION_PROFILE_EXPLICIT=1
fi
QUALIFICATION_PROFILE=${FR13_STREAMK_QUALIFICATION_PROFILE:-full_vocab}
if [[ ( "$GATE_CANDIDATE" == "identity_onen_b1" \
        || "$GATE_CANDIDATE" == "identity_onen_n5120_single_b1" ) \
      && ( "$QUALIFICATION_PROFILE_EXPLICIT" != "1" \
           || "$QUALIFICATION_PROFILE" != "k64_root" ) ]]; then
  if [[ "$GATE_CANDIDATE" == "identity_onen_b1" ]]; then
    echo "identity_onen_b1 diagnostic requires explicit k64_root qualification" >&2
  else
    echo "$GATE_CANDIDATE diagnostic requires explicit k64_root qualification" >&2
  fi
  exit 2
fi
DRAFT_VOCAB_BLOCKS_HOST=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
case "$QUALIFICATION_PROFILE" in
  full_vocab)
    LIVE_SCHEMA=$FULL_VOCAB_LIVE_SCHEMA
    RUN_CLASSIFICATION=one_real_swe_verified_b1_byte_diagnostic
    DRAFT_VOCAB_ROOT=0
    DRAFT_VOCAB_K=0
    NEEDS_ALLOW=FR13_DRAFT_VOCAB_K=0
    MANDATORY_WEIGHT_BYTES=42025179008
    MANDATORY_WEIGHT_FLOOR_MS=153.9383846446886
    ONE_SIDED_U95_CAP_MS=177.0291423413919
    MAX_COMPARISONS=256
    ARM_PROFILE_SUFFIX=
    LIVE_RESULT_NAME=cutlass_streamk_byte_gate.json
    ;;
  k64_root)
    [[ "$GATE_CANDIDATE" == "streamk_force_wide256" \
       || "$GATE_CANDIDATE" == "static_persistent_stocktile" \
       || "$GATE_CANDIDATE" == "divisor_static_stocktile" \
       || "$GATE_CANDIDATE" == "identity_stage2_static" \
       || "$GATE_CANDIDATE" == "identity_stage2_pingpong_b1" \
       || "$GATE_CANDIDATE" == "identity_onen_b1" \
       || "$GATE_CANDIDATE" == "identity_onen_n5120_single_b1" ]] || {
      echo "B1 k64_root qualification requires a pinned B1 projection candidate" >&2
      exit 2
    }
    LIVE_SCHEMA=$K64_ROOT_LIVE_SCHEMA
    RUN_CLASSIFICATION=one_real_swe_verified_b1_k64_root_byte_diagnostic
    DRAFT_VOCAB_ROOT=1
    DRAFT_VOCAB_K=65536
    NEEDS_ALLOW=
    MANDATORY_WEIGHT_BYTES=32666638208
    MANDATORY_WEIGHT_FLOOR_MS=119.658015414
    ONE_SIDED_U95_CAP_MS=137.6067177261
    MAX_COMPARISONS=320
    ARM_PROFILE_SUFFIX=_k64_root
    LIVE_RESULT_NAME=$K64_ROOT_RESULT_NAME
    ;;
  *)
    echo "FR13_STREAMK_QUALIFICATION_PROFILE must be full_vocab or k64_root" >&2
    exit 2
    ;;
esac

if [[ "$GATE_CANDIDATE" == "identity_onen_b1" \
      || "$GATE_CANDIDATE" == "identity_onen_n5120_single_b1" ]]; then
  .venv/bin/python scripts/fr13_cutlass_streamk_pass.py source-binding \
    --source-commit "$SOURCE_COMMIT" \
    --patch-source "$PATCH_SOURCE" \
    --candidate-selector "$GATE_CANDIDATE" \
    >/dev/null
fi

if [[ -e "$RUNROOT" || -L "$RUNROOT" ]]; then
  echo "CUTLASS Stream-K gate requires a fresh RUNROOT: $RUNROOT" >&2
  exit 2
fi

.venv/bin/python scripts/fr13_cutlass_wave_binary.py verify \
  "$CUTLASS_STREAMK_SO" \
  --selector "$DIAGNOSTIC_SELECTOR" \
  --qualification-profile "$QUALIFICATION_PROFILE" \
  >/dev/null
if [[ "$QUALIFICATION_PROFILE" == "k64_root" ]]; then
  [[ -f "$DRAFT_VOCAB_BLOCKS_HOST" \
     && ! -L "$DRAFT_VOCAB_BLOCKS_HOST" \
     && "$(sha256sum "$DRAFT_VOCAB_BLOCKS_HOST" | awk '{print $1}')" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
    || { echo "pinned root-64K draft-vocabulary block map drifted" >&2; exit 2; }
fi

export FR13_GATE_QROW16=0
export FR13_GATE_TAW_NATIVE=0
export FR13_GATE_DRAFT_HEAD_PAD=0
export FR13_GATE_DRAFT_HEAD_M32=0
export FR13_GATE_BM8=0
export FR13_GATE_GDN_BV=0
export FR13_DFWD_UNIFIED_BM8_LIVE_AB=0
export FR13_DFWD_UNIFIED_BM8_PRODUCTION=0
export ENFORCE_EAGER=1
export FR13_B1_WORKLOAD_PROFILE="$QUALIFICATION_PROFILE"
export FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE="$QUALIFICATION_PROFILE"
export FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT"
export FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K"
export FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER"
export FR13_NEEDS_ALLOW="$NEEDS_ALLOW"
export FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES"
export FR13_WEIGHT_FLOOR_MS="$MANDATORY_WEIGHT_FLOOR_MS"
export FR13_FIXED32_CUTLASS_WAVE="$DIAGNOSTIC_SELECTOR"
export FR13_FIXED32_CUTLASS_WAVE_SO="$CUTLASS_STREAMK_SO"
export FR13_FIXED32_CUTLASS_WAVE_BYTE_AB_JSONL="$CONTAINER_JSONL"

bash scripts/fr13_run_b1_kernel_live_gate.sh

ARM="hydra27_fixed32${ARM_PROFILE_SUFFIX}_${TAG}"
ARMDIR="$RUNROOT/$ARM"
TASK_ID=astropy__astropy-12907
CUTLASS_ARM_ARTIFACT="$ARMDIR/swe_out/verified/per_task/$TASK_ID/fixed32_cutlass_streamk_real_task_arm.json"
.venv/bin/python - \
  "$ARMDIR$CONTAINER_JSONL" \
  "$ARMDIR/logs/fr13_fixed32_cutlass_streamk_binary.json" \
  "$ARMDIR/$LIVE_RESULT_NAME" \
  "$PATCH_SOURCE" "$SOURCE_COMMIT" "$CUTLASS_ARM_ARTIFACT" \
  "$ARMDIR/container_env.txt" "$GATE_CANDIDATE" \
  "$DIAGNOSTIC_SELECTOR" "$RECORD_SCHEMA" "$LIVE_SCHEMA" \
  "$QUALIFICATION_PROFILE" "$RUN_CLASSIFICATION" \
  "$DRAFT_VOCAB_ROOT" "$DRAFT_VOCAB_K" \
  "$DRAFT_VOCAB_BLOCKS_CONTAINER" "$DRAFT_VOCAB_BLOCKS_SHA256" \
  "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$MAX_COMPARISONS" <<'PY'
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
expected_candidate = sys.argv[8]
expected_diagnostic_selector = sys.argv[9]
expected_record_schema = sys.argv[10]
expected_live_schema = sys.argv[11]
expected_profile = sys.argv[12]
expected_run_classification = sys.argv[13]
expected_draft_vocab_root = int(sys.argv[14])
expected_draft_vocab_k = int(sys.argv[15])
expected_draft_vocab_blocks = sys.argv[16]
expected_draft_vocab_blocks_sha256 = sys.argv[17]
expected_mandatory_weight_bytes = int(sys.argv[18])
expected_mandatory_weight_floor_ms = float(sys.argv[19])
expected_one_sided_u95_cap_ms = float(sys.argv[20])
expected_max_comparisons = int(sys.argv[21])
expected_candidate_sha256, expected_candidate_size, expected_candidate_family = (
    binary.candidate_identity(expected_diagnostic_selector)
)
lines = jsonl_path.read_text(encoding="utf-8").splitlines()
if not lines:
    raise SystemExit("CUTLASS Stream-K byte gate was vacuous")
records = [json.loads(line) for line in lines]
expected_task_id = "astropy__astropy-12907"
expected_task_marker = f"swe_verified:{expected_task_id}"
expected_shapes = {
    (34816, 5120),
    (5120, 17408),
    (5120, 6144),
    (16384, 5120),
    (14336, 5120),
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
if len(records) > expected_max_comparisons:
    errors.append(f"diagnostic exceeded its {expected_max_comparisons}-call bound")
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
    errors.append("at least one stock/candidate output differed")
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
if binary_record.get("selector") != expected_diagnostic_selector:
    errors.append("installed binary selector attestation mismatch")
if binary_record.get("candidate_family") != expected_candidate_family:
    errors.append("installed binary candidate-family attestation mismatch")
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
expected_environment = (
    f"FR13_DRAFT_VOCAB_ROOT={expected_draft_vocab_root}",
    f"FR13_DRAFT_VOCAB_K={expected_draft_vocab_k}",
    f"FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE={expected_profile}",
)
if expected_profile == "k64_root":
    expected_environment += (
        f"FR13_DRAFT_VOCAB_BLOCKS={expected_draft_vocab_blocks}",
    )
for expected_env in expected_environment:
    if container_env_lines.count(expected_env) != 1:
        errors.append(f"qualification environment mismatch: {expected_env}")
environment_prefixes = (
    "FR13_DRAFT_VOCAB_ROOT=",
    "FR13_DRAFT_VOCAB_K=",
    "FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE=",
)
if expected_profile == "k64_root":
    environment_prefixes += ("FR13_DRAFT_VOCAB_BLOCKS=",)
for prefix in environment_prefixes:
    if sum(line.startswith(prefix) for line in container_env_lines) != 1:
        errors.append(f"qualification environment is ambiguous: {prefix}")
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
    if identity.get("sha256") != expected_candidate_sha256:
        errors.append(f"installed binary {label} SHA-256 mismatch")
    if identity.get("bytes") != expected_candidate_size:
        errors.append(f"installed binary {label} size mismatch")
    if identity.get("candidate_family") != expected_candidate_family:
        errors.append(f"installed binary {label} candidate-family mismatch")
source_contract = qualification._source_contract(expected_candidate)
patch_source_sha256 = hashlib.sha256(patch_source.read_bytes()).hexdigest()
if patch_source_sha256 != source_contract["patch_source_sha256"]:
    errors.append("Stream-K patch source SHA-256 mismatch")
source_identity = None
if qualification.CANDIDATE_CONTRACTS[expected_candidate].get("source_binding") == "required":
    source_identity = qualification.validate_source_commit_binding(
        source_commit, patch_source, expected_candidate
    )

payload = {
    "schema": expected_live_schema,
    "status": "pass" if not errors else "fail",
    "run_classification": expected_run_classification,
    "acceptance_valid": False,
    "task_set": "one real SWE-Verified B1 diagnostic task",
    "task_count": 1,
    "task_ids": [expected_task_id],
    "task_marker": expected_task_marker,
    "real_task_arm_sha256": hashlib.sha256(arm_raw).hexdigest(),
    "container_env_sha256": hashlib.sha256(container_env_raw).hexdigest(),
    "draft_vocab_root": expected_draft_vocab_root,
    "draft_vocab_k": expected_draft_vocab_k,
    "mandatory_weight_bytes": expected_mandatory_weight_bytes,
    "mandatory_weight_floor_ms": expected_mandatory_weight_floor_ms,
    "one_sided_u95_cap_ms": expected_one_sided_u95_cap_ms,
    "comparator_timing_eligible": False,
    "batch_size": 1,
    "concurrency": 1,
    "fixed_rows": 32,
    "candidate": expected_candidate,
    "diagnostic_selector": expected_diagnostic_selector,
    "served_result": "stock",
    "production_enabled": False,
    "comparisons": len(records),
    "observed_m_values": sorted({record["m"] for record in records}),
    "observed_projection_nk": sorted([list(shape) for shape in observed_shapes]),
    "mismatching_comparisons": sum(
        record.get("byte_equal") is not True for record in records
    ),
    "differing_bytes": sum(record.get("mismatch_count", 0) for record in records),
    "candidate_family": expected_candidate_family,
    "candidate_sha256": expected_candidate_sha256,
    "candidate_bytes": expected_candidate_size,
    "patch_source_sha256": patch_source_sha256,
    "vllm_base_commit": qualification.VLLM_BASE_COMMIT,
    "patched_dispatch_sha256": source_contract["patched_dispatch_sha256"],
    "source_commit": source_commit,
    "binary_attestation_sha256": hashlib.sha256(binary_path.read_bytes()).hexdigest(),
    "errors": errors,
}
if source_identity is not None:
    payload["source_identity"] = source_identity
if expected_profile == "k64_root":
    payload.update(
        {
            "qualification_profile": expected_profile,
            "draft_vocab_blocks": expected_draft_vocab_blocks,
            "draft_vocab_blocks_sha256": expected_draft_vocab_blocks_sha256,
            "comparison_call_limit": expected_max_comparisons,
        }
    )
output_path.write_text(
    json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
if errors:
    raise SystemExit(4)
PY
