#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned FA2 shared object}"
: "${CUTLASS_STREAMK_SO:?set CUTLASS_STREAMK_SO to the pinned Stream-K shared object}"

.venv/bin/python scripts/fr13_cutlass_wave_binary.py verify \
  "$CUTLASS_STREAMK_SO" >/dev/null

export FR13_GATE_QROW16=0
export FR13_GATE_TAW_NATIVE=0
export FR13_GATE_DRAFT_HEAD_PAD=0
export FR13_GATE_GDN_BV=0
export ENFORCE_EAGER=1
export FR13_FIXED32_CUTLASS_WAVE=streamk_coop128_byte_ab
export FR13_FIXED32_CUTLASS_WAVE_SO="$CUTLASS_STREAMK_SO"
export FR13_FIXED32_CUTLASS_WAVE_BYTE_AB_JSONL=/logs/fr13_fixed32_cutlass_streamk_byte_ab.jsonl

bash scripts/fr13_run_b1_kernel_live_gate.sh

ARM="hydra27_fixed32_${TAG}"
ARMDIR="$RUNROOT/$ARM"
.venv/bin/python - \
  "$ARMDIR/logs/fr13_fixed32_cutlass_streamk_byte_ab.jsonl" \
  "$ARMDIR/logs/fr13_fixed32_cutlass_streamk_binary.json" \
  "$ARMDIR/cutlass_streamk_byte_gate.json" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import fr13_cutlass_wave_binary as binary

jsonl_path, binary_path, output_path = map(Path, sys.argv[1:])
lines = jsonl_path.read_text(encoding="utf-8").splitlines()
if not lines:
    raise SystemExit("CUTLASS Stream-K byte gate was vacuous")
records = [json.loads(line) for line in lines]
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
errors = []
if len(records) > 256:
    errors.append("diagnostic exceeded its 256-call bound")
if invocations != list(range(len(records))):
    errors.append("invocations are not contiguous from zero")
if not expected_shapes.issubset(observed_shapes):
    errors.append("not all five real projection shapes were exercised")
if any(record.get("m") != 32 for record in records):
    errors.append("a comparison did not use the fixed32 B1 row count")
if any(record.get("bytes") != 2 * record["m"] * record["n"] for record in records):
    errors.append("a comparison reported an invalid BF16 byte count")
if any(record.get("byte_equal") is not True for record in records):
    errors.append("at least one stock/Stream-K output differed")
if binary_record.get("selector") != "streamk_coop128_byte_ab":
    errors.append("installed binary selector attestation mismatch")
if binary_record.get("production_enabled") is not False:
    errors.append("binary attestation did not remain production-off")
destination = binary_record.get("destination") or {}
if destination.get("sha256") != binary.CANDIDATE_SHA256:
    errors.append("installed binary SHA-256 mismatch")
if destination.get("bytes") != binary.CANDIDATE_SIZE:
    errors.append("installed binary size mismatch")

payload = {
    "schema": "fr13.fixed32.cutlass_streamk_live_gate.v1",
    "status": "pass" if not errors else "fail",
    "acceptance_valid": False,
    "task_set": "one real SWE-Verified B1 diagnostic task",
    "candidate": "streamk_coop128",
    "served_result": "stock",
    "production_enabled": False,
    "comparisons": len(records),
    "observed_m_values": sorted({record["m"] for record in records}),
    "observed_projection_nk": sorted([list(shape) for shape in observed_shapes]),
    "mismatch_count": sum(record.get("byte_equal") is not True for record in records),
    "candidate_sha256": binary.CANDIDATE_SHA256,
    "candidate_bytes": binary.CANDIDATE_SIZE,
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
