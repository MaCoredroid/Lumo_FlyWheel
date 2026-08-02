#!/usr/bin/env bash
# One real SWE-Verified full-vocabulary B1 SFWD byte/timing diagnostic.
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
FULL_VOCAB_WEIGHT_BYTES=42025179008
FULL_VOCAB_FLOOR_MS=153.938384645
FULL_VOCAB_CAP_MS=177.02914234175
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
export FR13_DRAFT_VOCAB_ROOT=0
export FR13_DRAFT_VOCAB_K=0
export FR13_NEEDS_ALLOW='FR13_DRAFT_VOCAB_K=0'
export FR13_MANDATORY_WEIGHT_BYTES="$FULL_VOCAB_WEIGHT_BYTES"
export FR13_WEIGHT_FLOOR_MS="$FULL_VOCAB_FLOOR_MS"

bash scripts/fr13_run_b1_kernel_live_gate.sh

printf '%s\n' \
  'classification=one_real_swe_verified_full_vocab_b1_sfwd_state_fusion_byte_timing_diagnostic' \
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
  'draft_vocab_root=0' \
  'draft_vocab_k=0' \
  >> "$RUNROOT_ABS/launcher_meta.txt"

.venv/bin/python - \
  "$ARMDIR" "$SOURCE_COMMIT" "$TASK_ID" \
  "$FULL_VOCAB_WEIGHT_BYTES" "$FULL_VOCAB_FLOOR_MS" "$FULL_VOCAB_CAP_MS" <<'PY'
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
logs = arm_dir / "logs"
records_path = logs / "fr13_fixed32_sfwd_state_fusion.byte_ab.jsonl"
pass_path = logs / "fr13_fixed32_sfwd_state_fusion.live_pass.json"
marker_path = logs / "fr13_fixed32_sfwd_state_fusion.real_event.arm"
diagnostic_path = arm_dir / "fixed32_b1_diagnostic.json"
container_env_path = arm_dir / "container_env.txt"
engine_ledger_path = logs / "fr13_fixed32_engine_ingress.jsonl"
output_path = arm_dir / "sfwd_state_fusion_b1_gate.json"


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
marker_raw = regular(marker_path)
diagnostic_raw = regular(diagnostic_path)
container_env_raw = regular(container_env_path)
engine_ledger_raw = regular(engine_ledger_path)
records = [json.loads(line) for line in records_raw.decode("ascii").splitlines()]
live_pass = json.loads(pass_raw.decode("ascii"))
diagnostic = json.loads(diagnostic_raw.decode("ascii"))
marker = f"swe_verified:{task_id}"
errors = []

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
    "candidate": "fixed32_sfwd_state_fusion_rowgroup4_v2",
    "task_marker": marker,
    "batch": 1,
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

container_env = container_env_raw.decode("ascii").splitlines()
for expected in (
    "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=1",
    "FR13_DRAFT_VOCAB_ROOT=0",
    "FR13_DRAFT_VOCAB_K=0",
    "FR13_CONV_WB_BATCHED=1",
    "FR13_TREE_CONV_FUSED=1",
    "MAX_NUM_SEQS=1",
):
    if container_env.count(expected) != 1:
        errors.append(f"container environment mismatch: {expected}")

payload = {
    "schema": "fr13.fixed32.sfwd_state_fusion.b1_gate.v1",
    "status": "pass" if not errors else "fail",
    "run_classification": "one_real_swe_verified_full_vocab_b1_byte_timing_diagnostic",
    "task_set": "one",
    "task_count": 1,
    "task_ids": [task_id],
    "task_marker": marker,
    "real_task_authenticated": True,
    "reference_returned": True,
    "timing_eligible": False,
    "floor_acceptance_eligible": False,
    "production_enabled": False,
    "candidate": "fixed32_sfwd_state_fusion_rowgroup4_v2",
    "batch_size": 1,
    "physical_rows_per_request": 32,
    "candidate_conv_launches_per_layer": 1,
    "gdn_level_path_programs": [1, 11],
    "gdn_physical_launches_per_layer": 2,
    "draft_vocab_root": 0,
    "draft_vocab_k": 0,
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
    "container_env_sha256": hashlib.sha256(container_env_raw).hexdigest(),
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
