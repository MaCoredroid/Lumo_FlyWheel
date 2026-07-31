#!/usr/bin/env bash
# Real SWE-Verified exact4 B4 byte diagnostic for the batched wide-BV GDN.
# This runner is deliberately non-timing and cannot produce floor acceptance.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned FA2 shared object}"

FR13_GATE_BATCH_GDN_BV=${FR13_GATE_BATCH_GDN_BV:-64}
case "$FR13_GATE_BATCH_GDN_BV" in
  16|32|64|128) ;;
  *) echo "FR13_GATE_BATCH_GDN_BV must be 16, 32, 64, or 128" >&2; exit 2 ;;
esac

ARM="tail6_fixed32_${TAG}"
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
FA2_SHA256=$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')

[[ "$FA2_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical exact4 subset SHA-256 drift" >&2; exit 2; }
[[ ! -e "$RUNROOT" && ! -L "$RUNROOT" ]] \
  || { echo "RUNROOT must be new: $RUNROOT" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the gate" >&2; exit 2; }

export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_FLOOR_ORDER=TH

source scripts/fr13_canonical_env.sh
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh

mkdir -p "$RUNROOT"
printf 'classification=exact4_b4_byte_diagnostic\ntiming_eligible=0\nfloor_acceptance_eligible=0\nlauncher_pid=%s\nrunroot=%s\narm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nfa2_sha256=%s\ncandidate_bv=%s\nstarted=%s\n' \
  "$$" "$RUNROOT" "$ARM" "$(git rev-parse HEAD)" "$RUNNER_SHA256" \
  "$SUBSET_SHA256" "$FA2_SHA256" "$FR13_GATE_BATCH_GDN_BV" \
  "$(date -u +%FT%TZ)" \
  > "$RUNROOT/launcher_meta.txt"

.venv/bin/python scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT/runtime_manifest.at_launch.json"
.venv/bin/python scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT/external_manifest.at_launch.json"

if OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S= \
    FR13_FIXED32_B1_DIAGNOSTIC=0 \
    FR10_METRICS=1 ENFORCE_EAGER=1 \
    FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=1 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE="$FR13_GATE_BATCH_GDN_BV" \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_INSTANCE_ID= \
    FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 \
    FORKED_FA2_SO="$FORKED_FA2_SO" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" tail6_fixed32 "$SUBSET" \
      > "$RUNROOT/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi

printf 'serve_rc=%s ended=%s\n' "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT/launcher_meta.txt"
.venv/bin/python scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT/runtime_manifest.at_end.json"
.venv/bin/python scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT/external_manifest.at_end.json"
cmp -s "$RUNROOT/runtime_manifest.at_launch.json" "$RUNROOT/runtime_manifest.at_end.json" \
  || { echo "runtime/source manifest changed during diagnostic" >&2; exit 14; }
cmp -s "$RUNROOT/external_manifest.at_launch.json" "$RUNROOT/external_manifest.at_end.json" \
  || { echo "external manifest changed during diagnostic" >&2; exit 14; }
[[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
  || { echo "B4 diagnostic runner changed during execution" >&2; exit 14; }

(( serve_rc == 0 )) || exit "$serve_rc"

.venv/bin/python - \
  "$RUNROOT/$ARM" "$FR13_GATE_BATCH_GDN_BV" "$SUBSET_SHA256" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, "src")
from lumo_flywheel_serving.inference_proxy import (  # noqa: E402
    fixed32_canonical_task_set_sha256,
    fixed32_task_key_id,
    verify_fixed32_ingress_ledger,
)


arm = Path(sys.argv[1]).resolve()
candidate_bv = int(sys.argv[2])
subset_sha256 = sys.argv[3]
logs = arm / "logs"
record_path = logs / "fr13_fixed32_batch_gdn_byte_ab.jsonl"
pass_path = logs / "fr13_fixed32_batch_gdn_byte_ab.pass.json"
marker_path = logs / "fr13_fixed32_batch_gdn_byte_ab.real_event.arm"
ledger_path = logs / "fr13_fixed32_engine_ingress.jsonl"
verdict_path = arm.parent / "b4_gdn_wide_gate_verdict.json"
exact_tasks = {
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
}
surfaces = [
    "out",
    "ring_k",
    "ring_v",
    "ring_a",
    "ring_b",
    "state_export_compact",
    "state_export_untouched_tail",
    "flags",
    "invocation_counter",
]

marker_info = os.lstat(marker_path)
if (
    not stat.S_ISREG(marker_info.st_mode)
    or stat.S_ISLNK(marker_info.st_mode)
    or marker_info.st_nlink != 1
    or stat.S_IMODE(marker_info.st_mode) != 0o400
):
    raise SystemExit("real-event marker identity or mode is invalid")
marker_raw = marker_path.read_bytes()
try:
    marker_text = marker_raw.decode("ascii")
except UnicodeDecodeError as error:
    raise SystemExit("real-event marker is not ASCII") from error
if not marker_text.endswith("\n") or marker_text.count("\n") != 1:
    raise SystemExit("real-event marker framing is invalid")
task_marker = marker_text.removesuffix("\n")
prefix = "swe_verified:"
if not task_marker.startswith(prefix) or task_marker[len(prefix) :] not in exact_tasks:
    raise SystemExit("real-event marker is not bound to the canonical exact4 set")

payload = json.loads(pass_path.read_text(encoding="ascii"))
source_path = Path("src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py")
source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
expected_pass = {
    "schema": "fr13.fixed32.batch_gdn.live_pass.v2",
    "status": "pass",
    "task_marker": task_marker,
    "batch": 4,
    "layer_count": 48,
    "reference_always_served": True,
    "candidate": "fixed32_batch_gdn_bv_v2",
    "source_sha256": source_sha256,
    "mode": "tail6_fixed32",
    "physical_rows_per_request": 32,
    "reference_bv": 8,
    "candidate_bv": candidate_bv,
    "reference_physical_launches_per_layer": 8,
    "candidate_physical_launches_per_layer": 2,
    "compared_byte_surfaces": surfaces,
    "raw_byte_equal": True,
    "state_restored": True,
}
for key, expected in expected_pass.items():
    if payload.get(key) != expected:
        raise SystemExit(f"B4 PASS field mismatch: {key}")
layer_keys = payload.get("layer_keys")
if (
    not isinstance(layer_keys, list)
    or len(layer_keys) != 48
    or len(set(layer_keys)) != 48
    or not all(isinstance(key, str) and key.startswith("0x") for key in layer_keys)
):
    raise SystemExit("B4 PASS does not contain 48 distinct layer keys")

records = []
for line_number, line in enumerate(record_path.read_text(encoding="ascii").splitlines(), 1):
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid byte-gate JSONL line {line_number}") from error
    if not isinstance(record, dict):
        raise SystemExit(f"non-object byte-gate JSONL line {line_number}")
    records.append(record)
if any(record.get("status") == "mismatch_reference_served" for record in records):
    raise SystemExit("byte mismatch was observed in a real exact4 event")
b4 = [
    record
    for record in records
    if record.get("batch") == 4 and record.get("status") == "pass"
]
if len(b4) != 48 or {record.get("layer_key") for record in b4} != set(layer_keys):
    raise SystemExit("B4 JSONL does not contain exactly 48 distinct PASS records")
for record in b4:
    comparisons = record.get("comparisons")
    if (
        record.get("schema") != "fr13.fixed32.batch_gdn.byte_ab.v1"
        or record.get("task_marker") != task_marker
        or record.get("physical_rows_per_request") != 32
        or record.get("reference_bv") != 8
        or record.get("candidate_bv") != candidate_bv
        or record.get("legacy_physical_launches") != 8
        or record.get("candidate_physical_launches") != 2
        or record.get("carrier_nonzero") is not True
        or record.get("zero_diff") is not True
        or record.get("reference_restored_and_served") is not True
        or not isinstance(comparisons, list)
        or [comparison.get("name") for comparison in comparisons] != surfaces
        or any(comparison.get("byte_equal") is not True for comparison in comparisons)
    ):
        raise SystemExit("B4 JSONL PASS record is incomplete or inconsistent")

ledger_verification = verify_fixed32_ingress_ledger(
    ledger_path,
    expected_role="engine",
    require_finalized=True,
)
ledger_rows = [
    json.loads(line)
    for line in ledger_path.read_text(encoding="ascii").splitlines()
]
expected_task_set_sha256 = fixed32_canonical_task_set_sha256(tuple(exact_tasks))
if not any(
    row.get("event") == "campaign_begin"
    and row.get("evidence_sha256") == expected_task_set_sha256
    for row in ledger_rows
):
    raise SystemExit("engine ledger is not bound to the canonical exact4 task set")
marker_task_key = fixed32_task_key_id(task_marker[len(prefix) :])
if not any(
    row.get("event") == "request_accepted"
    and row.get("task_key_id") == marker_task_key
    for row in ledger_rows
):
    raise SystemExit("real-event marker has no matching engine acceptance record")

health = json.loads((arm / "health.json").read_text(encoding="utf-8"))
health_tasks = health.get("tasks")
if (
    not isinstance(health_tasks, list)
    or len(health_tasks) != 4
    or {task.get("instance_id") for task in health_tasks} != exact_tasks
    or health.get("swe_orchestrator_rc") != 0
):
    raise SystemExit("diagnostic did not complete the canonical exact4 task set")

observed_pass_layers = {
    str(batch): len(
        {
            record.get("layer_key")
            for record in records
            if record.get("batch") == batch and record.get("status") == "pass"
        }
    )
    for batch in (2, 3, 4)
}

verdict = {
    "schema": "fr13.fixed32.batch_gdn.b4_diagnostic.v1",
    "status": "pass",
    "run_classification": "exact4_b4_byte_diagnostic",
    "timing_eligible": False,
    "floor_acceptance_eligible": False,
    "subset_sha256": subset_sha256,
    "task_ids": sorted(exact_tasks),
    "task_marker": task_marker,
    "candidate_bv": candidate_bv,
    "b4_layer_passes": 48,
    "observed_pass_layers_by_batch": observed_pass_layers,
    "engine_ledger_chain_head_sha256": ledger_verification[
        "chain_head_sha256"
    ],
    "raw_byte_equal": True,
    "reference_always_served": True,
    "production_default_enabled": False,
}
temporary = verdict_path.with_name(verdict_path.name + ".tmp")
temporary.write_text(
    json.dumps(verdict, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    + "\n",
    encoding="ascii",
)
temporary.replace(verdict_path)
print(json.dumps(verdict, sort_keys=True))
PY
