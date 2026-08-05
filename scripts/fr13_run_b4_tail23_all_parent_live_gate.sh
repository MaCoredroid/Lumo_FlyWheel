#!/usr/bin/env bash
# Real SWE-Verified exact4 B4 byte qualification for source-v7 fixed32 all-parent TAW.
# The candidate is shadow-only; the exact reference is always returned.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned stock FA2 shared object}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
TAW_SOURCE=scripts/fr13_device_multidraft_kernel.py
TAW_SOURCE_SCHEMA=fr13-fixed32-taw-all-parent-v7
TAW_SOURCE_SHA256=694a3f4cd6e36ff1b6503ff19b2968b94a1ac226535a6efb44dcea1bb8a9a57b
TAIL_VALID_MASK=0x7a9ce7ff
TAIL_ACTIVE_DRAFTS=23
HYDRA_VALID_MASK=0x7abdffff
HYDRA_ACTIVE_DRAFTS=27
FIXED32_MODE=${FR13_FIXED32_ALL_PARENT_MODE:-tail6_fixed32}
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
B4_KV_CACHE_MEMORY_BYTES=42949672960
MANDATORY_WEIGHT_BYTES=32666638208
MANDATORY_WEIGHT_FLOOR_MS=119.658015414
ONE_SIDED_U95_CAP_MS=137.60671772610
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
case "$FIXED32_MODE" in
  tail6_fixed32)
    LOGICAL_TOPOLOGY=Tail23
    LOGICAL_SLUG=tail23
    VALID_MASK=$TAIL_VALID_MASK
    ACTIVE_DRAFTS=$TAIL_ACTIVE_DRAFTS
    VERDICT_SCHEMA=fr13.fixed32.tail23_all_parent.exact4_b4_live_gate.v1
    ;;
  hydra27_fixed32)
    LOGICAL_TOPOLOGY=Hydra27
    LOGICAL_SLUG=hydra27
    VALID_MASK=$HYDRA_VALID_MASK
    ACTIVE_DRAFTS=$HYDRA_ACTIVE_DRAFTS
    VERDICT_SCHEMA=fr13.fixed32.hydra27_all_parent.exact4_b4_live_gate.v1
    ;;
  *)
    echo "FR13_FIXED32_ALL_PARENT_MODE must be tail6_fixed32 or hydra27_fixed32" >&2
    exit 2
    ;;
esac
ARM="${FIXED32_MODE}_${LOGICAL_SLUG}_all_parent_b4_gate_${TAG}"
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
[[ "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" ]] \
  || { echo "pinned K64 draft-vocabulary block map drift" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the gate" >&2; exit 2; }

"$PYTHON_BIN" - \
  "$TAW_SOURCE_SCHEMA" "$TAW_SOURCE_SHA256" "$FIXED32_MODE" \
  "$VALID_MASK" "$ACTIVE_DRAFTS" <<'PY'
import importlib.util
import sys
from pathlib import Path

module_path = Path("scripts/fr13_device_multidraft_kernel.py")
spec = importlib.util.spec_from_file_location("fr13_tail23_gate_preflight", module_path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot import all-parent source")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
topology = module._fr13_fixed32_topology()
source_contract = module._fr13_fixed32_taw_source_contract(topology, batch_size=4)
mode = sys.argv[3]
if mode not in topology.VALID_MASK_BY_MODE:
    raise SystemExit("all-parent gate fixed32 mode is unsupported")
expected_active = module._fr13_fixed32_expected_active(topology, mode)
if (
    module._FR13_FIXED32_TAW_SOURCE_SCHEMA != sys.argv[1]
    or module._FR13_FIXED32_TAW_SOURCE_SHA256 != sys.argv[2]
    or source_contract.get("source_contract_sha256") != sys.argv[2]
    or int(topology.VALID_MASK_BY_MODE[mode]) != int(sys.argv[4], 0)
    or expected_active != int(sys.argv[5])
    or int(topology.PHYSICAL_DRAFTS) != 31
    or int(topology.PHYSICAL_ROWS) != 32
):
    raise SystemExit("fixed32 all-parent source-v7 preflight contract drifted")
PY

export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant

[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" \
   && "$FR13_DRAFT_VOCAB_K" == "65536" \
   && "$FR13_DRAFT_VOCAB_ROOT" == "1" \
   && "$FR13_DRAFT_VOCAB_BLOCKS" == "/workspace/scripts/fr13_dvk_subset_blocks.json" \
   && "$LUMO_SWE_AUTOCOMMIT" == "0" ]] \
  || { echo "fixed32 all-parent K64/ROOT=1 deployment contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
printf 'classification=real_swe_verified_exact4_b4_byte_diagnostic\nacceptance_valid=0\ntiming_eligible=0\nfloor_acceptance_eligible=0\nproduction_enabled=0\ncandidate_shadow_only=1\nreference_always_served=1\ncandidate=fixed32_all_parent_commit_v2\nsource_contract_schema=%s\nsource_contract_sha256=%s\ntopology=%s\nmode=%s\nactive_drafts=%s\nvalid_mask=%s\nphysical_drafts=31\nphysical_rows_root_inclusive=32\nrequired_observed_batches=1,2,3,4\ntask_count=4\nbatch_size=4\nconcurrency=4\ndraft_vocab_k=65536\ndraft_vocab_root=1\ndraft_vocab_blocks=/workspace/scripts/fr13_dvk_subset_blocks.json\ndraft_vocab_blocks_sha256=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nraw_prompt_response_autocommit=0\nlauncher_pid=%s\nrunroot=%s\narm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nstock_fa2_sha256=%s\nstock_fa2_bytes=%s\nenforce_eager=0\ncudagraph_mode=FULL_AND_PIECEWISE\nkv_cache_memory_bytes=%s\nstarted=%s\n' \
  "$TAW_SOURCE_SCHEMA" "$TAW_SOURCE_SHA256" "$LOGICAL_TOPOLOGY" \
  "$FIXED32_MODE" "$ACTIVE_DRAFTS" "$VALID_MASK" \
  "$BLOCK_MAP_SHA256" "$MANDATORY_WEIGHT_BYTES" \
  "$MANDATORY_WEIGHT_FLOOR_MS" "$ONE_SIDED_U95_CAP_MS" "$$" \
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
    LUMO_SWE_AUTOCOMMIT=0 \
    FR13_FIXED32_B1_DIAGNOSTIC=0 \
    FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_DRAFT_VOCAB_K=65536 FR13_DRAFT_VOCAB_ROOT=1 \
    FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json \
    FR13_DEVICE_MULTIDRAFT=1 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=1 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
    FR13_FIXED32_CUTLASS_WAVE=stock \
    FR13_FIXED32_CUTLASS_WAVE_SO= \
    FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 \
    FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_INSTANCE_ID= \
    FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 \
    FORKED_FA2_SO="$FORKED_FA2_SO" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" "$FIXED32_MODE" "$SUBSET" \
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
  || { echo "fixed32 all-parent gate runner changed during execution" >&2; exit 14; }
(( serve_rc == 0 )) || exit "$serve_rc"

"$PYTHON_BIN" - \
  "$ARMDIR" "$ARMDIR/logs/fr13_fixed32_taw_native_precompute.live_pass.json" \
  "$ARMDIR/${LOGICAL_SLUG}_all_parent_production_pass.json" \
  "$ARMDIR/${LOGICAL_SLUG}_all_parent_b4_byte_gate.json" \
  "$TAW_SOURCE" "$TAW_SOURCE_SCHEMA" "$TAW_SOURCE_SHA256" \
  "$SOURCE_COMMIT" "$SUBSET" "$SUBSET_SHA256" "$BLOCK_MAP" \
  "$BLOCK_MAP_SHA256" "$RUNNER_PATH" "$RUNNER_SHA256" \
  "$RUNROOT_ABS/runtime_manifest.at_end.json" "$STOCK_FA2_SHA256" \
  "$FIXED32_MODE" "$LOGICAL_TOPOLOGY" "$ACTIVE_DRAFTS" "$VALID_MASK" \
  "$VERDICT_SCHEMA" <<'PY'
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

arm = Path(sys.argv[1]).resolve()
live_path = Path(sys.argv[2])
production_path = Path(sys.argv[3])
verdict_path = Path(sys.argv[4])
source_path = Path(sys.argv[5])
source_schema = sys.argv[6]
source_contract_sha256 = sys.argv[7]
source_commit = sys.argv[8]
subset_path = Path(sys.argv[9])
subset_sha256 = sys.argv[10]
block_map_path = Path(sys.argv[11])
block_map_sha256 = sys.argv[12]
runner_path = Path(sys.argv[13])
runner_sha256 = sys.argv[14]
runtime_manifest_path = Path(sys.argv[15])
stock_fa2_sha256 = sys.argv[16]
fixed32_mode = sys.argv[17]
logical_topology = sys.argv[18]
active_drafts = int(sys.argv[19])
valid_mask = int(sys.argv[20], 0)
verdict_schema = sys.argv[21]
dataset_out = arm / "swe_out" / "verified"
campaign_arm_path = dataset_out / "fixed32_taw_campaign_arm.json"
campaign_proof_path = dataset_out / "fixed32_qwen_campaign_provenance.json"
health_path = arm / "health.json"
expected_tasks = [
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
]
expected_marker = f"swe_verified:campaign4_{subset_sha256}"

for label, path in (
    ("live bundle", live_path),
    ("campaign arm", campaign_arm_path),
    ("campaign proof", campaign_proof_path),
    ("health", health_path),
    ("runtime manifest", runtime_manifest_path),
):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"{label} is missing, non-regular, or symlinked: {path}")

spec = importlib.util.spec_from_file_location("fr13_tail23_gate_verdict", source_path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot import source-v7 all-parent implementation")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
topology = module._fr13_fixed32_topology()
bundle = module._fr13_fixed32_taw_native_production_pass(
    path=str(live_path),
    expected_mode=fixed32_mode,
)
if (
    bundle.get("status") != "production_ready"
    or bundle.get("qualified_batches") != [1, 2, 3, 4]
    or bundle.get("required_production_batches") != [1, 4]
    or bundle.get("source_contract_schema") != source_schema
    or bundle.get("source_contract_sha256") != source_contract_sha256
    or bundle.get("valid_mask") != valid_mask
    or module._fr13_fixed32_expected_active(topology, fixed32_mode) != active_drafts
    or int(topology.PHYSICAL_DRAFTS) != 31
    or int(topology.PHYSICAL_ROWS) != 32
):
    raise SystemExit("fixed32 all-parent source-v7 production bundle is incomplete or drifted")
for batch in (1, 2, 3, 4):
    record = bundle["batch_passes"][str(batch)]
    if (
        record.get("batch_size") != batch
        or record.get("covered_batches") != [batch]
        or record.get("task_marker") != expected_marker
        or record.get("probability_mismatches") != 0
        or record.get("product_mismatches") != 0
        or record.get("reference_returned") is not True
        or record.get("candidate_returned") is not False
    ):
        raise SystemExit(f"fixed32 all-parent B{batch} qualification is not independent and exact")

campaign_arm = json.loads(campaign_arm_path.read_text(encoding="ascii"))
if (
    campaign_arm.get("schema") != "fr13-fixed32-taw-campaign-arm-v1"
    or campaign_arm.get("state") != "ended"
    or campaign_arm.get("run_classification") != "b4_taw_diagnostic"
    or campaign_arm.get("batch_size") != 4
    or campaign_arm.get("concurrency") != 4
    or campaign_arm.get("task_count") != 4
    or campaign_arm.get("subset_sha256") != subset_sha256
    or campaign_arm.get("task_ids") != expected_tasks
    or campaign_arm.get("marker") != expected_marker
):
    raise SystemExit("TAW campaign arm is not the canonical exact4 B4 marker")

campaign_proof_raw = campaign_proof_path.read_bytes()
campaign_proof = json.loads(campaign_proof_raw.decode("ascii"))
campaign_proof_identity = {
    "path": str(campaign_proof_path.resolve()),
    "sha256": hashlib.sha256(campaign_proof_raw).hexdigest(),
    "bytes": len(campaign_proof_raw),
}
if (
    campaign_proof.get("schema")
    != "fr13-fixed32-qwen-campaign-provenance-v1"
    or campaign_proof.get("metric_scope") != "concurrent_campaign_union"
    or campaign_proof.get("concurrency") != 4
    or campaign_proof.get("task_ids") != expected_tasks
):
    raise SystemExit("Qwen campaign proof is not the canonical concurrent exact4 union")
for task_id in expected_tasks:
    metadata_path = dataset_out / "per_task" / task_id / "runner_metadata.json"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise SystemExit(f"canonical task metadata is missing: {task_id}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    provenance = metadata.get("fixed32_real_task_provenance")
    if (
        metadata.get("instance_id") != task_id
        or metadata.get("fixed32_qwen_campaign_proof") != campaign_proof_identity
        or not isinstance(provenance, dict)
        or provenance.get("schema") != "fr13-fixed32-real-task-provenance-v3"
        or provenance.get("instance_id") != task_id
        or provenance.get("qwen_metric_scope") != "campaign"
        or provenance.get("qwen_campaign_metric_proof") != campaign_proof_identity
        or provenance.get("aborted_logical_requests") != 0
        or provenance.get("failed_attempts") != 0
    ):
        raise SystemExit(f"canonical task provenance is incomplete: {task_id}")

health = json.loads(health_path.read_text(encoding="utf-8"))
health_tasks = health.get("tasks")
if (
    health.get("swe_orchestrator_rc") != 0
    or not isinstance(health_tasks, list)
    or len(health_tasks) != 4
    or [task.get("instance_id") for task in health_tasks] != expected_tasks
):
    raise SystemExit("health record does not cover the ordered canonical exact4 set")

runtime_manifest_raw = runtime_manifest_path.read_bytes()
runtime_manifest = json.loads(runtime_manifest_raw.decode("utf-8"))
unsigned_manifest = {
    key: value
    for key, value in runtime_manifest.items()
    if key != "overall_canonical_sha256"
}
computed_manifest_sha256 = hashlib.sha256(
    json.dumps(
        unsigned_manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()
if (
    runtime_manifest.get("schema") != "fr13-runtime-manifest-v1"
    or runtime_manifest.get("profile") != "fixed32"
    or runtime_manifest.get("overall_canonical_sha256")
    != computed_manifest_sha256
):
    raise SystemExit("runtime manifest identity or digest drifted")
closures = runtime_manifest.get("closures")
if not isinstance(closures, dict):
    raise SystemExit("runtime manifest closures are missing")
closure_records = [
    record
    for records in closures.values()
    if isinstance(records, list)
    for record in records
    if isinstance(record, dict)
]
closure_by_path = {
    record.get("path"): record
    for record in closure_records
    if isinstance(record.get("path"), str)
}
required_closure = {
    "scripts/fr13_run_b4_tail23_all_parent_live_gate.sh": runner_sha256,
    "scripts/fr13_device_multidraft_kernel.py": hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest(),
    "scripts/fr13_fixed32_topology.py": hashlib.sha256(
        Path("scripts/fr13_fixed32_topology.py").read_bytes()
    ).hexdigest(),
    "scripts/fr13_dvk_subset_blocks.json": block_map_sha256,
    "config/fr13_fixed32/subset_b4_four.json": subset_sha256,
}
for path, expected_sha256 in required_closure.items():
    if closure_by_path.get(path, {}).get("sha256") != expected_sha256:
        raise SystemExit(f"runtime manifest does not bind {path}")
if hashlib.sha256(runner_path.read_bytes()).hexdigest() != runner_sha256:
    raise SystemExit("gate runner changed before verdict creation")
if hashlib.sha256(block_map_path.read_bytes()).hexdigest() != block_map_sha256:
    raise SystemExit("K64 block map changed before verdict creation")
if hashlib.sha256(subset_path.read_bytes()).hexdigest() != subset_sha256:
    raise SystemExit("exact4 subset changed before verdict creation")

live_raw = live_path.read_bytes()
temporary = production_path.with_name(production_path.name + ".tmp")
temporary.write_bytes(live_raw)
os.chmod(temporary, 0o400)
os.replace(temporary, production_path)
production_raw = production_path.read_bytes()
if production_raw != live_raw:
    raise SystemExit("published production bundle differs from validated live bundle")

verdict = {
    "schema": verdict_schema,
    "status": "pass",
    "run_classification": "real_swe_verified_exact4_b4_byte_diagnostic",
    "acceptance_valid": False,
    "timing_eligible": False,
    "floor_acceptance_eligible": False,
    "reference_always_served": True,
    "candidate_returned": False,
    "production_default_enabled": False,
    "raw_prompt_response_published": False,
    "candidate": "fixed32_all_parent_commit_v2",
    "source_commit": source_commit,
    "source_contract_schema": source_schema,
    "source_contract_sha256": source_contract_sha256,
    "source_file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "mode": fixed32_mode,
    "logical_topology": logical_topology,
    "active_drafts": active_drafts,
    "valid_mask": hex(valid_mask),
    "physical_drafts": 31,
    "physical_rows_root_inclusive": 32,
    "qualified_batches": [1, 2, 3, 4],
    "required_production_batches": [1, 4],
    "independent_b1_record": True,
    "independent_b4_record": True,
    "draft_vocab_k": 65536,
    "draft_vocab_root": 1,
    "draft_vocab_blocks": "/workspace/scripts/fr13_dvk_subset_blocks.json",
    "draft_vocab_blocks_sha256": block_map_sha256,
    "mandatory_weight_bytes": 32666638208,
    "mandatory_weight_floor_ms": 119.658015414,
    "one_sided_u95_cap_ms": 137.60671772610,
    "subset_sha256": subset_sha256,
    "task_ids": expected_tasks,
    "task_marker": expected_marker,
    "stock_fa2_sha256": stock_fa2_sha256,
    "live_bundle_sha256": hashlib.sha256(live_raw).hexdigest(),
    "production_bundle_sha256": hashlib.sha256(production_raw).hexdigest(),
    "campaign_proof_sha256": campaign_proof_identity["sha256"],
    "runtime_manifest_sha256": computed_manifest_sha256,
    "gate_runner_sha256": runner_sha256,
    "probability_mismatches": 0,
    "product_mismatches": 0,
}
encoded = (
    json.dumps(verdict, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    + "\n"
).encode("ascii")
temporary = verdict_path.with_name(verdict_path.name + ".tmp")
temporary.write_bytes(encoded)
os.replace(temporary, verdict_path)
print(json.dumps(verdict, sort_keys=True))
PY
