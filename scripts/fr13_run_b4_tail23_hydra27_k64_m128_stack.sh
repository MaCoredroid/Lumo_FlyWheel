#!/usr/bin/env bash
# Qualify and time the physical32 K64 B4 Tail23/Hydra27 all-parent + M128 stack.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

EXPECTED_BRANCH=${EXPECTED_BRANCH:-agent/fixed32-b4-tail23-hydra27-k64-m128-review}
STOCK_FA2_SOURCE=${STOCK_FA2_SOURCE:-/home/mark/lumoFlyWheel-b4-sfwd-campaignfix/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so}
STOCK_FA2_SO=${STOCK_FA2_SO:-$REPO/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so}
CUTLASS_B4_SO=${CUTLASS_B4_SO:-/home/mark/fr13_streamk_build/bin/_C_stable_libtorch.persistent_b4_m128_stock_symbol_exact_compare320_gate_ready.abi3.so}
EXPECTED_STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
EXPECTED_CANDIDATE_SHA256=895495fe82cb0e0278d3b0a39b8e57e1281aa73a10bbba01a94085733c81d64f

[[ "$(git branch --show-current)" == "$EXPECTED_BRANCH" ]] \
  || { echo "run from $EXPECTED_BRANCH" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
if [[ ! -e "$STOCK_FA2_SO" && ! -L "$STOCK_FA2_SO" ]]; then
  mkdir -p "$(dirname "$STOCK_FA2_SO")"
  cp --reflink=auto -- "$STOCK_FA2_SOURCE" "$STOCK_FA2_SO"
fi
[[ -f "$STOCK_FA2_SO" && ! -L "$STOCK_FA2_SO" \
   && "$(sha256sum "$STOCK_FA2_SO" | awk '{print $1}')" == "$EXPECTED_STOCK_FA2_SHA256" ]] \
  || { echo "stock FA2 identity mismatch" >&2; exit 2; }
[[ -f "$CUTLASS_B4_SO" && ! -L "$CUTLASS_B4_SO" \
   && "$(sha256sum "$CUTLASS_B4_SO" | awk '{print $1}')" == "$EXPECTED_CANDIDATE_SHA256" ]] \
  || { echo "persistent-M128 candidate identity mismatch" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent" >&2; exit 2; }

stamp=$(date -u +%Y%m%dT%H%M%SZ)
campaign_root="$REPO/output/fr13_b4_tail23_hydra27_k64_m128_stack_${stamp}"
source_commit=$(git rev-parse HEAD)
mkdir -p "$campaign_root"

run_topology() {
  local mode=$1
  local slug=$2
  local gate_tag="${slug}_${stamp}"
  local taw_root="$campaign_root/${slug}_all_parent_gate"
  local taw_arm="$taw_root/${mode}_${slug}_all_parent_b4_gate_${gate_tag}"
  local taw_pass="$taw_arm/${slug}_all_parent_production_pass.json"
  local taw_verdict="$taw_arm/${slug}_all_parent_b4_byte_gate.json"

  FR13_FIXED32_ALL_PARENT_MODE="$mode" \
  RUNROOT="$taw_root" TAG="$gate_tag" FORKED_FA2_SO="$STOCK_FA2_SO" \
    bash scripts/fr13_run_b4_tail23_all_parent_live_gate.sh
  [[ -f "$taw_pass" && ! -L "$taw_pass" \
     && -f "$taw_verdict" && ! -L "$taw_verdict" ]] \
    || { echo "$mode all-parent exact4 PASS was not issued" >&2; return 4; }

  local m128_root="$campaign_root/${slug}_m128_gate"
  local m128_tag="${slug}_m128_${stamp}"
  local m128_arm="$m128_root/${mode}_cutlass_b4_m128_k64_root_gate_${m128_tag}"
  local m128_pass="$m128_arm/cutlass_b4_m128_k64_root_byte_gate.json"
  CUTLASS_B4_QUALIFICATION_PROFILE=k64_root \
  CUTLASS_B4_FIXED32_MODE="$mode" \
  RUNROOT="$m128_root" TAG="$m128_tag" FORKED_FA2_SO="$STOCK_FA2_SO" \
  CUTLASS_B4_SO="$CUTLASS_B4_SO" \
    bash scripts/fr13_run_b4_cutlass_persistent_m128_live_gate.sh
  [[ -f "$m128_pass" && ! -L "$m128_pass" ]] \
    || { echo "$mode persistent-M128 exact4 PASS was not issued" >&2; return 4; }

  local timing_root="$campaign_root/${slug}_timing"
  local timing_tag="${slug}_timing_${stamp}"
  local m128_sha256
  m128_sha256=$(sha256sum "$m128_pass" | awk '{print $1}')
  CUTLASS_B4_QUALIFICATION_PROFILE=k64_root \
  CUTLASS_B4_FIXED32_MODE="$mode" \
  CUTLASS_B4_QUALIFICATION_SOURCE_COMMIT="$source_commit" \
  FR13_FIXED32_ALL_PARENT_PASS_JSON="$taw_pass" \
  FR13_FIXED32_ALL_PARENT_VERDICT_JSON="$taw_verdict" \
  RUNROOT="$timing_root" TAG="$timing_tag" STOCK_FA2_SO="$STOCK_FA2_SO" \
  CUTLASS_B4_SO="$CUTLASS_B4_SO" CUTLASS_B4_PASS_JSON="$m128_pass" \
  CUTLASS_B4_PASS_SHA256="$m128_sha256" \
    bash scripts/fr13_run_b4_cutlass_persistent_m128_timing.sh
  [[ -f "$timing_root/timing_summary.json" \
     && ! -L "$timing_root/timing_summary.json" ]] \
    || { echo "$mode full-step timing summary was not issued" >&2; return 4; }
}

run_topology tail6_fixed32 tail23
run_topology hydra27_fixed32 hydra27

.venv/bin/python - \
  "$campaign_root/tail23_all_parent_gate/tail6_fixed32_tail23_all_parent_b4_gate_tail23_${stamp}/tail23_all_parent_b4_byte_gate.json" \
  "$campaign_root/tail23_m128_gate/tail6_fixed32_cutlass_b4_m128_k64_root_gate_tail23_m128_${stamp}/cutlass_b4_m128_k64_root_byte_gate.json" \
  "$campaign_root/tail23_timing/timing_summary.json" \
  "$campaign_root/hydra27_all_parent_gate/hydra27_fixed32_hydra27_all_parent_b4_gate_hydra27_${stamp}/hydra27_all_parent_b4_byte_gate.json" \
  "$campaign_root/hydra27_m128_gate/hydra27_fixed32_cutlass_b4_m128_k64_root_gate_hydra27_m128_${stamp}/cutlass_b4_m128_k64_root_byte_gate.json" \
  "$campaign_root/hydra27_timing/timing_summary.json" \
  "$campaign_root/paired_summary.json" "$source_commit" \
  "$campaign_root/tail23_timing/tail6_fixed32_cutlass_persistent_m128_k64_root_tail23_timing_${stamp}/logs/fr13_fixed32_work_census.jsonl" \
  "$campaign_root/hydra27_timing/hydra27_fixed32_cutlass_persistent_m128_k64_root_hydra27_timing_${stamp}/logs/fr13_fixed32_work_census.jsonl" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import fr13_derive_qwen_agent_bundle_cap256 as qwen_derivation
import fr13_fixed32_work_census as work_census
import fr13_floor_gate as floor_gate

paths = list(map(Path, sys.argv[1:7]))
out = Path(sys.argv[7])
source_commit = sys.argv[8]
tail_census_path = Path(sys.argv[9])
hydra_census_path = Path(sys.argv[10])
expected_work_route = "fixed32_native_precompute_production_candidate_return"
tail_census_raw = tail_census_path.read_bytes()
hydra_census_raw = hydra_census_path.read_bytes()
tail_census_sha256 = hashlib.sha256(tail_census_raw).hexdigest()
hydra_census_sha256 = hashlib.sha256(hydra_census_raw).hexdigest()
tail_records = work_census.load_jsonl_bytes(
    tail_census_raw, source=str(tail_census_path)
)
hydra_records = work_census.load_jsonl_bytes(
    hydra_census_raw, source=str(hydra_census_path)
)
tail_work = work_census.validate_arm(
    tail_records,
    expected_mode="tail6_fixed32",
    expected_route=expected_work_route,
    required_batches=(4,),
)
hydra_work = work_census.validate_arm(
    hydra_records,
    expected_mode="hydra27_fixed32",
    expected_route=expected_work_route,
    required_batches=(4,),
)
physical_work = work_census.validate_campaign(
    tail_records,
    hydra_records,
    required_batches=(4,),
)
for label, report in (
    ("Tail23", tail_work),
    ("Hydra27", hydra_work),
    ("paired", physical_work),
):
    if work_census.normalized_work_sha256(
        report["normalized_work_signature"]
    ) != report["normalized_work_signature_sha256"]:
        raise SystemExit(f"{label} normalized physical-work digest is invalid")
if not (
    tail_work["normalized_work_signature_sha256"]
    == hydra_work["normalized_work_signature_sha256"]
    == physical_work["normalized_work_signature_sha256"]
):
    raise SystemExit("Tail23/Hydra27 normalized physical-work signatures differ")
expected = (
    ("tail6_fixed32", "Tail23", 23, "0x7a9ce7ff"),
    ("hydra27_fixed32", "Hydra27", 27, "0x7abdffff"),
)
arms = {}
four_way_work_signatures = {}
for index, (mode, logical, active, mask) in enumerate(expected):
    taw_path, m128_path, timing_path = paths[index * 3:index * 3 + 3]
    taw_sha256 = hashlib.sha256(taw_path.read_bytes()).hexdigest()
    m128_sha256 = hashlib.sha256(m128_path.read_bytes()).hexdigest()
    taw = json.loads(taw_path.read_text(encoding="ascii"))
    m128 = json.loads(m128_path.read_text(encoding="ascii"))
    timing = json.loads(timing_path.read_text(encoding="ascii"))
    arm_work = tail_work if mode == "tail6_fixed32" else hydra_work
    timing_work = timing.get("work_census", {})
    stock_work_signature = timing_work.get("stock_reference", {}).get(
        "normalized_work_signature_sha256"
    )
    candidate_work_signature = timing_work.get("candidate", {}).get(
        "normalized_work_signature_sha256"
    )
    arm_census_sha256 = (
        tail_census_sha256 if mode == "tail6_fixed32" else hydra_census_sha256
    )
    expected_tasks = [
        "astropy__astropy-12907",
        "astropy__astropy-13033",
        "astropy__astropy-13236",
        "astropy__astropy-13398",
    ]
    if (
        taw.get("status") != "pass"
        or taw.get("source_commit") != source_commit
        or taw.get("mode") != mode
        or taw.get("logical_topology") != logical
        or taw.get("active_drafts") != active
        or taw.get("valid_mask") != mask
        or taw.get("physical_drafts") != 31
        or taw.get("physical_rows_root_inclusive") != 32
        or taw.get("qualified_batches") != [1, 2, 3, 4]
        or taw.get("required_production_batches") != [1, 4]
        or taw.get("draft_vocab_root") != 1
        or taw.get("draft_vocab_k") != 65536
        or taw.get("draft_vocab_blocks_sha256")
        != "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
        or taw.get("task_ids") != expected_tasks
        or taw.get("probability_mismatches") != 0
        or taw.get("product_mismatches") != 0
        or m128.get("status") != "pass"
        or m128.get("source_commit") != source_commit
        or m128.get("topology") != mode
        or m128.get("logical_topology") != logical
        or m128.get("active_drafts") != active
        or m128.get("valid_mask") != mask
        or m128.get("physical_drafts") != 31
        or m128.get("physical_rows_root_inclusive") != 32
        or m128.get("task_ids") != expected_tasks
        or m128.get("draft_vocab_root") != 1
        or m128.get("draft_vocab_k") != 65536
        or m128.get("draft_vocab_blocks_sha256")
        != "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
        or m128.get("mismatching_comparisons") != 0
        or m128.get("differing_bytes") != 0
        or m128.get("observed_m_values") != [128]
        or timing.get("status") != "complete"
        or timing.get("qualification_source_commit") != source_commit
        or timing.get("timing_harness_commit") != source_commit
        or timing.get("arm") != mode
        or timing.get("logical_topology") != logical
        or timing.get("active_drafts") != active
        or timing.get("valid_mask") != mask
        or timing.get("physical_rows_root_inclusive") != 32
        or timing.get("sfwd_projection_rows") != 128
        or stock_work_signature != arm_work["normalized_work_signature_sha256"]
        or candidate_work_signature != arm_work["normalized_work_signature_sha256"]
        or timing_work.get("candidate", {}).get("event_count")
        != arm_work["event_count"]
        or timing_work.get("candidate", {}).get("event_counts_by_batch")
        != arm_work["event_counts_by_batch"]
        or timing_work.get("candidate", {}).get("census_sha256")
        != arm_census_sha256
        or timing.get("all_parent_production") is not True
        or timing.get("all_parent_verdict_sha256") != taw_sha256
        or timing.get("all_parent_pass_sha256")
        != taw.get("production_bundle_sha256")
        or timing.get("task_ids") != expected_tasks
        or timing.get("draft_vocab_root") != 1
        or timing.get("draft_vocab_k") != 65536
        or timing.get("draft_vocab_blocks_sha256")
        != "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
        or timing.get("mandatory_weight_bytes") != 27977022848
        or not math.isclose(timing.get("mandatory_weight_floor_ms", math.nan), 102.479937172, abs_tol=1e-9)
        or not math.isclose(timing.get("one_sided_u95_cap_ms", math.nan), 117.8519277478, abs_tol=1e-9)
    ):
        raise SystemExit(f"{logical} B4 stack evidence is incomplete")
    if timing.get("candidate", {}).get("live_result_sha256") != m128_sha256:
        raise SystemExit(f"{logical} M128 timing is not bound to its exact4 gate")
    for arm_name in ("stock_reference", "candidate"):
        record = timing.get(arm_name, {})
        for key in (
            "step_wall_ms", "measured_tps_fullstep_wall",
            "accepted_drafts_per_event", "committed_tokens_per_event",
            "events_per_step", "wall_ms_per_event", "sfwd_gpu_ms_per_event",
            "sfwd_gpu_ms_per_step", "dfwd_gpu_ms_per_step",
            "cfwd_gpu_ms_per_step", "gpu_component_ms_per_step",
            "other_wall_ms_per_step", "step_wall_to_optimistic_floor_ratio",
        ):
            value = record.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (value < 0 if key == "other_wall_ms_per_step" else value <= 0)
            ):
                raise SystemExit(f"{logical} {arm_name} lacks valid finite {key}")
        if not math.isclose(
            record["sfwd_gpu_ms_per_step"],
            record["sfwd_gpu_ms_per_event"] * record["events_per_step"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise SystemExit(f"{logical} {arm_name} SFWD units do not reconcile")
        if not math.isclose(
            record["step_wall_ms"],
            record["wall_ms_per_event"] * record["events_per_step"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise SystemExit(f"{logical} {arm_name} wall units do not reconcile")
        if not math.isclose(
            record["measured_tps_fullstep_wall"],
            record["committed_tokens_per_event"] * 1000.0
            / record["wall_ms_per_event"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise SystemExit(f"{logical} {arm_name} TPS does not reconcile")
        if not math.isclose(
            record["gpu_component_ms_per_step"]
            + record["other_wall_ms_per_step"],
            record["step_wall_ms"],
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise SystemExit(f"{logical} {arm_name} phases do not reconcile")
    candidate = timing["candidate"]
    four_way_work_signatures[f"{logical}.stock_reference"] = stock_work_signature
    four_way_work_signatures[f"{logical}.persistent_m128"] = (
        candidate_work_signature
    )
    arms[logical] = {
        "mode": mode,
        "active_drafts": active,
        "valid_mask": mask,
        "all_parent_gate_sha256": taw_sha256,
        "m128_gate_sha256": m128_sha256,
        "timing_summary_sha256": hashlib.sha256(timing_path.read_bytes()).hexdigest(),
        "stock_cutlass_with_all_parent": timing["stock_reference"],
        "persistent_m128_with_all_parent": candidate,
    }

if set(four_way_work_signatures.values()) != {
    physical_work["normalized_work_signature_sha256"]
}:
    raise SystemExit(
        "Tail23/Hydra27 stock/M128 physical-work signatures are not identical"
    )

summary = {
    "schema": "fr13.fixed32.b4_tail23_hydra27_k64_m128_stack.exact4_pair.v1",
    "status": "complete",
    "run_classification": "real_swe_verified_exact4_b4_k64_root_timing_pair",
    "formal_floor_acceptance_eligible": False,
    "source_commit": source_commit,
    "task_count_per_arm": 4,
    "batch_size": 4,
    "concurrency": 4,
    "draft_vocab_root": 1,
    "draft_vocab_k": 65536,
    "target_verifier_vocabulary": "full",
    "qwen_code_version": "0.19.4",
    "qwen_turn_tool_call_cap": qwen_derivation.DERIVED_CAP,
    "qwen_bundle_tree_sha256": floor_gate.FIXED32_QWEN_BUNDLE_TREE[
        "manifest_sha256"
    ],
    "physical_rows_per_request": 32,
    "sfwd_projection_rows": 128,
    "mandatory_weight_floor_ms": 102.479937172,
    "one_sided_u95_cap_ms": 117.8519277478,
    "physical32_work_census": {
        "schema": physical_work["schema"],
        "status": physical_work["status"],
        "required_batch_sizes": physical_work["required_batch_sizes"],
        "event_counts": physical_work["event_counts"],
        "normalized_work_signature_sha256": physical_work[
            "normalized_work_signature_sha256"
        ],
        "four_way_normalized_work_signatures": four_way_work_signatures,
        "tail_census_sha256": tail_census_sha256,
        "hydra_census_sha256": hydra_census_sha256,
    },
    "arms": arms,
}
if (
    summary["qwen_turn_tool_call_cap"] != 256
    or summary["qwen_bundle_tree_sha256"]
    != "594cac41e2d5ed505e0646f318b263ff70e200bcffe97326fe1c042fdc220516"
):
    raise SystemExit("pinned Qwen cap-256 bundle contract drifted")
out.write_text(json.dumps(summary, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n", encoding="ascii")
print(json.dumps(summary, sort_keys=True))
PY

printf 'campaign_root=%s\npaired_summary=%s\n' \
  "$campaign_root" "$campaign_root/paired_summary.json"
