#!/usr/bin/env bash
# Qualify and time the physical32 K64 B4 Tail23/Hydra27 all-parent + M128 stack.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

EXPECTED_BRANCH=${EXPECTED_BRANCH:-agent/fixed32-b4-tail23-hydra27-k64-m128}
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
  "$campaign_root/paired_summary.json" "$source_commit" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

paths = list(map(Path, sys.argv[1:7]))
out = Path(sys.argv[7])
source_commit = sys.argv[8]
expected = (
    ("tail6_fixed32", "Tail23", 23, "0x7a9ce7ff"),
    ("hydra27_fixed32", "Hydra27", 27, "0x7abdffff"),
)
arms = {}
for index, (mode, logical, active, mask) in enumerate(expected):
    taw_path, m128_path, timing_path = paths[index * 3:index * 3 + 3]
    taw = json.loads(taw_path.read_text(encoding="ascii"))
    m128 = json.loads(m128_path.read_text(encoding="ascii"))
    timing = json.loads(timing_path.read_text(encoding="ascii"))
    if (
        taw.get("status") != "pass"
        or taw.get("mode") != mode
        or taw.get("active_drafts") != active
        or taw.get("valid_mask") != mask
        or taw.get("probability_mismatches") != 0
        or taw.get("product_mismatches") != 0
        or m128.get("status") != "pass"
        or m128.get("topology") != mode
        or m128.get("mismatching_comparisons") != 0
        or m128.get("differing_bytes") != 0
        or m128.get("observed_m_values") != [128]
        or timing.get("status") != "complete"
        or timing.get("arm") != mode
        or timing.get("logical_topology") != logical
        or timing.get("active_drafts") != active
        or timing.get("valid_mask") != mask
        or timing.get("physical_rows_root_inclusive") != 32
        or timing.get("sfwd_projection_rows") != 128
        or timing.get("all_parent_production") is not True
        or timing.get("draft_vocab_root") != 1
        or timing.get("draft_vocab_k") != 65536
        or timing.get("mandatory_weight_bytes") != 32666638208
        or not math.isclose(timing.get("mandatory_weight_floor_ms", math.nan), 119.658015414, abs_tol=1e-9)
        or not math.isclose(timing.get("one_sided_u95_cap_ms", math.nan), 137.6067177261, abs_tol=1e-9)
    ):
        raise SystemExit(f"{logical} B4 stack evidence is incomplete")
    candidate = timing.get("candidate", {})
    for key in (
        "step_wall_ms", "measured_tps_fullstep_wall",
        "accepted_drafts_per_event", "committed_tokens_per_event",
        "sfwd_gpu_ms_per_step", "dfwd_gpu_ms_per_step",
        "cfwd_gpu_ms_per_step", "other_wall_ms_per_step",
        "step_wall_to_optimistic_floor_ratio",
    ):
        value = candidate.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SystemExit(f"{logical} candidate lacks finite {key}")
    arms[logical] = {
        "mode": mode,
        "active_drafts": active,
        "valid_mask": mask,
        "all_parent_gate_sha256": hashlib.sha256(taw_path.read_bytes()).hexdigest(),
        "m128_gate_sha256": hashlib.sha256(m128_path.read_bytes()).hexdigest(),
        "timing_summary_sha256": hashlib.sha256(timing_path.read_bytes()).hexdigest(),
        "stock_cutlass_with_all_parent": timing["stock_reference"],
        "persistent_m128_with_all_parent": candidate,
    }

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
    "physical_rows_per_request": 32,
    "sfwd_projection_rows": 128,
    "mandatory_weight_floor_ms": 119.658015414,
    "one_sided_u95_cap_ms": 137.6067177261,
    "arms": arms,
}
out.write_text(json.dumps(summary, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n", encoding="ascii")
print(json.dumps(summary, sort_keys=True))
PY

printf 'campaign_root=%s\npaired_summary=%s\n' \
  "$campaign_root" "$campaign_root/paired_summary.json"
