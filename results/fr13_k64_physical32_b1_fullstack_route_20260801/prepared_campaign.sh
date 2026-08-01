#!/usr/bin/env bash
# Run reviewed B4 prerequisites through fresh TAW/GDN B1 gates and paired timing.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO"

: "${TAG:?set TAG to a unique campaign tag}"
: "${STOCK_FA2_SO:?set the exact-safe stock FA2 binary}"
: "${QROW16_FA2_SO:?set the pinned qrow16 binary}"
: "${TAIL23_REVIEWED_B4_TAW_PASS:?set the corrected Tail23 B4 pass}"
: "${TAIL23_REVIEWED_B4_TAW_VERDICT:?set the corrected Tail23 B4 verdict}"
: "${HYDRA27_REVIEWED_B4_TAW_PASS:?set the corrected Hydra27 B4 pass}"
: "${HYDRA27_REVIEWED_B4_TAW_VERDICT:?set the corrected Hydra27 B4 verdict}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
TAW_SOURCE=scripts/fr13_device_multidraft_kernel.py
CAMPAIGN_ROOT="$REPO/output/fr13_k64_physical32_b1_fullstack_${TAG}"
[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ ! -e "$CAMPAIGN_ROOT" && ! -L "$CAMPAIGN_ROOT" ]] \
  || { echo "campaign root must be new: $CAMPAIGN_ROOT" >&2; exit 2; }
for input in \
  "$STOCK_FA2_SO" "$QROW16_FA2_SO" \
  "$TAIL23_REVIEWED_B4_TAW_PASS" "$TAIL23_REVIEWED_B4_TAW_VERDICT" \
  "$HYDRA27_REVIEWED_B4_TAW_PASS" "$HYDRA27_REVIEWED_B4_TAW_VERDICT"; do
  [[ "$input" == /* && -f "$input" && ! -L "$input" ]] \
    || { echo "campaign input must be an absolute regular file: $input" >&2; exit 2; }
done
unset input
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }

"$PYTHON_BIN" scripts/fr13_taw_b1_credential.py validate-reviewed-b4 \
  --mode tail6_fixed32 \
  --source "$TAW_SOURCE" \
  --production-pass "$TAIL23_REVIEWED_B4_TAW_PASS" \
  --gate-verdict "$TAIL23_REVIEWED_B4_TAW_VERDICT" \
  >/dev/null
"$PYTHON_BIN" scripts/fr13_taw_b1_credential.py validate-reviewed-b4 \
  --mode hydra27_fixed32 \
  --source "$TAW_SOURCE" \
  --production-pass "$HYDRA27_REVIEWED_B4_TAW_PASS" \
  --gate-verdict "$HYDRA27_REVIEWED_B4_TAW_VERDICT" \
  >/dev/null
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the campaign" >&2; exit 2; }

mkdir -p "$CAMPAIGN_ROOT"
SOURCE_COMMIT=$(git rev-parse HEAD)

run_mode() {
  local mode=$1
  local slug=$2
  local reviewed_b4_pass=$3
  local reviewed_b4_verdict=$4
  local gate_tag="${slug}_b1_gate_${TAG}"
  local gate_root="$CAMPAIGN_ROOT/${slug}_b1_gate"
  local gate_arm="$gate_root/${mode}_taw_source_v7_b1_gate_${gate_tag}"
  local credential="$gate_arm/taw_source_v7_b1_credential.json"
  local b1_live="$gate_arm/taw_source_v7_b1_live_bundle.json"
  local merged_pass="$gate_arm/${slug}_taw_source_v7_merged_production_pass.json"
  local merge_binding="$gate_arm/${slug}_taw_source_v7_merge_binding.json"
  local gdn_gate_tag="${slug}_gdn_b1_gate_${TAG}"
  local gdn_gate_root="$CAMPAIGN_ROOT/${slug}_gdn_b1_gate"
  local gdn_gate_arm="$gdn_gate_root/${mode}_k64_gdn_level0_coeff_gate_${gdn_gate_tag}"
  local gdn_live_pass="$gdn_gate_arm/logs/fr13_fixed32_gdn_level0_coeff.live_pass.json"
  local gdn_gate_summary="$gdn_gate_root/gate_summary.json"

  MODE="$mode" RUNROOT="$gate_root" TAG="$gate_tag" \
  STOCK_FA2_SO="$STOCK_FA2_SO" \
    bash scripts/fr13_run_b1_k64_taw_source_v7_gate.sh
  [[ -f "$credential" && ! -L "$credential" \
     && -f "$b1_live" && ! -L "$b1_live" ]] \
    || { echo "$mode B1 graph credential was not issued" >&2; return 4; }

  "$PYTHON_BIN" scripts/fr13_taw_b1_credential.py merge \
    --mode "$mode" \
    --source "$TAW_SOURCE" \
    --credential "$credential" \
    --b1-live-bundle "$b1_live" \
    --b4-production-pass "$reviewed_b4_pass" \
    --b4-gate-verdict "$reviewed_b4_verdict" \
    --out "$merged_pass" \
    --binding-out "$merge_binding" \
    > "$gate_arm/${slug}_taw_source_v7_merge_validation.json"

  TOPOLOGY="$mode" RUNROOT="$gdn_gate_root" TAG="$gdn_gate_tag" \
  FORKED_FA2_SO="$STOCK_FA2_SO" \
    bash scripts/fr13_run_b1_gdn_level0_coeff_live_gate.sh
  [[ -f "$gdn_live_pass" && ! -L "$gdn_live_pass" \
     && -f "$gdn_gate_summary" && ! -L "$gdn_gate_summary" ]] \
    || { echo "$mode GDN coefficient B1 gate was not issued" >&2; return 4; }

  local pair_root="$CAMPAIGN_ROOT/${slug}_b1_pair"
  local pair_tag="${slug}_b1_pair_${TAG}"
  MODE="$mode" RUNROOT="$pair_root" TAG="$pair_tag" \
  QROW16_FA2_SO="$QROW16_FA2_SO" \
  TAW_B1_CREDENTIAL="$credential" \
  TAW_B1_CREDENTIAL_SHA256="$(sha256sum "$credential" | awk '{print $1}')" \
  TAW_B1_LIVE_BUNDLE="$b1_live" \
  TAW_B1_LIVE_BUNDLE_SHA256="$(sha256sum "$b1_live" | awk '{print $1}')" \
  TAW_REVIEWED_B4_PASS="$reviewed_b4_pass" \
  TAW_REVIEWED_B4_PASS_SHA256="$(sha256sum "$reviewed_b4_pass" | awk '{print $1}')" \
  TAW_REVIEWED_B4_VERDICT="$reviewed_b4_verdict" \
  TAW_REVIEWED_B4_VERDICT_SHA256="$(sha256sum "$reviewed_b4_verdict" | awk '{print $1}')" \
  TAW_MERGE_BINDING="$merge_binding" \
  TAW_MERGE_BINDING_SHA256="$(sha256sum "$merge_binding" | awk '{print $1}')" \
  TAW_PRODUCTION_PASS="$merged_pass" \
  TAW_PRODUCTION_PASS_SHA256="$(sha256sum "$merged_pass" | awk '{print $1}')" \
  GDN_LEVEL0_COEFF_LIVE_PASS="$gdn_live_pass" \
  GDN_LEVEL0_COEFF_LIVE_PASS_SHA256="$(sha256sum "$gdn_live_pass" | awk '{print $1}')" \
  GDN_LEVEL0_COEFF_GATE_SUMMARY="$gdn_gate_summary" \
  GDN_LEVEL0_COEFF_GATE_SUMMARY_SHA256="$(sha256sum "$gdn_gate_summary" | awk '{print $1}')" \
    bash scripts/fr13_run_b1_k64_physical32_fullstack_pair.sh
  [[ -f "$pair_root/timing_summary.json" \
     && ! -L "$pair_root/timing_summary.json" ]] \
    || { echo "$mode paired B1 summary was not issued" >&2; return 4; }
}

run_mode \
  tail6_fixed32 tail23 \
  "$TAIL23_REVIEWED_B4_TAW_PASS" "$TAIL23_REVIEWED_B4_TAW_VERDICT"
run_mode \
  hydra27_fixed32 hydra27 \
  "$HYDRA27_REVIEWED_B4_TAW_PASS" "$HYDRA27_REVIEWED_B4_TAW_VERDICT"

"$PYTHON_BIN" - \
  "$CAMPAIGN_ROOT/tail23_b1_pair/timing_summary.json" \
  "$CAMPAIGN_ROOT/hydra27_b1_pair/timing_summary.json" \
  "$CAMPAIGN_ROOT/paired_summary.json" "$SOURCE_COMMIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

tail_path, hydra_path, output_path = map(Path, sys.argv[1:4])
source_commit = sys.argv[4]
arms = {}
for path, mode, topology in (
    (tail_path, "tail6_fixed32", "Tail23"),
    (hydra_path, "hydra27_fixed32", "Hydra27"),
):
    raw = path.read_bytes()
    payload = json.loads(raw.decode("ascii"))
    gdn = payload.get("gdn_level0_coeff", {})
    if (
        payload.get("schema")
        != "fr13.fixed32.k64_physical32_fullstack.b1_pair.v1"
        or payload.get("status") != "complete"
        or payload.get("mode") != mode
        or payload.get("logical_topology") != topology
        or payload.get("source_commit") != source_commit
        or payload.get("qrow16_production") is not True
        or payload.get("sfwd_state_fusion_production") is not True
        or payload.get("candidate_all_parent_committer_production") is not True
        or payload.get("only_arm_delta")
        != "source_v7_all_parent_committer_production_0_to_1"
        or payload.get("formal_floor_acceptance_eligible") is not False
        or gdn.get("schema")
        != "fr13.fixed32.gdn_level0_coeff.fullstack_binding.v1"
        or gdn.get("status") != "bound"
        or gdn.get("candidate") != "fixed32_gdn_level0_coeff_v1"
        or gdn.get("mode") != mode
        or gdn.get("qualified_batches") != [1]
        or gdn.get("count_invocation") is not False
        or gdn.get("compared_bytes") != 4725178944
        or gdn.get("surfaces")
        != [
            "output",
            "export_non_scratch_rows",
            "ring_k",
            "ring_v",
            "ring_a",
            "ring_b",
            "flags",
            "counter",
        ]
        or gdn.get("b4_live_qualified") is not False
        or gdn.get("b4_deployable") is not False
        or gdn.get("b4_evidence_classification") != "static_only"
    ):
        raise SystemExit(f"{topology} B1 paired summary is incomplete")
    arms[topology] = {
        "summary_sha256": hashlib.sha256(raw).hexdigest(),
        "mode": mode,
        "stock": payload["stock"],
        "candidate": payload["candidate"],
        "candidate_minus_stock": payload["candidate_minus_stock"],
        "candidate_to_stock_full_wall_tps_ratio": payload[
            "candidate_to_stock_full_wall_tps_ratio"
        ],
        "gdn_level0_coeff": gdn,
    }

summary = {
    "schema": "fr13.fixed32.k64_physical32_fullstack.b1_tail23_hydra27_pair.v1",
    "status": "complete",
    "run_classification": "real_swe_verified_exact4_k64_b1_fullstack_pair",
    "source_commit": source_commit,
    "task_count_per_arm": 4,
    "batch_size": 1,
    "concurrency": 1,
    "draft_vocab_root": 1,
    "draft_vocab_k": 65536,
    "physical_rows_root_inclusive": 32,
    "mandatory_weight_floor_ms": 119.658015414,
    "one_sided_u95_cap_ms": 137.6067177261,
    "formal_floor_acceptance_eligible": False,
    "arms": arms,
}
output_path.write_text(
    json.dumps(summary, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    + "\n",
    encoding="ascii",
)
print(json.dumps(summary, sort_keys=True))
PY

printf 'campaign_root=%s\npaired_summary=%s\n' \
  "$CAMPAIGN_ROOT" "$CAMPAIGN_ROOT/paired_summary.json"
