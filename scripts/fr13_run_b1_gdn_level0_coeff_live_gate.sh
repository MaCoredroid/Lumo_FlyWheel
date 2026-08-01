#!/usr/bin/env bash
# One real SWE-Verified K64 B1 graph-replay byte gate; stock remains served.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path below output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned FA2 shared object}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
TOPOLOGY=${TOPOLOGY:-hydra27_fixed32}
case "$TOPOLOGY" in
  tail6_fixed32)
    LOGICAL_DRAFTS=23
    VALID_MASK=0x7a9ce7ff
    ;;
  hydra27_fixed32)
    LOGICAL_DRAFTS=27
    VALID_MASK=0x7abdffff
    ;;
  *)
    echo "TOPOLOGY must be tail6_fixed32 or hydra27_fixed32" >&2
    exit 2
    ;;
esac
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
TASK_ID=astropy__astropy-12907
MANDATORY_WEIGHT_BYTES=32666638208
MANDATORY_WEIGHT_FLOOR_MS=119.658015414
ONE_SIDED_U95_CAP_MS=137.6067177261
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
SOURCE_COMMIT=$(git rev-parse HEAD)
SOURCE_SHA256=$(sha256sum src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py | awk '{print $1}')
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
FA2_SHA256=$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
ARM="${TOPOLOGY}_k64_gdn_level0_coeff_gate_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable" >&2; exit 2; }
[[ "$FORKED_FA2_SO" == /* && -f "$FORKED_FA2_SO" && ! -L "$FORKED_FA2_SO" ]] \
  || { echo "FORKED_FA2_SO must be an absolute regular file" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" ]] \
  || { echo "K64 gate inputs drifted" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the gate" >&2; exit 2; }

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
export FR13_FLOOR_ORDER=HT
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" ]] \
  || { echo "ROOT=1 K64 hardware-floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS/sidecars"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"
printf 'classification=one_real_swe_verified_k64_b1_byte_diagnostic\nacceptance_valid=0\ntiming_eligible=0\nfloor_acceptance_eligible=0\nreference_always_served=1\ncandidate_shadow_only=1\ncandidate=fixed32_gdn_level0_coeff_v1\ntopology=%s\nphysical_rows=32\nlogical_drafts=%s\nvalid_mask=%s\ntask_id=%s\ndraft_vocab_root=1\ndraft_vocab_k=65536\ndraft_vocab_blocks_sha256=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nsource=%s\nsource_sha256=%s\nrunner_sha256=%s\nsubset_sha256=%s\nfa2_sha256=%s\nstarted=%s\n' \
  "$TOPOLOGY" "$LOGICAL_DRAFTS" "$VALID_MASK" \
  "$TASK_ID" "$BLOCK_MAP_SHA256" "$MANDATORY_WEIGHT_BYTES" \
  "$MANDATORY_WEIGHT_FLOOR_MS" "$ONE_SIDED_U95_CAP_MS" \
  "$SOURCE_COMMIT" "$SOURCE_SHA256" "$RUNNER_SHA256" "$SUBSET_SHA256" \
  "$FA2_SHA256" "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"

finalize_manifests() {
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json"
  cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json"
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json"
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]]
}

if env \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
    FR13_FIXED32_B1_DIAGNOSTIC=1 \
    FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
    FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json \
    FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES" \
    FR13_WEIGHT_FLOOR_MS="$MANDATORY_WEIGHT_FLOOR_MS" \
    ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR10_METRICS=1 FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
    FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 \
    FR13_FIXED32_GDN_LEVEL0_COEFF=0 \
    FR13_FIXED32_GDN_LEVEL0_COEFF_BYTE_AB=1 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
    FR13_FIXED32_CUTLASS_WAVE=stock FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
    FORKED_FA2_SO="$FORKED_FA2_SO" RUNROOT="$RUNROOT_ABS" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" "$TOPOLOGY" "$SUBSET" \
      > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi
printf 'serve_rc=%s ended=%s\n' "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
finalize_manifests
(( serve_rc == 0 )) || exit "$serve_rc"

LIVE_PASS="$RUNROOT_ABS/$ARM/logs/fr13_fixed32_gdn_level0_coeff.live_pass.json"
"$PYTHON_BIN" scripts/fr13_gdn_level0_coeff_pass.py \
  --live-result "$LIVE_PASS" \
  --kernel-source src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py \
  --expected-task-id "$TASK_ID" \
  --expected-mode "$TOPOLOGY" \
  > "$RUNROOT_ABS/live_pass_validation.json"
LIVE_PASS_SHA256=$(sha256sum "$LIVE_PASS" | awk '{print $1}')
"$PYTHON_BIN" - "$LIVE_PASS" "$RUNROOT_ABS/gate_summary.json" \
  "$SOURCE_COMMIT" "$RUNNER_SHA256" "$SUBSET_SHA256" \
  "$BLOCK_MAP_SHA256" "$FA2_SHA256" "$LIVE_PASS_SHA256" <<'PY'
import json
import sys
from pathlib import Path

live_path, out_path = map(Path, sys.argv[1:3])
source, runner_sha, subset_sha, block_sha, fa2_sha, live_sha = sys.argv[3:]
live = json.loads(live_path.read_text(encoding="ascii"))
summary = {
    "schema": "fr13.fixed32.gdn_level0_coeff.b1_gate.v1",
    "status": "pass",
    "run_classification": "one_real_swe_verified_k64_b1_byte_diagnostic",
    "acceptance_valid": False,
    "timing_eligible": False,
    "reference_served": True,
    "candidate_shadow_only": True,
    "task_id": "astropy__astropy-12907",
    "topology": live["mode"],
    "batch_size": 1,
    "physical_rows": 32,
    "draft_vocab_root": 1,
    "draft_vocab_k": 65536,
    "source_commit": source,
    "runner_sha256": runner_sha,
    "subset_sha256": subset_sha,
    "block_map_sha256": block_sha,
    "fa2_sha256": fa2_sha,
    "live_pass_sha256": live_sha,
    "records": live["records"],
    "compared_bytes": live["compared_bytes"],
    "surfaces": live["surfaces"],
    "scratch_rows": [live["scratch_row_start"]],
    "raw_byte_equal": live["raw_byte_equal"],
    "state_restored": live["state_restored"],
}
out_path.write_text(
    json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
PY
