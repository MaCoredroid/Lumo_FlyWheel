#!/usr/bin/env bash
# Exact4 real SWE-Verified B4 full-wall timing: stock CUTLASS vs a qualified candidate.
# This paired screen is not the formal statistical hardware-floor acceptance gate.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${STOCK_FA2_SO:?set STOCK_FA2_SO to the exact-safe stock FA2 binary}"
: "${CUTLASS_B4_SO:?set CUTLASS_B4_SO to the pinned candidate binary}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
QUALIFICATION_PROFILE=${CUTLASS_B4_QUALIFICATION_PROFILE:-full_vocab}
FIXED32_MODE=${CUTLASS_B4_FIXED32_MODE:-hydra27_fixed32}
CANDIDATE_SELECTOR=${CUTLASS_B4_CANDIDATE_SELECTOR:-persistent_b4_m128}
CUTLASS_B4_PASS_JSON=${CUTLASS_B4_PASS_JSON:-}
CUTLASS_B4_PASS_SHA256=${CUTLASS_B4_PASS_SHA256:-}
CUTLASS_B4_TAIL23_PASS_JSON=${CUTLASS_B4_TAIL23_PASS_JSON:-}
CUTLASS_B4_TAIL23_PASS_SHA256=${CUTLASS_B4_TAIL23_PASS_SHA256:-}
CUTLASS_B4_HYDRA27_PASS_JSON=${CUTLASS_B4_HYDRA27_PASS_JSON:-}
CUTLASS_B4_HYDRA27_PASS_SHA256=${CUTLASS_B4_HYDRA27_PASS_SHA256:-}
ALL_PARENT_PASS_JSON=${FR13_FIXED32_ALL_PARENT_PASS_JSON:-}
ALL_PARENT_VERDICT_JSON=${FR13_FIXED32_ALL_PARENT_VERDICT_JSON:-}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
PATCH_SOURCE=scripts/fr13_patch_cutlass_fixed32_wave.py
DRAFT_VOCAB_BLOCKS_HOST=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
DUAL_CANDIDATE=0
SOURCE_BOUND_CANDIDATE=0
case "$CANDIDATE_SELECTOR" in
  persistent_b4_m128)
    CANDIDATE_SHA256=895495fe82cb0e0278d3b0a39b8e57e1281aa73a10bbba01a94085733c81d64f
    CANDIDATE_BYTES=112698512
    QUALIFIED_PATCH_SOURCE_SHA256=656c53b20497fc08cc7fdfb18256235b07cfad9868fde2faa70e6b0b9dfca41a
    CANDIDATE_ARM_SLUG=persistent_m128
    ONLY_ARM_DELTA_META=only_arm_delta=CUTLASS_stock_to_persistent_b4_m128
    ONLY_ARM_DELTA=${ONLY_ARM_DELTA_META#*=}
    ;;
  identity_stockshape_stage2_b4)
    [[ "$QUALIFICATION_PROFILE" == "k64_root" ]] || {
      echo "Stage2 production timing requires CUTLASS_B4_QUALIFICATION_PROFILE=k64_root" >&2
      exit 2
    }
    CANDIDATE_SHA256=c5da32258e678494cd2b6b34da0b2aa96e70096b215db0938ed1e0750aa43d29
    CANDIDATE_BYTES=117488608
    QUALIFIED_PATCH_SOURCE_SHA256=841b5dfc0741aa5051037e3eda005e5a65f7a425b76235a61e8a0779bd31a155
    CANDIDATE_ARM_SLUG=identity_stockshape_stage2
    ONLY_ARM_DELTA_META=only_arm_delta=CUTLASS_stock_to_identity_stockshape_stage2_b4
    ONLY_ARM_DELTA=${ONLY_ARM_DELTA_META#*=}
    DUAL_CANDIDATE=1
    ;;
  identity_twom_b4)
    [[ "$QUALIFICATION_PROFILE" == "k64_root" ]] || {
      echo "two-M production timing requires CUTLASS_B4_QUALIFICATION_PROFILE=k64_root" >&2
      exit 2
    }
    CANDIDATE_SHA256=c5da32258e678494cd2b6b34da0b2aa96e70096b215db0938ed1e0750aa43d29
    CANDIDATE_BYTES=117488608
    QUALIFIED_PATCH_SOURCE_SHA256=841b5dfc0741aa5051037e3eda005e5a65f7a425b76235a61e8a0779bd31a155
    CANDIDATE_ARM_SLUG=identity_twom
    ONLY_ARM_DELTA_META=only_arm_delta=CUTLASS_stock_to_identity_twom_b4
    ONLY_ARM_DELTA=${ONLY_ARM_DELTA_META#*=}
    DUAL_CANDIDATE=1
    ;;
  identity_hybrid_n5120_b4)
    [[ "$QUALIFICATION_PROFILE" == "k64_root" ]] || {
      echo "hybrid N5120 production timing requires CUTLASS_B4_QUALIFICATION_PROFILE=k64_root" >&2
      exit 2
    }
    CANDIDATE_SHA256=65250ccb46057e4726f68b6056eab3e46f71a1bee2ce25eca306d4d889a66ecc
    CANDIDATE_BYTES=119471552
    QUALIFIED_PATCH_SOURCE_SHA256=b9dba4077f63425dd4c245d9de33a8b2413e223f44a7fb0f6b4abe6e003d24a0
    CANDIDATE_ARM_SLUG=identity_hybrid_n5120
    ONLY_ARM_DELTA_META=only_arm_delta=CUTLASS_stock_to_identity_hybrid_n5120_b4
    ONLY_ARM_DELTA=${ONLY_ARM_DELTA_META#*=}
    DUAL_CANDIDATE=1
    SOURCE_BOUND_CANDIDATE=1
    ;;
  *)
    echo "CUTLASS_B4_CANDIDATE_SELECTOR is unsupported for B4 timing" >&2
    exit 2
    ;;
esac
case "$QUALIFICATION_PROFILE" in
  full_vocab)
    QUALIFICATION_SOURCE_COMMIT=${CUTLASS_B4_QUALIFICATION_SOURCE_COMMIT:-0f2a31ed298758cba72fad7e77fc3e13e27d545a}
    LAUNCH_CLASSIFICATION=real_swe_verified_exact4_b4_timing_candidate
    RUN_CLASSIFICATION=real_swe_verified_exact4_b4_timing
    SUMMARY_SCHEMA=fr13.fixed32.cutlass_persistent_b4_m128.full_wall_timing_pair.v1
    BINDING_SCHEMA=fr13.fixed32.cutlass_b4.production_binding.v1
    DRAFT_VOCAB_ROOT=0
    DRAFT_VOCAB_K=0
    NEEDS_ALLOW=FR13_DRAFT_VOCAB_K=0
    MANDATORY_WEIGHT_BYTES=42025179008
    MANDATORY_WEIGHT_FLOOR_MS=153.9383846446886
    ONE_SIDED_U95_CAP_MS=177.0291423413919
    ARM_PROFILE_SUFFIX=
    ;;
  k64_root)
    : "${CUTLASS_B4_QUALIFICATION_SOURCE_COMMIT:?set it to the K64 live-gate source commit}"
    QUALIFICATION_SOURCE_COMMIT=$CUTLASS_B4_QUALIFICATION_SOURCE_COMMIT
    LAUNCH_CLASSIFICATION=real_swe_verified_exact4_b4_k64_root_timing_candidate
    RUN_CLASSIFICATION=real_swe_verified_exact4_b4_k64_root_timing
    if [[ "$CANDIDATE_SELECTOR" == "identity_stockshape_stage2_b4" ]]; then
      SUMMARY_SCHEMA=fr13.fixed32.cutlass_identity_stockshape_stage2_b4.k64_root.dual_topology.full_wall_timing_pair.v1
      BINDING_SCHEMA=fr13.fixed32.cutlass_b4.identity_stockshape_stage2.k64_root.dual_topology.production_binding.v1
    elif [[ "$CANDIDATE_SELECTOR" == "identity_twom_b4" ]]; then
      SUMMARY_SCHEMA=fr13.fixed32.cutlass_identity_twom_b4.k64_root.dual_topology.full_wall_timing_pair.v1
      BINDING_SCHEMA=fr13.fixed32.cutlass_b4.identity_twom.k64_root.dual_topology.production_binding.v1
    elif [[ "$CANDIDATE_SELECTOR" == "identity_hybrid_n5120_b4" ]]; then
      SUMMARY_SCHEMA=fr13.fixed32.cutlass_identity_hybrid_n5120_b4.k64_root.dual_topology.full_wall_timing_pair.v1
      BINDING_SCHEMA=fr13.fixed32.cutlass_b4.identity_hybrid_n5120.k64_root.dual_topology.production_binding.v1
    else
      SUMMARY_SCHEMA=fr13.fixed32.cutlass_persistent_b4_m128.k64_root.full_wall_timing_pair.v1
      BINDING_SCHEMA=fr13.fixed32.cutlass_b4.k64_root.production_binding.v1
    fi
    DRAFT_VOCAB_ROOT=1
    DRAFT_VOCAB_K=65536
    NEEDS_ALLOW=
    MANDATORY_WEIGHT_BYTES=32666638208
    MANDATORY_WEIGHT_FLOOR_MS=119.658015414
    ONE_SIDED_U95_CAP_MS=137.6067177261
    ARM_PROFILE_SUFFIX=_k64_root
    ;;
  *)
    echo "CUTLASS_B4_QUALIFICATION_PROFILE must be full_vocab or k64_root" >&2
    exit 2
    ;;
esac
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
TIMING_HARNESS_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
B4_KV_CACHE_MEMORY_BYTES=49392123904
case "$FIXED32_MODE" in
  tail6_fixed32)
    LOGICAL_TOPOLOGY=Tail23
    ACTIVE_DRAFTS=23
    VALID_MASK=0x7a9ce7ff
    ;;
  hydra27_fixed32)
    LOGICAL_TOPOLOGY=Hydra27
    ACTIVE_DRAFTS=27
    VALID_MASK=0x7abdffff
    ;;
  *)
    echo "CUTLASS_B4_FIXED32_MODE must be tail6_fixed32 or hydra27_fixed32" >&2
    exit 2
    ;;
esac
TIMING_KIND=$FIXED32_MODE
STOCK_ARM="${FIXED32_MODE}_cutlass_stock_b4${ARM_PROFILE_SUFFIX}_${TAG}"
CANDIDATE_ARM="${FIXED32_MODE}_cutlass_${CANDIDATE_ARM_SLUG}${ARM_PROFILE_SUFFIX}_${TAG}"
ALL_PARENT_PRODUCTION=0
ALL_PARENT_PASS_SHA256=
ALL_PARENT_VERDICT_SHA256=
# The hot-path TAW commit route is a function of the committer this run selects.
# Without an all-parent credential both arms run the reference committer, so the
# work census records the reference route on every event.
WORK_CENSUS_ROUTE=fixed32_pytorch_exact_float_triton_integer_commit

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for input in "$STOCK_FA2_SO" "$CUTLASS_B4_SO"; do
  [[ "$input" == /* && -f "$input" && ! -L "$input" ]] \
    || { echo "timing input must be an absolute regular non-symlink file: $input" >&2; exit 2; }
done
if (( DUAL_CANDIDATE == 1 )); then
  [[ -z "$CUTLASS_B4_PASS_JSON" && -z "$CUTLASS_B4_PASS_SHA256" ]] || {
    echo "dual-topology timing forbids the single-topology CUTLASS PASS input" >&2
    exit 2
  }
  for input in "$CUTLASS_B4_TAIL23_PASS_JSON" "$CUTLASS_B4_HYDRA27_PASS_JSON"; do
    [[ "$input" == /* && -f "$input" && ! -L "$input" ]] \
      || { echo "dual-topology timing requires absolute regular Tail23 and Hydra27 PASS files" >&2; exit 2; }
  done
else
  [[ -z "$CUTLASS_B4_TAIL23_PASS_JSON" \
     && -z "$CUTLASS_B4_TAIL23_PASS_SHA256" \
     && -z "$CUTLASS_B4_HYDRA27_PASS_JSON" \
     && -z "$CUTLASS_B4_HYDRA27_PASS_SHA256" \
     && "$CUTLASS_B4_PASS_JSON" == /* \
     && -f "$CUTLASS_B4_PASS_JSON" \
     && ! -L "$CUTLASS_B4_PASS_JSON" ]] || {
    echo "persistent-M128 timing requires only one absolute regular live PASS" >&2
    exit 2
  }
fi
if [[ -n "$ALL_PARENT_PASS_JSON" || -n "$ALL_PARENT_VERDICT_JSON" ]]; then
  [[ "$QUALIFICATION_PROFILE" == "k64_root" ]] \
    || { echo "all-parent exact4 credential is scoped to k64_root" >&2; exit 2; }
  [[ "$ALL_PARENT_PASS_JSON" == /* && -f "$ALL_PARENT_PASS_JSON" \
     && ! -L "$ALL_PARENT_PASS_JSON" ]] \
    || { echo "all-parent pass must be an absolute regular non-symlink file" >&2; exit 2; }
  [[ "$ALL_PARENT_VERDICT_JSON" == /* && -f "$ALL_PARENT_VERDICT_JSON" \
     && ! -L "$ALL_PARENT_VERDICT_JSON" ]] \
    || { echo "all-parent verdict must be an absolute regular non-symlink file" >&2; exit 2; }
  "$PYTHON_BIN" - \
    "$ALL_PARENT_PASS_JSON" "$ALL_PARENT_VERDICT_JSON" "$FIXED32_MODE" \
    "$LOGICAL_TOPOLOGY" "$ACTIVE_DRAFTS" "$VALID_MASK" \
    "$TIMING_HARNESS_COMMIT" "$SUBSET_SHA256" "$DRAFT_VOCAB_BLOCKS_SHA256" \
    "$STOCK_FA2_SHA256" <<'PY'
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

source = Path("scripts/fr13_device_multidraft_kernel.py")
spec = importlib.util.spec_from_file_location("fr13_b4_pair_taw_pass", source)
if spec is None or spec.loader is None:
    raise SystemExit("cannot import all-parent credential validator")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bundle = module._fr13_fixed32_taw_native_production_pass(
    path=sys.argv[1], expected_mode=sys.argv[3], expected_batch=4,
)
if bundle.get("qualified_batches") != [1, 2, 3, 4]:
    raise SystemExit("all-parent credential lacks independent B1/B2/B3/B4 closure")
pass_sha256 = hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
verdict = json.loads(Path(sys.argv[2]).read_text(encoding="ascii"))
mode, logical_topology = sys.argv[3:5]
active_drafts = int(sys.argv[5])
valid_mask = int(sys.argv[6], 0)
source_commit, subset_sha256, block_map_sha256, stock_fa2_sha256 = sys.argv[7:11]
expected_tasks = [
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
]
expected_schema = {
    "tail6_fixed32": "fr13.fixed32.tail23_all_parent.exact4_b4_live_gate.v1",
    "hydra27_fixed32": "fr13.fixed32.hydra27_all_parent.exact4_b4_live_gate.v1",
}[mode]
sha_fields = (
    "campaign_proof_sha256",
    "runtime_manifest_sha256",
    "gate_runner_sha256",
)
if (
    verdict.get("schema") != expected_schema
    or verdict.get("status") != "pass"
    or verdict.get("run_classification")
    != "real_swe_verified_exact4_b4_byte_diagnostic"
    or verdict.get("acceptance_valid") is not False
    or verdict.get("timing_eligible") is not False
    or verdict.get("floor_acceptance_eligible") is not False
    or verdict.get("reference_always_served") is not True
    or verdict.get("candidate_returned") is not False
    or verdict.get("production_default_enabled") is not False
    or verdict.get("raw_prompt_response_published") is not False
    or verdict.get("candidate") != "fixed32_all_parent_commit_v2"
    or verdict.get("source_commit") != source_commit
    or verdict.get("source_contract_schema")
    != module._FR13_FIXED32_TAW_SOURCE_SCHEMA
    or verdict.get("source_contract_sha256")
    != module._FR13_FIXED32_TAW_SOURCE_SHA256
    or verdict.get("source_file_sha256")
    != hashlib.sha256(source.read_bytes()).hexdigest()
    or verdict.get("mode") != mode
    or verdict.get("logical_topology") != logical_topology
    or verdict.get("active_drafts") != active_drafts
    or verdict.get("valid_mask") != hex(valid_mask)
    or verdict.get("physical_drafts") != 31
    or verdict.get("physical_rows_root_inclusive") != 32
    or verdict.get("qualified_batches") != [1, 2, 3, 4]
    or verdict.get("required_production_batches") != [1, 4]
    or verdict.get("independent_b1_record") is not True
    or verdict.get("independent_b4_record") is not True
    or verdict.get("draft_vocab_k") != 65536
    or verdict.get("draft_vocab_root") != 1
    or verdict.get("draft_vocab_blocks")
    != "/workspace/scripts/fr13_dvk_subset_blocks.json"
    or verdict.get("draft_vocab_blocks_sha256") != block_map_sha256
    or verdict.get("mandatory_weight_bytes") != 32666638208
    or verdict.get("mandatory_weight_floor_ms") != 119.658015414
    or verdict.get("one_sided_u95_cap_ms") != 137.6067177261
    or verdict.get("subset_sha256") != subset_sha256
    or verdict.get("task_ids") != expected_tasks
    or verdict.get("task_marker") != f"swe_verified:campaign4_{subset_sha256}"
    or verdict.get("stock_fa2_sha256") != stock_fa2_sha256
    or verdict.get("live_bundle_sha256") != pass_sha256
    or verdict.get("production_bundle_sha256") != pass_sha256
    or verdict.get("probability_mismatches") != 0
    or verdict.get("product_mismatches") != 0
    or any(
        re.fullmatch(r"[0-9a-f]{64}", str(verdict.get(field, ""))) is None
        for field in sha_fields
    )
):
    raise SystemExit(
        "all-parent credential is not bound to this source and canonical exact4 verdict"
    )
PY
  ALL_PARENT_PRODUCTION=1
  ALL_PARENT_PASS_SHA256=$(sha256sum "$ALL_PARENT_PASS_JSON" | awk '{print $1}')
  ALL_PARENT_VERDICT_SHA256=$(sha256sum "$ALL_PARENT_VERDICT_JSON" | awk '{print $1}')
  WORK_CENSUS_ROUTE=fixed32_native_precompute_production_candidate_return
fi
[[ "$(stat -c '%s' "$STOCK_FA2_SO")" == "$STOCK_FA2_BYTES" \
   && "$(sha256sum "$STOCK_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "STOCK_FA2_SO is not the exact-safe stock reference" >&2; exit 2; }
[[ "$(stat -c '%s' "$CUTLASS_B4_SO")" == "$CANDIDATE_BYTES" \
   && "$(sha256sum "$CUTLASS_B4_SO" | awk '{print $1}')" == "$CANDIDATE_SHA256" ]] \
  || { echo "CUTLASS_B4_SO is not the pinned selected candidate" >&2; exit 2; }
[[ "$QUALIFICATION_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "CUTLASS B4 qualification source commit is invalid" >&2; exit 2; }
CURRENT_PATCH_SOURCE_SHA256=$(sha256sum "$PATCH_SOURCE" | awk '{print $1}')
QUALIFICATION_PATCH_SOURCE_SHA256=$(
  git show "${QUALIFICATION_SOURCE_COMMIT}:${PATCH_SOURCE}" | sha256sum | awk '{print $1}'
)
[[ "$CURRENT_PATCH_SOURCE_SHA256" == "$QUALIFIED_PATCH_SOURCE_SHA256" \
   && "$QUALIFICATION_PATCH_SOURCE_SHA256" == "$QUALIFIED_PATCH_SOURCE_SHA256" ]] \
  || { echo "qualified CUTLASS patch source changed after the live gate" >&2; exit 2; }
if (( SOURCE_BOUND_CANDIDATE == 1 )); then
  [[ "$QUALIFICATION_SOURCE_COMMIT" == "$TIMING_HARNESS_COMMIT" ]] \
    || { echo "source-bound timing requires the qualification commit to equal HEAD" >&2; exit 2; }
else
  git merge-base --is-ancestor "$QUALIFICATION_SOURCE_COMMIT" "$TIMING_HARNESS_COMMIT" \
    || { echo "timing harness does not descend from the qualification source" >&2; exit 2; }
fi
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical exact4 subset SHA-256 drift" >&2; exit 2; }
if [[ "$QUALIFICATION_PROFILE" == "k64_root" ]]; then
  [[ -f "$DRAFT_VOCAB_BLOCKS_HOST" && ! -L "$DRAFT_VOCAB_BLOCKS_HOST" \
     && "$(sha256sum "$DRAFT_VOCAB_BLOCKS_HOST" | awk '{print $1}')" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
    || { echo "pinned root-64K draft-vocabulary block map drifted" >&2; exit 2; }
fi
if (( DUAL_CANDIDATE == 1 )); then
  [[ "$CUTLASS_B4_TAIL23_PASS_SHA256" =~ ^[0-9a-f]{64}$ \
     && "$CUTLASS_B4_HYDRA27_PASS_SHA256" =~ ^[0-9a-f]{64}$ \
     && "$(sha256sum "$CUTLASS_B4_TAIL23_PASS_JSON" | awk '{print $1}')" == "$CUTLASS_B4_TAIL23_PASS_SHA256" \
     && "$(sha256sum "$CUTLASS_B4_HYDRA27_PASS_JSON" | awk '{print $1}')" == "$CUTLASS_B4_HYDRA27_PASS_SHA256" ]] \
    || { echo "CUTLASS dual-topology live PASS identity mismatch" >&2; exit 2; }
else
  [[ "$CUTLASS_B4_PASS_SHA256" =~ ^[0-9a-f]{64}$ \
     && "$(sha256sum "$CUTLASS_B4_PASS_JSON" | awk '{print $1}')" == "$CUTLASS_B4_PASS_SHA256" ]] \
    || { echo "CUTLASS B4 live PASS identity mismatch" >&2; exit 2; }
fi
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }

if (( DUAL_CANDIDATE == 1 )); then
  LIVE_PASS_BINDING_JSON=$("$PYTHON_BIN" scripts/fr13_cutlass_b4_pass.py dual-validate \
    --tail23-live-result "$CUTLASS_B4_TAIL23_PASS_JSON" \
    --expected-tail23-live-sha256 "$CUTLASS_B4_TAIL23_PASS_SHA256" \
    --hydra27-live-result "$CUTLASS_B4_HYDRA27_PASS_JSON" \
    --expected-hydra27-live-sha256 "$CUTLASS_B4_HYDRA27_PASS_SHA256" \
    --candidate-so "$CUTLASS_B4_SO" --patch-source "$PATCH_SOURCE" \
    --expected-source-commit "$QUALIFICATION_SOURCE_COMMIT" \
    --candidate-selector "$CANDIDATE_SELECTOR" \
    --draft-vocab-blocks "$DRAFT_VOCAB_BLOCKS_HOST")
else
  LIVE_PASS_BINDING_JSON=$("$PYTHON_BIN" scripts/fr13_cutlass_b4_pass.py validate \
    --live-result "$CUTLASS_B4_PASS_JSON" \
    --expected-live-sha256 "$CUTLASS_B4_PASS_SHA256" \
    --candidate-so "$CUTLASS_B4_SO" --patch-source "$PATCH_SOURCE" \
    --expected-source-commit "$QUALIFICATION_SOURCE_COMMIT" \
    --candidate-selector "$CANDIDATE_SELECTOR" \
    --qualification-profile "$QUALIFICATION_PROFILE" \
    --fixed32-mode "$FIXED32_MODE")
fi
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT"
export FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K"
export FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER"
export FR13_NEEDS_ALLOW="$NEEDS_ALLOW"
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" ]] \
  || { echo "canonical B4 qualification floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
printf '%s\n' "$LIVE_PASS_BINDING_JSON" \
  > "$RUNROOT_ABS/cutlass_b4_live_pass_binding.at_launch.json"
LIVE_PASS_BINDING_SHA256=$(
  sha256sum "$RUNROOT_ABS/cutlass_b4_live_pass_binding.at_launch.json" | awk '{print $1}'
)
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

printf 'classification=%s\nqualification_profile=%s\ntiming_eligible=1\nformal_floor_acceptance_eligible=0\nonly_arm_delta=%s\ncandidate_selector=%s\ntopology=%s\nlogical_topology=%s\nactive_drafts=%s\nvalid_mask=%s\nphysical_drafts=31\nphysical_rows_root_inclusive=32\nbatch_size=4\nconcurrency=4\nfixed_rows=128\nall_parent_production=%s\nall_parent_pass_sha256=%s\nall_parent_verdict_sha256=%s\ndraft_vocab_root=%s\ndraft_vocab_k=%s\nfr13_needs_allow=%s\ndraft_vocab_blocks=%s\ndraft_vocab_blocks_sha256=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nlauncher_pid=%s\nrunroot=%s\nstock_arm=%s\ncandidate_arm=%s\nsource=%s\nqualification_source_commit=%s\ntiming_harness_commit=%s\nqualified_patch_source_sha256=%s\nrunner_sha256=%s\nsubset_sha256=%s\nstock_fa2_sha256=%s\nstock_fa2_bytes=%s\ncandidate_sha256=%s\ncandidate_bytes=%s\nlive_pass_sha256=%s\ntail23_live_pass_sha256=%s\nhydra27_live_pass_sha256=%s\nlive_pass_binding_sha256=%s\nenforce_eager=0\ncudagraph_mode=FULL_AND_PIECEWISE\nkv_cache_memory_bytes=%s\nstarted=%s\n' \
  "$LAUNCH_CLASSIFICATION" "$QUALIFICATION_PROFILE" "$ONLY_ARM_DELTA" \
  "$CANDIDATE_SELECTOR" \
  "$FIXED32_MODE" "$LOGICAL_TOPOLOGY" "$ACTIVE_DRAFTS" "$VALID_MASK" \
  "$ALL_PARENT_PRODUCTION" "$ALL_PARENT_PASS_SHA256" \
  "$ALL_PARENT_VERDICT_SHA256" \
  "$DRAFT_VOCAB_ROOT" "$DRAFT_VOCAB_K" "$NEEDS_ALLOW" \
  "$DRAFT_VOCAB_BLOCKS_CONTAINER" "$DRAFT_VOCAB_BLOCKS_SHA256" \
  "$FR13_MANDATORY_WEIGHT_BYTES" "$FR13_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$$" \
  "$RUNROOT_ABS" "$STOCK_ARM" "$CANDIDATE_ARM" "$TIMING_HARNESS_COMMIT" \
  "$QUALIFICATION_SOURCE_COMMIT" "$TIMING_HARNESS_COMMIT" \
  "$QUALIFIED_PATCH_SOURCE_SHA256" "$RUNNER_SHA256" "$SUBSET_SHA256" \
  "$STOCK_FA2_SHA256" "$STOCK_FA2_BYTES" \
  "$CANDIDATE_SHA256" "$CANDIDATE_BYTES" "$CUTLASS_B4_PASS_SHA256" \
  "$CUTLASS_B4_TAIL23_PASS_SHA256" "$CUTLASS_B4_HYDRA27_PASS_SHA256" \
  "$LIVE_PASS_BINDING_SHA256" "$B4_KV_CACHE_MEMORY_BYTES" "$(date -u +%FT%TZ)" \
  > "$RUNROOT_ABS/launcher_meta.txt"

MANIFEST_FINALIZED=0
finalize_manifests() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json" || return $?
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json" || return $?
  cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || { echo "runtime/source manifest changed during timing" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
    || { echo "B4 CUTLASS timing runner changed during execution" >&2; return 14; }
  [[ "$(sha256sum "$RUNROOT_ABS/cutlass_b4_live_pass_binding.at_launch.json" | awk '{print $1}')" == "$LIVE_PASS_BINDING_SHA256" ]] \
    || { echo "B4 CUTLASS live-PASS binding changed during timing" >&2; return 14; }
  if (( ALL_PARENT_PRODUCTION == 1 )); then
    [[ "$(sha256sum "$ALL_PARENT_PASS_JSON" | awk '{print $1}')" == "$ALL_PARENT_PASS_SHA256" ]] \
      || { echo "all-parent production PASS changed during timing" >&2; return 14; }
    [[ "$(sha256sum "$ALL_PARENT_VERDICT_JSON" | awk '{print $1}')" == "$ALL_PARENT_VERDICT_SHA256" ]] \
      || { echo "all-parent exact4 verdict changed during timing" >&2; return 14; }
  fi
  MANIFEST_FINALIZED=1
}

runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    if finalize_manifests; then :; else
      local manifest_rc=$?
      (( rc == 0 )) && rc=$manifest_rc
    fi
  fi
  exit "$rc"
}
trap runner_exit EXIT

run_arm() {
  local arm=$1
  local production=$2
  local selector=stock
  local candidate_so=""
  local pass_json=""
  local pass_sha=""
  local tail23_pass_json=""
  local tail23_pass_sha=""
  local hydra27_pass_json=""
  local hydra27_pass_sha=""
  local twom_tail23_pass_json=""
  local twom_tail23_pass_sha=""
  local twom_hydra27_pass_json=""
  local twom_hydra27_pass_sha=""
  local hybrid_tail23_pass_json=""
  local hybrid_tail23_pass_sha=""
  local hybrid_hydra27_pass_json=""
  local hybrid_hydra27_pass_sha=""
  local qualification_source_commit=""
  if [[ "$production" == "1" ]]; then
    selector=$CANDIDATE_SELECTOR
    candidate_so=$CUTLASS_B4_SO
    if (( DUAL_CANDIDATE == 1 )); then
      if [[ "$CANDIDATE_SELECTOR" == "identity_stockshape_stage2_b4" ]]; then
        tail23_pass_json=$CUTLASS_B4_TAIL23_PASS_JSON
        tail23_pass_sha=$CUTLASS_B4_TAIL23_PASS_SHA256
        hydra27_pass_json=$CUTLASS_B4_HYDRA27_PASS_JSON
        hydra27_pass_sha=$CUTLASS_B4_HYDRA27_PASS_SHA256
      elif [[ "$CANDIDATE_SELECTOR" == "identity_twom_b4" ]]; then
        twom_tail23_pass_json=$CUTLASS_B4_TAIL23_PASS_JSON
        twom_tail23_pass_sha=$CUTLASS_B4_TAIL23_PASS_SHA256
        twom_hydra27_pass_json=$CUTLASS_B4_HYDRA27_PASS_JSON
        twom_hydra27_pass_sha=$CUTLASS_B4_HYDRA27_PASS_SHA256
      else
        hybrid_tail23_pass_json=$CUTLASS_B4_TAIL23_PASS_JSON
        hybrid_tail23_pass_sha=$CUTLASS_B4_TAIL23_PASS_SHA256
        hybrid_hydra27_pass_json=$CUTLASS_B4_HYDRA27_PASS_JSON
        hybrid_hydra27_pass_sha=$CUTLASS_B4_HYDRA27_PASS_SHA256
      fi
    else
      pass_json=$CUTLASS_B4_PASS_JSON
      pass_sha=$CUTLASS_B4_PASS_SHA256
    fi
    qualification_source_commit=$QUALIFICATION_SOURCE_COMMIT
  fi
  echo "===== $arm: exact4 B4 CUTLASS production=$production ====="
  if env \
      RUNROOT="$RUNROOT_ABS" \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S=5400 \
      KV_CACHE_MEMORY_BYTES="$B4_KV_CACHE_MEMORY_BYTES" \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT" \
      FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K" \
      FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER" \
      FR13_NEEDS_ALLOW="$NEEDS_ALLOW" \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
      FR13_SCAN_ALIGN=0 FR13_NPAD_INVARIANT=0 \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json" \
      FR13_FIXED32_CUTLASS_WAVE="$selector" \
      FR13_FIXED32_CUTLASS_WAVE_SO="$candidate_so" \
      FR13_FIXED32_CUTLASS_WAVE_PRODUCTION="$production" \
      FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_JSON="$pass_json" \
      FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_SHA256="$pass_sha" \
      FR13_FIXED32_CUTLASS_STAGE2_TAIL23_LIVE_PASS_JSON="$tail23_pass_json" \
      FR13_FIXED32_CUTLASS_STAGE2_TAIL23_LIVE_PASS_SHA256="$tail23_pass_sha" \
      FR13_FIXED32_CUTLASS_STAGE2_HYDRA27_LIVE_PASS_JSON="$hydra27_pass_json" \
      FR13_FIXED32_CUTLASS_STAGE2_HYDRA27_LIVE_PASS_SHA256="$hydra27_pass_sha" \
      FR13_FIXED32_CUTLASS_TWOM_TAIL23_LIVE_PASS_JSON="$twom_tail23_pass_json" \
      FR13_FIXED32_CUTLASS_TWOM_TAIL23_LIVE_PASS_SHA256="$twom_tail23_pass_sha" \
      FR13_FIXED32_CUTLASS_TWOM_HYDRA27_LIVE_PASS_JSON="$twom_hydra27_pass_json" \
      FR13_FIXED32_CUTLASS_TWOM_HYDRA27_LIVE_PASS_SHA256="$twom_hydra27_pass_sha" \
      FR13_FIXED32_CUTLASS_HYBRID_N5120_TAIL23_LIVE_PASS_JSON="$hybrid_tail23_pass_json" \
      FR13_FIXED32_CUTLASS_HYBRID_N5120_TAIL23_LIVE_PASS_SHA256="$hybrid_tail23_pass_sha" \
      FR13_FIXED32_CUTLASS_HYBRID_N5120_HYDRA27_LIVE_PASS_JSON="$hybrid_hydra27_pass_json" \
      FR13_FIXED32_CUTLASS_HYBRID_N5120_HYDRA27_LIVE_PASS_SHA256="$hybrid_hydra27_pass_sha" \
      FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_SOURCE_COMMIT="${qualification_source_commit:-}" \
      FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE="$QUALIFICATION_PROFILE" \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION="$ALL_PARENT_PRODUCTION" \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON="$ALL_PARENT_PASS_JSON" \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_INSTANCE_ID= \
      FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
      FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
      FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
      FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$STOCK_FA2_SO" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh \
        "$arm" "$TIMING_KIND" "$SUBSET" \
        > "$RUNROOT_ABS/$arm.runlog" 2>&1; then
    :
  else
    local serve_rc=$?
    printf 'arm=%s serve_rc=%s ended=%s\n' \
      "$arm" "$serve_rc" "$(date -u +%FT%TZ)" \
      >> "$RUNROOT_ABS/launcher_meta.txt"
    return "$serve_rc"
  fi
  local container_env="$RUNROOT_ABS/$arm/container_env.txt"
  [[ -f "$container_env" && ! -L "$container_env" ]] \
    || { echo "$arm lacks a regular container environment artifact" >&2; return 4; }
  [[ "$(grep -Fxc "FR13_FIXED32_MODE=$FIXED32_MODE" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_DRAFT_VOCAB_ROOT=$DRAFT_VOCAB_ROOT" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_DRAFT_VOCAB_K=$DRAFT_VOCAB_K" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=$ALL_PARENT_PRODUCTION" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE=$QUALIFICATION_PROFILE" "$container_env")" -eq 1 ]] \
    || { echo "$arm did not run the selected B4 qualification profile" >&2; return 4; }
  if [[ "$QUALIFICATION_PROFILE" == "k64_root" ]]; then
    [[ "$(grep -Fxc "FR13_DRAFT_VOCAB_BLOCKS=$DRAFT_VOCAB_BLOCKS_CONTAINER" "$container_env")" -eq 1 ]] \
      || { echo "$arm did not run the pinned root-64K block map" >&2; return 4; }
  fi
  local work_census="$RUNROOT_ABS/$arm/logs/fr13_fixed32_work_census.jsonl"
  local work_report="$RUNROOT_ABS/$arm/work_census_validation.json"
  if (( ALL_PARENT_PRODUCTION == 1 )); then
    local all_parent_marker="$RUNROOT_ABS/$arm/logs/fr13_fixed32_taw_native_precompute_production.arm"
    local all_parent_credential="$RUNROOT_ABS/$arm/logs/fr13_fixed32_taw_native_precompute.production_pass.json"
    [[ -f "$all_parent_marker" && ! -L "$all_parent_marker" \
       && "$(<"$all_parent_marker")" == "1" ]] \
      || { echo "$arm lacks all-parent production engagement" >&2; return 4; }
    [[ -f "$all_parent_credential" && ! -L "$all_parent_credential" \
       && "$(sha256sum "$all_parent_credential" | awk '{print $1}')" == "$ALL_PARENT_PASS_SHA256" ]] \
      || { echo "$arm used the wrong all-parent production credential" >&2; return 4; }
  fi
  # The paired summary binds one work-census report per arm regardless of the
  # committer selector, so every arm must reduce its census here.
  "$PYTHON_BIN" - \
    "$work_census" "$work_report" "$FIXED32_MODE" "$WORK_CENSUS_ROUTE" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import fr13_fixed32_work_census as census

census_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
mode = sys.argv[3]
expected_route = sys.argv[4]
if not census_path.is_file() or census_path.is_symlink():
    raise SystemExit("hot-path work census is missing")
report = census.validate_arm(
    census.load_jsonl(census_path),
    expected_mode=mode,
    expected_route=expected_route,
    required_batches=(4,),
)
raw = census_path.read_bytes()
report["census_sha256"] = hashlib.sha256(raw).hexdigest()
report["census_bytes"] = len(raw)
temporary = report_path.with_name(report_path.name + ".tmp")
temporary.write_text(
    json.dumps(report, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    + "\n",
    encoding="ascii",
)
os.replace(temporary, report_path)
PY
  # B=4 co-residency: every task opens its /metrics bracket on the same engine
  # state, so the brackets NEST and a naive sum multiply-counts the shared
  # prefix. fr13_measure.py reduces on the arm-authoritative bracket instead and
  # cross-gates that choice against this arm's own engine-side work census, which
  # is topology-blind. The census is mandatory here: an ungated B4 aggregate is
  # exactly the artifact the alignment study invalidated. This branch already
  # binds the census on every arm, so $work_census is always the right witness.
  [[ -f "$work_census" && ! -L "$work_census" ]] \
    || { echo "$arm lacks the work census the B4 bracket reduction is gated on" >&2; return 4; }
  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$arm" --out-root "$RUNROOT_ABS/$arm/swe_out" \
    --expected-tok-per-draft 31 --batch-size 4 \
    --work-census "$work_census" \
    --out "$RUNROOT_ABS/$arm/deploy_speed_fullwall.json"
  printf 'arm=%s serve_rc=0 container_env_sha256=%s ended=%s\n' \
    "$arm" "$(sha256sum "$container_env" | awk '{print $1}')" \
    "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
}

run_arm "$STOCK_ARM" 0
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "Docker state was not clean after the stock reference" >&2; exit 2; }
run_arm "$CANDIDATE_ARM" 1

STOCK_ATTESTATION="$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_cutlass_streamk_binary.json"
CANDIDATE_ATTESTATION="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_cutlass_streamk_binary.json"
CANDIDATE_SIDECAR="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_cutlass_streamk.production_pass.json"
CANDIDATE_SELECTOR_PATH="$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_cutlass_wave.selector"
[[ ! -e "$STOCK_ATTESTATION" && ! -L "$STOCK_ATTESTATION" ]] \
  || { echo "stock arm emitted a CUTLASS candidate attestation" >&2; exit 4; }
[[ -f "$CANDIDATE_ATTESTATION" && ! -L "$CANDIDATE_ATTESTATION" \
   && -f "$CANDIDATE_SIDECAR" && ! -L "$CANDIDATE_SIDECAR" \
   && -f "$CANDIDATE_SELECTOR_PATH" && ! -L "$CANDIDATE_SELECTOR_PATH" \
   && "$(<"$CANDIDATE_SELECTOR_PATH")" == "$CANDIDATE_SELECTOR" ]] \
  || { echo "candidate arm lacks selected CUTLASS identity artifacts" >&2; exit 4; }
CANDIDATE_SIDECAR_SHA256=$(sha256sum "$CANDIDATE_SIDECAR" | awk '{print $1}')
if (( DUAL_CANDIDATE == 1 )); then
  "$PYTHON_BIN" scripts/fr13_cutlass_b4_pass.py dual-verify \
    --sidecar "$CANDIDATE_SIDECAR" \
    --expected-sidecar-sha256 "$CANDIDATE_SIDECAR_SHA256" \
    --candidate-so "$CUTLASS_B4_SO" --patch-source "$PATCH_SOURCE" \
    --candidate-selector "$CANDIDATE_SELECTOR" \
    --draft-vocab-blocks "$DRAFT_VOCAB_BLOCKS_HOST" >/dev/null
else
  "$PYTHON_BIN" scripts/fr13_cutlass_b4_pass.py verify \
    --sidecar "$CANDIDATE_SIDECAR" \
    --expected-sidecar-sha256 "$CANDIDATE_SIDECAR_SHA256" \
    --candidate-so "$CUTLASS_B4_SO" --patch-source "$PATCH_SOURCE" \
    --candidate-selector "$CANDIDATE_SELECTOR" \
    --qualification-profile "$QUALIFICATION_PROFILE" \
    --fixed32-mode "$FIXED32_MODE" >/dev/null
fi
"$PYTHON_BIN" scripts/fr13_cutlass_b4_pass.py attestation \
  --attestation "$CANDIDATE_ATTESTATION" \
  --expected-sidecar-sha256 "$CANDIDATE_SIDECAR_SHA256" \
  --qualification-profile "$QUALIFICATION_PROFILE" \
  --fixed32-mode "$FIXED32_MODE" \
  --patch-source "$PATCH_SOURCE" \
  > "$RUNROOT_ABS/$CANDIDATE_ARM/cutlass_b4_production_binding.json"

finalize_manifests

"$PYTHON_BIN" - \
  "$SUBSET" "$RUNROOT_ABS/$STOCK_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/deploy_speed_fullwall.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/cutlass_b4_production_binding.json" \
  "$RUNROOT_ABS/timing_summary.json" "$CANDIDATE_SIDECAR_SHA256" \
  "$CUTLASS_B4_PASS_SHA256" "$CANDIDATE_SHA256" "$STOCK_FA2_SHA256" \
  "$QUALIFICATION_SOURCE_COMMIT" "$TIMING_HARNESS_COMMIT" \
  "$QUALIFIED_PATCH_SOURCE_SHA256" "$LIVE_PASS_BINDING_SHA256" \
  "$QUALIFICATION_PROFILE" "$SUMMARY_SCHEMA" "$RUN_CLASSIFICATION" \
  "$BINDING_SCHEMA" "$DRAFT_VOCAB_ROOT" "$DRAFT_VOCAB_K" \
  "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$DRAFT_VOCAB_BLOCKS_CONTAINER" \
  "$DRAFT_VOCAB_BLOCKS_SHA256" "$FIXED32_MODE" "$LOGICAL_TOPOLOGY" \
  "$ACTIVE_DRAFTS" "$VALID_MASK" "$ALL_PARENT_PRODUCTION" \
  "$ALL_PARENT_PASS_SHA256" "$ALL_PARENT_VERDICT_SHA256" \
  "$RUNROOT_ABS/$STOCK_ARM/work_census_validation.json" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/work_census_validation.json" \
  "$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_work_census.jsonl" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_work_census.jsonl" \
  "$CANDIDATE_SELECTOR" "$CUTLASS_B4_TAIL23_PASS_SHA256" \
  "$CUTLASS_B4_HYDRA27_PASS_SHA256" \
  "$RUNROOT_ABS/$STOCK_ARM/logs/fr13_fixed32_engine_ingress.jsonl" \
  "$RUNROOT_ABS/$CANDIDATE_ARM/logs/fr13_fixed32_engine_ingress.jsonl" \
  "$WORK_CENSUS_ROUTE" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from fr13_b4_timing_math import phase_breakdown, positive, promotion_verdict
import fr13_fixed32_work_census as work_census
from lumo_flywheel_serving.inference_proxy import (
    fixed32_canonical_task_set_sha256,
    fixed32_task_key_id,
    verify_fixed32_ingress_ledger,
)

subset_path, stock_path, candidate_path, binding_path, out_path = map(Path, sys.argv[1:6])
sidecar_sha256, live_sha256, candidate_sha256, fa2_sha256 = sys.argv[6:10]
qualification_source_commit, timing_harness_commit = sys.argv[10:12]
patch_source_sha256, live_binding_sha256 = sys.argv[12:14]
qualification_profile, summary_schema, run_classification = sys.argv[14:17]
binding_schema = sys.argv[17]
draft_vocab_root, draft_vocab_k, mandatory_weight_bytes = map(int, sys.argv[18:21])
mandatory_weight_floor_ms = float(sys.argv[21])
one_sided_u95_cap_ms = float(sys.argv[22])
draft_vocab_blocks, draft_vocab_blocks_sha256 = sys.argv[23:25]
fixed32_mode, logical_topology = sys.argv[25:27]
active_drafts = int(sys.argv[27])
valid_mask = int(sys.argv[28], 0)
all_parent_production = bool(int(sys.argv[29]))
all_parent_pass_sha256 = sys.argv[30]
all_parent_verdict_sha256 = sys.argv[31]
stock_work_path, candidate_work_path = map(Path, sys.argv[32:34])
stock_census_path, candidate_census_path = map(Path, sys.argv[34:36])
candidate_selector, tail23_live_sha256, hydra27_live_sha256 = sys.argv[36:39]
stock_ingress_path, candidate_ingress_path = map(Path, sys.argv[39:41])
work_census_route = sys.argv[41]
expected_work_census_route = (
    "fixed32_native_precompute_production_candidate_return"
    if all_parent_production
    else "fixed32_pytorch_exact_float_triton_integer_commit"
)
if work_census_route != expected_work_census_route:
    raise SystemExit("work-census route does not match the selected committer")
task_ids = sorted(json.loads(subset_path.read_text(encoding="ascii"))["instance_ids"])
stock = json.loads(stock_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
binding = json.loads(binding_path.read_text(encoding="ascii"))
expected_task_keys = sorted(fixed32_task_key_id(task_id) for task_id in task_ids)
expected_task_set_sha256 = fixed32_canonical_task_set_sha256(tuple(task_ids))


def validate_ingress(path, label):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"{label} engine ingress ledger is not a regular file")
    verification = verify_fixed32_ingress_ledger(
        path, expected_role="engine", require_finalized=True
    )
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("ascii").splitlines()]
    accepted = sorted(
        {
            row.get("task_key_id")
            for row in rows
            if row.get("event") == "request_accepted"
        }
    )
    completed = sorted(
        {
            row.get("task_key_id")
            for row in rows
            if row.get("event") == "request_complete"
            and row.get("outcome") == "completed"
        }
    )
    if accepted != expected_task_keys or completed != expected_task_keys:
        raise SystemExit(f"{label} engine ingress is not canonical exact4")
    if not any(
        row.get("event") == "campaign_begin"
        and row.get("evidence_sha256") == expected_task_set_sha256
        for row in rows
    ):
        raise SystemExit(f"{label} engine ingress lacks the exact4 campaign binding")
    return {
        "ledger_sha256": hashlib.sha256(raw).hexdigest(),
        "chain_head_sha256": verification["chain_head_sha256"],
        "accepted_task_key_ids": accepted,
        "completed_task_key_ids": completed,
    }


stock_ingress = validate_ingress(stock_ingress_path, "stock")
candidate_ingress = validate_ingress(candidate_ingress_path, "candidate")

def load_work_report(report_path, census_path, label):
    for kind, path in (("report", report_path), ("census", census_path)):
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"{label} work-census {kind} is not a regular file")
    report_raw = report_path.read_bytes()
    census_raw = census_path.read_bytes()
    report = json.loads(
        report_raw.decode("ascii"),
        object_pairs_hook=work_census._duplicate_checked_object,
    )
    report = work_census.validate_bound_arm_report(
        report,
        census_raw=census_raw,
        census_source=str(census_path),
        expected_mode=fixed32_mode,
        expected_route=work_census_route,
        required_batches=(4,),
    )
    return report, hashlib.sha256(report_raw).hexdigest()

work_census_summary = None
if all_parent_production:
    stock_work, stock_work_sha256 = load_work_report(
        stock_work_path, stock_census_path, "stock"
    )
    candidate_work, candidate_work_sha256 = load_work_report(
        candidate_work_path, candidate_census_path, "candidate"
    )
    if (
        stock_work["normalized_work_signature_sha256"]
        != candidate_work["normalized_work_signature_sha256"]
    ):
        raise SystemExit("stock/candidate physical-work signatures differ")
    work_census_summary = {
        "stock_reference": {
            "report_sha256": stock_work_sha256,
            "census_sha256": stock_work["census_sha256"],
            "event_count": stock_work["event_count"],
            "event_counts_by_batch": stock_work["event_counts_by_batch"],
            "normalized_work_signature_sha256": stock_work[
                "normalized_work_signature_sha256"
            ],
        },
        "candidate": {
            "report_sha256": candidate_work_sha256,
            "census_sha256": candidate_work["census_sha256"],
            "event_count": candidate_work["event_count"],
            "event_counts_by_batch": candidate_work["event_counts_by_batch"],
            "normalized_work_signature_sha256": candidate_work[
                "normalized_work_signature_sha256"
            ],
        },
    }

def validate(record, label):
    if (
        record.get("schema") != "fr13.measure.deploy_speed.v1"
        or record.get("regime") != "deployment"
        or record.get("instrument") != "OFF"
        or record.get("batch_size") != 4
        or record.get("n_tasks") != 4
        or sorted(record.get("task_instance_ids", [])) != task_ids
        or record.get("draft_vocab_root") != draft_vocab_root
        or record.get("draft_vocab_k") != draft_vocab_k
        or record.get("mandatory_weight_bytes") != mandatory_weight_bytes
        or not math.isclose(
            float(record.get("weight_floor_ms", math.nan)),
            mandatory_weight_floor_ms,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or record.get("floor_is_full_step_hardware_floor") is not False
    ):
        raise SystemExit(f"{label} deploy-speed provenance is not exact4 B4")
    # The B4 aggregate is only citable if the per-task bracket reduction was
    # cross-gated against the arm's own engine-side work census. An ungated
    # reduction is what carried the nested-bracket inflation.
    reduction = record.get("bracket_reduction")
    if not isinstance(reduction, dict) or reduction.get("topology") not in {
        "nested",
        "disjoint",
    }:
        raise SystemExit(f"{label} deploy-speed carries no bracket-topology provenance")
    if (reduction.get("work_census_gate") or {}).get("status") != "pass":
        raise SystemExit(
            f"{label} deploy-speed bracket reduction was not work-census gated"
        )
    for key in (
        "measured_tps_fullstep_wall", "step_wall_ms", "accept_per_event",
        "committed_per_event", "wall_steps_measured", "events_per_step",
        "s_per_fwd_gpu", "s_per_fwd_gpu_per_forward", "events_per_step",
        "wall_s_per_event", "drafter_gpu_ms_per_step", "committer_gpu_ms_per_step",
        "weight_floor_ms", "floor_ms", "floor_ratio",
    ):
        positive(record, key)

validate(stock, "stock")
validate(candidate, "candidate")
if (
    binding.get("schema") != binding_schema
    or binding.get("status") != "BOUND"
    or binding.get("selector") != candidate_selector
    or binding.get("candidate_sha256") != candidate_sha256
    or binding.get("patch_source_sha256") != patch_source_sha256
    or binding.get("production_sidecar_sha256") != sidecar_sha256
    or binding.get("qualification_source_commit") != qualification_source_commit
    or binding.get("qualified_fixed_rows") != 128
    or binding.get("qualified_comparison_call_limit") != 320
    or binding.get("qualified_draft_vocab_root") != draft_vocab_root
    or binding.get("qualified_draft_vocab_k") != draft_vocab_k
    or binding.get("mandatory_weight_bytes") != mandatory_weight_bytes
    or not math.isclose(
        float(binding.get("mandatory_weight_floor_ms", math.nan)),
        mandatory_weight_floor_ms,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
):
    raise SystemExit("candidate lacks the selected CUTLASS production binding")
if candidate_selector in {
    "identity_stockshape_stage2_b4",
    "identity_twom_b4",
    "identity_hybrid_n5120_b4",
}:
    expected_live_results = {
        "tail6_fixed32": tail23_live_sha256,
        "hydra27_fixed32": hydra27_live_sha256,
    }
    if (
        binding.get("qualification_profile") != "k64_root"
        or binding.get("qualification_topologies")
        != ["tail6_fixed32", "hydra27_fixed32"]
        or binding.get("live_result_sha256_by_topology") != expected_live_results
        or binding.get("qualified_topology") is not None
    ):
        raise SystemExit(
            "dual candidate lacks independent Tail23 and Hydra27 production binding"
        )
    if candidate_selector == "identity_hybrid_n5120_b4":
        source_identity = binding.get("qualification_source_identity")
        if (
            qualification_source_commit != timing_harness_commit
            or not isinstance(source_identity, dict)
            or source_identity.get("source_commit") != timing_harness_commit
            or binding.get("authenticated_task_count") != 4
            or binding.get("authenticated_task_ids") != task_ids
            or binding.get("authenticated_task_set_sha256")
            != expected_task_set_sha256
            or binding.get("engine_ingress_accepted_task_key_ids")
            != expected_task_keys
            or binding.get("engine_ingress_completed_task_key_ids")
            != expected_task_keys
        ):
            raise SystemExit(
                "hybrid N5120 binding is not source-bound canonical exact4"
            )
else:
    if (
        binding.get("live_result_sha256") != live_sha256
        or binding.get("qualified_topology") != fixed32_mode
        or tail23_live_sha256
        or hydra27_live_sha256
    ):
        raise SystemExit("persistent-M128 candidate binding is not single-topology")
if qualification_profile == "k64_root" and (
    binding.get("qualification_profile") != qualification_profile
    or binding.get("qualified_draft_vocab_blocks") != draft_vocab_blocks
    or binding.get("qualified_draft_vocab_blocks_sha256")
    != draft_vocab_blocks_sha256
):
    raise SystemExit("candidate lacks the pinned root-64K block-map binding")
stock_wall = positive(stock, "step_wall_ms")
candidate_wall = positive(candidate, "step_wall_ms")
stock_tps = positive(stock, "measured_tps_fullstep_wall")
candidate_tps = positive(candidate, "measured_tps_fullstep_wall")
stock_floor = positive(stock, "floor_ms")
candidate_floor = positive(candidate, "floor_ms")
if not math.isclose(stock_floor, candidate_floor, rel_tol=0.0, abs_tol=1e-9):
    raise SystemExit("stock and candidate floor values differ")

stock_phases = phase_breakdown(stock, "stock")
candidate_phases = phase_breakdown(candidate, "candidate")
if candidate_selector in {
    "identity_stockshape_stage2_b4",
    "identity_twom_b4",
    "identity_hybrid_n5120_b4",
}:
    candidate_live_binding = {
        "live_result_sha256_by_topology": {
            "tail6_fixed32": tail23_live_sha256,
            "hydra27_fixed32": hydra27_live_sha256,
        },
        "qualification_topologies": ["tail6_fixed32", "hydra27_fixed32"],
    }
else:
    candidate_live_binding = {"live_result_sha256": live_sha256}
summary = {
    "schema": summary_schema,
    "status": "complete",
    "run_classification": run_classification,
    "qualification_profile": qualification_profile,
    "task_count": 4,
    "batch_size": 4,
    "concurrency": 4,
    "arm": fixed32_mode,
    "logical_topology": logical_topology,
    "active_drafts": active_drafts,
    "valid_mask": hex(valid_mask),
    "physical_drafts": 31,
    "physical_rows_root_inclusive": 32,
    "sfwd_projection_rows": 128,
    "all_parent_production": all_parent_production,
    "all_parent_pass_sha256": all_parent_pass_sha256 or None,
    "all_parent_verdict_sha256": all_parent_verdict_sha256 or None,
    "common_committer_selector": (
        "fixed32_all_parent_commit_v2" if all_parent_production else "reference"
    ),
    "work_census": work_census_summary,
    "authenticated_task_count": 4,
    "authenticated_task_ids": task_ids,
    "authenticated_task_set_sha256": expected_task_set_sha256,
    "authenticated_task_provenance": {
        "stock_reference": stock_ingress,
        "candidate": candidate_ingress,
    },
    "task_ids": task_ids,
    "decision_metric": "measured_tps_fullstep_wall",
    "promotion": promotion_verdict(stock_phases, candidate_phases),
    "draft_vocab_root": draft_vocab_root,
    "draft_vocab_k": draft_vocab_k,
    "target_verifier_vocabulary": "full",
    "mandatory_weight_bytes": mandatory_weight_bytes,
    "mandatory_weight_floor_ms": mandatory_weight_floor_ms,
    "one_sided_u95_cap_ms": one_sided_u95_cap_ms,
    "qualification_source_commit": qualification_source_commit,
    "timing_harness_commit": timing_harness_commit,
    "qualified_patch_source_sha256": patch_source_sha256,
    "live_pass_binding_sha256": live_binding_sha256,
    "qualification_source_identity": binding.get(
        "qualification_source_identity"
    ),
    "stock_reference": {
        "selector": "stock",
        "fa2_sha256": fa2_sha256,
        "step_wall_ms": stock_wall,
        "measured_tps_fullstep_wall": stock_tps,
        "accepted_drafts_per_event": float(stock["accept_per_event"]),
        "committed_tokens_per_event": float(stock["committed_per_event"]),
        **stock_phases,
        "step_wall_to_optimistic_floor_ratio": float(stock["floor_ratio"]),
    },
    "candidate": {
        "selector": candidate_selector,
        "candidate_sha256": candidate_sha256,
        **candidate_live_binding,
        "production_sidecar_sha256": sidecar_sha256,
        "step_wall_ms": candidate_wall,
        "measured_tps_fullstep_wall": candidate_tps,
        "accepted_drafts_per_event": float(candidate["accept_per_event"]),
        "committed_tokens_per_event": float(candidate["committed_per_event"]),
        **candidate_phases,
        "step_wall_to_optimistic_floor_ratio": float(candidate["floor_ratio"]),
    },
    "optimistic_floor_ms": stock_floor,
    "optimistic_floor_is_full_step_hardware_floor": False,
    "candidate_to_stock_full_wall_tps_ratio": candidate_tps / stock_tps,
    "stock_to_candidate_step_wall_ratio": stock_wall / candidate_wall,
    "formal_floor_acceptance_eligible": False,
    "formal_floor_acceptance_reason": (
        "paired exact4 timing candidate only; run the canonical statistical "
        "Tail/Hydra floor gate after a positive screen"
    ),
    "production_default_enabled": False,
}
if qualification_profile == "k64_root":
    summary.update(
        {
            "draft_vocab_blocks": draft_vocab_blocks,
            "draft_vocab_blocks_sha256": draft_vocab_blocks_sha256,
        }
    )
temporary = out_path.with_name(out_path.name + ".tmp")
temporary.write_text(
    json.dumps(summary, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n",
    encoding="ascii",
)
temporary.replace(out_path)
print(json.dumps(summary, sort_keys=True))
PY

printf 'timing_summary=%s completed=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
