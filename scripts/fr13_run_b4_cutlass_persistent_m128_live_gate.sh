#!/usr/bin/env bash
# Real SWE-Verified exact4 B4 byte gate for fixed32 CUTLASS candidates.
# The diagnostic always serves the stock result and contributes no timing samples.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned FA2 shared object}"
: "${CUTLASS_B4_SO:?set CUTLASS_B4_SO to the pinned persistent-M128 shared object}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
QUALIFICATION_PROFILE=${CUTLASS_B4_QUALIFICATION_PROFILE:-full_vocab}
FIXED32_MODE=${CUTLASS_B4_FIXED32_MODE:-hydra27_fixed32}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
PATCH_SOURCE=scripts/fr13_patch_cutlass_fixed32_wave.py
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
CANDIDATE_SELECTOR=${CUTLASS_B4_CANDIDATE_SELECTOR:-persistent_b4_m128}
RESOURCE_CREDENTIAL=${CUTLASS_B4_RESOURCE_CREDENTIAL:-}
RESOURCE_CREDENTIAL_SHA256=${CUTLASS_B4_RESOURCE_CREDENTIAL_SHA256:-}
case "$CANDIDATE_SELECTOR" in
  identity_divisor_b4)
    DIAGNOSTIC_SELECTOR=identity_divisor_b4_byte_ab
    RECORD_SCHEMA=fr13.fixed32.cutlass_identity_divisor_b4_byte_ab.v1
    CONTAINER_JSONL=/logs/fr13_fixed32_cutlass_identity_divisor_b4_byte_ab.jsonl
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_divisor_b4_live_gate.v1
    K64_ROOT_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_divisor_b4_k64_root_live_gate.v1
    CANDIDATE_ARM_NAME=identity_divisor
    [[ -z "$RESOURCE_CREDENTIAL" && -z "$RESOURCE_CREDENTIAL_SHA256" ]] || {
      echo "divisor identity gate forbids a static resource credential" >&2
      exit 2
    }
    ;;
  identity_stockshape_b4)
    DIAGNOSTIC_SELECTOR=identity_stockshape_b4_byte_ab
    RECORD_SCHEMA=fr13.fixed32.cutlass_identity_stockshape_b4_byte_ab.v1
    CONTAINER_JSONL=/logs/fr13_fixed32_cutlass_identity_stockshape_b4_byte_ab.jsonl
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_stockshape_b4_live_gate.v1
    K64_ROOT_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_stockshape_b4_k64_root_live_gate.v1
    CANDIDATE_ARM_NAME=identity_stockshape
    [[ -z "$RESOURCE_CREDENTIAL" && -z "$RESOURCE_CREDENTIAL_SHA256" ]] || {
      echo "stock-shape identity gate forbids a static resource credential" >&2
      exit 2
    }
    ;;
  identity_stockshape_stage2_b4)
    DIAGNOSTIC_SELECTOR=identity_stockshape_stage2_b4_byte_ab
    RECORD_SCHEMA=fr13.fixed32.cutlass_identity_stockshape_stage2_b4_byte_ab.v1
    CONTAINER_JSONL=/logs/fr13_fixed32_cutlass_identity_stockshape_stage2_b4_byte_ab.jsonl
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_stockshape_stage2_b4_live_gate.v1
    K64_ROOT_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_stockshape_stage2_b4_k64_root_live_gate.v1
    CANDIDATE_ARM_NAME=identity_stockshape_stage2
    [[ -z "$RESOURCE_CREDENTIAL" && -z "$RESOURCE_CREDENTIAL_SHA256" ]] || {
      echo "stock-shape Stage2 identity gate forbids a static resource credential" >&2
      exit 2
    }
    ;;
  identity_twom_b4)
    DIAGNOSTIC_SELECTOR=identity_twom_b4_byte_ab
    RECORD_SCHEMA=fr13.fixed32.cutlass_identity_twom_b4_byte_ab.v1
    CONTAINER_JSONL=/logs/fr13_fixed32_cutlass_identity_twom_b4_byte_ab.jsonl
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_twom_b4_live_gate.v1
    K64_ROOT_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_twom_b4_k64_root_live_gate.v1
    CANDIDATE_ARM_NAME=identity_twom
    [[ -z "$RESOURCE_CREDENTIAL" && -z "$RESOURCE_CREDENTIAL_SHA256" ]] || {
      echo "two-M identity gate forbids a static resource credential" >&2
      exit 2
    }
    ;;
  identity_hybrid_n5120_b4)
    DIAGNOSTIC_SELECTOR=identity_hybrid_n5120_b4_byte_ab
    RECORD_SCHEMA=fr13.fixed32.cutlass_identity_hybrid_n5120_b4_byte_ab.v1
    CONTAINER_JSONL=/logs/fr13_fixed32_cutlass_identity_hybrid_n5120_b4_byte_ab.jsonl
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_hybrid_n5120_b4_live_gate.v1
    K64_ROOT_LIVE_SCHEMA=fr13.fixed32.cutlass_identity_hybrid_n5120_b4_k64_root_live_gate.v1
    CANDIDATE_ARM_NAME=identity_hybrid_n5120
    [[ -z "$RESOURCE_CREDENTIAL" && -z "$RESOURCE_CREDENTIAL_SHA256" ]] || {
      echo "hybrid N5120 identity gate forbids a static resource credential" >&2
      exit 2
    }
    [[ "$QUALIFICATION_PROFILE" == "k64_root" ]] || {
      echo "hybrid N5120 identity gate requires k64_root" >&2
      exit 2
    }
    ;;
  persistent_b4_m128)
    DIAGNOSTIC_SELECTOR=persistent_b4_m128_byte_ab
    RECORD_SCHEMA=fr13.fixed32.cutlass_persistent_b4_m128_byte_ab.v1
    CONTAINER_JSONL=/logs/fr13_fixed32_cutlass_persistent_b4_m128_byte_ab.jsonl
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_persistent_b4_m128_live_gate.v1
    K64_ROOT_LIVE_SCHEMA=fr13.fixed32.cutlass_persistent_b4_m128_k64_root_live_gate.v1
    CANDIDATE_ARM_NAME=m128
    [[ -z "$RESOURCE_CREDENTIAL" && -z "$RESOURCE_CREDENTIAL_SHA256" ]] || {
      echo "incumbent persistent M128 gate forbids a static resource credential" >&2
      exit 2
    }
    ;;
  persistent_b4_m128_static)
    DIAGNOSTIC_SELECTOR=persistent_b4_m128_static_byte_ab
    RECORD_SCHEMA=fr13.fixed32.cutlass_persistent_b4_m128_static_byte_ab.v1
    CONTAINER_JSONL=/logs/fr13_fixed32_cutlass_persistent_b4_m128_static_byte_ab.jsonl
    FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_persistent_b4_m128_static_live_gate.v1
    K64_ROOT_LIVE_SCHEMA=fr13.fixed32.cutlass_persistent_b4_m128_static_k64_root_live_gate.v1
    CANDIDATE_ARM_NAME=m128_static
    [[ "$RESOURCE_CREDENTIAL_SHA256" == "7ab2c3223366f4591fc2324a47c805aa0a1e9d4a106743af4256d4089054a2dc" ]] || {
      echo "static M128 gate requires the pinned host-build resource credential SHA-256" >&2
      exit 2
    }
    ;;
  *)
    echo "CUTLASS_B4_CANDIDATE_SELECTOR is unsupported" >&2
    exit 2
    ;;
esac
DRAFT_VOCAB_BLOCKS_HOST=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
B4_KV_CACHE_MEMORY_BYTES=49392123904
COMPARISON_CALL_LIMIT=320
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
case "$QUALIFICATION_PROFILE" in
  full_vocab)
    RUN_CLASSIFICATION=real_swe_verified_exact4_b4_byte_diagnostic
    LIVE_SCHEMA=$FULL_VOCAB_LIVE_SCHEMA
    DRAFT_VOCAB_ROOT=0
    DRAFT_VOCAB_K=0
    NEEDS_ALLOW=FR13_DRAFT_VOCAB_K=0
    MANDATORY_WEIGHT_BYTES=25430574256
    MANDATORY_WEIGHT_FLOOR_MS=93.15228665201465
    ONE_SIDED_U95_CAP_MS=107.12512964981684
    LIVE_RESULT_NAME=cutlass_b4_${CANDIDATE_ARM_NAME}_byte_gate.json
    PRODUCTION_PASS_NAME=cutlass_b4_${CANDIDATE_ARM_NAME}.production_pass.json
    ARM_PROFILE_SUFFIX=
    ;;
  k64_root)
    RUN_CLASSIFICATION=real_swe_verified_exact4_b4_k64_root_byte_diagnostic
    LIVE_SCHEMA=$K64_ROOT_LIVE_SCHEMA
    DRAFT_VOCAB_ROOT=1
    DRAFT_VOCAB_K=65536
    NEEDS_ALLOW=
    MANDATORY_WEIGHT_BYTES=25210209416
    MANDATORY_WEIGHT_FLOOR_MS=92.345089436
    ONE_SIDED_U95_CAP_MS=106.1968528514
    LIVE_RESULT_NAME=cutlass_b4_${CANDIDATE_ARM_NAME}_k64_root_byte_gate.json
    PRODUCTION_PASS_NAME=cutlass_b4_${CANDIDATE_ARM_NAME}_k64_root.production_pass.json
    ARM_PROFILE_SUFFIX=_k64_root
    ;;
  *)
    echo "CUTLASS_B4_QUALIFICATION_PROFILE must be full_vocab or k64_root" >&2
    exit 2
    ;;
esac
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
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
ARM="${FIXED32_MODE}_cutlass_b4_${CANDIDATE_ARM_NAME}${ARM_PROFILE_SUFFIX}_gate_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for input in "$FORKED_FA2_SO" "$CUTLASS_B4_SO"; do
  [[ "$input" == /* && -f "$input" && ! -L "$input" ]] \
    || { echo "gate input must be an absolute regular non-symlink file: $input" >&2; exit 2; }
done
if [[ "$CANDIDATE_SELECTOR" == "persistent_b4_m128_static" ]]; then
  [[ "$RESOURCE_CREDENTIAL" == /* \
     && -f "$RESOURCE_CREDENTIAL" \
     && ! -L "$RESOURCE_CREDENTIAL" ]] || {
    echo "static M128 gate requires an absolute regular resource credential" >&2
    exit 2
  }
fi
[[ "$(stat -c '%s' "$FORKED_FA2_SO")" == "$STOCK_FA2_BYTES" \
   && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "FORKED_FA2_SO is not the exact-safe stock reference" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical exact4 subset SHA-256 drift" >&2; exit 2; }
if [[ "$QUALIFICATION_PROFILE" == "k64_root" ]]; then
  [[ -f "$DRAFT_VOCAB_BLOCKS_HOST" && ! -L "$DRAFT_VOCAB_BLOCKS_HOST" \
     && "$(sha256sum "$DRAFT_VOCAB_BLOCKS_HOST" | awk '{print $1}')" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
    || { echo "pinned root-64K draft-vocabulary block map drifted" >&2; exit 2; }
fi
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the gate" >&2; exit 2; }

RESOURCE_CREDENTIAL_ARGS=()
if [[ "$CANDIDATE_SELECTOR" == "persistent_b4_m128_static" ]]; then
  RESOURCE_CREDENTIAL_ARGS=(
    --resource-credential "$RESOURCE_CREDENTIAL"
    --expected-resource-credential-sha256 "$RESOURCE_CREDENTIAL_SHA256"
  )
fi
"$PYTHON_BIN" scripts/fr13_cutlass_wave_binary.py verify \
  "$CUTLASS_B4_SO" --selector "$DIAGNOSTIC_SELECTOR" \
  --qualification-profile "$QUALIFICATION_PROFILE" \
  "${RESOURCE_CREDENTIAL_ARGS[@]}" >/dev/null

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
SOURCE_IDENTITY_PATH=
SOURCE_IDENTITY_SHA256=
if [[ "$CANDIDATE_SELECTOR" == "identity_hybrid_n5120_b4" ]]; then
  SOURCE_IDENTITY_PATH="$RUNROOT_ABS/cutlass_b4_source_identity.at_launch.json"
  "$PYTHON_BIN" scripts/fr13_cutlass_b4_pass.py source-binding \
    --source-commit "$SOURCE_COMMIT" --patch-source "$PATCH_SOURCE" \
    --candidate-selector "$CANDIDATE_SELECTOR" > "$SOURCE_IDENTITY_PATH"
  chmod 0444 "$SOURCE_IDENTITY_PATH"
  SOURCE_IDENTITY_SHA256=$(sha256sum "$SOURCE_IDENTITY_PATH" | awk '{print $1}')
fi
printf 'classification=%s\nqualification_profile=%s\ntiming_eligible=0\nfloor_acceptance_eligible=0\nreference_always_served=1\ntopology=%s\nlogical_topology=%s\nactive_drafts=%s\nvalid_mask=%s\nphysical_drafts=31\nphysical_rows_root_inclusive=32\nbatch_size=4\nconcurrency=4\nfixed_rows=128\neager_builder_capacity=128\ncomparison_call_limit=%s\ndraft_vocab_root=%s\ndraft_vocab_k=%s\nfr13_needs_allow=%s\ndraft_vocab_blocks=%s\ndraft_vocab_blocks_sha256=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nlauncher_pid=%s\nrunroot=%s\narm=%s\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nstock_fa2_sha256=%s\nstock_fa2_bytes=%s\ncandidate_selector=%s\ndiagnostic_selector=%s\nresource_credential_sha256=%s\nsource_identity_sha256=%s\nenforce_eager=1\nkv_cache_memory_bytes=%s\nstarted=%s\n' \
  "$RUN_CLASSIFICATION" "$QUALIFICATION_PROFILE" \
  "$FIXED32_MODE" "$LOGICAL_TOPOLOGY" "$ACTIVE_DRAFTS" "$VALID_MASK" \
  "$COMPARISON_CALL_LIMIT" "$DRAFT_VOCAB_ROOT" "$DRAFT_VOCAB_K" \
  "$NEEDS_ALLOW" "$DRAFT_VOCAB_BLOCKS_CONTAINER" \
  "$DRAFT_VOCAB_BLOCKS_SHA256" "$FR13_MANDATORY_WEIGHT_BYTES" \
  "$FR13_WEIGHT_FLOOR_MS" "$ONE_SIDED_U95_CAP_MS" "$$" \
  "$RUNROOT_ABS" "$ARM" "$SOURCE_COMMIT" "$RUNNER_SHA256" \
  "$SUBSET_SHA256" "$STOCK_FA2_SHA256" "$STOCK_FA2_BYTES" \
  "$CANDIDATE_SELECTOR" "$DIAGNOSTIC_SELECTOR" \
  "$RESOURCE_CREDENTIAL_SHA256" "$SOURCE_IDENTITY_SHA256" \
  "$B4_KV_CACHE_MEMORY_BYTES" \
  "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

if env \
    RUNROOT="$RUNROOT_ABS" \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S=5400 \
    KV_CACHE_MEMORY_BYTES="$B4_KV_CACHE_MEMORY_BYTES" \
    FR13_FIXED32_B1_DIAGNOSTIC=0 \
    FR10_METRICS=0 ENFORCE_EAGER=1 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT" \
    FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K" \
    FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER" \
    FR13_NEEDS_ALLOW="$NEEDS_ALLOW" \
    FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
    FR13_SCAN_ALIGN=0 FR13_NPAD_INVARIANT=0 \
    FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_CUTLASS_WAVE="$DIAGNOSTIC_SELECTOR" \
    FR13_FIXED32_CUTLASS_WAVE_SO="$CUTLASS_B4_SO" \
    FR13_FIXED32_CUTLASS_WAVE_RESOURCE_CREDENTIAL="$RESOURCE_CREDENTIAL" \
    FR13_FIXED32_CUTLASS_WAVE_RESOURCE_CREDENTIAL_SHA256="$RESOURCE_CREDENTIAL_SHA256" \
    FR13_FIXED32_CUTLASS_WAVE_BYTE_AB_JSONL="$CONTAINER_JSONL" \
    FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
    FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_JSON= \
    FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_SHA256= \
    FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE="$QUALIFICATION_PROFILE" \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
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
  || { echo "B4 CUTLASS gate runner changed during execution" >&2; exit 14; }
(( serve_rc == 0 )) || exit "$serve_rc"

# The immutable real-event marker is created by the root-owned serving
# container with mode 0400. Keep that ownership boundary intact and run only
# the closed-over credential reducer with read privilege after teardown.
sudo -n -- "$PYTHON_BIN" - \
  "$ARMDIR" "$ARMDIR$CONTAINER_JSONL" \
  "$ARMDIR/logs/fr13_fixed32_cutlass_streamk_binary.json" \
  "$ARMDIR/$LIVE_RESULT_NAME" "$PATCH_SOURCE" \
  "$SOURCE_COMMIT" "$SUBSET_SHA256" "$DIAGNOSTIC_SELECTOR" \
  "$CANDIDATE_SELECTOR" "$RECORD_SCHEMA" "$LIVE_SCHEMA" \
  "$STOCK_FA2_SHA256" \
  "$COMPARISON_CALL_LIMIT" "$QUALIFICATION_PROFILE" \
  "$DRAFT_VOCAB_BLOCKS_SHA256" "$FIXED32_MODE" "$LOGICAL_TOPOLOGY" \
  "$ACTIVE_DRAFTS" "$VALID_MASK" "$RESOURCE_CREDENTIAL_SHA256" \
  "$SOURCE_IDENTITY_PATH" "$SOURCE_IDENTITY_SHA256" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import fr13_cutlass_b4_pass as qualification
import fr13_cutlass_wave_binary as binary
from lumo_flywheel_serving.inference_proxy import (
    fixed32_canonical_task_set_sha256,
    fixed32_task_key_id,
    verify_fixed32_ingress_ledger,
)

arm = Path(sys.argv[1]).resolve()
jsonl_path = Path(sys.argv[2])
binary_path = Path(sys.argv[3])
output_path = Path(sys.argv[4])
patch_source = Path(sys.argv[5])
source_commit = sys.argv[6]
subset_sha256 = sys.argv[7]
diagnostic_selector = sys.argv[8]
candidate_selector = sys.argv[9]
record_schema = sys.argv[10]
live_schema = sys.argv[11]
stock_fa2_sha256 = sys.argv[12]
comparison_call_limit = int(sys.argv[13])
qualification_profile = sys.argv[14]
draft_vocab_blocks_sha256 = sys.argv[15]
fixed32_mode = sys.argv[16]
logical_topology = sys.argv[17]
active_drafts = int(sys.argv[18])
valid_mask = int(sys.argv[19], 0)
resource_credential_sha256 = sys.argv[20]
source_identity_path_text = sys.argv[21]
source_identity_sha256 = sys.argv[22]
if fixed32_mode not in qualification.QUALIFIED_FIXED32_MODES:
    raise SystemExit("CUTLASS B4 fixed32 topology is unsupported")
if comparison_call_limit != qualification.MAX_COMPARISONS:
    raise SystemExit("CUTLASS B4 comparison-call limit contract drifted")
try:
    profile = qualification.QUALIFICATION_PROFILES[qualification_profile]
except KeyError as error:
    raise SystemExit("CUTLASS B4 qualification profile is invalid") from error
candidate_contract = qualification._candidate_contract(candidate_selector)
if candidate_contract["diagnostic_selector"] != diagnostic_selector:
    raise SystemExit("CUTLASS B4 candidate/diagnostic selector contract drifted")
expected_live_schema = qualification._contract_profile_value(
    candidate_contract, "live_schemas", qualification_profile
)
if live_schema != expected_live_schema:
    raise SystemExit("CUTLASS B4 live schema/profile contract drifted")
if (
    qualification_profile == "k64_root"
    and draft_vocab_blocks_sha256 != qualification.DRAFT_VOCAB_BLOCKS_SHA256
):
    raise SystemExit("CUTLASS B4 root-64K block-map contract drifted")
logs = arm / "logs"
marker_path = logs / "fr13_fixed32_cutlass_b4_byte_ab.real_event.arm"
ledger_path = logs / "fr13_fixed32_engine_ingress.jsonl"
container_env_path = arm / "container_env.txt"
expected_tasks = list(qualification.EXPECTED_TASK_IDS)
expected_task_set = set(expected_tasks)
expected_task_keys = {
    fixed32_task_key_id(task_id) for task_id in expected_tasks
}
expected_shapes = set(qualification.EXPECTED_PROJECTION_NK)
expected_sha256, expected_size, expected_family = binary.candidate_identity(
    diagnostic_selector
)

marker_info = os.lstat(marker_path)
if (
    not stat.S_ISREG(marker_info.st_mode)
    or stat.S_ISLNK(marker_info.st_mode)
    or marker_info.st_nlink != 1
    or stat.S_IMODE(marker_info.st_mode) != 0o400
):
    raise SystemExit("CUTLASS B4 real-event marker identity or mode is invalid")
marker_raw = marker_path.read_bytes()
try:
    marker_text = marker_raw.decode("ascii")
except UnicodeDecodeError as error:
    raise SystemExit("CUTLASS B4 real-event marker is not ASCII") from error
if not marker_text.endswith("\n") or marker_text.count("\n") != 1:
    raise SystemExit("CUTLASS B4 real-event marker framing is invalid")
task_marker = marker_text.removesuffix("\n")
prefix = "swe_verified:"
if not task_marker.startswith(prefix) or task_marker[len(prefix):] not in expected_task_set:
    raise SystemExit("CUTLASS B4 marker is not bound to canonical exact4")

lines = jsonl_path.read_text(encoding="ascii").splitlines()
if not lines:
    raise SystemExit("CUTLASS B4 byte gate was vacuous")
records = [json.loads(line) for line in lines]
observed_shapes = {(record.get("n"), record.get("k")) for record in records}
invocations = [record.get("invocation") for record in records]
errors = []
if len(records) > comparison_call_limit:
    errors.append(
        f"diagnostic exceeded its {comparison_call_limit}-call bound"
    )
if any(record.get("schema") != record_schema for record in records):
    errors.append("comparison record schema mismatch")
if any(record.get("task_marker") != task_marker for record in records):
    errors.append("comparison record is not bound to the authenticated exact4 marker")
if invocations != list(range(len(records))):
    errors.append("invocations are not contiguous from zero")
if not expected_shapes.issubset(observed_shapes):
    errors.append("not all five real projection shapes were exercised")
if any(record.get("m") != 128 for record in records):
    errors.append("a comparison did not use the exact B4 physical row count")
if any(record.get("bytes") != 2 * record["m"] * record["n"] for record in records):
    errors.append("a comparison reported an invalid BF16 byte count")
if any(record.get("byte_equal") is not True for record in records):
    errors.append("at least one stock/persistent-M128 output differed")
if any(
    not isinstance(record.get("mismatch_count"), int)
    or isinstance(record.get("mismatch_count"), bool)
    or record["mismatch_count"] != 0
    or record.get("first_mismatch") is not None
    for record in records
):
    errors.append("byte mismatch accounting is invalid")

binary_raw = binary_path.read_bytes()
binary_record = json.loads(binary_raw.decode("ascii"))
if binary_record.get("schema") != qualification.ATTESTATION_SCHEMA:
    errors.append("installed binary attestation schema mismatch")
if binary_record.get("selector") != diagnostic_selector:
    errors.append("installed binary selector attestation mismatch")
if binary_record.get("candidate_family") != expected_family:
    errors.append("installed binary candidate-family attestation mismatch")
if binary_record.get("production_enabled") is not False:
    errors.append("diagnostic binary unexpectedly enabled production")
if binary_record.get("installed_mode") != "0555":
    errors.append("installed binary mode attestation mismatch")
for label, expected_path in (
    ("source", str(binary.CONTAINER_SOURCE)),
    ("destination", str(binary.CONTAINER_DESTINATION)),
):
    identity = binary_record.get(label) or {}
    if (
        identity.get("path") != expected_path
        or identity.get("regular") is not True
        or identity.get("symlink") is not False
        or identity.get("sha256") != expected_sha256
        or identity.get("bytes") != expected_size
        or identity.get("candidate_family") != expected_family
    ):
        errors.append(f"installed binary {label} identity mismatch")

resource_binding = {}
if candidate_contract["requires_resource_credential"] is True:
    if (
        resource_credential_sha256
        != binary.STATIC_B4_M128_RESOURCE_CREDENTIAL_SHA256
    ):
        errors.append("static M128 resource-credential input is not pinned")
    for label in ("source", "destination"):
        identity = binary_record.get(label) or {}
        resource = identity.get("resource_credential") or {}
        if (
            resource.get("sha256") != resource_credential_sha256
            or resource.get("schema")
            != binary.STATIC_B4_M128_RESOURCE_CREDENTIAL_SCHEMA
            or resource.get("candidate_sha256") != expected_sha256
            or resource.get("candidate_bytes") != expected_size
            or resource.get("regular") is not True
            or resource.get("symlink") is not False
        ):
            errors.append(
                f"installed binary {label} resource-credential binding mismatch"
            )
    try:
        resource_binding = qualification._resource_binding(
            binary_record.get("source") or {}
        )
    except qualification.QualificationError as error:
        errors.append(str(error))
else:
    if resource_credential_sha256:
        errors.append("incumbent M128 diagnostic received a static resource credential")

patch_sha256 = hashlib.sha256(patch_source.read_bytes()).hexdigest()
expected_patch_sha256, expected_dispatch_sha256 = (
    qualification._candidate_source_hashes(candidate_selector)
)
if patch_sha256 != expected_patch_sha256:
    errors.append("CUTLASS patch source SHA-256 mismatch")

ledger_verification = verify_fixed32_ingress_ledger(
    ledger_path, expected_role="engine", require_finalized=True
)
ledger_rows = [
    json.loads(line)
    for line in ledger_path.read_text(encoding="ascii").splitlines()
]
expected_set_sha256 = fixed32_canonical_task_set_sha256(tuple(expected_tasks))
if not any(
    row.get("event") == "campaign_begin"
    and row.get("evidence_sha256") == expected_set_sha256
    for row in ledger_rows
):
    errors.append("engine ledger is not bound to canonical exact4")
accepted_task_keys = {
    row.get("task_key_id")
    for row in ledger_rows
    if row.get("event") == "request_accepted"
}
completed_task_keys = {
    row.get("task_key_id")
    for row in ledger_rows
    if row.get("event") == "request_complete"
    and row.get("outcome") == "completed"
}
if accepted_task_keys != expected_task_keys:
    errors.append("engine ledger did not accept all and only canonical exact4 tasks")
if completed_task_keys != expected_task_keys:
    errors.append("engine ledger did not complete all and only canonical exact4 tasks")
marker_task_key = fixed32_task_key_id(task_marker[len(prefix):])
if not any(
    row.get("event") == "request_accepted"
    and row.get("task_key_id") == marker_task_key
    for row in ledger_rows
):
    errors.append("real-event marker has no matching accepted request")

health = json.loads((arm / "health.json").read_text(encoding="utf-8"))
health_tasks = health.get("tasks")
if (
    not isinstance(health_tasks, list)
    or len(health_tasks) != 4
    or {task.get("instance_id") for task in health_tasks} != expected_task_set
    or health.get("swe_orchestrator_rc") != 0
):
    errors.append("diagnostic did not complete canonical exact4")

source_identity = None
if candidate_contract.get("source_binding") == "required":
    source_identity_path = Path(source_identity_path_text)
    if (
        not source_identity_path.is_absolute()
        or not source_identity_path.is_file()
        or source_identity_path.is_symlink()
        or not source_identity_sha256
    ):
        errors.append("hybrid N5120 source identity is unavailable")
    else:
        source_identity_raw = source_identity_path.read_bytes()
        if hashlib.sha256(source_identity_raw).hexdigest() != source_identity_sha256:
            errors.append("hybrid N5120 source identity changed during diagnostic")
        try:
            source_identity = json.loads(source_identity_raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            errors.append("hybrid N5120 source identity is invalid")
        if not isinstance(source_identity, dict):
            errors.append("hybrid N5120 source identity is not an object")
        elif (
            source_identity.get("schema") != qualification.SOURCE_BINDING_SCHEMA
            or source_identity.get("source_commit") != source_commit
        ):
            errors.append("hybrid N5120 source identity is stale")
elif source_identity_path_text or source_identity_sha256:
    errors.append("non-source-bound candidate received a source identity")

container_env_raw = container_env_path.read_bytes()
container_env_lines = container_env_raw.decode("ascii").splitlines()
for expected_env in (
    f"FR13_FIXED32_MODE={fixed32_mode}",
    f"FR13_DRAFT_VOCAB_ROOT={profile['draft_vocab_root']}",
    f"FR13_DRAFT_VOCAB_K={profile['draft_vocab_k']}",
    (
        "FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE="
        f"{qualification_profile}"
    ),
):
    if container_env_lines.count(expected_env) != 1:
        errors.append(f"B4 environment mismatch: {expected_env}")

payload = {
    "schema": live_schema,
    "status": "pass" if not errors else "fail",
    "run_classification": profile["run_classification"],
    "acceptance_valid": False,
    "task_set": "canonical real SWE-Verified exact4 B4",
    "topology": fixed32_mode,
    "logical_topology": logical_topology,
    "active_drafts": active_drafts,
    "valid_mask": hex(valid_mask),
    "physical_drafts": 31,
    "physical_rows_root_inclusive": 32,
    "task_count": 4,
    "task_ids": expected_tasks,
    "task_marker": task_marker,
    "subset_sha256": subset_sha256,
    "real_task_arm_sha256": hashlib.sha256(marker_raw).hexdigest(),
    "container_env_sha256": hashlib.sha256(container_env_raw).hexdigest(),
    "engine_ledger_chain_head_sha256": ledger_verification["chain_head_sha256"],
    "draft_vocab_root": profile["draft_vocab_root"],
    "draft_vocab_k": profile["draft_vocab_k"],
    "mandatory_weight_bytes": profile["mandatory_weight_bytes"],
    "mandatory_weight_floor_ms": profile["mandatory_weight_floor_ms"],
    "one_sided_u95_cap_ms": profile["one_sided_u95_cap_ms"],
    "comparator_timing_eligible": False,
    "batch_size": 4,
    "concurrency": 4,
    "fixed_rows": 128,
    "eager_builder_capacity": 128,
    "candidate": candidate_selector,
    "diagnostic_selector": diagnostic_selector,
    "served_result": "stock",
    "production_enabled": False,
    "comparison_call_limit": comparison_call_limit,
    "comparisons": len(records),
    "observed_m_values": sorted({record["m"] for record in records}),
    "observed_projection_nk": [list(shape) for shape in sorted(observed_shapes)],
    "mismatching_comparisons": sum(record.get("byte_equal") is not True for record in records),
    "differing_bytes": sum(record.get("mismatch_count", 0) for record in records),
    "candidate_family": expected_family,
    "candidate_sha256": expected_sha256,
    "candidate_bytes": expected_size,
    "stock_fa2_sha256": stock_fa2_sha256,
    "patch_source_sha256": patch_sha256,
    "vllm_base_commit": qualification.VLLM_BASE_COMMIT,
    "patched_dispatch_sha256": expected_dispatch_sha256,
    "source_commit": source_commit,
    "binary_attestation_sha256": hashlib.sha256(binary_raw).hexdigest(),
    "errors": errors,
}
payload.update(resource_binding)
if candidate_contract.get("source_binding") == "required":
    payload.update(
        {
            "authenticated_task_count": len(expected_tasks),
            "authenticated_task_ids": expected_tasks,
            "authenticated_task_set_sha256": expected_set_sha256,
            "engine_ingress_accepted_task_key_ids": sorted(accepted_task_keys),
            "engine_ingress_completed_task_key_ids": sorted(completed_task_keys),
            "source_identity": source_identity,
        }
    )
if qualification_profile == "k64_root":
    expected_blocks_env = (
        "FR13_DRAFT_VOCAB_BLOCKS="
        f"{qualification.DRAFT_VOCAB_BLOCKS_CONTAINER_PATH}"
    )
    if container_env_lines.count(expected_blocks_env) != 1:
        errors.append(f"B4 environment mismatch: {expected_blocks_env}")
    payload.update(
        {
            "qualification_profile": qualification_profile,
            "draft_vocab_blocks": (
                qualification.DRAFT_VOCAB_BLOCKS_CONTAINER_PATH
            ),
            "draft_vocab_blocks_sha256": draft_vocab_blocks_sha256,
        }
    )
    payload["status"] = "pass" if not errors else "fail"
output_path.write_text(
    json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
if errors:
    raise SystemExit(4)
PY

LIVE_RESULT="$ARMDIR/$LIVE_RESULT_NAME"
LIVE_SHA256=$(sha256sum "$LIVE_RESULT" | awk '{print $1}')
QUALIFICATION_RESOURCE_ARGS=()
if [[ "$CANDIDATE_SELECTOR" == "persistent_b4_m128_static" ]]; then
  QUALIFICATION_RESOURCE_ARGS=(
    --resource-credential "$RESOURCE_CREDENTIAL"
    --expected-resource-credential-sha256 "$RESOURCE_CREDENTIAL_SHA256"
  )
fi
"$PYTHON_BIN" scripts/fr13_cutlass_b4_pass.py issue \
  --live-result "$LIVE_RESULT" --expected-live-sha256 "$LIVE_SHA256" \
  --candidate-so "$CUTLASS_B4_SO" --patch-source "$PATCH_SOURCE" \
  --expected-source-commit "$SOURCE_COMMIT" \
  --candidate-selector "$CANDIDATE_SELECTOR" \
  "${QUALIFICATION_RESOURCE_ARGS[@]}" \
  --qualification-profile "$QUALIFICATION_PROFILE" \
  --fixed32-mode "$FIXED32_MODE" \
  --out "$ARMDIR/$PRODUCTION_PASS_NAME"

printf 'live_result=%s\nlive_sha256=%s\nproduction_pass=%s\nproduction_pass_sha256=%s\n' \
  "$LIVE_RESULT" "$LIVE_SHA256" \
  "$ARMDIR/$PRODUCTION_PASS_NAME" \
  "$(sha256sum "$ARMDIR/$PRODUCTION_PASS_NAME" | awk '{print $1}')"
