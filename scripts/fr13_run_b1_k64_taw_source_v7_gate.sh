#!/usr/bin/env bash
# One real SWE-Verified K64 B1 graph-replay byte gate for one source-v7 TAW mode.
# The candidate is shadow-only and the exact reference is always served.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${MODE:?set MODE to tail6_fixed32 or hydra27_fixed32}"
: "${STOCK_FA2_SO:?set STOCK_FA2_SO to the exact-safe stock FA2 binary}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
TAW_SOURCE=scripts/fr13_device_multidraft_kernel.py
TAW_TOPOLOGY=scripts/fr13_fixed32_topology.py
TAW_SOURCE_SCHEMA=fr13-fixed32-taw-all-parent-v7
TAW_SOURCE_CONTRACT_SHA256=2b1cc55c6ec3d45c2d6ad0a21be4dc76685df4c974ae7fcfa421d5824a5c1ffb
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
MANDATORY_WEIGHT_BYTES=32666638208
MANDATORY_WEIGHT_FLOOR_MS=119.658015414
ONE_SIDED_U95_CAP_MS=137.6067177261
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")

case "$MODE" in
  tail6_fixed32)
    LOGICAL_TOPOLOGY=Tail23
    LOGICAL_DRAFTS=23
    VALID_MASK=0x7a9ce7ff
    ;;
  hydra27_fixed32)
    LOGICAL_TOPOLOGY=Hydra27
    LOGICAL_DRAFTS=27
    VALID_MASK=0x7abdffff
    ;;
  *)
    echo "MODE must be tail6_fixed32 or hydra27_fixed32" >&2
    exit 2
    ;;
esac
ARM="${MODE}_taw_source_v7_b1_gate_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ "$STOCK_FA2_SO" == /* && -f "$STOCK_FA2_SO" && ! -L "$STOCK_FA2_SO" ]] \
  || { echo "STOCK_FA2_SO must be an absolute regular file" >&2; exit 2; }
[[ "$(stat -c '%s' "$STOCK_FA2_SO")" == "$STOCK_FA2_BYTES" \
   && "$(sha256sum "$STOCK_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "STOCK_FA2_SO is not the exact-safe stock binary" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" ]] \
  || { echo "canonical B1 task or K64 block map drifted" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the B1 gate" >&2; exit 2; }

"$PYTHON_BIN" - \
  "$MODE" "$VALID_MASK" "$LOGICAL_DRAFTS" \
  "$TAW_SOURCE_SCHEMA" "$TAW_SOURCE_CONTRACT_SHA256" <<'PY'
import importlib.util
import sys
from pathlib import Path

path = Path("scripts/fr13_device_multidraft_kernel.py")
spec = importlib.util.spec_from_file_location("fr13_taw_source_v7_b1_preflight", path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot import source-v7 all-parent implementation")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mode = sys.argv[1]
topology = module._fr13_fixed32_topology()
contract = module._fr13_fixed32_taw_source_contract(topology, batch_size=1)
if (
    int(topology.VALID_MASK_BY_MODE[mode]) != int(sys.argv[2], 0)
    or module._fr13_fixed32_expected_active(topology, mode) != int(sys.argv[3])
    or module._FR13_FIXED32_TAW_SOURCE_SCHEMA != sys.argv[4]
    or module._FR13_FIXED32_TAW_SOURCE_SHA256 != sys.argv[5]
    or contract.get("source_contract_sha256") != sys.argv[5]
    or int(topology.PHYSICAL_DRAFTS) != 31
    or int(topology.PHYSICAL_ROWS) != 32
):
    raise SystemExit("source-v7 B1 preflight contract drifted")
PY

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER"
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_DRAFT_VOCAB_ROOT" == "1" \
   && "$FR13_DRAFT_VOCAB_K" == "65536" \
   && "$FR13_DRAFT_VOCAB_BLOCKS" == "$BLOCK_MAP_CONTAINER" \
   && "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" \
   && "$LUMO_SWE_AUTOCOMMIT" == "0" ]] \
  || { echo "K64 ROOT=1 B1 floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"
printf 'classification=one_real_swe_verified_k64_b1_graph_byte_gate\ntiming_eligible=0\nfloor_acceptance_eligible=0\nreference_always_served=1\ncandidate_returned=0\nsource_contract_schema=%s\nsource_contract_sha256=%s\nmode=%s\nlogical_topology=%s\nlogical_drafts=%s\nvalid_mask=%s\nphysical_drafts=31\nphysical_rows_root_inclusive=32\nbatch_size=1\nconcurrency=1\ntask_count=1\ntask_id=astropy__astropy-12907\ndraft_vocab_root=1\ndraft_vocab_k=65536\ndraft_vocab_blocks=%s\ndraft_vocab_blocks_sha256=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nenforce_eager=0\ncudagraph_mode=FULL_AND_PIECEWISE\nsource=%s\nrunner_sha256=%s\nsubset_sha256=%s\nstock_fa2_sha256=%s\nstarted=%s\n' \
  "$TAW_SOURCE_SCHEMA" "$TAW_SOURCE_CONTRACT_SHA256" "$MODE" \
  "$LOGICAL_TOPOLOGY" "$LOGICAL_DRAFTS" "$VALID_MASK" \
  "$BLOCK_MAP_CONTAINER" "$BLOCK_MAP_SHA256" "$MANDATORY_WEIGHT_BYTES" \
  "$MANDATORY_WEIGHT_FLOOR_MS" "$ONE_SIDED_U95_CAP_MS" \
  "$SOURCE_COMMIT" "$RUNNER_SHA256" "$SUBSET_SHA256" \
  "$STOCK_FA2_SHA256" "$(date -u +%FT%TZ)" \
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
    || { echo "runtime/source manifest changed during B1 gate" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during B1 gate" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
    || { echo "B1 gate runner changed during execution" >&2; return 14; }
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

if env \
    RUNROOT="$RUNROOT_ABS" \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
    LUMO_SWE_AUTOCOMMIT=0 \
    FR13_FIXED32_B1_DIAGNOSTIC=1 \
    FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
    FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER" \
    FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES" \
    FR13_WEIGHT_FLOOR_MS="$MANDATORY_WEIGHT_FLOOR_MS" \
    FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=1 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
    FR13_FIXED32_CONV_SOURCE_BATCH=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_CUTLASS_WAVE=stock \
    FR13_FIXED32_CUTLASS_WAVE_SO= \
    FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 \
    FORKED_FA2_SO="$STOCK_FA2_SO" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" "$MODE" "$SUBSET" \
      > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  :
else
  rc=$?
  printf 'serve_rc=%s ended=%s\n' "$rc" "$(date -u +%FT%TZ)" \
    >> "$RUNROOT_ABS/launcher_meta.txt"
  exit "$rc"
fi

CONTAINER_ENV="$ARMDIR/container_env.txt"
for expected in \
  "FR13_FIXED32_MODE=$MODE" \
  'FR13_FIXED32_B1_DIAGNOSTIC=1' \
  'FR13_DRAFT_VOCAB_ROOT=1' \
  'FR13_DRAFT_VOCAB_K=65536' \
  "FR13_DRAFT_VOCAB_BLOCKS=$BLOCK_MAP_CONTAINER" \
  'MAX_NUM_SEQS=1' \
  'ENFORCE_EAGER=0' \
  'FR13_SFWD_GPU_TIMER=1' \
  'FR13_DFWD_GPU_TIMER=1' \
  'FR13_CFWD_GPU_TIMER=1' \
  'FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=1' \
  'FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0'; do
  [[ "$(grep -Fxc "$expected" "$CONTAINER_ENV")" -eq 1 ]] \
    || { echo "container lacks exact B1 gate pin: $expected" >&2; exit 4; }
done
unset expected

finalize_manifests
LIVE_BUNDLE="$ARMDIR/logs/fr13_fixed32_taw_native_precompute.live_pass.json"
CURATED_LIVE="$ARMDIR/taw_source_v7_b1_live_bundle.json"
CREDENTIAL="$ARMDIR/taw_source_v7_b1_credential.json"
"$PYTHON_BIN" scripts/fr13_taw_b1_credential.py issue \
  --mode "$MODE" \
  --source "$TAW_SOURCE" \
  --topology "$TAW_TOPOLOGY" \
  --subset "$SUBSET" \
  --block-map "$BLOCK_MAP" \
  --live-bundle "$LIVE_BUNDLE" \
  --runtime-manifest "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  --health "$ARMDIR/health.json" \
  --traffic-audit "$ARMDIR/fixed32_chat_traffic_audit.json" \
  --runner "$RUNNER_PATH" \
  --source-commit "$SOURCE_COMMIT" \
  --curated-live-out "$CURATED_LIVE" \
  --out "$CREDENTIAL" \
  > "$ARMDIR/taw_source_v7_b1_credential_validation.json"

printf 'serve_rc=0 credential=%s credential_sha256=%s live_bundle=%s live_bundle_sha256=%s ended=%s\n' \
  "$CREDENTIAL" "$(sha256sum "$CREDENTIAL" | awk '{print $1}')" \
  "$CURATED_LIVE" "$(sha256sum "$CURATED_LIVE" | awk '{print $1}')" \
  "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
