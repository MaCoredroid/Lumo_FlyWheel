#!/usr/bin/env bash
# Real SWE-Verified exact4 B4 same-EngineCore byte gate for FA2 qrow32.
# The stock FULL graph is always served; this script emits no timing samples.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the qualified qrow32 FA2 SO}"

PYTHON_BIN=${PYTHON_BIN:-/home/mark/lumoFlyWheel-b4-compactionfix-rerun/.venv/bin/python}
FIXED32_MODE=${FR13_QROW32_FIXED32_MODE:-hydra27_fixed32}
SUBSET=config/fr13_fixed32/subset_b4_four.json
SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
TASK_IDS=astropy__astropy-12907,astropy__astropy-13033,astropy__astropy-13236,astropy__astropy-13398
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
B4_KV_CACHE_MEMORY_BYTES=42949672960
SOURCE_COMMIT=$(git rev-parse HEAD)
CANDIDATE_SHA256=$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")

case "$FIXED32_MODE" in
  tail6_fixed32)
    LOGICAL_TOPOLOGY=Tail23
    ;;
  hydra27_fixed32)
    LOGICAL_TOPOLOGY=Hydra27
    ;;
  *)
    echo "FR13_QROW32_FIXED32_MODE must be tail6_fixed32 or hydra27_fixed32" >&2
    exit 2
    ;;
esac
ARM="${FIXED32_MODE}_fa2_qrow32_gate_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ "$FORKED_FA2_SO" == /* && -f "$FORKED_FA2_SO" && ! -L "$FORKED_FA2_SO" ]] \
  || { echo "FORKED_FA2_SO must be an absolute regular non-symlink file" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ && "$CANDIDATE_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical exact4 subset SHA-256 drift" >&2; exit 2; }
[[ "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" ]] \
  || { echo "pinned K64 block-map SHA-256 drift" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the gate" >&2; exit 2; }

export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh
unset -f run_variant

mkdir -p "$RUNROOT_ABS"
printf 'classification=real_swe_verified_exact4_b4_fa2_qrow32_byte_diagnostic\nacceptance_valid=0\ntiming_eligible=0\nfloor_acceptance_eligible=0\nproduction_enabled=0\nreference_always_served=1\ntopology=%s\nlogical_topology=%s\ntask_ids=%s\nsubset_sha256=%s\nbatch_size=4\nconcurrency=4\nphysical_rows_per_slot=32\ntotal_query_rows=128\ndraft_vocab_k=65536\ndraft_vocab_root=1\ncandidate_so_sha256=%s\nsource_commit=%s\nrunner_sha256=%s\nlauncher_pid=%s\nstarted=%s\n' \
  "$FIXED32_MODE" "$LOGICAL_TOPOLOGY" "$TASK_IDS" "$SUBSET_SHA256" \
  "$CANDIDATE_SHA256" "$SOURCE_COMMIT" "$RUNNER_SHA256" "$$" \
  "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
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
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
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
    FR13_DRAFT_HEAD_PAD_ROWS=0 \
    FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 \
    FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 \
    FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FA2_QROW32_LIVE_PAGED_AB=1 \
    FR13_FA2_QROW32_LIVE_PAGED_AB_TASK_IDS="$TASK_IDS" \
    FR13_FA2_QROW32_LIVE_PAGED_AB_SUBSET_SHA256="$SUBSET_SHA256" \
    FR13_FA2_QROW32_LIVE_PAGED_AB_JSON=/logs/fr13_fa2_qrow32_live_paged_ab.json \
    FR13_FA2_QROW32_SO_SHA256="$CANDIDATE_SHA256" \
    FR13_FA2_QROW32_SOURCE_COMMIT="$SOURCE_COMMIT" \
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
  --repo "$PWD" --profile fixed32 \
  --sequence scripts/fr13_fixed32_floor_timers_seq.sh \
  --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json"
cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
  "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  || { echo "runtime/source manifest changed during qrow32 gate" >&2; exit 14; }
cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
  "$RUNROOT_ABS/external_manifest.at_end.json" \
  || { echo "external manifest changed during qrow32 gate" >&2; exit 14; }
[[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
  || { echo "qrow32 gate runner changed during execution" >&2; exit 14; }
(( serve_rc == 0 )) || exit "$serve_rc"

"$PYTHON_BIN" scripts/fr13_fa2_qrow32_gate.py verify-live \
  --result "$ARMDIR/logs/fr13_fa2_qrow32_live_paged_ab.json" \
  --campaign-arm "$ARMDIR/swe_out/verified/fixed32_taw_campaign_arm.json" \
  --campaign-provenance "$ARMDIR/swe_out/verified/fixed32_qwen_campaign_provenance.json" \
  --candidate-so "$FORKED_FA2_SO" \
  --fixed32-mode "$FIXED32_MODE" \
  --source-commit "$SOURCE_COMMIT" \
  > "$ARMDIR/qrow32_live_verification.json"
