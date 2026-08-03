#!/usr/bin/env bash
# Real SWE-Verified K64/root1 B1 or exact4 B4 graph tree-conv byte gate.
# The incumbent kernel is restored and served; this emits no timing samples.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned stock FA2 binary}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
BATCH=${TREECONV_GATE_BATCH:-4}
MODE=${TREECONV_GATE_MODE:-hydra27_fixed32}
SOURCE_COMMIT=$(git rev-parse HEAD)
SOURCE=src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
BLOCKS=scripts/fr13_dvk_subset_blocks.json
BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STOCK_FA2_BYTES=299183936
RUNROOT_ABS=$(realpath -m "$RUNROOT")

case "$BATCH" in
  1)
    SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
    B1=1
    CLASS=one_real_swe_verified_b1_k64_root_treeconv_graph_gate
    ;;
  4)
    SUBSET=config/fr13_fixed32/subset_b4_four.json
    B1=0
    CLASS=real_swe_verified_exact4_b4_k64_root_treeconv_graph_gate
    ;;
  *) echo "TREECONV_GATE_BATCH must be exactly 1 or 4" >&2; exit 2 ;;
esac
case "$MODE" in
  tail6_fixed32|hydra27_fixed32) ;;
  *) echo "TREECONV_GATE_MODE must be tail6_fixed32 or hydra27_fixed32" >&2; exit 2 ;;
esac
[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* && ! -e "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be a new path below repository output" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python environment is unavailable" >&2; exit 2; }
[[ -f "$FORKED_FA2_SO" && ! -L "$FORKED_FA2_SO" ]] \
  || { echo "FORKED_FA2_SO must be a regular non-symlink file" >&2; exit 2; }
[[ "$(stat -c '%s' "$FORKED_FA2_SO")" == "$STOCK_FA2_BYTES" \
   && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" == "$STOCK_FA2_SHA256" ]] \
  || { echo "FORKED_FA2_SO is not the pinned stock reference" >&2; exit 2; }
[[ "$(sha256sum "$BLOCKS" | awk '{print $1}')" == "$BLOCKS_SHA256" ]] \
  || { echo "root-64K block map drifted" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the gate" >&2; exit 2; }

ARM="${MODE}_treeconv_zero_tail_b${BATCH}_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"
export BSIZE="$BATCH" CONC="$BATCH" WALL=0
export FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS="$BLOCKS_CONTAINER" FR13_NEEDS_ALLOW=
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant

mkdir -p "$RUNROOT_ABS"
printf 'classification=%s\ntiming_eligible=0\nacceptance_valid=0\nreference_always_served=1\nbatch_size=%s\nconcurrency=%s\nmode=%s\nphysical_drafts=31\nphysical_rows_root_inclusive=32\ndraft_vocab_root=1\ndraft_vocab_k=65536\nsource_commit=%s\nstarted=%s\n' \
  "$CLASS" "$BATCH" "$BATCH" "$MODE" "$SOURCE_COMMIT" \
  "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"

if env \
    RUNROOT="$RUNROOT_ABS" \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR="$BATCH" SWE_CONCURRENCY="$BATCH" \
    AGENT_WALL_S=5400 KV_CACHE_MEMORY_BYTES=42949672960 \
    FR13_FIXED32_B1_DIAGNOSTIC="$B1" \
    FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
    FR13_DRAFT_VOCAB_BLOCKS="$BLOCKS_CONTAINER" FR13_NEEDS_ALLOW= \
    FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
    FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
    FR13_FIXED32_CONV_COMMIT_ZERO_TAIL=0 \
    FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB=1 \
    FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB_LIMIT=320 \
    FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB_PATH=/logs/fr13_fixed32_treeconv_zero_tail.byte_ab.jsonl \
    FR13_FIXED32_CUTLASS_WAVE=stock \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_TIMING_AB=0 \
    FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB=0 \
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 \
    FR13_FIXED32_COMMITTER_LAYER_BATCH_QUALIFICATION=0 \
    FORKED_FA2_SO="$FORKED_FA2_SO" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" "$MODE" "$SUBSET" > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi
printf 'serve_rc=%s ended=%s\n' "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
  "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  || { echo "runtime/source manifest changed during gate" >&2; exit 14; }
(( serve_rc == 0 )) || exit "$serve_rc"

QWEN_ARGS=()
if [[ "$BATCH" == "4" ]]; then
  QWEN_ARGS=(--qwen-campaign "$ARMDIR/swe_out/verified/fixed32_qwen_campaign_provenance.json")
fi
"$PYTHON_BIN" scripts/fr13_treeconv_zero_tail_credential.py \
  --comparator "$ARMDIR/logs/fr13_fixed32_treeconv_zero_tail.byte_ab.jsonl" \
  --subset "$SUBSET" --health "$ARMDIR/health.json" \
  --proxy-ledger "$ARMDIR/logs/fr13_fixed32_proxy_ingress.jsonl" \
  --engine-ledger "$ARMDIR/logs/fr13_fixed32_engine_ingress.jsonl" \
  --work-census "$ARMDIR/logs/fr13_fixed32_work_census.jsonl" \
  --final-flush "$ARMDIR/fixed32_final_flush.json" \
  --boundary-snapshot-base "$ARMDIR/logs/fr13_fixed32_boundary_snapshot" \
  --runtime-manifest-launch "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
  --runtime-manifest-end "$RUNROOT_ABS/runtime_manifest.at_end.json" \
  --runtime-git-head "$ARMDIR/git_head.txt" \
  --source "$SOURCE" --source-commit "$SOURCE_COMMIT" \
  --repo "$PWD" --container-env "$ARMDIR/container_env.txt" \
  --task-root "$ARMDIR/swe_out/verified/per_task" \
  --mode "$MODE" --batch-size "$BATCH" "${QWEN_ARGS[@]}" \
  --output "$ARMDIR/treeconv_zero_tail.credential.json"
