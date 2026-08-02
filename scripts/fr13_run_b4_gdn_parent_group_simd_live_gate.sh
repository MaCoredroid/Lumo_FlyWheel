#!/usr/bin/env bash
# Real SWE-Verified B4 byte qualification for fixed32_gdn_parent_group_dense_simd_v3.
# The candidate is shadow-only and the incumbent bytes are always served.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new directory below output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned stock FA2 shared object}"

CAMPAIGN=${CAMPAIGN:-exact4}
GATE=${GATE:-graph}
FIXED32_MODE=${FIXED32_MODE:-hydra27_fixed32}
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
RUNROOT_ABS=$(realpath -m "$RUNROOT")

case "$CAMPAIGN" in
  exact4)
    SUBSET=config/fr13_fixed32/subset_b4_four.json
    SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
    ;;
  exact16)
    SUBSET=config/fr13_fixed32/subset_b4_sixteen.json
    SUBSET_SHA256=47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c
    ;;
  *) echo "CAMPAIGN must be exact4 or exact16" >&2; exit 2 ;;
esac
case "$GATE" in
  eager)
    GATE_ENFORCE_EAGER=1
    GROUP_EAGER=1
    GROUP_GRAPH=0
    ;;
  graph)
    GATE_ENFORCE_EAGER=0
    GROUP_EAGER=0
    GROUP_GRAPH=1
    ;;
  *) echo "GATE must be eager or graph" >&2; exit 2 ;;
esac
case "$FIXED32_MODE" in
  tail6_fixed32|hydra27_fixed32) ;;
  *) echo "FIXED32_MODE must be tail6_fixed32 or hydra27_fixed32" >&2; exit 2 ;;
esac

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ "$FORKED_FA2_SO" == /* && -f "$FORKED_FA2_SO" \
   && ! -L "$FORKED_FA2_SO" ]] \
  || { echo "FORKED_FA2_SO must be an absolute regular non-symlink file" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "$CAMPAIGN subset SHA-256 drift" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the gate" >&2; exit 2; }

export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_ROOT=0
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh
unset -f run_variant

ARM="${FIXED32_MODE}_gdn_parent_group_simd_${CAMPAIGN}_${GATE}_${TAG}"
mkdir -p "$RUNROOT_ABS"

env \
  RUNROOT="$RUNROOT_ABS" \
  OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S= \
  LUMO_SWE_AUTOCOMMIT=0 \
  FR13_FIXED32_B1_DIAGNOSTIC=0 \
  FR13_DRAFT_VOCAB_K=65536 FR13_DRAFT_VOCAB_ROOT=0 \
  FR10_METRICS=1 FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
  ENFORCE_EAGER="$GATE_ENFORCE_EAGER" CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
  FR13_FIXED32_GDN_PARENT_GROUP=1 \
  FR13_FIXED32_GDN_PARENT_GROUP_BYTE_AB=0 \
  FR13_FIXED32_GDN_PARENT_GROUP_SIMD_B4_EAGER="$GROUP_EAGER" \
  FR13_FIXED32_GDN_PARENT_GROUP_SIMD_B4_GRAPH="$GROUP_GRAPH" \
  FR13_FIXED32_GDN_PARENT_GROUP_SIMD_PRODUCTION=0 \
  FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
  FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
  FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
  FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
  FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
  FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
  FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
  FR13_FIXED32_BATCH_GDN_BV8_TIMING=0 \
  FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
  FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
  FR13_FIXED32_CUTLASS_WAVE=stock \
  FR13_FIXED32_CUTLASS_WAVE_SO= \
  FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
  FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
  FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 \
  FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 \
  FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
  FR13_DRAFT_HEAD_M32_LIVE_AB=0 \
  FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
  FR13_FA2_QROW16_LIVE_PAGED_AB=0 \
  FR13_FA2_QROW16_PRODUCTION=0 \
  FR13_FIXED32_ATTRIBUTION_ONLY=0 \
  FORKED_FA2_SO="$FORKED_FA2_SO" \
  bash scripts/fr13_bigdenom_swe_serve_variant.sh \
    "$ARM" "$FIXED32_MODE" "$SUBSET"

PASS_PATH="$RUNROOT_ABS/$ARM/logs/fr13_fixed32_gdn_parent_group_simd.${CAMPAIGN}.${GATE}.pass.json"
"$PYTHON_BIN" - "$PASS_PATH" "$CAMPAIGN" "$GATE" "$FIXED32_MODE" <<'PY'
import json
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
campaign, gate, mode = sys.argv[2:]
info = os.lstat(path)
payload = json.loads(path.read_text(encoding="ascii"))
if (
    not stat.S_ISREG(info.st_mode)
    or stat.S_ISLNK(info.st_mode)
    or info.st_nlink != 1
    or stat.S_IMODE(info.st_mode) != 0o444
    or payload.get("status") != "pass"
    or payload.get("candidate") != "fixed32_gdn_parent_group_dense_simd_v3"
    or payload.get("kernel") != "tree_gdn_parent_group_dense_simd_width4_v3"
    or payload.get("campaign") != campaign
    or payload.get("gate") != gate
    or payload.get("mode") != mode
    or payload.get("batch") != 4
    or payload.get("raw_byte_equal") is not True
    or payload.get("state_restored") is not True
    or payload.get("reference_served") is not True
    or payload.get("production_default_enabled") is not False
):
    raise SystemExit("grouped SIMD reduced PASS validation failed")
encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True)
if "swe_verified:" in encoded or "astropy__" in encoded:
    raise SystemExit("grouped SIMD reduced PASS exposed raw task identity")
print(path)
PY
