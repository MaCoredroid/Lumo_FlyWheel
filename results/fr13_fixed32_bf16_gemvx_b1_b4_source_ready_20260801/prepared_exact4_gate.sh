#!/usr/bin/env bash
# Real SWE-Verified exact4 shadow byte gate. This is not a timing run.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO"

: "${PYTHON_BIN:?set PYTHON_BIN to pinned Torch 2.10.0+cu130 Python}"
: "${FORKED_FA2_SO:?set the absolute canonical FA2 SO path}"
: "${FR13_DRAFT_HEAD_M1_SO:?set the absolute B1-B4 candidate SO path}"
: "${FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION:?set the absolute build attestation path}"

SOURCE_COMMIT=6ca0e1428144b8756722861149d5e97bea7f8b37
CAMPAIGN_FIX_SOURCE_COMMIT=0f2a31ed298758cba72fad7e77fc3e13e27d545a
CAMPAIGN_FIX_SOURCE_PATCH_ID=90d32f2bea4e910ee49e8f8ede6a64fe9ae6e4a7
INTEGRATED_CAMPAIGN_FIX_COMMIT=1add76690849bc18f67af59c332b9e808bcc7c0a
SOURCE_SHA256=b1e9c5ce798f6b16b652be86b9a5c38b4e0f8040d881401e853705200a4638f1
PATCHER_SHA256=b536d139bba5888f2bd7fd1fd5de5665553f144d77e0b63b6bdcd9a415471afb
LAUNCHER_SHA256=951404b17805d07c9b3256096c72f3a54818b31347e5e5b3d8243102b6eb0623
RUNNER_SHA256=68960a2f94d779174855a8bf277b8918099e229e4236f63e56b159680da671a0

git merge-base --is-ancestor "$SOURCE_COMMIT" HEAD \
  || { echo "B1-B4 source commit is not integrated" >&2; exit 2; }
git merge-base --is-ancestor "$INTEGRATED_CAMPAIGN_FIX_COMMIT" HEAD \
  || { echo "integrated runner-owned B4 endpoint fix is absent" >&2; exit 2; }
[[ "$(git show "$INTEGRATED_CAMPAIGN_FIX_COMMIT" --pretty=format: | git patch-id --stable | cut -d' ' -f1)" \
   == "$CAMPAIGN_FIX_SOURCE_PATCH_ID" ]] \
  || { echo "integrated B4 endpoint fix differs from $CAMPAIGN_FIX_SOURCE_COMMIT" >&2; exit 2; }
[[ -z "$(git status --porcelain=v1)" ]] \
  || { echo "exact4 gate requires a clean checkout" >&2; exit 2; }
[[ "$(sha256sum csrc/fr13_bf16_gemvx_b1_b4.cu | awk '{print $1}')" == "$SOURCE_SHA256" \
   && "$(sha256sum scripts/fr13_phase4_patch_vllm_tree_gdn_b1_b4.py | awk '{print $1}')" == "$PATCHER_SHA256" \
   && "$(sha256sum scripts/fr13_launch_forked_fa2_tree_server.sh | awk '{print $1}')" == "$LAUNCHER_SHA256" \
   && "$(sha256sum scripts/fr13_run_b4_draft_head_b1_b4_live.sh | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
  || { echo "B1-B4 exact4 source or route identity drifted" >&2; exit 2; }

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT="output/fr13_b1_b4_exact4_shadow_$STAMP" \
TAG="b1_b4_shadow_$STAMP" \
PYTHON_BIN="$PYTHON_BIN" \
FORKED_FA2_SO="$FORKED_FA2_SO" \
FR13_DRAFT_HEAD_M1_SO="$FR13_DRAFT_HEAD_M1_SO" \
FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION="$FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION" \
FR13_DRAFT_HEAD_M1_EXPECTED_SOURCE_COMMIT="$(git rev-parse HEAD)" \
  bash scripts/fr13_run_b4_draft_head_b1_b4_live.sh
