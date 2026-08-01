#!/usr/bin/env bash
# Offline compilation only. This does not run a GPU kernel or issue qualification.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO"

: "${PYTHON_BIN:?set PYTHON_BIN to pinned Torch 2.10.0+cu130 Python}"
SOURCE_COMMIT=6ca0e1428144b8756722861149d5e97bea7f8b37
SOURCE_SHA256=b1e9c5ce798f6b16b652be86b9a5c38b4e0f8040d881401e853705200a4638f1
BUILDER_SHA256=9ce31e68ed7937f7c7cd9ef002efd9ef286c9ecc35947f774b5056cd37e609b2

[[ -x "$PYTHON_BIN" ]] || { echo "PYTHON_BIN is not executable" >&2; exit 2; }
git merge-base --is-ancestor "$SOURCE_COMMIT" HEAD \
  || { echo "bound source commit is not an ancestor of HEAD" >&2; exit 2; }
[[ "$(sha256sum csrc/fr13_bf16_gemvx_b1_b4.cu | awk '{print $1}')" == "$SOURCE_SHA256" ]] \
  || { echo "B1-B4 CUDA source drifted" >&2; exit 2; }
[[ "$(sha256sum scripts/fr13_build_bf16_gemvx_b1_b4.py | awk '{print $1}')" == "$BUILDER_SHA256" ]] \
  || { echo "B1-B4 builder drifted" >&2; exit 2; }
"$PYTHON_BIN" - <<'PY'
import torch

if torch.__version__ != "2.10.0+cu130":
    raise SystemExit(f"requires torch 2.10.0+cu130, got {torch.__version__}")
PY

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT="$REPO/output/fr13_bf16_gemvx_b1_b4_build_$STAMP"
mkdir -p "$RUNROOT"
"$PYTHON_BIN" scripts/fr13_build_bf16_gemvx_b1_b4.py \
  --output "$RUNROOT/fr13_bf16_gemvx_b1_b4.abi3.so" \
  --build-dir "$RUNROOT/build" \
  --attestation "$RUNROOT/build_attestation.json"
sha256sum \
  "$RUNROOT/fr13_bf16_gemvx_b1_b4.abi3.so" \
  "$RUNROOT/build_attestation.json" \
  > "$RUNROOT/SHA256SUMS"
printf 'BUILT_UNQUALIFIED=%s\n' "$RUNROOT"
