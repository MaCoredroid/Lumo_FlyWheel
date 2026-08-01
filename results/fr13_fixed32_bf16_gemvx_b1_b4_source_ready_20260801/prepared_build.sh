#!/usr/bin/env bash
# Offline compilation only. This does not run a GPU kernel or issue qualification.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO"

: "${PYTHON_BIN:?set PYTHON_BIN to pinned Torch 2.10.0+cu130 Python}"
SOURCE_COMMIT=3ed2d6d8d2c7fe68b44a5d34835bbcfa68bc2101
SOURCE_SHA256=4412fc292ffc5e9a7786deb857fdeb99a7283b2f0ba4833df5c2141668f3902c
BUILDER_SHA256=2963c60cbd46c0bd314a1907a8ebb293b4324a339176d93e644cd3ad308602e7

[[ -x "$PYTHON_BIN" ]] || { echo "PYTHON_BIN is not executable" >&2; exit 2; }
git merge-base --is-ancestor "$SOURCE_COMMIT" HEAD \
  || { echo "bound source commit is not an ancestor of HEAD" >&2; exit 2; }
[[ "$(sha256sum csrc/fr13_bf16_gemvx_m1.cu | awk '{print $1}')" == "$SOURCE_SHA256" ]] \
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
