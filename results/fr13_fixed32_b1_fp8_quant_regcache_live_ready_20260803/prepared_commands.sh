#!/usr/bin/env bash
set -euo pipefail

REPO=/home/mark/lumoFlyWheel-b1-fp8-quant-regcache-livegate-20260803
FP8_QUANT_SO=/home/mark/fr13_fp8_quant_regcache_live_build_20260803/runtime-v2/_C_stable_libtorch.fp8_quant_regcache.sm121a.abi3.so
FP8_QUANT_SO_SHA256=847599fc7e3250cd56963592d4786d5f32fe5a391da107b4a791198a7d59c110
: "${FORKED_FA2_SO:?set FORKED_FA2_SO to the pinned stock FA2 binary}"

cd "$REPO"

# Gate first: exactly one real SWE-Verified K64/root1 B1 task, no timing.
RUNROOT=output/fr13_fixed32_b1_fp8_quant_regcache_byte_gate_20260803 \
TAG=bytegate_20260803 \
FORKED_FA2_SO="$FORKED_FA2_SO" \
FP8_QUANT_SO="$FP8_QUANT_SO" \
FP8_QUANT_SO_SHA256="$FP8_QUANT_SO_SHA256" \
bash scripts/fr13_run_b1_fp8_quant_regcache_live_gate.sh

# After PASS, set these from the gate's final three output lines.
: "${FP8_QUANT_PASS:?set FP8_QUANT_PASS to the gate production sidecar}"
: "${FP8_QUANT_PASS_SHA256:?set FP8_QUANT_PASS_SHA256 to its SHA-256}"

# Screen next: standing exact four-task set, paired stock/candidate full-step TPS.
RUNROOT=output/fr13_fixed32_b1_fp8_quant_regcache_exact4_20260803 \
TAG=exact4_20260803 \
FORKED_FA2_SO="$FORKED_FA2_SO" \
FP8_QUANT_SO="$FP8_QUANT_SO" \
FP8_QUANT_SO_SHA256="$FP8_QUANT_SO_SHA256" \
FP8_QUANT_PASS="$FP8_QUANT_PASS" \
FP8_QUANT_PASS_SHA256="$FP8_QUANT_PASS_SHA256" \
bash scripts/fr13_run_b1_fp8_quant_regcache_timing.sh
