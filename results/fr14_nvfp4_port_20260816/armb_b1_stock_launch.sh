#!/usr/bin/env bash
# FR14 ARM B — the stock B1 SWE serve, the bytes-ablation deliverable.
#
# Byte-identical in shape to arm A's fr14_b1_stock_20260816T204931Z/tail6 arm
# (BSIZE=1, CONC=1, ROOT=1, K=65536, empty selectors, subset_b4_four,
# fr13_fixed32_floor_timers_seq). The ONLY variable is the checkpoint and the
# floor that follows from it -- which is the whole point of the ablation, so
# nothing else may move.
#
#   arm A: /models/qwen3.8-27b-nvfp4          27,977,022,848 B / 102.479937172 ms (K64)
#   arm B: /models/qwen3.8-27b-nvfp4-radixark 25,430,574,256 B /  93.15228665201465 ms (K0)
#
# K0 (FULL-VOCAB DRAFTING) -- MARK'S RULING, 2026-08-17. K64 is PARKED FOREVER;
# full-vocab drafting through the NVFP4 head is the production config. This
# supersedes the earlier "K64-as-built" staging of this script.
#
# The ablation that produced the ruling (armb_k64_ablation.md): under the NVFP4
# head K64's byte advantage is only 0.807 ms of floor, while it costs 0.476
# accepted tokens/event -- net -10.5% throughput on the 1024/1024 shape. K64's
# remaining justification was ~3.0 ms/step of DFWD compute, smaller than the
# acceptance it gives up. K0 also retires the DVK shim, the Phase-1 boot dequant
# and the 128-id block map from the serving path.
#
# The DVK shim is INERT BY CONSTRUCTION here: _fr13_dvk_prepare returns at its
# own `_fr13_dvk_configured <= 0` early return, so neither the shim nor the
# Phase-1 dequant runs. Verified by needle -- ZERO [FR13_DRAFT_VOCAB] /
# [FR14_DVK_DEQUANT] lines in the boot log (the same check the K64/K0 bench
# used). NOTE it is the K-value that makes it inert, NOT the root flag: there
# are two _fr13_dvk_prepare call sites and the second is taken when
# `not _fr13_dvk_root`.
#
# FR13_NEEDS_ALLOW is the launcher's sanctioned override for the full-vocab arm
# (it gates 0:0 behind an explicit opt-in so nobody drifts off K64 by accident).
# FOLLOW-UP CONFIG TRAIN: promote K=0 to the canonical default and retire the
# K64 machinery from the serving path, so this override is no longer needed.
# The code stays -- git preserves it -- but the serving path should stop
# carrying a subset head nothing selects.
#
# Reduction env (the campaign driver's reducer picks these up from the sequence
# file, so they are stated here only for the record):
#   FR13_MANDATORY_WEIGHT_BYTES=25430574256
#   FR13_WEIGHT_FLOOR_MS=93.15228665201465
#
# Usage: bash armb_b1_stock_launch.sh   (launch detached; the parent monitors)
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
cd "$REPO"

TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT=output/fr14_b1_stock_$TS
mkdir -p "$RUNROOT"
printf '%s\n' "$RUNROOT" > /home/mark/shared/tmp-scratch/fr14_armb_b1_stock_runroot.txt

echo "[b1radix] memory preflight"
free -g
awk '/^MemFree:/{exit ($2/1048576 < 82.3)}' /proc/meminfo \
  || { echo "[b1radix] FAIL: unified-memory preflight"; exit 2; }
docker ps -q | grep -q . && { echo "[b1radix] FAIL: docker not empty"; exit 2; }

echo "[b1radix] runroot=$RUNROOT head=$(git rev-parse HEAD)"
BSIZE=1 CONC=1 TAG=b1radix RUNROOT="$RUNROOT" WALL=5400 \
  SUBSET=config/fr13_fixed32/subset_b4_four.json \
  SEQUENCE_FILE=scripts/fr13_fixed32_floor_timers_seq.sh \
  FR13_DRAFT_VOCAB_K=0 FR13_DRAFT_VOCAB_ROOT=0 \
  FR13_NEEDS_ALLOW="FR13_DRAFT_VOCAB_K=0" \
  TMPDIR=/home/mark/shared/tmp-scratch \
  bash scripts/fr13_b4_campaign_driver.sh > "$RUNROOT/driver.log" 2>&1
echo "[b1radix] driver rc=$? $(date -u +%H:%M:%SZ)"
