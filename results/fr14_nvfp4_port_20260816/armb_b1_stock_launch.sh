#!/usr/bin/env bash
# FR14 ARM B — the stock B1 SWE serve, the bytes-ablation deliverable.
#
# Byte-identical in shape to arm A's fr14_b1_stock_20260816T204931Z/tail6 arm
# (BSIZE=1, CONC=1, ROOT=1, K=65536, empty selectors, subset_b4_four,
# fr13_fixed32_floor_timers_seq). The ONLY variable is the checkpoint and the
# floor that follows from it -- which is the whole point of the ablation, so
# nothing else may move.
#
#   arm A: /models/qwen3.8-27b-nvfp4          27,977,022,848 B / 102.479937172 ms
#   arm B: /models/qwen3.8-27b-nvfp4-radixark 25,210,209,416 B /  92.345089436 ms
#
# K64-AS-BUILT ON PURPOSE. The K64-vs-K0 ablation runs separately and informs
# the NEXT config train; mixing a K0 decision into this serve would confound the
# bytes ablation with a drafting-strategy change.
#
# Reduction env (the campaign driver's reducer picks these up from the sequence
# file, so they are stated here only for the record):
#   FR13_MANDATORY_WEIGHT_BYTES=25210209416
#   FR13_WEIGHT_FLOOR_MS=92.345089436
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
  FR13_DRAFT_VOCAB_ROOT=1 \
  TMPDIR=/home/mark/shared/tmp-scratch \
  bash scripts/fr13_b4_campaign_driver.sh > "$RUNROOT/driver.log" 2>&1
echo "[b1radix] driver rc=$? $(date -u +%H:%M:%SZ)"
