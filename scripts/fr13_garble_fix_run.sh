#!/bin/bash
# FR13 garble-fix iteration runner: cat8+cache (stateless-tree trio + APC) + a FORWARD-DRIFT FIX,
# on a task subset. Gate = give-ups drop toward native (0 garble). Launched detached (setsid).
#   env: FIX_FLAGS="LUMO_FB_KERNEL_ROWS=1 ..."  ARM=<name>  SUBSET=<json>  RUNROOT=<dir>
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
ARM=${ARM:?arm}; SUBSET=${SUBSET:?subset}; RUNROOT=${RUNROOT:?runroot}
mkdir -p "$RUNROOT" output/fr13_sfwd_sidecar
# cat8+cache baked config (from cachefirst_seq Arm 1) + the fix flags exported by the caller
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1
echo "=== garble-fix run ARM=$ARM SUBSET=$SUBSET FIX: LUMO_FB_KERNEL_ROWS=${LUMO_FB_KERNEL_ROWS:-unset} LUMO_FB_PROJ_PAD_ROWS=${LUMO_FB_PROJ_PAD_ROWS:-unset} ==="
RUNROOT="$RUNROOT" bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" cat8 "$SUBSET"
echo "=== garble-fix run $ARM DONE rc=$? ==="
