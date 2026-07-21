# SCAN_ALIGN FIX TEST (2026-07-21): pb chain-fold seed = custom tree-GDN kernel (SCAN_ALIGN OFF)
# diverges from the deployed NATIVE committer (FR13_COMMITTER_NATIVE=1, launcher default) that non-pb
# uses for col-0. FR13_SCAN_ALIGN=1 aligns the tree kernel to native SASS (l2norm div-by-sqrt + beta
# bf16 round-trip) -> chain-fold matches the native committer -> seed matches non-pb -> deep arctic
# tail recovers. ONLY change vs bc5 (SCAN_ALIGN OFF: 3.725/3.864/4.712); non-pb target 5.027/5.306.
# WIN: pb+base-col+SCAN_ALIGN accept -> ~5. Cache-independent (seed carrier). PARENT_GATHER=1 light compile.
# Launch: RUNROOT=output/fr13_scanalign TAG=sa1 SUBSET=subset_collapse3.json WALL=0 BSIZE=4 CONC=4
#   HEALTH_TIMEOUT_S=3600 SEQUENCE_FILE=scripts/fr13_pb_scanalign_seq.sh
#   bash scripts/fr13_campaign_tmux.sh scanalign
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_PARENT_GATHER=1
export FR13_PB_BASE_COL_INVARIANT=1
export FR13_SCAN_ALIGN=1
export FR13_ENABLE_APC=0
run_variant tail6pb_scanalign_${TAG}  tail6_pb  29  1
