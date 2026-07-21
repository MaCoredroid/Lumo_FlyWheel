# pb+fix+CACHE-ON on the 3 deep collapse tasks (user 2026-07-21: drop the control arm -- use
# HISTORIC non-pb cache-ON pbab1 [14539=5.027, 14995=5.306] as the target; small subset for the trend).
# QUESTION: does FR13_PB_BASE_COL_INVARIANT + CACHE ON recover pb toward the non-pb cache-ON ~5.0-5.3
# on the collapse cases? (bc5 pb+fix CACHE-OFF was 3.73/4.71/3.86 -- does cache-ON move it?)
# One arm, 3 tasks (~1h). PARENT_GATHER=1 (byte-identical O(N) kernel, light compile). Cache-ON config.
# Launch: RUNROOT=output/fr13_pbcacheon TAG=pco1 SUBSET=subset_collapse3.json WALL=0 BSIZE=4 CONC=4
#   HEALTH_TIMEOUT_S=3600 SEQUENCE_FILE=scripts/fr13_pb_cacheon_deep_seq.sh
#   bash scripts/fr13_campaign_tmux.sh pbcacheon
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_PARENT_GATHER=1
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_pbcacheon/cf_pco1.json
export FR13_PB_BASE_COL_INVARIANT=1
run_variant tail6pb_cacheon_${TAG}  tail6_pb  29  1
