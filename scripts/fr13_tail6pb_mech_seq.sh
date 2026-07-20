# TAIL6_PB MECHANICAL VALIDATION (quick, 4-task subset): first boot of the
# ported piggyback on the deliverable shape. Gates: engage needles ('extended
# drafter engaged' 29 cols + 'committer GDN replay DROPPED'), drafts==29,
# 0 fatal, committer CFWD collapse (~10-16ms vs tail6's 85.5), no raises.
# Launch: RUNROOT=output/fr13_resolve TAG=t6pb1 SUBSET=subset_b4_four.json
#   WALL=0 BSIZE=4 CONC=4 SEQUENCE_FILE=scripts/fr13_tail6pb_mech_seq.sh
#   bash scripts/fr13_campaign_tmux.sh t6pbmech
# CACHE-ON (post-rg-pair policy): forked-launcher APC stack
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1
unset FR13_SERVE_BATCH_FLAGS
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_resolve/tail6pb_cfwd.json
run_variant tail6pb_${TAG}  tail6_pb  29  1
