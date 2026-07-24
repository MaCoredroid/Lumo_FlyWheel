# REGATE 2h (regate_queue.sh): FR13_CONV_WB_BATCHED alone (B2c committer
# host-gap attack; pool route — NODEBANK family parked on the boundary-copy
# design). Staging preseeded at builder init (capacity-keyed). Legs:
# loop-watch, accept band (bv1x 6.058/4.718), garble eyeball, cfwd span
# delta = the win metric (timers armed, attribution-labeled).
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
export FR13_CONV_WB_BATCHED=1
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
run_variant wb2h_batch  tail6  21  1
