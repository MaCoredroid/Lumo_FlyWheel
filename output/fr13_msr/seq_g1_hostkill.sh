# GATE g1 (post-BAR17, 4-task): sampler sync-kill (baked-in code path — the
# deployed device committer, no flag) + HCxPG compat (FR13_HC_INTERNAL=1 +
# FR13_PARENT_GATHER=1 together, first combo run) + floor-ratio fields.
# Legs: loop-watch, accept band (bv1x 6.058/4.718), garble eyeball, wall +
# floor_ratio readout. Timers armed (attribution-labeled).
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
export FR13_PARENT_GATHER=1
export FR13_HC_INTERNAL=1
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
run_variant g1_hostkill  tail6  21  1
