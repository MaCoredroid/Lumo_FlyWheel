# THE BAR SHOT: full-stack tail6 16-task @0.70 vs native5's 50.99.
# Stack (all baked): B1+B2a+B2b+ssi-prebuild+syncfree+guard+repair+
# conv-pregather+parent-gather. Bar = native5_f70_r8 measured_tps_fullstep_wall
# 50.99 (valid: native unchanged since r8 — guard/repair were already in it;
# all later fixes are tree-side).
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
run_variant tail6_bar16  tail6  21  1
