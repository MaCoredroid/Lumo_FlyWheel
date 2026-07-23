# 16-task gate: graph committer (FR13_COMMITTER_GRAPH=1; takes dispatch
# precedence over baked batched). Baselines @0.70: tail6_batched_f70
# (tps 37.37, committer 36.0ms, comb 3.592, 8P/8F), native5_f70_r8 bar 50.99.
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
export FR13_COMMITTER_GRAPH=1
run_variant tail6_graph_f70  tail6  21  1
