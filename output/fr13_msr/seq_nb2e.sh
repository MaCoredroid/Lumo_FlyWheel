# REGATE 2e (regate_queue.sh): FR13_CONV_NODEBANK alone (spec-page surgery
# piece 1+2). Byte gates: route CPU 36/36 + GPU 36/36 ALL-IDENTICAL (incl.
# ordinal-perm composition cases). Legs: loop-watch, accept band (bv1x
# 6.058/4.718), garble eyeball, AND the arm-specific watch: turn boundaries
# exercise the ordinal-perm machinery — composition-change steps are where
# any residual keying bug would bite. Timers armed (attribution-labeled).
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
export FR13_CONV_NODEBANK=1
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
run_variant nb2e_bank  tail6  21  1
