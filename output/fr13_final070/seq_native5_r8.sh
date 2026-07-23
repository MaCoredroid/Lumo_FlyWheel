# native5 @0.70 re-baseline r6 = REAL measurement attempt.
# Fix stack: FR13_INPUTPREP_GUARD (draft rescue + sample assert) +
# FR13_REPLAY_DRAFT_REQKEY repair gate OPENED to native (was tree_mtp-only;
# dbg8 stranded-sampled-token class = the 4/4 crash series).
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
export FR13_INPUTPREP_GUARD=1
run_native  native5_f70_r8  5  5  1
