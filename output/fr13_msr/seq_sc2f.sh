# REGATE 2f (regate_queue.sh): NODEBANK + SPEC_BLOCKS_CAP=12 (surgery piece 3
# — the payoff). num_speculative_blocks 21->12 => 13 pages/request, ~+9
# pages/request reclaimed for cached history. Judge the nodebank FAMILY here:
# hit-rate (expect toward 85%), prefill_frac (toward 0.13-native-direction),
# wall recovery vs 2e's 28.05 bank-tax reading, accept band, no OOB, and the
# enforced stateless assert now live. Legs: loop-watch, garble eyeball.
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
export FR13_CONV_NODEBANK=1
export FR13_SPEC_BLOCKS_CAP=12
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
run_variant sc2f_cap  tail6  21  1
