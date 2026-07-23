# FR13 FINAL ROUND at GPU_UTIL=0.70 — attempt 2 (arm order SWAPPED after
# native5_f70 crash1: vectorized_gather index-OOB device assert in compiled
# forward at ~35min B=4; log: native5_f70_crash1.log; tail6@0.70 has 1.5h
# clean probe evidence so it runs first; native5 retries second).
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1

export FR13_COMMITTER_BATCHED=1
run_variant tail6_batched_f70  tail6  21  1

unset FR13_COMMITTER_BATCHED
export FR13_COMMITTER_BATCHED=0
run_native  native5_f70_r2  5  5  1
