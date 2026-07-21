# FR13 direction-2: single-arm burn-off tail6 run WITH the committer GPU timer live, so we can poll
# vllm:fr13_committer_gpu_seconds directly on /metrics (no need to wait for full task completion --
# it's an async-cuda-event prometheus Counter, updated continuously during decode).
export FR13_BURN_REDUNDANCY_TEST=1
export FR13_COMMITTER_GRAPH=0
export FR13_CFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
run_variant cfwd_${TAG}  tail6  21  1
