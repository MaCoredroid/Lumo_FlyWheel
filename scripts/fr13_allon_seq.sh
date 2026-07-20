# ALL-ON ARM (user directive 2026-07-20): the ship config in one shot —
# tail6_pb (ported piggyback, 29 cols) + APC cache + --async-scheduling.
# Strategy = all-flags-forward (the pb-debug playbook): boot needles fail fast
# if the port is broken; on-band resolve => NEW BASELINE, iterate from here.
# Peel order if broken: (1) drop async, (2) drop cache (FR13_ENABLE_APC=0),
# (3) tail6_pb -> tail6 (pb off).
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
# allon4 postmortem 2026-07-20: at 0.72 the engine-ready unified-mem margin
# is INSIDE boot noise (avail dipped 8644MiB < the 9000MiB gpu_oom_guard
# fence -> clean-boot kill at ready; allon3 same config measured ~9.1GB).
# Fence-graze precedent (0.82->0.78): shrink the workload, never the fence.
# NOTE config delta vs rg1/pureprobe comparators (0.72): KV/APC capacity
# only, no per-token semantics; resolve band absorbs it.
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_allon/tail6pb_cfwd.json
run_variant tail6pb_${TAG}  tail6_pb  29  1
