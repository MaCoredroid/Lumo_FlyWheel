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
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_allon/tail6pb_cfwd.json
run_variant tail6pb_${TAG}  tail6_pb  29  1
