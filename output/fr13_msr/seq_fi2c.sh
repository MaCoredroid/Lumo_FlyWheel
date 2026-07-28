# REGATE 2c (regate_queue.sh): FR13_FLAGS_INKERNEL alone — scan kernel writes
# the staging-freshness flags itself (replaces 2 aten fills/layer/step = 96
# launches). First gate ever for this lever. Legs: loop-watch (same-counter
# vs bv1 32/71/197/114), accept band vs bv1x total 6.058 / comb 4.718,
# garble eyeball, engagement = generated-source const True (grep) + accept
# band intact (stale flags would collapse accept, not hide).
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
export FR13_FLAGS_INKERNEL=1
run_variant fi2c_flags  tail6  21  1
