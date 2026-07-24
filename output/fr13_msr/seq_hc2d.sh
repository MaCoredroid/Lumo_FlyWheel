# REGATE 2d leg ii (regate_queue.sh): FR13_HC_INTERNAL alone UNDER GRAPH
# CAPTURE (eager byte gate PASSED 18:44Z: bit-identical, mask=0x1f2493,
# 32->16 rows). Legs: loop-watch (same-counter vs bv1 32/71/197/114),
# accept band (bv1x total 6.058 / comb 4.718), garble eyeball, derivation
# needle present. Deferred span timers armed — timer numbers are
# ATTRIBUTION-labeled, never the speed-of-record (two-kinds rule).
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
export FR13_HC_INTERNAL=1
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
run_variant hc2d_capture  tail6  21  1
