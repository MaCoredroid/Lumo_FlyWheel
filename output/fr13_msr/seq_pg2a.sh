# REGATE 2a (regate_queue.sh): FR13_PARENT_GATHER alone UNDER GRAPH CAPTURE.
# The lever's bit-identity was proven EAGER-only (offline gate + selfcheck
# both eager); deployment is graph-captured — this arm is the capture-mode
# gate the revert demanded. Mandatory legs: loop-watch (same-counter
# comparison vs bv1 baseline events 32/71/197/114), accept-inflation check,
# report "X pass, Y fail, Z finished". Deploy stack otherwise golden
# (default graph, cache-ON, committer batched); one lever only.
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
export FR13_PARENT_GATHER=1
run_variant pg2a_capture  tail6  21  1
