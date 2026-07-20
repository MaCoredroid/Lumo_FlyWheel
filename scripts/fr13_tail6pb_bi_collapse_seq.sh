# FIX-ATTEMPT (user: fix + rerun): the residual -1.1 is likely the 29-col fused
# forward's autotune numerics diverging from 21-col (within-floor ULP tipping
# temp-0.6 ties on the deep tasks). BATCH_INVARIANT=1 pins the kernels
# deterministic -> stops the autotune fork. tail6_pb cache-OFF + NEW mask on the
# 3 collapse tasks. Recover to ~5 => autotune-fork was the carrier (this is the
# fix); flat ~3.9 => inherent 29-vs-21 cost, build the localization gate.
export BATCH_INVARIANT=1
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_ENABLE_APC=0
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
run_variant tail6pb_bi_${TAG}  tail6_pb  29  1
