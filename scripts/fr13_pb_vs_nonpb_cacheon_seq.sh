# CONTROLLED pb-vs-nonpb A/B on tail6, BOTH CACHE-ON (user directive 2026-07-20):
# isolate piggyback at the SHIP regime (cache-ON), no cache-off/on confound. On
# the 3 collapse tasks. arm A = tail6 (non-pb, committer replay) cache-ON; arm B
# = tail6_pb (pb, new generalized mask) cache-ON. Expect: tail6 ~5 (deep tail
# works), tail6_pb ~3.9 (mask-recovered but deep-tail divergence remains) =>
# delta = PIGGYBACK deep-tail non-losslessness, at matched cache-ON.
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
# arm A: NON-PB (plain tail6, committer replay) cache-ON
run_variant tail6_nonpb_${TAG}   tail6     21  1
# arm B: PB (tail6_pb, new mask) cache-ON
run_variant tail6pb_${TAG}       tail6_pb  29  1
