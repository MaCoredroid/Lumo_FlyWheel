# ACCEPT REGRESSION ISOLATION (2026-07-20): allon5 tail6_pb (piggyback + APC
# cache, async, stale-root fix ON) accept 4.417 vs rg2c plain-tail6 (piggyback
# OFF, cache OFF, async, SAME stale-root fix ON) 5.36 -- a -0.94 accept drop.
# The fix (FR13_REPLAY_ROUTE=1) is engaged in BOTH; the delta is the two things
# rg2c lacked: PIGGYBACK and CACHE. Attribute via the 2x2 factorial (tail6 x
# {pb, cache}); rg2c and allon5 are two diagonal corners, these two arms fill
# the other two. Async + fix-ON held fixed across all four. WALL-FREE, 16 tasks.
#
# DECISIVE READ (arm 1, tail6_pb cache-OFF):
#   accept ~= 5.3  -> PIGGYBACK is innocent; the stale-root fix IS effective
#                     under pb; CACHE is the sole accept carrier (a trajectory
#                     effect, not a losslessness bug in my pb code).
#   accept ~= 4.4  -> PIGGYBACK is the carrier: the pb packer's chain-fill
#                     defeats the _fr13_all_neg first-spec signature, so pb
#                     rows fall back to the buggy CPU-gate. => localize + force
#                     the root repair on pb spec rows (a real fix I own).
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
# arm 1: tail6_pb, CACHE-OFF (pb ON, cache OFF). The decisive bisect: vs rg2c
#        isolates PIGGYBACK, vs allon5 isolates CACHE. Mirrors rg2c's regime
#        (FR13_ENABLE_APC=0, async) except piggyback is armed.
export FR13_ENABLE_APC=0
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_acciso/tail6pb_nocache_cfwd.json
run_variant tail6pb_nocache_${TAG}  tail6_pb  29  1
# arm 2: plain tail6, CACHE-ON (pb OFF, cache ON). Completes the 2x2: vs rg2c
#        isolates CACHE on the plain (no-pb) tail6. If arm 1 says cache is the
#        carrier, this confirms it on the pb-free shape.
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_acciso/tail6_cache_cfwd.json
run_variant tail6_cache_${TAG}  tail6  21  1
