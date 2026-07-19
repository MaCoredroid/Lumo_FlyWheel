# NATIVE MTP5 WALL-FREE pair (user directive 2026-07-19): re-anchor the BAR.
# Evidence the standing bar (27.9 / accept 3.415, nt1) was effectively walled:
# nt1 native task durations truncate at 1918s (wall-free runs reach 5400s) --
# the same right-censoring that suppressed tail6 accept (walled 4.31 vs
# wall-free ~5.47 lad2) also suppresses the native bar. Fair endgame bar =
# native wall-free (+ async twin, the ladder's honest-guard endgame).
# Runs AFTER the rg1/rg2 tail6 pair; same subset/code, back-to-back.
# CACHE-ON (user directive 2026-07-19: from this pair onward all arms run APC
# cache ON — (a) validate the stack under the ship config, (b) prefix reuse
# makes campaigns faster). nativemtp5apc kind = NATIVE_ENABLE_APC=1 + block
# flags. rg1/rg2 stay cache-OFF as their own matched pair.
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1
unset FR13_SERVE_BATCH_FLAGS
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_resolve/native_cfwd.json
run_variant native_${TAG}        nativemtp5apc  5  1
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_resolve/native_async_cfwd.json
run_variant native_async_${TAG}  nativemtp5apc  5  1
# DEPTH-MATCHED baseline (user): native MTP at tail6's depth 11 — decomposes
# the tree accept edge (depth budget vs branching vs mixed drafter). Cache ON
# like everything post-rg-pair.
unset FR13_SERVE_BATCH_FLAGS
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_resolve/native11_cfwd.json
run_variant native11_${TAG}      nativemtp11apc 11 1
