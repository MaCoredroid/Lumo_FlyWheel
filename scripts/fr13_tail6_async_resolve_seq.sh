# TAIL6+ASYNC resolve/accept arm (R1 pair completion, user-prioritized lead:
# as1 cross-run showed accept 4.953 vs tail6 4.306 — never same-session
# confirmed; lad2's async arm resolved 7/12 before the pair was preempted).
# Runs back-to-back with the rg1 tail6 baseline on the same subset/code:
# gates = accept delta (approx 5 vs 4.3?), resolve band, wall TPS, lossless
# watch (async's optimistic num_computed_tokens could change trajectories).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_resolve/tail6_async_cfwd.json
run_variant tail6_async_${TAG}  tail6  21  1
