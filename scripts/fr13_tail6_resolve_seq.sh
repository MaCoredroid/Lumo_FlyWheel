# TAIL6 RESOLVE-GATE run (user directive 2026-07-19): golden signal FIRST.
# Single tail6 arm on the FIXED code (bonus-repair CPU-placeholder gate,
# handoff trim, rowids-fresh, spec-keying corrections all baked in the
# patcher at boot). Purpose: verify the resolve rate lands ~8/16-ish on the
# 16-task subset per the user's tail6 band. No async (matches the historical
# tail6 resolve basis), no debug poison, standard GPU_UTIL.
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1
unset FR13_SERVE_BATCH_FLAGS
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_resolve/tail6_cfwd.json
run_variant tail6_${TAG}  tail6  21  1
