# BEAT-NATIVE LADDER async validation campaign (FR13_BEAT_NATIVE_LADDER.md) — R1 ASYNC pair (completes
# task #40's interrupted A/B; its baseline arm was reaped at start so the same-session confirm never
# landed): tail6+--async-scheduling vs tail6. Gate: per_req + fullstep UP, accept ~= baseline (LOSSLESS —
# async's optimistic num_computed_tokens could change accepted tokens => gate hard), no garble. PASS =>
# bake --async-scheduling into the deployed config.
# All arms: GPU_UTIL 0.72, no prewarm, B=4 CONC=4 qwen-code nudge-free (driver env).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1

export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_ladder/async_cfwd.json
run_variant tail6_async_${TAG}  tail6  21  1
unset FR13_SERVE_BATCH_FLAGS
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_ladder/tail6_cfwd.json
run_variant tail6_base_${TAG}   tail6  21  1
