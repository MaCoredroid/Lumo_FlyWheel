# BEAT-NATIVE LADDER combined validation campaign (FR13_BEAT_NATIVE_LADDER.md) — ONE campaign, two rungs:
#   R1 ASYNC pair (arms 1-2, FIRST — completes task #40's interrupted A/B; its baseline arm was reaped at
#      start so the same-session confirm never landed): tail6+--async-scheduling vs tail6. Gate: per_req +
#      fullstep UP, accept ~= baseline (LOSSLESS — async's optimistic num_computed_tokens could change
#      accepted tokens => gate hard), no garble. PASS => bake --async-scheduling into the deployed config.
#   R2 PIGGYBACK pair (arms 3-4 — requires the piggyback bundle applied + cat9_pb kind wired first;
#      DO NOT run this seq before the bundle lands): cat9_pb (FR13_PIGGYBACK=1, extended 17-node tree,
#      committer replay SKIPPED, chain-end export) vs cat9 baseline. Gates: ENGAGED needle (piggyback
#      export firing), accept vs depth-matched E5 basis (cat9 depth-5), committer CFWD collapse
#      (~99 -> ~16ms; CF2 timer on both arms), no garble, 0 fatal.
# Async pair runs FIRST so an interrupted campaign still yields the R1 bake decision (highest
# value-per-risk). All arms: GPU_UTIL 0.72, no prewarm, B=4 CONC=4 qwen-code nudge-free (driver env).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1

export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_ladder/async_cfwd.json
run_variant tail6_async_${TAG}  tail6  21  1
unset FR13_SERVE_BATCH_FLAGS
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_ladder/tail6_cfwd.json
run_variant tail6_base_${TAG}   tail6  21  1

# R2 arms = V2 MECHANICAL GATES ONLY (bundle Risk 1: chain-token attention-KV double-write is UNRESOLVED
# until the phase-3 KV/conv surgery => cat9_pb OUTPUT text/accept/resolve are NOT meaningful yet). Read:
# engage needles (drafter engaged + "committer GDN replay DROPPED"), 0 fatal, drafts==17 ratio, and the
# committer CFWD collapse (cat9pb ~16ms expected vs cat9f same-session baseline). cat9f = base cat9 on the
# FORKED launcher (piggyback OFF) — the valid same-session baseline (LOCKED cat9 kind bakes golden flags).
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_ladder/cat9pb_cfwd.json
run_variant cat9pb_${TAG}       cat9_pb 17  1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_ladder/cat9f_cfwd.json
run_variant cat9f_${TAG}        cat9f    9  1
