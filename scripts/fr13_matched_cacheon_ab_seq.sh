# MATCHED CACHE-ON SPEED GATE (2026-07-20): the decisive tail6_pb-vs-cat9pb A/B
# at the SHIP regime (APC cache ON + --async-scheduling), byte-identical env,
# ONLY the tree KIND differs. Resolves the confound that blocks an allon5
# verdict: allon5 tail6_pb measured 23.67 wall TPS at CACHE-ON pf 0.616, but the
# baked cat9pb 27.82 was measured CACHE-OFF -- the cross-campaign gap conflates
# deep-tree host cost (drafter 103 ms/step + committer 53.7 ms/step over 29
# cols) with the cache-ON chunked-prefill regime. This A/B runs BOTH shapes in
# ONE cache-ON async campaign so pf/eff-conc match. Answers: does tail6_pb's
# accept 4.417 beat cat9pb's ~3.385 on measured_tps_fullstep_wall, or does the
# 29-col tree's host cost eat the +1 accept? WALL-FREE (WALL=0), 16 tasks.
#
# Ship rule note: cat9_pb + tail6_pb BOTH arm FR13_PIGGYBACK, which is BAKED
# (R2, user decision 2026-07-19) -- V0/V1 gates already green pre-bake; this is
# a post-bake speed measurement, not a first arming.
#
# Launch: RUNROOT=output/fr13_matched TAG=mab1 SUBSET=subset_b4_sixteen.json
#   WALL=0 BSIZE=4 CONC=4 HEALTH_TIMEOUT_S=3600
#   SEQUENCE_FILE=scripts/fr13_matched_cacheon_ab_seq.sh
#   bash scripts/fr13_campaign_tmux.sh mab
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
# arm 1: BAKED baseline shape (17 cols = 8 chain + 9 base cat9). Reference for
# the R3/R4 A/Bs; the incumbent tail6_pb must beat on measured wall TPS.
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_matched/cat9pb_cfwd.json
run_variant cat9pb_${TAG}   cat9_pb   17  1
# arm 2: DEEP-TAIL deliverable shape (29 cols = 8 chain + 21 base tail6). Same
# regime as arm 1; the ONLY delta vs allon5 is that its comparator now shares
# the campaign pf.
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_matched/tail6pb_cfwd.json
run_variant tail6pb_${TAG}  tail6_pb  29  1
