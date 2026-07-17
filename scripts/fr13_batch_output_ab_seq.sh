# FR13_COMMIT_BATCH_OUTPUT A/B (phase-3 committer opt): legacy per-element output write (bo0) vs batched
# host-build + ONE H2D copy (bo1). Both tail6_mt + FR13_COMMIT_FULL_GPU_TIMER (whole-committer span).
# BYTE-IDENTICAL output => accept must be identical; the win is committer_ms(bo0) - committer_ms(bo1).
# Baseline (separate legacy run) whole-committer = 80.0ms. temp 0.6, subset_b4_four (timers accumulate fast).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_COMMIT_BATCH_OUTPUT=0
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/bo0_cf2.json
run_variant tail6_bo0  tail6_mt  21  1
cp -f output/fr13_sfwd_sidecar/tail6_mt_md.json output/fr13_sfwd_sidecar/bo0_md.json 2>/dev/null || true
export FR13_COMMIT_BATCH_OUTPUT=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/bo1_cf2.json
run_variant tail6_bo1  tail6_mt  21  1
cp -f output/fr13_sfwd_sidecar/tail6_mt_md.json output/fr13_sfwd_sidecar/bo1_md.json 2>/dev/null || true
