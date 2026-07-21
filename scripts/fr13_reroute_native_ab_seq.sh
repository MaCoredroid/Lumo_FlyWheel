# REROUTE-THROUGH-NATIVE A/B (user 2026-07-21 "reroute pbs through native kernel" + "do the fix then
# restart arm"). STRUCTURAL FINDING: rerouting pb's SEED through the native kernel necessarily removes
# the chain fold -> the subtree roots from node-8 = the IN-KERNEL custom chain-fold output; writing col-0
# native only fixes the NEXT step's h0, which the next chain re-folds custom -> subtree ALWAYS custom-
# rooted. The only native-rooted subtree = NO chain = the tail6 (non-pb) geometry. So "reroute pb through
# native" == non-pb tail6 (native committer seed), already proven lossless (rg1 5.03/5.31 deep).
# This A/B proves it on the B4 16-task GATE, ZERO config drift (identical env, only the geometry/pb flag
# differs), + committer CFWD both arms (to size the async-overlap = the real speed win, dominates the fold).
#   arm1 = tail6_pb  : pb CHAIN FOLD (custom, LOSSY baseline) -- expect deep-task collapse.
#   arm2 = tail6     : non-pb NATIVE SEED (the fix) -- expect ~5 on deep tasks == native-lossless.
# Matched: COMMITTER_NATIVE=1 (deployed), cache-OFF (accept is cache-independent), PARENT_GATHER=1,
# --async-scheduling, FR13_COMMIT_FULL_GPU_TIMER=1. B4, subset_b4_sixteen (the speed gate).
# Launch: RUNROOT=output/fr13_reroute TAG=rr1 SUBSET=subset_b4_sixteen.json WALL=0 BSIZE=4 CONC=4
#   HEALTH_TIMEOUT_S=3600 SEQUENCE_FILE=scripts/fr13_reroute_native_ab_seq.sh
#   bash scripts/fr13_campaign_tmux.sh reroute
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_PARENT_GATHER=1
export FR13_ENABLE_APC=0
export FR13_COMMIT_FULL_GPU_TIMER=1
# ---- arm1: pb CHAIN FOLD (lossy custom-seed baseline) ----
export FR13_PB_BASE_COL_INVARIANT=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_reroute/cf_pbfold_${TAG}.json
run_variant tail6pb_fold_${TAG}  tail6_pb  29  1
# ---- arm2: non-pb NATIVE SEED (the reroute = the fix) ----
unset FR13_PB_BASE_COL_INVARIANT
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_reroute/cf_native_${TAG}.json
run_variant tail6_native_${TAG}  tail6  21  1
