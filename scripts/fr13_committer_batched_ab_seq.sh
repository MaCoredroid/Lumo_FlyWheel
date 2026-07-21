# COMMITTER BATCHED speed A/B with the RIGHT timer FR13_SFWD_GPU_TIMER (gives fr13_committer_gpu span +
# step_wall + verify; readable LIVE at :9950/metrics). COMMIT_FULL was the sampler = wrong metric.
# FR13_COMMITTER_NATIVE_BATCHED=1 routes the LIVE committer to the batched fn (validated coherent, accept
# ~4.9 in cb2). Measures committer_gpu drop (default cross-est ~88ms -> ?).
#   arm1 = BATCHED (COMMITTER_NATIVE_BATCHED=1)  | arm2 = DEFAULT (per-layer, same campaign).
# Both COMMITTER_NATIVE=1, APC_SNAP_FIX=0, cache-off, PARENT_GATHER=1, --async-scheduling.
# Launch: RUNROOT=output/fr13_combatch3 TAG=cb3 SUBSET=subset_collapse3.json WALL=0 BSIZE=4 CONC=4
#   HEALTH_TIMEOUT_S=3600 SEQUENCE_FILE=scripts/fr13_committer_batched_ab_seq.sh
#   bash scripts/fr13_campaign_tmux.sh combatch3
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_PARENT_GATHER=1
export FR13_ENABLE_APC=0
export FR13_APC_SNAP_FIX=0
export FR13_COMMITTER_NATIVE=1
export FR13_SFWD_GPU_TIMER=1
# ---- arm1: BATCHED ----
export FR13_COMMITTER_NATIVE_BATCHED=1
export FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_combatch3/sfwd_batched_${TAG}.json
run_variant tail6_batched_${TAG}  tail6  21  1
# ---- arm2: DEFAULT ----
export FR13_COMMITTER_NATIVE_BATCHED=0
export FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_combatch3/sfwd_default_${TAG}.json
run_variant tail6_default_${TAG}  tail6  21  1
