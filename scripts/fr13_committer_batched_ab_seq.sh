# COMMITTER BATCHED A/B (user 2026-07-21: "keep the replay, make it faster"). The native committer
# replay is HOST-bound (measured: 14.5ms fused_sigmoid + ~73ms host = 48x layout recompute + 192 gathers).
# _fr13_native_committer_all_layers_batched hoists the layout (1 .tolist) + batches the gathers (4 cat over
# stacked rings) -- BYTE-IDENTICAL committed state (same fused_sigmoid ops, same order). Both arms route
# through SAMPLED_REPLAY_BATCHED=1 (-> launch_tree_gdn_replay_all_layers native branch); only the inner loop
# differs, so this isolates the batched effect.
#   arm1 = COMMITTER_NATIVE_BATCHED=0 : per-layer native loop (the ~88ms baseline).
#   arm2 = COMMITTER_NATIVE_BATCHED=1 : hoisted+batched (needle '[FR13_COMMITTER_NATIVE_BATCHED ENGAGED]').
# LOSSLESS CHECK: accept must HOLD (~5, no collapse/garble => committer byte-identical); committer_gpu must
# DROP. Both COMMITTER_NATIVE=1, APC_SNAP_FIX=0 (SAMPLED_REPLAY_BATCHED gate), cache-off, PARENT_GATHER=1,
# --async-scheduling, FR13_COMMIT_FULL_GPU_TIMER=1. subset_collapse3 (3 tasks), B4.
# Launch: RUNROOT=output/fr13_combatch TAG=cb1 SUBSET=subset_collapse3.json WALL=0 BSIZE=4 CONC=4
#   HEALTH_TIMEOUT_S=3600 SEQUENCE_FILE=scripts/fr13_committer_batched_ab_seq.sh
#   bash scripts/fr13_campaign_tmux.sh combatch
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_PARENT_GATHER=1
export FR13_ENABLE_APC=0
export FR13_APC_SNAP_FIX=0
export FR13_COMMITTER_NATIVE=1
export FR13_SAMPLED_REPLAY_BATCHED=1
export FR13_COMMIT_FULL_GPU_TIMER=1
# ---- arm1: per-layer native loop (baseline ~88ms) ----
export FR13_COMMITTER_NATIVE_BATCHED=0
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_combatch/cf_perlayer_${TAG}.json
run_variant tail6_perlayer_${TAG}  tail6  21  1
# ---- arm2: hoisted + batched committer ----
export FR13_COMMITTER_NATIVE_BATCHED=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_combatch/cf_batched_${TAG}.json
run_variant tail6_batched_${TAG}  tail6  21  1
