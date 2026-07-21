# COMMITTER BATCHED A/B on the DEPLOYED path (user: don't leave the win in the un-baked SAMPLED_REPLAY_BATCHED
# path). FR13_COMMITTER_NATIVE_BATCHED=1 now triggers the batched dispatch in the LIVE _lumo_tree_commit_gdn
# committer (sbr gate) -> launch_tree_gdn_replay_all_layers native branch -> _fr13_native_committer_all_layers_batched
# (hoisted layout + batched gather; BYTE-IDENTICAL committed state). Native committer replay is HOST-bound
# (measured 14.5ms fused_sigmoid + ~73ms host).
#   arm1 = COMMITTER_NATIVE_BATCHED=0 : DEFAULT per-layer live committer (the ~88ms deployed baseline).
#   arm2 = COMMITTER_NATIVE_BATCHED=1 : batched (needle '[FR13_COMMITTER_NATIVE_BATCHED ENGAGED]').
# LOSSLESS CHECK: accept must HOLD (~5, no collapse/garble); committer_gpu must DROP. Both COMMITTER_NATIVE=1,
# APC_SNAP_FIX=0 (sbr gate needs it), cache-off, PARENT_GATHER=1, --async-scheduling, FR13_COMMIT_FULL_GPU_TIMER=1.
# subset_collapse3 (3 tasks), B4. If validated -> deploy COMMITTER_NATIVE_BATCHED=1 (single live flag).
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
export FR13_COMMIT_FULL_GPU_TIMER=1
# ---- arm1: DEFAULT per-layer live committer (deployed baseline ~88ms) ----
export FR13_COMMITTER_NATIVE_BATCHED=0
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_combatch/cf_default_${TAG}.json
run_variant tail6_default_${TAG}  tail6  21  1
# ---- arm2: batched committer (live path via my flag) ----
export FR13_COMMITTER_NATIVE_BATCHED=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_combatch/cf_batched_${TAG}.json
run_variant tail6_batched_${TAG}  tail6  21  1
