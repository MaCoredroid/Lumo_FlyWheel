# COMMITTER-OVERHEAD DIAGNOSTIC (workflow wog6j3yyb 2026-07-21): SIZE the eager 48-layer committer
# rebuild dispatch, to decide if the INLINE-FOLD build is worth it + resolve the 72ms-vs-+26ms/step
# magnitude contradiction. The committer eager loop (~48 per-layer native fused_sigmoid replays, ~48
# blocking .tolist() D2H syncs) is the dominant OURS-ONLY cost per the workflow; native folds it inline
# (inplace_final_state + num_accepted_tokens) at zero dispatch. This A/B measures the end-to-end TPS/
# per_req delta the eager loop costs (== the inline-fold ceiling):
#   arm1 = COMMITTER_NATIVE=1                          : deployed, per-layer native loop (bit-exact, SLOW).
#   arm2 = COMMITTER_NATIVE=0 + SAMPLED_REPLAY_BATCHED=1: custom all-layers kernel, ONE launch (FAST, NOT
#          bit-exact -- SPEED-ONLY diagnostic, do NOT deploy). Measures the batched ceiling.
# arm1 - arm2 TPS delta = the committer eager-dispatch tax. Big => inline-fold worth building. Also the
# full committer CFWD decomposition (CFWD/COMMIT_FULL/MULTIDRAFT GPU timers) both arms -> the real magnitude.
# BATCHED gates: needs FR13_APC_SNAP_FIX!=1 + no APC-publish (cache-OFF satisfies) + stacks present.
# Both non-pb tail6 (the deployed geometry), cache-OFF, PARENT_GATHER=1, --async-scheduling. Timers read
# early (few k drafts) => small subset (subset_collapse3, 3 tasks) suffices. B4.
# Launch (AFTER rr1 frees GPU): RUNROOT=output/fr13_commdiag TAG=cd1 SUBSET=subset_collapse3.json WALL=0
#   BSIZE=4 CONC=4 HEALTH_TIMEOUT_S=3600 SEQUENCE_FILE=scripts/fr13_committer_overhead_diag_seq.sh
#   bash scripts/fr13_campaign_tmux.sh commdiag
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_PARENT_GATHER=1
export FR13_ENABLE_APC=0
export FR13_APC_SNAP_FIX=0
export FR13_CFWD_GPU_TIMER=1
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_MULTIDRAFT_GPU_TIMER=1
# ---- arm1: deployed per-layer NATIVE committer (bit-exact, slow) ----
export FR13_COMMITTER_NATIVE=1
unset FR13_SAMPLED_REPLAY_BATCHED
export FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_commdiag/cfwd_native_${TAG}.json
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_commdiag/cf_native_${TAG}.json
export FR13_MULTIDRAFT_GPU_TIMER_JSON=/workspace/output/fr13_commdiag/md_native_${TAG}.json
run_variant tail6_native_${TAG}  tail6  21  1
# ---- arm2: batched all-layers custom kernel (fast, NOT bit-exact -- SPEED CEILING only) ----
export FR13_COMMITTER_NATIVE=0
export FR13_SAMPLED_REPLAY_BATCHED=1
export FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_commdiag/cfwd_batched_${TAG}.json
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_commdiag/cf_batched_${TAG}.json
export FR13_MULTIDRAFT_GPU_TIMER_JSON=/workspace/output/fr13_commdiag/md_batched_${TAG}.json
run_variant tail6_batched_${TAG}  tail6  21  1
