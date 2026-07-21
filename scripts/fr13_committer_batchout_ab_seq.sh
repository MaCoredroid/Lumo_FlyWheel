# COMMITTER OVERHEAD A/B (2026-07-21, committer front): measure the current committer decomposition
# on the deployed pb config + the CHEAP byte-identical FR13_COMMIT_BATCH_OUTPUT lever, toward the
# piggyback ~16ms target (TPS ~31, +11% vs native). pb committer CFWD=53.7ms; the ~30-42ms
# (CFWD - commit_full) is the deferrable DtoH/wait (=> the async-overlap, the BIG lever, next).
#   arm1: FR13_COMMIT_BATCH_OUTPUT=0 (legacy per-element output_token_ids writes = ~B*len syncs/step).
#   arm2: FR13_COMMIT_BATCH_OUTPUT=1 (ONE H2D copy, BYTE-IDENTICAL) -- measures the sync-kill win.
# Three nested timers => full decomposition: CFWD (full _sample dispatch) / COMMIT_FULL (kernel+
# assembly+GDN publish) / MULTIDRAFT (rejection kernel). Timers accumulate over decode steps ->
# readable EARLY (a few k drafts), so 2 tasks/arm suffices. cache-OFF (committer is cache-independent).
# base-col + PARENT_GATHER=1 = deployment-representative + light compile.
# Launch: RUNROOT=output/fr13_commbatch TAG=cb1 SUBSET=subset_collapse3.json WALL=0 BSIZE=4 CONC=4
#   HEALTH_TIMEOUT_S=3600 SEQUENCE_FILE=scripts/fr13_committer_batchout_ab_seq.sh
#   bash scripts/fr13_campaign_tmux.sh commbatch
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_PARENT_GATHER=1
export FR13_PB_BASE_COL_INVARIANT=1
export FR13_ENABLE_APC=0
export FR13_CFWD_GPU_TIMER=1
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_MULTIDRAFT_GPU_TIMER=1
# ---- arm1: legacy per-element output writes ----
export FR13_COMMIT_BATCH_OUTPUT=0
export FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_commbatch/cfwd_legacy_${TAG}.json
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_commbatch/cf_legacy_${TAG}.json
export FR13_MULTIDRAFT_GPU_TIMER_JSON=/workspace/output/fr13_commbatch/md_legacy_${TAG}.json
run_variant tail6pb_legacy_${TAG}  tail6_pb  29  1
# ---- arm2: batched output write (byte-identical) ----
export FR13_COMMIT_BATCH_OUTPUT=1
export FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_commbatch/cfwd_batch_${TAG}.json
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_commbatch/cf_batch_${TAG}.json
export FR13_MULTIDRAFT_GPU_TIMER_JSON=/workspace/output/fr13_commbatch/md_batch_${TAG}.json
run_variant tail6pb_batch_${TAG}  tail6_pb  29  1
