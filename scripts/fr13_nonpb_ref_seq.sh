# NON-PB MATCHED REFERENCE for the FR13_PB_BASE_COL_INVARIANT A/B (user directive 2026-07-21:
# compare the fix to tail6 NON-PB, not to collapse3 which is COLLAPSED-PB ~3.6). This arm is
# the TARGET: tail6 with NO piggyback (holds ~4.9-5.7 on these 3 tasks per rg1). The fix WINS
# iff bc5 (tail6_pb + FR13_PB_BASE_COL_INVARIANT=1) matches THIS non-pb decode-bracketed accept,
# not merely beats collapse3's 3.6.
# CONFIG MATCHED TO bc5: PARENT_GATHER=1 (byte-identical O(N) kernel, avoids the O(N^2) compile
# that trips the gpu_oom_guard), cache-OFF (FR13_ENABLE_APC=0), --async-scheduling, same 3 tasks.
# ONLY DIFFERENCE vs bc5: NO piggyback (tail6 not tail6_pb) + NO FR13_PB_BASE_COL_INVARIANT.
# Launch (AFTER bc5 finishes, GPU serialized):
#   RUNROOT=output/fr13_bcinv TAG=npref SUBSET=subset_collapse3.json WALL=0 BSIZE=4 CONC=4
#   HEALTH_TIMEOUT_S=3600 SEQUENCE_FILE=scripts/fr13_nonpb_ref_seq.sh
#   bash scripts/fr13_campaign_tmux.sh bcinv
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_PARENT_GATHER=1
# NON-PB: FR13_PB_BASE_COL_INVARIANT UNSET; tail6 (21) not tail6_pb (29) => no piggyback armed.
export FR13_ENABLE_APC=0
export FR13_COMMIT_FULL_GPU_TIMER=1
run_variant tail6_nonpb_${TAG}  tail6  21  1
