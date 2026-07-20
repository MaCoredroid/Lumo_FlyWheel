# COMMITTER OVERHEAD DECOMPOSITION (user directive 2026-07-20: work the committer
# overhead + add to arm to validate). CORRECTION: the temp-0.6 committer is
# fr13_device_multidraft_commit (FR13_DEVICE_MULTIDRAFT, BAKED default-ON, already
# on-device) -- NOT the greedy FR13_GPU_COMMITTER (greedy LCP, off the temp-0.6
# path). So the 53.7ms CFWD committer span is NOT a host compute loop; it is:
#   multidraft rejection KERNEL time  +  result DtoH  +  verify-wait/pipeline gap.
# This arm SPLITS it with three nested timers on the tail6_pb SHIP config:
#   FR13_MULTIDRAFT_GPU_TIMER  -> the rejection-kernel GPU time (reducible by kernel opt).
#   FR13_COMMIT_FULL_GPU_TIMER -> the GDN col-0 commit + assembly span (17.8ms in allon5).
#   FR13_CFWD_GPU_TIMER        -> the full _sample rejection-dispatch (53.7ms).
# READ: CFWD - COMMIT_FULL = DtoH+verify-wait residual; COMMIT_FULL - MULTIDRAFT =
#   host assembly/publish. multidraft HIGH => optimize the kernel; residual HIGH =>
#   async/overlap the result DtoH. Decides the committer strategy (measure-first).
# Also re-reads ship accept (~4.4, overflow-affected) + resolve. Timers accumulate
# over decode steps -> readable EARLY (few k drafts), no need for 16 tasks.
# Launch: RUNROOT=output/fr13_commdecomp TAG=cd1 SUBSET=subset_b4_sixteen.json
#   WALL=0 BSIZE=4 CONC=4 HEALTH_TIMEOUT_S=3600
#   SEQUENCE_FILE=scripts/fr13_committer_decomp_seq.sh
#   bash scripts/fr13_campaign_tmux.sh commdecomp
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_CFWD_GPU_TIMER=1
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_commdecomp/commit_full_cd1.json
export FR13_MULTIDRAFT_GPU_TIMER=1
export FR13_MULTIDRAFT_GPU_TIMER_JSON=/workspace/output/fr13_commdecomp/multidraft_cd1.json
run_variant tail6pb_cd_${TAG}  tail6_pb  29  1
