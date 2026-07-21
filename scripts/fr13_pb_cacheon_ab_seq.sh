# 16-TASK CACHE-ON A/B (user directive 2026-07-21): get pb accept back to ~5.4 on the FULL
# 16-task DEPLOYMENT config (cache ON) BEFORE the committer levers (a=batch-output, b=async-overlap).
# My prior bc5 read was 3 deep tasks + cache-OFF (wrong config). This is the real measurement.
#   EXPERIMENT arm1: tail6_pb + FR13_PB_BASE_COL_INVARIANT=1 (base-col accept fix) + CACHE ON.
#   CONTROL   arm2: tail6 NO-PB + CACHE ON  <-- the ~5.4 target the experiment must match.
# WIN CONDITION: arm1 accept ~= arm2 accept ~= 5.4. arm1 << arm2 => the base-col fix is not enough,
#   more accept work needed before the committer levers.
# Both PARENT_GATHER=1 (byte-identical O(N) GDN kernel -> light compile, avoids the O(N^2) host-RAM
# spike that trips gpu_oom_guard). Cache-ON = collapse3 arm2 config.
# Launch: RUNROOT=output/fr13_cacheon_ab TAG=co1 SUBSET=subset_b4_sixteen.json WALL=0 BSIZE=4 CONC=4
#   HEALTH_TIMEOUT_S=3600 SEQUENCE_FILE=scripts/fr13_pb_cacheon_ab_seq.sh
#   bash scripts/fr13_campaign_tmux.sh cacheon_ab
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_PARENT_GATHER=1
# CACHE ON (deployment config, matched both arms):
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMIT_FULL_GPU_TIMER=1
# ---- arm1: EXPERIMENT = tail6_pb + base-col fix + cache ON ----
export FR13_PB_BASE_COL_INVARIANT=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_cacheon_ab/cf_pb_${TAG}.json
run_variant tail6pb_cacheon_${TAG}  tail6_pb  29  1
# ---- arm2: CONTROL = tail6 NO-PB + cache ON (the target) ----
unset FR13_PB_BASE_COL_INVARIANT
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_cacheon_ab/cf_nonpb_${TAG}.json
run_variant tail6_nonpb_cacheon_${TAG}  tail6  21  1
