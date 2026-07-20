# ACCEPT-COLLAPSE CONTROL (user directive 2026-07-20): allon5 (tail6_pb cache-ON
# async) collapses on 3 END-OF-RUN tasks -- 14539/14598/14995 -> accept 3.6-3.9 --
# while rg1 (plain tail6 cache-OFF sync) HOLDS 4.9-5.7 on those SAME tasks (they are
# rg1's HIGHEST, not inherently hard). The col-0 workflow (wgrmh5q1l) found ZERO
# divergences, so it is NOT an overflow col-0 bug. This CONTROL runs tail6_pb
# CACHE-OFF async to COMPLETION -- the only variable vs allon5 is the cache:
#   3 tasks still collapse  => PIGGYBACK/overflow (investigate concretely on those
#                              tasks: per-pos accept + overflow rate, no hand-waving).
#   3 tasks HOLD (~5+)      => CACHE-ON late-run degradation is the carrier (ship-
#                              config issue) => fix the cache path.
# Committer timers folded in => doubles as the committer decomposition (cache-OFF,
# ship-representative). MUST run the full 16 to reach 14539/14598/14995.
# Launch: RUNROOT=output/fr13_acctrl TAG=ac1 SUBSET=subset_b4_sixteen.json
#   WALL=0 BSIZE=4 CONC=4 HEALTH_TIMEOUT_S=3600
#   SEQUENCE_FILE=scripts/fr13_accept_control_seq.sh
#   bash scripts/fr13_campaign_tmux.sh acctrl
export FR13_ENABLE_APC=0
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
export FR13_CFWD_GPU_TIMER=1
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_acctrl/commit_full_ac1.json
export FR13_MULTIDRAFT_GPU_TIMER=1
export FR13_MULTIDRAFT_GPU_TIMER_JSON=/workspace/output/fr13_acctrl/multidraft_ac1.json
run_variant tail6pb_ctrl_${TAG}  tail6_pb  29  1
