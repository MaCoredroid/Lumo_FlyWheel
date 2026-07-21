# FR13 BURN-REDUNDANCY A/B sequence (direction-2 committer verify).
# Hypothesis: the committer BURN (zeroing tree spec rows) is 87% of the committer and REDUNDANT --
# kernel comment (fr10_gdn_tree_kernel.py:1210-1213): "nothing downstream reads them (h0+snapshot read
# col0; next scan writes nodes fresh)". If true, dropping it is a standalone committer speed win, lossless.
# GATE (no config drift -- SAME cat9f kind + golden flags, vary ONLY the burn):
#   losslessness = burn-OFF resolve + garble + accept MATCH burn-ON (within-floor, temp 0.6, live SWE);
#   speed        = burn-OFF s_per_fwd_gpu / derived_tps_gpu BETTER than burn-ON (committer minus the burn).
# burn-OFF runs FIRST to fail-fast: if the burn is load-bearing, burn-OFF garbles on the first task.
# Must exercise B>1 + APC-snapshot readers (the cases the kernel comment might not cover) -- CONC=4, cache on.

# arm 1: BURN OFF  (commit/init stay ON; FR13_BURN_REDUNDANCY_TEST bypasses the tri-flag assert)
export FR13_BURN_REDUNDANCY_TEST=1
run_variant burnoff_${TAG}  cat9f  9  1

# arm 2: BURN ON  (deployed baseline)
export FR13_BURN_REDUNDANCY_TEST=0
run_variant burnon_${TAG}   cat9f  9  1
