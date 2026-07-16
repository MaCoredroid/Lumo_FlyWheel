# HW-LIMIT baseline decomposition: native MTP-5 + tail6 (spine tail), SAME-SESSION on subset_b4_sixteen,
# component timers ON (run_variant sets FR13_DFWD/CFWD/SFWD_GPU_TIMER=1 for every arm). Gives the two
# missing pieces for the HW-limit plan: (1) NATIVE's per-stage decomposition (drafter/verify/committer) --
# never measured (its timers were off in every prior native run); (2) a clean SAME-SESSION native-vs-tail6
# per-stream tps + kernel tps (does our fastest tree beat native?). No new/risky flags => no losslessness
# concern (native is the reference; tail6 is already-validated lossless). NO config drift.
#   nativemtp5 = flash_ns5_nocache (forked launcher, FLASH_ATTN, num_spec=5, no tree)  E5  -- the BAR
#   tail6      = MTP head + arctic spine tail (21 nodes, no branches)                  E21 -- fastest tree
# The b7 clean result already showed tail6b (branched) LOSES ~8% tps to tail6 => more branches is an
# anti-speed lever; the geometry-widen arms (tail6c/tail6e) are DEPRIORITIZED. The real TPS levers are the
# HW-limit ones (committer sync-kill, async pipelining, graphed drafter) -- see FR13_PIPELINE_OVERHEAD_ACCOUNTING.md.
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
run_variant nativemtp5_${TAG}  flash_ns5_nocache  5   1
run_variant tail6_${TAG}       tail6              21  1
