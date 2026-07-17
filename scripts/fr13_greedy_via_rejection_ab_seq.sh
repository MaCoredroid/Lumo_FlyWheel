# PHASE-1 LOSSLESS GATE: route all_greedy -> rejection committer (point-mass). temp-0 (DEPLOY_FORCE_TEMP=0.0)
# so all_greedy engages. gv0 = FR13_GREEDY_VIA_REJECTION=0 (old greedy path-LCP, baseline). gv1 =
# FR13_GREEDY_VIA_REJECTION=1 (rejection point-mass, + FR13_DEDUP_SIBLINGS=1 default so no argmax ties).
# GATE: accept gv0 == gv1 (rejection@temp0 == old greedy) => routing lossless => safe to DELETE greedy.
# tail6 (deployed geometry; byte-exact by construction). subset_b4_four (fast).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_GREEDY_VIA_REJECTION=0
run_variant tail6_gv0  tail6  21  1
export FR13_GREEDY_VIA_REJECTION=1
run_variant tail6_gv1  tail6  21  1
