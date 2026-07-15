# SAME-SESSION COLD arm for the GATE-0.5 A/B: tail6 WITHOUT pre-warm, identical config otherwise
# (GPU_UTIL 0.72, 16-task subset_b4_sixteen). Isolates the pre-warm accept lift from run-to-run variance
# (the cross-session g4c cold 4.277 is a weaker reference). Run AFTER the prewarm arm completes; compare
# deploy_speed accept_per_event prewarm-vs-cold. No FR13_PREWARM_TRIE = cold trie (within-task arctic only).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
run_variant tail6_cold_${TAG} tail6 21 1
