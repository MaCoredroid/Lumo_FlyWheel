# COMMITTER DECOMPOSITION: whole-committer span (FR13_COMMIT_FULL_GPU_TIMER = _lumo_tree_canonical_
# multidraft_sample: device walk + output assembly + GDN publish) vs inner multidraft walk
# (FR13_MULTIDRAFT_GPU_TIMER, baked in tail6_mt XFLAGS). DELTA = surrounding host assembly/publish.
# Settles whether the historical ~94ms committer is the WALK (optimize kernel) or the SURROUNDING
# (optimize assembly). SHORT run -- timers accumulate over decode steps (json every 50 spans).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/tail6_decomp_cf2.json
run_variant tail6_decomp  tail6_mt  21  1
