# REPLAY-TIMER probe: tail6_rt (FR13_REPLAY_GPU_TIMER=1) -> sidecar output/fr13_sfwd_sidecar/tail6_rt_replay.json.
# Settles the 94ms-committer decomposition: replay ~=80ms => reducible (stateless-tree gather = real lever);
# replay ~=11ms => cost is sync-wait/DtoH => native wins. Short run: read sidecar after ~50 decode steps, then kill.
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
run_variant tail6_rt_${TAG}  tail6_rt  21  1
