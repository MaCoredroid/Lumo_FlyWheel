# PHASE-3 LEVER 2 A/B: batched replay (FR13_SAMPLED_REPLAY_BATCHED=1 => one launch_tree_gdn_replay_all_layers
# kernel) vs per-layer (=0, the 48-launch storm). Both with FR13_COMMIT_FULL_GPU_TIMER to measure the whole
# committer CFWD span. Gate: accept_per_event IDENTICAL (batched is a launch-batching, same numerics => no
# config drift) + committer CFWD REDUCED on batched (trims the 48-launch host overhead, ~5-10ms of the ~25ms
# host portion; the 66-72ms latency-bound replay COMPUTE is unchanged -- only the piggyback touches that).
# subset_b4_four for a quick indicative read; promote to subset_b4_sixteen if the CFWD delta is clean.
# run_variant is driver-sourced. GPU_UTIL 0.72 (established tail6 config).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_SAMPLED_REPLAY_BATCHED=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_lever2/batched_cfwd.json
run_variant tail6_batched  tail6  21  1
export FR13_SAMPLED_REPLAY_BATCHED=0
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_lever2/perlayer_cfwd.json
run_variant tail6_perlayer tail6  21  1
