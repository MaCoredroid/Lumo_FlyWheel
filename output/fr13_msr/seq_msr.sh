# COMMITTER BREAKDOWN measurement campaign (user-directed 2026-07-23):
# measure -> design -> impl -> BATCHED 16-task gates. 4-task set (spans need
# events, not pass bands). Two arms:
#  M1 msr_batched: baked batched committer + narrow timers ->
#     4-way split: cfwd(whole) - replay_only(gathers+sg loop) - sg(sg loop)
#     => accept/remap/host remainder by subtraction.
#  M2 msr_graph: graph committer + FR13_GRAPH_TIMER (fill/replay/burn print)
#     -> graph's split + graph-vs-batched span verdict CHEAP (no 16-task).
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1

export FR13_COMMITTER_SG_TIMER=1
export FR13_COMMITTER_SG_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/msr_batched_sg.json
export FR13_REPLAY_ONLY_GPU_TIMER=1
export FR13_REPLAY_ONLY_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/msr_batched_replayonly.json
run_variant msr_batched  tail6  21  1

unset FR13_COMMITTER_SG_TIMER FR13_COMMITTER_SG_TIMER_JSON
unset FR13_REPLAY_ONLY_GPU_TIMER FR13_REPLAY_ONLY_GPU_TIMER_JSON
export FR13_COMMITTER_GRAPH=1
export FR13_GRAPH_TIMER=1
run_variant msr_graph  tail6  21  1
