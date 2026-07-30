# Hydra23 hardware-floor gate: candidate first, then a matched tail6 control.
# Both arms use the same true deployment temperature, 16-task subset, cache
# regime, batch geometry, and clean measurement-off serving path.
#
# Launch:
#   RUNROOT=output/fr13_hydra23_gate TAG=hydra23_tail6_clean16 \
#   SUBSET=output/fr13_b1_gold_swe/subset_b4_sixteen.json \
#   WALL=0 BSIZE=4 CONC=4 \
#   HEALTH_TIMEOUT_S=3600 \
#   SEQUENCE_FILE=scripts/fr13_hydra23_tail6_clean_seq.sh \
#   bash scripts/fr13_campaign_tmux.sh hydra23_tail6_clean16
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export DEPLOY_FORCE_TEMP=0.6
export FR10_METRICS=0
export BATCH_INVARIANT=0
# The hydra arm enables its marker inside the per-arm child process. Keeping
# the campaign baseline at zero guarantees it cannot leak into the tail6 boot.
export FR13_HYDRA23=0
export FR13_TAIL_BRANCHES=0
export FR13_TAIL_BRANCH_DEPTHS=0

# Clean timing surface. The campaign driver historically defaults the first
# three timers on; explicit zeros exercise its measurement-off override.
export FR13_SFWD_GPU_TIMER=0
export FR13_DFWD_GPU_TIMER=0
export FR13_CFWD_GPU_TIMER=0
export FR13_MULTIDRAFT_GPU_TIMER=0
export FR13_REPLAY_GPU_TIMER=0
export FR13_COMMIT_FULL_GPU_TIMER=0
export FR13_COMMITTER_SG_TIMER=0
export FR13_REPLAY_ONLY_GPU_TIMER=0
export FR13_GRAPH_TIMER=0
export FR13_KVREMAP_TIMER=0
export FR13_STATEREMAP_TIMER=0

# Fail closed against inherited diagnostic, self-check, and profiler state.
export FR13_BRANCH_ACCEPT_DIAG=0
export FR13_FORCE_SPINE_COMMIT=0
export FR13_FIX1_SELFCHECK=0
export FR13_COMMIT_ARGMAX_GATE=0
export FR13_FORK_MARGIN_DUMP=0
export FR13_CHASE_DIAG=0
export FR13_REPLAY_BOUNDARY_LOG=0
export FR13_GDN_SUBOP_MAB=0
export FR13_CONV_SUBOP_MAB=0
export FR13_FA2_MAB=0
export FR13_REPLAY_DURABLE_AB=0
export FR13_TREE_POSREAD_PROBE=0
export FR13_LEAK_PROBE=0
export FR13_SERVE_LOG=0
export FR13_TORCH_DET_WARN=0
export FR13_TCF_DIAG_OVERRIDE=0
export FR13_TCF_SELFCHECK=0
export FR13_SUBTREE_PARALLEL_SELFCHECK=0
export FR13_PARENT_GATHER_SELFCHECK=0
export FR13_TORCHPROF=0
export FR13_TORCH_PROF=""
export FR13_DVK_DRAFTID_DUMP=""
export LUMO_NSYS_WRAP_VLLM=0
export LUMO_FA_ACTIVATION_REPLAY_BATCH4_DIAG=0
export LUMO_FA_REPLAY_COMMIT_BATCH4_RUNNER_DIAG=0
export LUMO_IR_DIAGNOSTIC_UNISOLATED=0
export LUMO_IR_ALLOW_UNVERIFIED_SPINES2_MEASUREMENT=0

run_variant hydra23_${TAG} hydra23 23 1
run_variant tail6_${TAG} tail6 21 1
