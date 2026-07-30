# Hydra23 physical-floor screen: matched Tail6/Hydra23 arms with only the
# asynchronous stage timers enabled. The order is selectable so a second run
# can balance boot/order effects.
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
# The official SWE-Verified dataset is already cached under the runner's
# repository-local HF_HOME. Keep the two expensive arms independent of Hub
# availability and fail if that exact cached dataset is absent.
export HF_HOME=/home/mark/shared/lumoFlyWheel/.cache/huggingface
export HF_HUB_OFFLINE=1
export DEPLOY_FORCE_TEMP=0.6
export FR10_METRICS=0
export BATCH_INVARIANT=0
export FR13_HYDRA23=0
export FR13_TAIL_BRANCHES=0
export FR13_TAIL_BRANCH_DEPTHS=0

# Low-overhead stage timers required for physical-step accounting.
export FR13_SFWD_GPU_TIMER=1
export FR13_DFWD_GPU_TIMER=1
export FR13_CFWD_GPU_TIMER=1
export FR13_SFWD_GPU_TIMER_MAXPENDING=256
export FR13_SFWD_GPU_TIMER_SAMPLES_MAX=200000
export FR13_SFWD_GPU_TIMER_DUMP_S=1
export FR13_SFWD_SAMPLES_DUMP_S=30
export FR13_SPAN_GPU_TIMER_DUMP_S=1
export FR13_STEP_WALL_CAP_S=1.5
export FR13_WEIGHT_FLOOR_MS=98.6
export FR13_COMPUTE_MS_PER_ROW=0.54

# Coarse/synchronizing timers and alternate execution routes stay off.
export FR13_MULTIDRAFT_GPU_TIMER=0
export FR13_REPLAY_GPU_TIMER=0
export FR13_COMMIT_FULL_GPU_TIMER=0
export FR13_COMMITTER_SG_TIMER=0
export FR13_REPLAY_ONLY_GPU_TIMER=0
export FR13_GRAPH_TIMER=0
export FR13_KVREMAP_TIMER=0
export FR13_STATEREMAP_TIMER=0
export FR13_DFWD_SPLIT_NEEDLE=0
export FR13_STEP_GRAPH=0
export FR13_COMMIT_OVERLAP=0
export FR13_REPLAY_MULTISTREAM=0
export CUDA_LAUNCH_BLOCKING=0

# Fail closed against inherited diagnostics, selfchecks, and profilers.
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

case "${FR13_FLOOR_ORDER:-TH}" in
  TH)
    run_variant tail6_${TAG} tail6 21 1
    run_variant hydra23_${TAG} hydra23 23 1
    ;;
  HT)
    run_variant hydra23_${TAG} hydra23 23 1
    run_variant tail6_${TAG} tail6 21 1
    ;;
  *)
    echo "FR13_FLOOR_ORDER must be TH or HT, got: ${FR13_FLOOR_ORDER:-}" >&2
    exit 2
    ;;
esac
