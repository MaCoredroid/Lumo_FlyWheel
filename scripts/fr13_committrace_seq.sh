# Baseline cat33333 (MTP-only) on REAL SWE-Verified tasks with the committer commit-trace ON, to measure
# on the LIVE gate: (1) branch-accept-rate per depth (do rank-2/3 help?), (2) MTP top-1 p (argmax_prob)
# nucleus distribution per depth (how often is top-1 confident => branches wasted, reallocate to tail).
# LUMO_TREE_SAMPLER_DEBUG_LOG forces the eager per-node walk (LOSSLESS distributions, SLOW -> do NOT read
# TPS; only accept/branch/prob stats). Trace -> /logs/commit_trace.jsonl (host <arm>/logs/).
export LUMO_PROXY_SSE_HEARTBEAT_S=15
export FR10_METRICS=0
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1 FR13_ATTN_KV_REMAP=1 FR13_SLOT_REORDER=1
export FR13_DEVICE_MULTIDRAFT=0
export LUMO_TREE_SAMPLER_DEBUG_LOG=/logs/commit_trace.jsonl
unset FR13_DRAFT_SOURCE
run_variant base_ctrace_t33333_${TAG}  t33333  15  1
