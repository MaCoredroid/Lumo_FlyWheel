# GATE 0.5 (design §6b): does the HARNESS-AWARE PRE-WARM lift arctic's suffix coverage on real SWE tasks?
# Arm 1 = merged + PRE-WARM (FR13_PREWARM_TRIE -> launcher copies corpus to /logs -> maybe_prewarm seeds).
# Arm 2 = merged + COLD (no pre-warm). Same tree/committer/cache; only diff = the pre-warm sidecar.
# Read the [FR13_MERGED ENGAGED] needle match_full/match_partial (arctic COVERAGE) + deploy_speed accept.
# Pre-warm is monotone-lossless (ADD only) so it can only raise coverage; magnitude is the question.
export LUMO_PROXY_SSE_HEARTBEAT_S=15
export FR10_METRICS=0
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1 FR13_ATTN_KV_REMAP=1 FR13_SLOT_REORDER=1
export FR13_DRAFT_SOURCE=merged

# Arm 1: PRE-WARM
export FR13_PREWARM_TRIE=/home/mark/shared/lumoFlyWheel/output/fr13_prewarm/corpus_harness.jsonl
run_variant merged_prewarm_t33333_${TAG}  t33333  15  1

# Arm 2: COLD
unset FR13_PREWARM_TRIE
run_variant merged_cold_t33333_${TAG}     t33333  15  1
