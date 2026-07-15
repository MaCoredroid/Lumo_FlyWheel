# FR13 merged-drafter LIVE agentic A/B (gate f) — MTP-k+Arctic-suffix cat33333 vs MTP-only t33333.
# Same ship cache env + tree flags on BOTH arms; identical tasks/B; the only diff is
# FR13_DRAFT_SOURCE=merged (arm 2). The driver's qwen-code request-dump gives the garble A/B +
# resolve/give-ups; the dfwd/sfwd sidecars give the speed A/B. Merged arm runs FIRST so the
# [FR13_MERGED ENGAGED] docker log (match_full/skip_fired on REAL agentic repetition) lands early
# and decides whether the full A/B is worth completing. run_variant is in scope (driver-sourced).
#
# CANONICAL offload/proxy env (was DRIFTED-MISSING -> caused the empty-patch flake): the merged
# drafter's arctic overhead trips the GB10 emit-wedge; heartbeat masks the mid-stream idle so
# qwen-code's stream-idle abort doesn't cut the agent off mid-edit. FR10_METRICS=0 = speed regime.
export LUMO_PROXY_SSE_HEARTBEAT_S=15
export FR10_METRICS=0
# Ship cache env (mirrors fr13_cat8_cat6_native_cachefirst_seq.sh + the SLOT_REORDER fix):
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1 FR13_ATTN_KV_REMAP=1 FR13_SLOT_REORDER=1

# ---- Arm 1: t33333 + FR13_DRAFT_SOURCE=merged (Arctic suffix grow) — the DELIVERABLE ----
export FR13_DRAFT_SOURCE=merged
run_variant merged_arctic_t33333_${TAG}  t33333  15  1

# ---- Arm 2: t33333 MTP-only baseline (same tree, no Arctic) — the never-regress bar ----
unset FR13_DRAFT_SOURCE
run_variant merged_base_t33333_${TAG}    t33333  15  1
