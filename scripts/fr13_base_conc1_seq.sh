# CONC=1 BASELINE (pure MTP-only t33333, NO merged/arctic) on the same 4 resolved tasks as gc1diag, to
# complete the apples-to-apples sparse-serving A/B: gated-CONC1 (accept 2.99, fullstep 14.85) vs THIS.
# If gated fullstep > baseline fullstep => confidence gate wins in sparse serving; else it washes there too.
# Identical ship-cache env; the ONLY diff vs the gated arm is FR13_DRAFT_SOURCE unset (no arctic, no skip).
export LUMO_PROXY_SSE_HEARTBEAT_S=15
export FR10_METRICS=0
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1 FR13_ATTN_KV_REMAP=1 FR13_SLOT_REORDER=1

unset FR13_DRAFT_SOURCE
run_variant base_t33333_${TAG}  t33333  15  1
