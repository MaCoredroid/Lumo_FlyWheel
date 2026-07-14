# FR13_SLOT_REORDER live goal-gate arm: cat8 + cache (EXACT ship env of
# fr13_cat8_cat6_native_cachefirst_seq.sh Arm 1) + FR13_SLOT_REORDER=1.
# Compare vs existing clean refs: native+cache 3.050 accept / 8/16 resolve / 1 give-up
# (FR13_B4_CACHE_MATRIX_RESULTS.md) + fix-on-remap cat8 ~3.3 (FR13_REMAP_SHIP_RESULTS.md).
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1 FR13_ATTN_KV_REMAP=1 FR13_SLOT_REORDER=1
run_variant slreorder_cat8_cache_${TAG} cat8 8 1
