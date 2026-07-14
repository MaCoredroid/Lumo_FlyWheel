# P1 B=4 A/B: two arms on 4 live SWE tasks each (short), ship cache env + timers,
# FR13_DM_DEPTHSYNC off/on. cfwd under REAL B=4 concurrency = the decisive number.
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1 FR13_ATTN_KV_REMAP=1 FR13_SLOT_REORDER=1
export FR13_DM_DEPTHSYNC=0
run_variant p1b4_legacy_${TAG}  cat8 8 1
export FR13_DM_DEPTHSYNC=1
run_variant p1b4_dson_${TAG}    cat8 8 1
