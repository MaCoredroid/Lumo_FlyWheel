# FR13 B=4 carrier localization — two arms on the give-up subset (B=4/CONC=4).
# Baseline already measured: B=4 + cache + BI=0 => 4/4 give-up (§81/82).
#   Cell B: NO-CACHE (FR13_ENABLE_APC=0) baseline-numerics  -> tests the concurrent
#           EXACT_SEED cache-restore carrier. Resolves => cache is it. Give-up => not cache.
#   Cell A: CACHE on + BATCH-INVARIANT (BI=1, BI_TREE_ATTN=1) -> tests batch-variant
#           kernel numerics. Resolves => batch-variance is it. Give-up => not batch-variance.
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
export MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32

# --- Cell B: no-cache, baseline numerics ---
export FR13_ENABLE_APC=0 FR13_APC_EXACT_SEED=0 BATCH_INVARIANT=0 FR13_BI_TREE_ATTN=0
run_variant cellB_nocache_${TAG}   cat8  9  1

# --- Cell A: cache on + batch-invariant numerics ---
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 BATCH_INVARIANT=1 FR13_BI_TREE_ATTN=1
run_variant cellA_batchinv_${TAG}  cat8  9  1
