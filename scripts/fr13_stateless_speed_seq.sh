# FR13 STATELESS-TREE speed + give-up matrix (user 2026-07-07):
# "stateless tree with and without cache compare with native mtp5" (native CACHE ON);
# tree = cat8 AND spine5(chain5). 16-task set (subset_b4_sixteen). GRAPH mode only
# (no ENFORCE_EAGER — eager degrades). Each arm: fresh boot, B=$BSIZE co-residency, CONC
# codex tasks -> give-ups (concurrency/carrier-B gate) + deploy-speed (matrix). Sourced by
# fr13_b4_campaign_driver.sh (run_native/run_variant in scope). Ordered so the BAR
# (native+cache) and the DELIVERABLE (cat8+cache) complete first.
# Sizing (same for native+cache and tree+cache => apples): MAMBA_BLOCK_SIZE=1024,
# MAMBA_SSM_CACHE_DTYPE=float32; EXACT_SEED=0 (its chunked-staging retired).

# ---- Arm 1: native MTP-5 + CACHE ON (the fair bar; NO tree/stateless flags) ----
# nativemtp5apc = LAUNCHER=native + NATIVE_ENABLE_APC=1 + MAMBA/APC block flags (XFLAGS).
run_variant nativemtp5_${TAG}         nativemtp5     5  1  # native+APC crashes on GDN@B=4 (stock vLLM device assert); no-cache is native's working bar

# ---- stateless-tree 3-flag lifecycle (tree arms ONLY; set AFTER the native arm) ----
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32

# ---- Arm 2: cat8 (branch) + cache  [THE DELIVERABLE] ----
export FR13_ENABLE_APC=1
run_variant sl_cat8_cache_${TAG}      cat8   8  1
# ---- Arm 3: cat8 (branch) + no-cache ----
export FR13_ENABLE_APC=0
run_variant sl_cat8_nocache_${TAG}    cat8   8  1
# ---- Arm 4: spine5 (chain5) + cache ----
export FR13_ENABLE_APC=1
run_variant sl_spine5_cache_${TAG}    chain5 5  1
# ---- Arm 5: spine5 (chain5) + no-cache ----
export FR13_ENABLE_APC=0
run_variant sl_spine5_nocache_${TAG}  chain5 5  1

# defensive cleanup of the driver env (driver is about to end)
unset FR13_ENABLE_APC FR13_APC_EXACT_SEED FR13_APC_COMMIT_TO_RUNNING_ROW \
      FR13_TREE_RUNROW_INIT FR13_APC_BURN_NODE_BANK MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE
