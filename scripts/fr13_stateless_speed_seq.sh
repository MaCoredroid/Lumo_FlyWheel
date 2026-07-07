# FR13 STATELESS-TREE speed + give-up matrix (user 2026-07-07):
# "stateless tree with and without cache compare with native mtp5"; tree = cat8 AND spine5.
# GRAPH mode only (no ENFORCE_EAGER — eager degrades behavior). Each arm: fresh boot,
# B=$BSIZE co-residency, CONC codex tasks -> give-ups (concurrency/carrier-B gate) +
# deploy-speed (the matrix). Sourced by fr13_b4_campaign_driver.sh (run_native/run_variant
# in scope). native-mtp5 = the incumbent bar; stateless tree = COMMIT_TO_RUNNING_ROW +
# TREE_RUNROW_INIT + BURN_NODE_BANK (dep-guard: all 3), EXACT_SEED=0.

# ---- Arm 1: native MTP-5 bar (no tree flags) ----
run_native nativemtp5_${TAG}          5      5  1

# ---- stateless-tree flags (the 3-flag lifecycle; block/dtype for the cache path) ----
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32

# ---- Arm 2: cat8 (branch) + cache ----
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

# clean the tree flags out of the driver env (defensive; the driver is about to end)
unset FR13_ENABLE_APC FR13_APC_EXACT_SEED FR13_APC_COMMIT_TO_RUNNING_ROW \
      FR13_TREE_RUNROW_INIT FR13_APC_BURN_NODE_BANK MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE
