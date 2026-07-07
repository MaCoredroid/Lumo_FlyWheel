# FR13 STATELESS-TREE 4-arm NO-WALL speed+quality matrix (user 2026-07-07):
# rerun with NO agent/eval wall cap. WALL + EVAL_TIMEOUT_S set huge by the launch env;
# hang protection = the stall-watchdog (stream_idle_timeout 600s) per
# feedback_no_agent_wall_on_gates (total-time caps right-censor the give-up gate).
# Dropped both spine5 arms (user). 4 arms only:
#   1 native MTP-5 nocache (the bar)          2 cat8 + cache  (THE DELIVERABLE)
#   3 cat8 + nocache (clean decode superset)  4 native + OUR forked cache (apples cache-on bar)
# Arm 4 = nativemtp5_exseed: EXACT_SEED is INERT for native (verified patch:6017 forces the
# DEFAULT chunked path == base cache; native has no node-bank so the SNAP_FIX redirect is a no-op).
# So this number == the post-cleanup native + FR13_ENABLE_APC(exact_seed=0) config; the crash-relevant
# lever is the sizing bundle (max_num_batched=block_size=1024 #45238 overshoot fix + fp32), NOT exact_seed.
# GRAPH mode only (no ENFORCE_EAGER; eager degrades). Sourced by fr13_b4_campaign_driver.sh.

# ---- Arm 1: native MTP-5 + NO cache (the bar; no tree/stateless flags) ----
run_variant nativemtp5_${TAG}         nativemtp5     5  1

# ---- stateless-tree 3-flag lifecycle (tree arms ONLY; set AFTER the native bar) ----
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32

# ---- Arm 2: cat8 (branch) + cache  [THE DELIVERABLE] ----
export FR13_ENABLE_APC=1
run_variant sl_cat8_cache_${TAG}      cat8   8  1
# ---- Arm 3: cat8 (branch) + no-cache  [CLEAN SUPERSET] ----
export FR13_ENABLE_APC=0
run_variant sl_cat8_nocache_${TAG}    cat8   8  1

# ---- Arm 4: native + OUR forked cache (apples cache-on bar) ----
# unset the tree-lifecycle env so it cannot leak into the native arm; nativemtp5_exseed
# sets its own XFLAGS (FR13_ENABLE_APC=1 + FLASH_ATTN + naive_mtp + tree-off flags).
unset FR13_APC_COMMIT_TO_RUNNING_ROW FR13_TREE_RUNROW_INIT FR13_APC_BURN_NODE_BANK \
      FR13_ENABLE_APC FR13_APC_EXACT_SEED MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE
run_variant native_ourcache_${TAG}    nativemtp5_exseed  5  1
