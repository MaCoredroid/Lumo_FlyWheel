# FR13 STATELESS-TREE 4-arm — CACHED-FIRST, qwen-code, NO-WALL, NO-TURN-LIMIT, NUDGE-OFF
# (user 2026-07-07). Honest give-up gate:
#   - agent = qwen-code (harness default now qwen_code; codex nudge confounded prior data)
#   - NUDGE OFF (LUMO_PROXY_AUTO_CONTINUE=0, default now) — proven off by serve/offload assert
#   - NO wall (WALL=0), NO turn limit (qwen --max-session-turns 100000); backstop = 600s stall-watchdog
#   - official SWE-bench per-instance images (SWE_AGENT_ENV=instance_image), temp 0.6
# CACHED arms FRONT (prefix cache skips the re-prefill => fast => deliverable lands first);
# the slow nocache bars run last.
#   1 cat8+cache (DELIVERABLE)       2 native-MTP5 + our cache (apples cache-on bar)
#   3 cat8+nocache (clean superset)  4 native-MTP5 nocache (the bar)
# Arm 2 nativemtp5_exseed: exact_seed INERT for native (base cache number). GRAPH mode.

# ---- Arm 1: cat8 (branch) + cache  [THE DELIVERABLE — cached, fast] ----
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1
run_variant sl_cat8_cache_${TAG}      cat8   8  1

# ---- Arm 2: native MTP-5 + our cache  [apples cache-on bar — cached, fast] ----
unset FR13_APC_COMMIT_TO_RUNNING_ROW FR13_TREE_RUNROW_INIT FR13_APC_BURN_NODE_BANK \
      FR13_ENABLE_APC FR13_APC_EXACT_SEED MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE
run_variant native_ourcache_${TAG}    nativemtp5_exseed  5  1

# ---- Arm 3: cat8 (branch) + no-cache  [CLEAN SUPERSET — slow] ----
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=0
run_variant sl_cat8_nocache_${TAG}    cat8   8  1

# ---- Arm 4: native MTP-5 + no-cache  [THE BAR — slow] ----
unset FR13_APC_COMMIT_TO_RUNNING_ROW FR13_TREE_RUNROW_INIT FR13_APC_BURN_NODE_BANK \
      FR13_ENABLE_APC FR13_APC_EXACT_SEED MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE
run_variant nativemtp5_${TAG}         nativemtp5     5  1
