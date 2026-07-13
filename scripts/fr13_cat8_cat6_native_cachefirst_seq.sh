# FR13 SEQUENCE: cat8 + cat6 + native, CACHE-ON (APC) — the ship-config garble+speed A/B.
# GOAL = cat8 AND cat6 branched garble-free on the ship config. This runs BOTH branched
# trees + the native cache-on bar on the SAME 16 tasks at B4, so the qwen-code request-dump
# JSONLs give the garble A/B (cat8-vs-native, cat6-vs-native) AND the speed reduce per arm.
# Mirrors fr13_stateless_4arm_cachefirst_seq.sh cat8+cache env; adds cat6root; drops the slow
# nocache superset arms (cache-ON is the ship config + the garble gate). run_variant is in
# scope (sourced by fr13_b4_campaign_driver.sh). FR13_ATTN_KV_REMAP forwards to cat8 AND cat6
# (both forked launcher); inert for the native chain arm.

# ---- Arm 1: cat8 (branch) + cache  [DELIVERABLE — cached, fast] ----
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1
run_variant sl_cat8_cache_${TAG}      cat8       8  1

# ---- Arm 2: cat6 (branch) + cache  [DELIVERABLE — cached, fast] ----
run_variant sl_cat6_cache_${TAG}      cat6root   6  1

# ---- Arm 3: native MTP-5 + our cache  [apples cache-on bar — cached, fast] ----
unset FR13_APC_COMMIT_TO_RUNNING_ROW FR13_TREE_RUNROW_INIT FR13_APC_BURN_NODE_BANK \
      FR13_ENABLE_APC FR13_APC_EXACT_SEED MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE
run_variant native_ourcache_${TAG}    nativemtp5_exseed  5  1
