# FR13 remaining-3 arms (user 2026-07-07: "complete where we left off" after exact_seed
# DEPRECATION). cat8+cache already DONE (16/16, kept). Re-run:
#   1 native+cache (nativemtp5_exseed, now EXACT_SEED=0 hard-deprecated -> NO es_ckpt leak;
#     validates the leak fix: memory must plateau at the KV ceiling, not OOM at 3.5h)
#   2 cat8+nocache (the race-skipped clean superset, task #15)
#   3 native+nocache (the bar)
# qwen-code, NUDGE OFF, no wall (WALL=0), no turn limit. Sourced by fr13_b4_campaign_driver.sh.

# ---- Arm 1: native MTP-5 + our cache (exact_seed=0 base forked cache) ----
run_variant native_ourcache_qc4    nativemtp5_exseed  5  1

# ---- Arm 2: cat6root (branch, 6-node, depth-5) + CACHE ----
# user 2026-07-08: CANCELLED the two no-cache arms (cat8+nocache, native+nocache);
# ADDED cat6+cache. Final matrix = all cache-ON: native+cache / cat8+cache / cat6+cache.
# spine5 not needed. Same stateless-tree cache flags as the cat8+cache deliverable
# (COMMIT_TO_RUNNING_ROW + TREE_RUNROW_INIT + BURN_NODE_BANK, EXACT_SEED deprecated-off).
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1
run_variant sl_cat6_cache_qc4       cat6root   6  1
