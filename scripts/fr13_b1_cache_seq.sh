# FR13 B=1 cache-ON pair (user 2026-07-08): cat6+cache and native+cache at TRUE B=1.
# The tree's FAVORABLE regime — genuine single-stream decode (BSIZE=1 -> --max-num-seqs 1,
# CONC=1 -> one SWE task at a time), HBM-bound (weight-read floor dominates), accept-per-forward
# is the ONLY speed lever and the tree's extra verify compute is nearly free. This is the decisive
# test of whether the tree's accept edge beats native MTP-5 per committed token when NOT fighting
# the B=4 max_num_batched_tokens=1024 concurrency throttle.
# Run AFTER the B=4 cat6 arm completes. Driver env: BSIZE=1 CONC=1 TAG=qc1
#   RUNROOT=output/fr13_qwencode_cachefirst_b1  SUBSET=subset_b4_sixteen  WALL=0
# NOTE: CONC=1 + no wall + hard tasks (13398 ran 4.5h) => this is SLOW; the deploy_speed reduce
# (accept/event + s_per_fwd_gpu) is robust after a few tasks even if the full 16 take long.

# ---- Arm 1: cat6+cache @ B=1 (sl_cat6_cache_qc1) ----
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1
run_variant sl_cat6_cache_qc1       cat6root   6  1

# ---- Arm 2: native MTP-5 + cache @ B=1 (native_ourcache_qc1) ----
# KIND nativemtp5_exseed XFLAGS provide the base forked APC cache; unset the tree-cache flags.
unset FR13_APC_COMMIT_TO_RUNNING_ROW FR13_TREE_RUNROW_INIT FR13_APC_BURN_NODE_BANK \
      FR13_ENABLE_APC FR13_APC_EXACT_SEED MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE
run_variant native_ourcache_qc1     nativemtp5_exseed  5  1
