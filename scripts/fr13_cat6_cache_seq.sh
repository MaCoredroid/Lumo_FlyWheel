# FR13 cat6root + cache — STANDALONE single-arm sequence (user 2026-07-08).
# Relaunch target if the hot-edited remaining3_seq boundary did NOT pick up cat6
# (bash may have buffered the stale no-cache arms while sourcing). All cache-ON,
# same stateless-tree flags as the cat8+cache deliverable. cat6root = 6-node depth-5
# branch tree via the forked launcher TREE override. Final matrix cell: cat6+cache.
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1
run_variant sl_cat6_cache_qc4       cat6root   6  1
