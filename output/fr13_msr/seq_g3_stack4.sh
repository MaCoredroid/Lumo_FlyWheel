# GATE g3 (post-seal, 4-task DEPLOYMENT gate): the byte-sealed bar candidate
# — bar17r2 six-lever stack with SUBTREE_PARALLEL replacing HC (subtree
# shadows HC on the tree-decode launch; never sum their wins). Locally
# validated 2026-07-25: graph combo PASS (stack4) + eager byte-seal PASS
# (stack_eager_sc) after dst-rows-range + path-kernel-flags fixes.
# Legs: engagement needles (CPG_SERVE, SUBTREE preseed), accept band vs
# bv1x (comb_ev 5.40-6.07), garble eyeball, eps + measured_tps + floor_ratio.
# CLEAN run (zero instruments) — speed-of-record per the two-kinds rule.
export FR13_ENABLE_APC=1
export MAMBA_BLOCK_SIZE=1024
export APC_BLOCK_SIZE=1024
export MAMBA_SSM_CACHE_DTYPE=float32
export FR13_COMMITTER_BATCHED=1
export FR13_PARENT_GATHER=1
export FR13_CONV_PREGATHER=1
export FR13_FLAGS_INKERNEL=1
export FR13_HC_INTERNAL=0
export FR13_CONV_WB_BATCHED=1
export FR13_CONV_NODEBANK=1
export FR13_SPEC_BLOCKS_CAP=12
export FR13_SUBTREE_PARALLEL=1
export FR13_SUBTREE_PARALLEL_SELFCHECK=0
run_variant g3_stack4  tail6  21  1
