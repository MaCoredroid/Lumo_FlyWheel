# FR13_SLOT_REORDER live goal-gate 3-ARM sequence (user 2026-07-14): cat8 / cat6 / 333,
# ALL with the fix ON + EXACT ship cache env (fr13_cat8_cat6_native_cachefirst_seq.sh).
# Fix is tree-agnostic (pi from SPEC_CONFIG); expect per-arm engagement pi:
#   cat8 [0,1,3,5,7,8,2,4,6] | 333 [0,1,4,7,2,3,5,6,8,9] | cat6root per its tree.
# SUPERSET question (user): spines now canonical bit-exact (S1 KPERM) => predict
# cat8 = spine+3branches > cat6 = spine+1branch by ~ the extra branch rescue.
# Refs: native+cache 3.050 / 8/16 resolve / 1 give-up (matrix); fix-on-remap cat8 ~3.3.
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1 FR13_ATTN_KV_REMAP=1 FR13_SLOT_REORDER=1
run_variant slreorder_cat8_cache_${TAG}  cat8      8  1
run_variant slreorder_cat6_cache_${TAG}  cat6root  6  1
run_variant slreorder_333_cache_${TAG}   333       9  1
