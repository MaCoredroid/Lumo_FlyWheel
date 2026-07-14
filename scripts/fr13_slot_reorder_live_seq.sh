# FR13_SLOT_REORDER live goal-gate 4-ARM sequence (user 2026-07-14):
#   cat8+fix / cat6+fix / native-mtp5 (fresh same-binary bar) / 333+fix
# Tree arms: EXACT ship cache env (fr13_cat8_cat6_native_cachefirst_seq.sh) + the fix.
# NATIVE arm: tree flags UNSET (FR13_SLOT_REORDER=1 on a treeless boot fail-louds by
# design — pi underivable; remap is tree-only) — mirrors the cachefirst seq pattern.
# Engagement pi expectations: cat8 [0,1,3,5,7,8,2,4,6] | 333 [0,1,4,7,2,3,5,6,8,9].
# SUPERSET (user): spines canonical bit-exact (S1 KPERM) => predict cat8 ≈ cat6 + ~0.17.
# Refs (garble-era-clean): native+cache 3.050 / 8/16 resolve / 1 give-up (matrix);
# fix-on-remap cat8 ~3.3 — the fresh native arm supersedes these as the primary bar.
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1 FR13_ATTN_KV_REMAP=1 FR13_SLOT_REORDER=1
run_variant slreorder_cat8_cache_${TAG}  cat8      8  1
run_variant slreorder_cat6_cache_${TAG}  cat6root  6  1
run_variant slreorder_333_cache_${TAG}   333       9  1
unset FR13_APC_COMMIT_TO_RUNNING_ROW FR13_TREE_RUNROW_INIT FR13_APC_BURN_NODE_BANK \
      FR13_ENABLE_APC FR13_APC_EXACT_SEED MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE \
      FR13_ATTN_KV_REMAP FR13_SLOT_REORDER
run_variant native_ourcache_${TAG}       nativemtp5_exseed  5  1
