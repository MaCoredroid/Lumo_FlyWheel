# FR13_SLOT_REORDER live goal-gate 4-ARM sequence (user 2026-07-14):
#   ORDER = risk-aware: cat8+fix (deliverable) -> native mtp5 (fresh same-binary bar)
#   -> cat6+fix (superset A/B) -> 33333+fix (depth-5, MTP-5-comparable, 2 branches/depth). A partial campaign still yields the
#   decisive cat8-vs-native answer after two arms.
# Tree arms: EXACT ship cache env (fr13_cat8_cat6_native_cachefirst_seq.sh) + the fix.
# NATIVE arm: tree flags UNSET (FR13_SLOT_REORDER=1 on a treeless boot fail-louds by
# design — pi underivable; remap is tree-only) — mirrors the cachefirst seq pattern.
# Engagement pi expectations: cat8 [0,1,3,5,7,8,2,4,6] | 33333 [0,1,4,7,10,13,2,3,5,6,8,9,11,12,14,15].
# SUPERSET (user): spines canonical bit-exact (S1 KPERM) => predict cat8 ≈ cat6 + ~0.17.
_SR_TREE_ENV() {
  export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
  export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
  export FR13_ENABLE_APC=1 FR13_ATTN_KV_REMAP=1 FR13_SLOT_REORDER=1
}
_SR_NATIVE_ENV() {
  unset FR13_APC_COMMIT_TO_RUNNING_ROW FR13_TREE_RUNROW_INIT FR13_APC_BURN_NODE_BANK \
        FR13_ENABLE_APC FR13_APC_EXACT_SEED MAMBA_BLOCK_SIZE MAMBA_SSM_CACHE_DTYPE \
        FR13_ATTN_KV_REMAP FR13_SLOT_REORDER
}
_SR_TREE_ENV
run_variant slreorder_cat8_cache_${TAG}  cat8      8  1
_SR_NATIVE_ENV
run_variant native_ourcache_${TAG}       nativemtp5_exseed  5  1
_SR_TREE_ENV
run_variant slreorder_cat6_cache_${TAG}  cat6root  6  1
run_variant slreorder_33333_cache_${TAG} t33333    15 1
