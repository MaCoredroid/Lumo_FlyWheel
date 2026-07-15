# FR13 merged drafter FLAVOR-B arm (always mtp_k + suffix-fill whole tree, no adaptive skip gate).
# Run AFTER the merge16b A/B (which gives Flavor-A + MTP-only baseline on the SAME 16 tasks), so the
# 3-way comparison = Flavor-A (never-regress) vs Flavor-B (always-skip) vs baseline, via the new
# derived_tps_fullstep_gpu (drafter-inclusive) + resolve/give-up/garble. Same speed-gate cache env.
export FR13_APC_COMMIT_TO_RUNNING_ROW=1 FR13_TREE_RUNROW_INIT=1 FR13_APC_BURN_NODE_BANK=1
export FR13_APC_EXACT_SEED=0 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_ENABLE_APC=1 FR13_ATTN_KV_REMAP=1 FR13_SLOT_REORDER=1
export FR13_DRAFT_SOURCE=merged FR13_MERGED_FLAVOR=always FR13_MERGED_TREE_SPEC=1
run_variant merged_flavorB_t33333_${TAG}  t33333  15  1
