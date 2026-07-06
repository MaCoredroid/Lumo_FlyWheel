# FR13 BATCH-AXIS control (user 2026-07-06: "could be B4 vs B1, account for it").
# tree+cache at B=1 on split4 vs the existing tree+cache B=4 (0/4). SAME config, ONLY batch differs.
#   tree+cache B=1 ~4/4  => the tree's degradation is a B=4 CO-RESIDENCY effect (native tolerates B=4, tree doesn't)
#   tree+cache B=1 <4/4  => the tree degrades regardless of batch (fundamental)
# Also a tree+nocache B=1 arm to see if the RAMBLE is B=4-specific too.
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
run_variant cat8cache_b1_${TAG}   cat8   9  1
export FR13_ENABLE_APC=0 FR13_APC_EXACT_SEED=0
run_variant cat8nocache_b1_${TAG} cat8   9  1
