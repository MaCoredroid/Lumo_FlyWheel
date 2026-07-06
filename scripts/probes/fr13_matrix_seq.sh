# FR13 #7 MATRIX sequence — the SOLVED-PRODUCT comparison at n=16:
#   tree(cat8) + EXACT_SEED cache   vs   native MTP-5 + EXACT_SEED cache
# both with the baked carrier fixes (default-ON) + official instance_image env
# (default) + hot-path logging off. Sourced by fr13_b4_campaign_driver.sh with
# run_variant/run_native + $TAG/$BSIZE/$CONC/$WALL/$SUBSET in scope.
# APC + EXACT_SEED are per-arm (not baked); enable for both cache arms:
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR10_METRICS=0            # hot-path metrics OFF (speed regime; brackets via /metrics)
export LUMO_PROXY_SSE_HEARTBEAT_S=15   # heartbeat-only offload (survives emit wedges; ~0 cost)
run_variant cat8cache_${TAG}  cat8              9  1   # tree(cat8) + cache  (Product: tree+cache)
run_variant natcache_${TAG}   nativemtp5_exseed 5  1   # native MTP-5 + cache (reference)
