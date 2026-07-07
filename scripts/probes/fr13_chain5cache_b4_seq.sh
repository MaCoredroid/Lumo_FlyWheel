# FR13 missing cell (user 2026-07-06): chain5(spine5) + EXACT_SEED cache at B=4.
# The CLEAN cache-on shape control: vs cat8+cache@B=4 (0/4) — same cache, same B=4, only branches->spine.
#   chain5+cache B=4 ~3/4 => branches are the carrier WITH cache too (spine+cache clean under co-residency)
#   chain5+cache B=4 <<3/4 => the cache DOES hurt the spine under B=4 (cache+co-residency interaction beyond branches)
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
run_variant chain5cache_b4_${TAG}  chain5  5  1
