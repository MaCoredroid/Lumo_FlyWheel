# FR13 Cell C — CONFOUND CHECK: do the B=4 give-up tasks give up at B=1 too?
# Same locked cat8+cache build as the matrix, but B=1/CONC=1 (SOLVED regime, no
# co-residency). If these tasks RESOLVE / stop giving up here => B=4 is the carrier.
# If they give up at B=1 too => task-hardness, NOT B=4 (thesis refuted).
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
run_variant cat8cache_${TAG}  cat8  9  1
