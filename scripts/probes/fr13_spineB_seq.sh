# FR13 §120 CLEAN carrier-B confirmation on the SPINE (chain5 isolates B — no A').
# spine+cache CONC=1 = 4/4 (clean) vs CONC=4 = 0/4 (B fires). Does the positional-global
# fix (FR13_FREE_TREE_POSGLOBALS=1, engagement-proven: nonempty_clears>0) recover CONC=4
# toward 4/4? 0/4 -> ~4/4 is decisive above n=4 noise (unlike the A'-floored branch).
# Arm 3 = CONC=1 flag-ON control: does clearing globals HURT single-agent? (12907 regressed once.)
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
# Arm 1: spine + cache CONC=4, flag OFF (baseline ~0/4)
export FR13_FREE_TREE_POSGLOBALS=0
run_variant chain5cache_c4off_${TAG}  chain5  5  1
# Arm 2: spine + cache CONC=4, flag ON (the B fix — recover?)
export FR13_FREE_TREE_POSGLOBALS=1
run_variant chain5cache_c4on_${TAG}   chain5  5  1
