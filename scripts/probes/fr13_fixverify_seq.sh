# FR13 carrier-B fix GATE: chain5+cache @ B=4 with FR13_FREE_TREE_POSGLOBALS=1.
# Expect 0/4 (flag-off baseline) -> ~4/4 (flag-on). If it stays 0/4 => read dead under baked config, carrier elsewhere.
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_FREE_TREE_POSGLOBALS=1
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
run_variant chain5cache_fixon_${TAG}  chain5  5  1
