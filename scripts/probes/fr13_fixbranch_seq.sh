# FR13 carrier-B fix on the BRANCH + spine re-test, both B=4 (seqs4,conc4), FR13_FREE_TREE_POSGLOBALS=1.
# cat8+cache flag-on: does the fix help the branch (0/4 baseline)? residual give-ups => carrier A' (within-request).
# chain5+cache flag-on (2nd run): is 14096's residual give-up a flake or a real 2nd seam?
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_FREE_TREE_POSGLOBALS=1
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
run_variant cat8cache_fixon_${TAG}    cat8    9  1
run_variant chain5cache_fix2_${TAG}   chain5  5  1
