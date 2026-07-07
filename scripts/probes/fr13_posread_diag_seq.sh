# FR13 diagnostic: single spine arm, REQKEY=1 + PROBE=1 + the SEEN trace. The first
# 12 postprocess reads log (maplen, sid_in, j, bi, phase) to distinguish: map-empty/
# sid-miss (j=None, probe no-op) vs j==bi (misindex absent) vs j!=bi (misindex present).
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
export FR13_TREE_POSREAD_PROBE=1 FR13_TREE_POSREAD_REQKEY=1
run_variant chain5cache_diag_${TAG}  chain5  5  1
