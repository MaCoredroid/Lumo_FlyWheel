# FR13 §121 CORRECTED carrier-B fix (FR13_TREE_POSREAD_REQKEY). Spine isolates B.
# FIX ARM FIRST (decisive: does spec-row keying recover the spine?). PROBE on both
# (autotune-immune misindex counter): baseline ~0/4 (from prior control) already known.
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
export FR13_TREE_POSREAD_PROBE=1
# Arm 1: the corrected fix (REQKEY=1) — should recover 0/4 -> ~4/4, counter>0 (misindex corrected)
export FR13_TREE_POSREAD_REQKEY=1
run_variant chain5cache_reqkey_${TAG}  chain5  5  1
# Arm 2: baseline (REQKEY=0) — same-run control, counter>0 confirms misindex present in failure
export FR13_TREE_POSREAD_REQKEY=0
run_variant chain5cache_probe_${TAG}   chain5  5  1
