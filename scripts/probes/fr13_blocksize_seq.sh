# FR13 §127 A' fix candidate: raise MAMBA_BLOCK_SIZE 1024->8192 so chunked-prefill crosses fewer
# align-mode block boundaries (the carrier per project_fr13_apc_spec_specific_carrier). GRAPH mode,
# B=1, split4. Gate: does 14096/14309 flip give-up -> resolve vs the block_size=1024 baseline (1/4)?
# max_num_batched auto-couples to block_size in the launcher (overshoot fix).
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_SSM_CACHE_DTYPE=float32
export MAMBA_BLOCK_SIZE=8192
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
run_variant cat8cache_bs8192_${TAG}  cat8  8  1
