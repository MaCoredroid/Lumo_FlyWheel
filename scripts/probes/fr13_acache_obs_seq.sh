# FR13 §126 A-cache residual-fallback localization. GRAPH mode (no eager confound), B=1,
# baked config (conv redirect ON, as deployed). Read per-pid obs: redirect_fallback /
# conv_leafmap_miss > 0 on the deterministically-failing tasks (14096/14309) = the residual
# A-cache defect (redirect falls back to wrong row). conv_leafmap_hit = redirect worked.
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
run_variant cat8cache_obs_${TAG}  cat8  8  1
