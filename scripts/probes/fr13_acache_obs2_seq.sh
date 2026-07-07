# FR13 §126b A-cache obs localization, GRAPH mode + FR13_SERVE_LOG=1 (routes FR13_OBS_SUMMARY
# to /logs/fr13_apc_exact_seed_eng.log every 60s). Read conv_leafmap_miss/hit + redirect_
# engaged/used/fallback: if fallback/miss >> 0, the baked conv redirect is materially partial
# (the residual A' lead); if ~0, the redirect works and A' is elsewhere. Cumulative across
# 4 tasks (12907 R + 14096/14309 give) — a first mechanism signal, not per-task attribution.
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
export FR13_SERVE_LOG=1
run_variant cat8cache_obs2_${TAG}  cat8  8  1
