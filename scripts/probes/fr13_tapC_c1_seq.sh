# CONC=1 spine+cache Tap C control. If stale_read ~0 @CONC=1 => the 480 @CONC=4 = carrier B (concurrency).
# If stale_read ~= 480 @CONC=1 => NOT concurrency-specific => the zeroed-source is the A-cache cache-restore
# defect (get_temporal_copy_spec source wrong), single-agent.
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
export FR13_REPLAY_BOUNDARY_LOG=1
export FR13_REPLAY_BOUNDARY_LAYERS=layers.0.linear_attn
run_variant chain5cache_tapCc1_${TAG}  chain5  5  1
