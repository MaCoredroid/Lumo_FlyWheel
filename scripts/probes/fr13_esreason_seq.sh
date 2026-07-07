# FR13 §129 exact-seed fallback-REASON diagnostic. bs=1024 (the TARGET — must work with exact-seed),
# GRAPH, SERVE_LOG=1 for the es_fb_<reason> obs. NO_FALLBACK stays OFF (soft: read the reason, don't
# crash). Reads WHY redirect_used=0: es_fb_no_record (no checkpoint at restore pos) vs es_fb_pos_mismatch
# (checkpoint at a different boundary). That localizes the capture-vs-restore position alignment fix.
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
export FR13_SERVE_LOG=1
run_variant cat8cache_esr_${TAG}  cat8  8  1
