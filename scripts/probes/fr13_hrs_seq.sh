# FR13 HRS measurement (user priority): recompute-suffix (SGLang-aligned) vs the vacuous exact-seed.
# B1, split4, GRAPH. EXACT_SEED=0 (else it DISABLES HRS). The two baked give-up fixes (ZERO_MAMBA,
# COPY_SRC) + CONV_LEAF_COMPLETE + SNAP_FIX stay at baked defaults (on; SNAP_FIX auto-no-ops on the
# cache-hit prefill where HRS fires). SERVE_LOG for HRS engagement obs + give-up counts. Gate:
# (1) does HRS cut give-ups vs HRS-off? (2) SPEED TAX = elapsed(HRS-on) - elapsed(HRS-off) (the
# suffix-recompute is a TTFT/prefill cost on each cache hit).
export FR13_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR13_APC_EXACT_SEED=0
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
export FR13_SERVE_LOG=1
# Arm 1: HRS ON (cap=64)
export FR13_APC_HIT_RECURRENT_SUFFIX=1 FR13_APC_HIT_SUFFIX_CAP=64
run_variant cat8cache_hrson_${TAG}  cat8  8  1
# Arm 2: HRS OFF (recurrent-leaf baseline, same otherwise)
export FR13_APC_HIT_RECURRENT_SUFFIX=0
run_variant cat8cache_hrsoff_${TAG}  cat8  8  1
