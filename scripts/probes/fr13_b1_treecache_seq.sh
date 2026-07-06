# FR13 BATCH x SHAPE control (user 2026-07-06: "separate spine5 and cat8" for the B1-vs-B4 question).
# Crosses shape (chain5 spine5 / cat8 branches) x batch, cache held OFF for the clean shape compare
# (+ a cat8+cache B=1 arm for the direct §66-style give-up test). Launch with BSIZE=1.
#   cat8 B=1 ~4/4 (recovers) + chain5 same B1==B4  => the BRANCHES' B=4 co-residency is the carrier (reshape-fixable)
#   cat8 B=1 <4/4 (still degrades)                 => cat8 degrades regardless of batch (fundamental)
#   chain5 B=1 == chain5 B=4                        => spine is batch-robust (no co-resident branches)
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
# --- cat8 (branches), cache OFF, B=1 ---
export FR13_ENABLE_APC=0 FR13_APC_EXACT_SEED=0
run_variant cat8nocache_b1_${TAG}  cat8    9  1
# --- chain5 (spine5), cache OFF, B=1 ---
run_variant chain5nc_b1_${TAG}     chain5  5  1
# --- cat8 + EXACT_SEED cache, B=1 (direct §66-style give-up test) ---
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
run_variant cat8cache_b1_${TAG}    cat8    9  1
