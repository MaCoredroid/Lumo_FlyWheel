# FR13 BATCH x SHAPE x CACHE control (user 2026-07-06: "separate spine5 and cat8" + "add spine5+cache").
# Launch with BSIZE=1. Completes shape(chain5 spine5 / cat8 branches) x cache(EXACT_SEED / off) at B=1.
# SHIPPABLE CANDIDATE = chain5(spine5)+cache: native-drift-level kernel WITH the cache = both speedups + (hoped) correctness.
#   chain5+cache B=1 ~4/4  => the shippable config WORKS at B=1 (then test B=4 for co-residency).
#   cat8+cache  B=1 ~4/4   => cat8's B=4 give-up was co-residency (recovers at B=1) — §66-style give-up test.
#   chain5 B1==B4 (nc)     => spine batch-robust; cat8 B=1 recovers => branches' B=4 co-residency is the carrier.
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
export MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
# --- SHIPPABLE candidate: spine5 + EXACT_SEED cache, B=1 (run first) ---
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1
run_variant chain5cache_b1_${TAG}  chain5  5  1
# --- cat8 + EXACT_SEED cache, B=1 (§66-style give-up test) ---
run_variant cat8cache_b1_${TAG}    cat8    9  1
# --- spine5 nocache B=1 (vs spine5@B=4 = cachesep arm2) ---
export FR13_ENABLE_APC=0 FR13_APC_EXACT_SEED=0
run_variant chain5nc_b1_${TAG}     chain5  5  1
# --- cat8 nocache B=1 (vs cat8@B=4 nc = 1/4) ---
run_variant cat8nocache_b1_${TAG}  cat8    9  1
