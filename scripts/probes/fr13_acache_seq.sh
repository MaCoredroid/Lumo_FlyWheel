# FR13 §126 A-cache (A') flag-flip test. GRAPH mode (no eager confound), B=1 (isolates A'),
# split4. Hypothesis: the conv-snapshot branch redirect is default-OFF (SSM twin default-ON) =>
# branch-winner conv restore reads the wrong (col-0/spine) row => 14096/14309 give up with cache.
# Fix arm flips FR13_APC_CONV_SNAP_FIX=1 + FR13_APC_CONV_LEAF_COMPLETE=1. Gate: 14096/14309 give->R.
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
# Arm 1: the A-cache fix (conv redirect ON) — does 14096/14309 flip give-up -> resolve?
export FR13_APC_CONV_SNAP_FIX=1 FR13_APC_CONV_LEAF_COMPLETE=1
run_variant cat8cache_acon_${TAG}   cat8  8  1
# Arm 2: baseline (conv redirect OFF) — same-run control (expect 14096/14309 give up)
export FR13_APC_CONV_SNAP_FIX=0 FR13_APC_CONV_LEAF_COMPLETE=0
run_variant cat8cache_acoff_${TAG}  cat8  8  1
