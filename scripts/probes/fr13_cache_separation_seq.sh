# FR13 cache-separation (user 2026-07-06): complete the 2x2. native+cache = the missing cell.
# Have: native+nc=4/4, tree+nc=1/4, tree+cache=0/4. native+cache answers "does the cache degrade NATIVE too?"
#   native+cache ~4/4  => cache is FINE on native => the tree is the carrier (cache harm is tree-specific)
#   native+cache <4/4  => the cache is an INDEPENDENT carrier (degrades native too) => fix cache AND tree
# Arm 2 = chain5 spine (forked, cache-off) — tree-shape/branch localization (branches vs spine).
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
run_variant nativeapc_${TAG}   nativemtp5apc  5  1
export FR13_ENABLE_APC=0 FR13_APC_EXACT_SEED=0
run_variant chain5nc_${TAG}    chain5         5  1
