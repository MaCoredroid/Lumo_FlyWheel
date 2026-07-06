# FR13 cache-separation (user 2026-07-06): complete the 2x2 with the STRICT cache match.
# arm1 = native MTP-5 + EXACT_SEED cache (nativemtp5_exseed) = the SAME cache config as tree+cache
#   (FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 block_size=1024 ssm float32) => only spec-method differs vs tree+cache.
#   native+exseed ~4/4 => the EXACT_SEED cache is FINE on native => cache harm is TREE-SPECIFIC (tree is the carrier).
#   native+exseed <4/4 => the EXACT_SEED cache is an INDEPENDENT carrier (degrades native too).
# arm2 = chain5 forked pure-spine + cache-off => tree branches vs spine.
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
run_variant nativeexseed_${TAG}  nativemtp5_exseed  5  1
export FR13_ENABLE_APC=0 FR13_APC_EXACT_SEED=0
run_variant chain5nc_${TAG}      chain5             5  1
