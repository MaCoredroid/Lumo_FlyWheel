# FR13 tree-vs-cache split: TREE cat8 with cache OFF (FR13_ENABLE_APC=0), same 8 fr9-resolved tasks.
# vs native (4/4 R, cache-off) and tree+cache (0/8). Resolves => APC cache is the agentic carrier;
# fails => the tree spec-decode itself degrades agentic output.
export FR13_ENABLE_APC=0 FR13_APC_EXACT_SEED=0
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
run_variant cat8nocache_${TAG}  cat8  9  1
