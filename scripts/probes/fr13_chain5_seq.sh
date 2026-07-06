# FR13 mechanism confirm: chain5 = FORKED tree kernel, PURE 5-spine (no branches), cache-OFF.
# 3-way (cache-off, subset_split4): native 4/4 vs cat8(branches) 1/4 vs chain5(spine) ?
# chain5 ~4/4 => branches/co-residency drift is the agentic carrier (fixable via tree-reshape);
# chain5 ~1/4 => the forked kernel itself degrades even spine-only.
export FR13_ENABLE_APC=0 FR13_APC_EXACT_SEED=0
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
run_variant chain5nocache_${TAG}  chain5  5  1
