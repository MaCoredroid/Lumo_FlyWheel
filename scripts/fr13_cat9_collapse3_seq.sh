# CHAIN A/B on the CLEANEST shape: cat9 (non-pb, locked) on the 3 collapse tasks.
# cat9pb (chain, depth-5, no overflow) already got 2.9-3.25 on these. cat9 (non-pb,
# SAME depth-5 shape, no chain) isolates the CHAIN with ZERO confound (no overflow,
# no arctic, no deep tail). cat9~4.5-5 => chain is the carrier (collapse confirmed
# on a clean shape); cat9~3 => the cat9 shape itself is weak on these tasks.
# Cache-OFF async to match the pb runs' regime.
export GPU_UTIL=0.70
unset FR13_PREWARM_TRIE
export FR13_ENABLE_APC=0
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
run_variant cat9_${TAG}  cat9  9  1
