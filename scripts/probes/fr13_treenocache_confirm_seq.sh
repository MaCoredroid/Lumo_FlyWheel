# FR13 tree+NO-cache give-up baseline on the 4-task split4, B1, GRAPH. Both SPINE (chain5) and
# BRANCH (cat8) — both use the forked tree pipeline (user point). No cache => isolates whether the
# TREE PIPELINE ITSELF (decode) causes give-ups, independent of the cache. Correct expects (chain5=5,
# cat8=8). Gate: give-up count. If ~0 => give-ups are cache-specific (fix the cache/refold). If >0 =>
# the tree decode itself gives up (carrier A quality issue).
export FR13_ENABLE_APC=0 FR13_APC_EXACT_SEED=0
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
run_variant chain5nocache_${TAG}  chain5  5  1
run_variant cat8nocache_${TAG}    cat8    8  1
