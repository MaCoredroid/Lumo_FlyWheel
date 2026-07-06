# FR13 §117 CLEAN carrier-B verification (fixes §116 faults).
# Product = the BRANCH = golden cat9 tree (kind=cat9, expect=9 -> assert_engaged PASSES;
# my §115 fixbranch seq wrongly used kind=cat8 expect=9 => tok/draft 8!=9 FAIL-LOUD).
# Paired flag OFF vs ON, SAME boot conditions back-to-back, CONC=4 (driver CONC=4).
# GATE: the flagon arm's docker log MUST contain 'FR13_POSGLOBALS ENGAGED fires=N
# nonempty_clears=M' (engagement proof, added d6b91543). If nonempty_clears=0 => the
# clear is a NO-OP (positional globals empty at free) => carrier-B is NOT these globals
# => pivot. If fires marker absent => flag never engaged => discard.
export FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32
export FR10_METRICS=0
export LUMO_PROXY_SSE_HEARTBEAT_S=15
# arm 1: golden branch cat9 + cache, flag OFF (baseline)
export FR13_FREE_TREE_POSGLOBALS=0
run_variant cat9cache_flagoff_${TAG}  cat9  9  1
# arm 2: golden branch cat9 + cache, flag ON (fix) — REQUIRE the POSGLOBALS marker
export FR13_FREE_TREE_POSGLOBALS=1
run_variant cat9cache_flagon_${TAG}   cat9  9  1
