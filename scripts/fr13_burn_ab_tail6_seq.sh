# FR13 DIRECTION-2 committer win A/B on TAIL6 (depth-11, ~5 accept anchor, R1 speed kind).
# CLEAN config: burn-OFF on the deployed PER-LAYER committer, cache-ON, NO graph (measured: batched/graph
# reach ~6ms but are SNAP_FIX-blocked with cache-on; per-layer+burn-off = ~15ms, cache-on-clean, no leaf-map
# fix needed). Burn is the dominant win (~45ms). GATE: burn-OFF LOSSLESS (resolve+garble+accept MATCH baseline,
# within-floor, temp 0.6, live SWE, B=4 cache-on) AND FASTER (s_per_fwd/TPS). arm 1 = burn-off first.

# arm 1: BURN-OFF (per-layer committer, cache-on) -- the deployable committer win
export FR13_BURN_REDUNDANCY_TEST=1
export FR13_COMMITTER_GRAPH=0
run_variant burnoff_${TAG}  tail6  21  1

# arm 2: deployed baseline (burn-ON)
export FR13_BURN_REDUNDANCY_TEST=0
export FR13_COMMITTER_GRAPH=0
run_variant baseline_${TAG}  tail6  21  1
