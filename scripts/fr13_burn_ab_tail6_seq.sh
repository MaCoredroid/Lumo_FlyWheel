# FR13 DIRECTION-2 committer-optimization A/B on TAIL6 (depth-11, the ~5 accept anchor, R1 speed kind).
# Measures the full committer optimization (burn-OFF + graph-captured fused_sigmoid) vs the deployed baseline.
# GATE: optimized arm must be LOSSLESS (resolve + garble + accept MATCH baseline, within-floor, temp 0.6,
#   live SWE, B=4 cache-on) AND FASTER (s_per_fwd / TPS). arm 1 (graph) runs first to fail-fast: watch for
#   '[FR13_COMMITTER_GRAPH ENGAGED]', 0 fatal, no garble. Burn redundancy already structurally proven (red-team).

# arm 1: BURN-OFF + GRAPH committer (the direction-2 optimized committer)
export FR13_BURN_REDUNDANCY_TEST=1
export FR13_COMMITTER_GRAPH=1
run_variant optgraph_${TAG}  tail6  21  1

# arm 2: deployed baseline (burn-ON, no graph)
export FR13_BURN_REDUNDANCY_TEST=0
export FR13_COMMITTER_GRAPH=0
run_variant baseline_${TAG}  tail6  21  1
