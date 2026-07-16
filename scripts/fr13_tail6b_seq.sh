# Direction-2 d6-BRANCH A/B: tail6b (25-node, d6/d7 arctic-branched, depth-11) on the SAME subset_b4_sixteen.
# Identical config to tail6 (GPU_UTIL=0.72, n_pad=32/BV=8, no prewarm) + FR13_TAIL_BRANCHES=2/DEPTHS=2 ->
# NO config drift; only the tail branching differs. Targets the handoff conditional (0.666). Compare accept
# to the locked tail6 ~5.1-5.2 AND tps (same depth-11 => tps ~= tail6, unlike tailx10's depth-15 -21%).
# Monotone-lossless by construction. Assert TAIL[fired>0] engagement before trusting the number.
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
run_variant tail6b_${TAG} tail6b 25 1
