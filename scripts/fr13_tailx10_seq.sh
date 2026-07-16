# Direction-2 tail-DEPTH A/B: tailx10 (25-node, depth-15 spine tail) on the SAME subset_b4_sixteen as the
# tail6 speed gate. Identical config to tail6 (GPU_UTIL=0.72 for n_pad=32, no prewarm) -> NO config drift;
# only the tree depth differs (tail_len 6->10). Compare accept to the locked tail6 ~5.1-5.2. Tests whether
# the rising deep-tail conditional (d7-11 0.848->0.950) keeps paying past d11. Watch for graph-capture stall
# (depth-11 booted; depth-21 hung) -- if it stalls at capture, fall back to x=8 (depth 13).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
run_variant tailx10_${TAG} tailx10 25 1
