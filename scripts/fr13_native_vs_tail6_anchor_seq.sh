# Direction-2 STRATEGIC ANCHOR: native MTP-5 vs the BEST tree (tail6 spine, 21-node) SAME-SESSION on
# subset_b4_sixteen. Resolves the cross-run confound in FR13_TREE_VS_NATIVE_VERDICT (native-wins rested
# on fr13_native_tail6_decomp, a DIFFERENT campaign). This is the premise underlying ALL of direction-2:
# if the BEST tree (tail6, derived_tps 5.06 in b7 — beats the branched tail6b 4.76) already loses to
# native on the SAME boot/subset, then raising accept via more/branched nodes (measured net-negative:
# needs >0.138 accept/node, arctic delivers 0.046) cannot rescue the tree — it would need to leap PAST
# native (+~1 accept), impossible by node-adding. If tail6 >= native, direction-2 is worth escalating.
#
# native FIRST so an interrupted campaign still yields the incumbent bar. Cache regime MATCHED: tail6
# (no APC flag == b7 cache-off) and flash_ns5_nocache (explicit nocache) — decode s_per_fwd_gpu is
# cache-independent anyway. Both forked launcher, FLASH_ATTN, GPU_UTIL=0.72. The ONLY difference is
# num_spec=5 no-tree (native incumbent) vs the 21-node tail6 tree => NO config drift. run_variant
# is driver-sourced. Reads (bracketed deploy_speed): accept_per_event, derived_tps_gpu, s_per_fwd_gpu,
# per_request_decode_tps, MATCHED prefill_frac/effective_concurrency.
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
run_variant native_${TAG}  flash_ns5_nocache  5   1
run_variant tail6_${TAG}   tail6              21  1
