# Direction-2 SEAM-GEOMETRY 3-arm sweep, ALL same-session on subset_b4_sixteen (one campaign => mutually
# comparable, no cross-boot confound, and tail6b is run ONCE as the shared reference instead of twice
# across two separate 2-arm A/Bs). The three arms differ ONLY in FR13_TAIL_BRANCHES/DEPTHS + which tree
# nodes carry the seam branches; launcher/geom(BV=8)/tail-mode/draft-source/no-prewarm IDENTICAL => NO drift.
#   tail6e = 3 @ d6,d7 (widen)      27 nodes, n_pad=32, fill width 4
#   tail6c = 4 @ d6   (concentrate) 25 nodes, n_pad=32, fill width 5
#   tail6b = 2 @ d6,d7 (spread ref) 25 nodes, n_pad=32, fill width 3  (== the b7 arm-1 geometry, shared bar)
# All three n_pad=32 => same tps envelope; all monotone-lossless. Reads (bracketed deploy_speed
# accept_per_event): does MORE arctic seam width keep paying (tail6e > tail6c > tail6b) or plateau
# (all ~=)? Plateau => arctic siblings correlation-capped => MTP-d6-seam is the next (decorrelated) lever.
# CPU-hardened boot-clean (fill width=max(3,tb+1); pad-fallback covers all ranks). run_variant driver-sourced.
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
run_variant tail6e_${TAG}  tail6e  27  1
run_variant tail6c_${TAG}  tail6c  25  1
run_variant tail6b_${TAG}  tail6b  25  1
