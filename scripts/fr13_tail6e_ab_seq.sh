# Direction-2 seam-WIDEN same-session A/B: tail6e (3 branches @ d6,d7) vs tail6b (2 @ d6,d7),
# back-to-back in ONE driver run on subset_b4_sixteen. Same launcher/geom/tail-mode/draft-source/
# no-prewarm; the ONLY difference is FR13_TAIL_BRANCHES (3 vs 2) + 2 extra branch nodes (27 vs 25,
# both n_pad=32 => same tps) => NO config drift. Tests whether MORE arctic width at the seam keeps
# paying past 2-wide (b7 interim ~5.49 implies ~35% recovery => rank-3 may still add mass). Both
# monotone-lossless. Needs fill width=max(3,tail_branches+1)=4. run_variant is driver-sourced.
# Run when b7 confirms branches help (tail6b > tail6); tail6b is the same-session reference arm here.
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
run_variant tail6e_${TAG}  tail6e  27  1
run_variant tail6b_${TAG}  tail6b  25  1
