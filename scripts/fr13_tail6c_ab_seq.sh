# Direction-2 CONCENTRATE-vs-SPREAD same-session A/B: tail6c (4 seam branches @ d6 ONLY) vs tail6b
# (2+2 branches @ d6,d7), back-to-back in ONE driver run on subset_b4_sixteen. BOTH are 25 nodes /
# n_pad=32 / identical tps; the ONLY difference is FR13_TAIL_BRANCHES/DEPTHS (4/1 vs 2/2) + which
# tree nodes carry the seam branches => NO config drift. Tests whether concentrating arctic width at
# THE seam (d6 leak 0.334 >> d7 0.152) beats spreading it across d6+d7. Both monotone-lossless vs tail6.
# Calibration: tail6c 5.46-5.58, tail6b 5.38-5.54 (recover r=.30/.45). Needs the fr13_merged_fill width
# fix (width=max(3,tail_branches+1)) so tail6c's ranks 3,4 aren't dropped. run_variant is driver-sourced.
# ONLY run after b7 confirms arctic seam branches help (tail6b > tail6); else concentrate is moot.
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
run_variant tail6c_${TAG}  tail6c  25  1
run_variant tail6b_${TAG}  tail6b  25  1
