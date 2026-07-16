# Direction-2 d6-BRANCH same-session A/B: tail6b (25-node, d6/d7 arctic-branched) vs tail6 (21-node
# spine tail), back-to-back in ONE driver run on subset_b4_sixteen. The ONLY difference is
# FR13_TAIL_BRANCHES=2 / FR13_TAIL_BRANCH_DEPTHS=2 (+ the 4 branch nodes in the tree, absolute depth
# unchanged at 11). GPU_UTIL / geom(BV=8) / tail-mode / draft-source(merged) / no-prewarm are IDENTICAL
# on BOTH arms => NO config drift. tail6b runs FIRST so the [FR13_MERGED ENGAGED] br_real needle lands
# early: br_real>0 proves the branch columns carry REAL arctic runner-up tokens (not pad-fallback).
# By the monotone-lossless committer, tail6b >= tail6 on the SAME spans; delta = the d6/d7 handoff lift
# (targets the measured weakest link: MTP->arctic handoff conditional 0.666). tps ~= tail6 (same depth-11,
# unlike tailx10's depth-15 -21%). run_variant is in scope (driver-sourced).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
run_variant tail6b_${TAG}  tail6b  25  1
run_variant tail6_${TAG}   tail6   21  1
