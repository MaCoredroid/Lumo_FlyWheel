# PHASE-3 native-committer A/B: FR13_COMMITTER_NATIVE=1 (native fused committed-path replay via
# fused_sigmoid_gating, bit-exact to no-spec) vs =0 (custom _tree_gdn_replay_kernel), SAME campaign,
# committer-timed each. GATE: accept-preserving (native is bit-exact-to-no-spec, must stay ~4.32) +
# committer CFWD REDUCED (native < custom). cn1 cross-run showed native 90.06 vs phase-2 custom 98.9
# (~8.8ms); this confirms it SAME-campaign (removes the cross-boot confound). subset_b4_four = quick clean
# delta; promote to subset_b4_sixteen for the bake gate if it holds. run_variant driver-sourced.
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_COMMITTER_NATIVE=1
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_commnative_ab/native_cfwd.json
run_variant tail6_native  tail6  21  1
export FR13_COMMITTER_NATIVE=0
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_commnative_ab/custom_cfwd.json
run_variant tail6_custom  tail6  21  1
