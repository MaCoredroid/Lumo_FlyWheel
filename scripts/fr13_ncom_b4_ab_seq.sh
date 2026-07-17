# CHEAP ATTACK: native-kernel committer replay vs custom Triton replay (B=4, the gate batch).
# ncom_b4 = FR13_COMMITTER_NATIVE=1 (native fused_sigmoid_gating for accepted-path advance, bit-exact 1.19e-7).
# ccom_b4 = custom _tree_gdn_replay_kernel (the 76.6ms baseline). Both CF2 => committer gpu_s A/B.
# GPU_UTIL=0.72 (tail6 n_pad=32 capture headroom). ncom FIRST (fail-fast: does native-replay boot past capture?).
export GPU_UTIL=0.72
run_variant ncom_b4 tail6_ncom 21 1
run_variant ccom_b4 tail6_cf2  21 1
