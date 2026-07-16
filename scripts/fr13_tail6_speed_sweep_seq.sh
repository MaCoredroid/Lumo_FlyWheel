# Direction-2 SPEED-GATE + seam-geometry sweep, ALL same-session on subset_b4_sixteen (one campaign =>
# mutually comparable, no cross-boot/cross-subset confound). Answers BOTH axes the user cares about:
#   (1) SPEED GATE: does ANY tree geometry beat native MTP-5 on per-stream/kernel tps, or is the tree
#       only an accept play? (b7 cross-run hinted tree ~TIES native: +35% accept but +43% slower forward.)
#   (2) ACCEPT: which seam geometry maximizes accept (spread vs concentrate vs widen).
#
# native + tail6b run FIRST (the speed-gate pair) so an interrupted campaign still yields the verdict.
# Cache regime MATCHED: b7's tree arms run FR13_ENABLE_APC=0 (cache-off) and flash_ns5_nocache is
# cache-off too => no cache confound. Decode s_per_fwd_gpu is cache-independent anyway. All tree arms
# n_pad=32 (same tps envelope); native uses FLASH_ATTN 5-tok verify (the honest incumbent forward).
#   nativemtp5  = flash_ns5_nocache (forked launcher, FLASH_ATTN, num_spec=5, NO tree)  E5  -- the BAR
#   tail6b      = 2 @ d6,d7 (spread, == b7 arm1)   25 nodes  E25
#   tail6e      = 3 @ d6,d7 (widen)                27 nodes  E27
#   tail6c      = 4 @ d6    (concentrate)          25 nodes  E25
# Reads (bracketed deploy_speed): accept_per_event + per_request_decode_tps + derived_tps_gpu +
# s_per_fwd_gpu, MATCHED prefill_frac/effective_concurrency. NO config drift (only the tree/branch flags
# vary across tree arms; native is the incumbent no-tree bar). CPU-hardened boot-clean. run_variant driver-sourced.
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
run_variant nativemtp5_${TAG}  flash_ns5_nocache  5   1
run_variant tail6b_${TAG}      tail6b             25  1
run_variant tail6e_${TAG}      tail6e             27  1
run_variant tail6c_${TAG}      tail6c             25  1
