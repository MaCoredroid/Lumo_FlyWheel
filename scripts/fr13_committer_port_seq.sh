# COMMITTER-PORT A/B: tail6_gc (FR13_GPU_COMMITTER=1, device LCP kernel) vs tail6 (host LCP loop),
# same-session on subset_b4_sixteen, component timers ON. THE lever: native decomposition showed the tree
# committer = 94ms/step vs native 7ms = 87ms of FR13's OWN host path-LCP overhead (the single biggest
# reducible piece; low architectural risk -- our code, no APC-boundary exposure). The ONLY difference is
# FR13_GPU_COMMITTER=1 => NO config drift. tail6_gc runs FIRST so a boot-crash (NEVER live-run) surfaces
# early. Reads (bracketed deploy_speed):
#   - committer_gpu_ms_per_step: 94 -> ~10 == compute port works.
#   - accept_per_event + per_request_decode_tps: accept ~= tail6 (lossless-equiv) + tps UP.
#   - swe resolve / empty-patch rate: ~= tail6 => lossless-EQUIVALENT (rejection-sampler convergence, NOT
#     byte-exact; GB10 has no cross-boot byte gate). If accept/quality DIVERGE => the device LCP is not
#     lossless => localize before trusting the speed.
# Projected: committer 94->10 => tail6 per-step compute 286->202ms => per_req 4.89->~5.3 (near native 5.49)
# WHILE keeping tail6's +0.9 accept over native. run_variant is driver-sourced.
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
run_variant tail6_gc_${TAG}  tail6_gc  21  1
run_variant tail6_${TAG}     tail6     21  1
