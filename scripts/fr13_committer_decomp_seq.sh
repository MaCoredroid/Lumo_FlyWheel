# COMMITTER-SPAN DECOMPOSITION: tail6_gc_sk (GPU_COMMITTER + SYNCKILL, DtoH/sync deferred to side stream)
# vs tail6_gc (GPU_COMMITTER, sync inline). SAME-session, only FR13_COMMITTER_SYNCKILL differs (no drift).
# Settles the last uncertainty in the 94ms committer span WITHOUT a 3rd premature no-go:
#   committer_gpu_ms(sk) << committer_gpu_ms(gc) => the DtoH/sync/materialise is the reducible bottleneck
#     (real per-stream lever -- the committer's 94ms is host-blocking, not GPU-inherent).
#   committer_gpu_ms(sk) ~= committer_gpu_ms(gc) ~94ms => the on-GPU work (GDN replay 48-layer + LCP
#     kernel) is inherent => the tree committer cannot approach native's 7ms => native genuinely wins,
#     and the tree's value is accept/losslessness not throughput on GB10.
# Lossless-gate both (accept ~= tail6 4.317). tail6_gc_sk FIRST (never-live-run => surface boot crash early).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
run_variant tail6_gc_sk_${TAG}  tail6_gc_sk  21  1
run_variant tail6_gc_${TAG}     tail6_gc     21  1
