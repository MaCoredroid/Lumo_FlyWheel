# COMPOSITION PROBE (the accept-why mechanism experiment): tail6, NO async,
# chunked-prefill co-scheduling DISABLED -> decode steps stay pure. Decides
# between the two surviving explanations for lad2's accept 5.47 vs rg1 4.89:
#   accept ~= 4.9  -> step-composition numerics exonerated; the async lift is
#                     the speed->agent trajectory-feedback loop (bankable
#                     config property, not artifact).
#   accept >> 4.9  -> composition/numerics carries; async's lift is mostly
#                     mediated by shapes, and pure scheduling is itself a lever.
# Cache-ON (post-rg-pair policy).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_COMMIT_FULL_GPU_TIMER=1
export FR13_SERVE_BATCH_FLAGS="--no-enable-chunked-prefill"
export FR13_COMMIT_FULL_GPU_TIMER_JSON=/workspace/output/fr13_resolve/tail6_pure_cfwd.json
run_variant tail6_pure_${TAG}  tail6  21  1
