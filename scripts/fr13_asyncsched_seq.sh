# ASYNC-SCHEDULING probe (cheap test of the align-escape lever): tail6 + --async-scheduling vs tail6
# baseline, SAME-session subset_b4_sixteen. --async-scheduling is a SETTABLE vLLM flag (arg_utils:1259)
# that runs step_with_batch_queue => overlaps host schedule/prepare/sample of step N+1 with the GPU forward
# of step N, hiding the ~250ms host stall the workflow measured. The redteam ASSUMED align-mode's per-step
# num_accepted.cpu() sync blocks it -- but never RAN it. This tests directly (like the batch-fill test).
# READ: (1) boots on GDN-hybrid? (2) per_request_decode_tps UP (host stall hidden)? (3) accept + generated
# output ~= baseline (LOSSLESS -- async's optimistic num_computed_tokens could change accepted tokens =>
# gate hard). If per_req UP + lossless => the last lever WORKS as a cheap flag (tree could reach ~5.9 > native
# 5.49). If crash/no-op/lossy => align-escape needs the deep rewrite (cost-gated). No drift (one flag, A/B).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--async-scheduling"
run_variant tail6_async_${TAG}  tail6  21  1
unset FR13_SERVE_BATCH_FLAGS
run_variant tail6_sbase_${TAG}   tail6  21  1
