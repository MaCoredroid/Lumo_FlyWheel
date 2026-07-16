# AGGREGATE-THROUGHPUT probe: tail6 + --enable-chunked-prefill --max-num-batched-tokens 8192 (batch-fill)
# vs tail6 baseline (vLLM defaults), SAME-session on subset_b4_sixteen. Only FR13_SERVE_BATCH_FLAGS differs
# => no drift. Tests whether the batch under-fill (effective_concurrency ~2.0 at B=4) is BUDGET-STARVED
# (raising max-num-batched lets more streams' prefill+decode co-reside => aggregate_decode_tps UP) or
# AGENTIC-IDLE (streams paused between tool calls => no change). aggregate tps = the multi-user serving
# metric (the per-stream tps gate already showed native > tree; this is the orthogonal throughput axis).
export GPU_UTIL=0.72
unset FR13_PREWARM_TRIE
export FR13_SERVE_BATCH_FLAGS="--enable-chunked-prefill --max-num-batched-tokens 8192"
run_variant tail6_bf_${TAG}    tail6  21  1
unset FR13_SERVE_BATCH_FLAGS
run_variant tail6_base_${TAG}  tail6  21  1
