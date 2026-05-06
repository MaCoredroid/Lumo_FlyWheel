# Candidate 001 Analysis

- speed_thesis: Use vLLM's existing max_num_seqs=4 warm batching to amortize the dense FP8 weight stream across concurrent cache-hit Codex turns, targeting aggregate decode throughput >=15 tok/s without changing target weights.
- expected_affected_counter: aggregate generation tokens per wall second should rise while vllm prefix-cache hit/query counters remain high on the shared-prefix workload.
- quality_risk: batching is a scheduler-level strong-equivalence mutation; greedy per-request outputs should remain semantically equivalent, and any B-1 drift would indicate request isolation or cache-state corruption.
- why_not_prior_failure: prior CUTLASS failures changed schedule/tile/stage internals while leaving B-weight bytes per single stream unchanged; this mutation changes serving shape so one weight stream serves multiple active sequences.
