# Candidate 000 Analysis

- speed_thesis: Round 0 enables aggressive prefix reuse so cache-hit agent turns skip repeated long prefill.
- expected_affected_counter: vllm prefix-cache hit/query counters should rise and per-turn wall time should drop on turns 2-N.
- quality_risk: strong-equivalence path should have near-zero B-1 KL; any drift indicates cache/state or rejection-sampling correctness bugs.
- why_not_prior_failure: this is a Track B config/decoding path, not another Track A tile/schedule mutation that leaves bytes-per-token unchanged.
