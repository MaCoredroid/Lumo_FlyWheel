Measurement caveat: this candidate targets aggregate warm decode throughput by increasing batching pressure. If the gate reports per-request latency separately, expect possible tail-latency regression even if aggregate tokens/sec improves.

No live benchmark was run by this worker; controller gates should measure the real vLLM workload window.
