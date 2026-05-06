# Track B Iteration Brief

Round: `qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z`
Objective: Ship prefix caching + LMCache CPU/disk-tier validation for cache-hit Codex turns.

## Hard Success Target

- Decode throughput must reach at least 37.50 tok/s.
- Baseline is 7.50 tok/s, so this is a 5.00x gate.
- Speed is measured on the real workload window: 5 completions, first completion cold/discarded, next 4 warm completions counted.

## Self Verify

- Measure prefix-cache hit rate, B-1 KL near zero, and latency reduction on turns 2-N.
- Do not modify quality fixtures, gate runners, Track B ledgers, or controller files from a candidate checkout.
