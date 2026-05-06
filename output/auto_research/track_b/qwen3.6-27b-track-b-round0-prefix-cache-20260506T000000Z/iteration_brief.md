# Track B Iteration Brief

Round: `qwen3.6-27b-track-b-round0-prefix-cache-20260506T000000Z`
Objective: Ship prefix caching + LMCache CPU/disk-tier validation for cache-hit Codex turns.

## Hard Success Target

- Decode throughput must reach at least 15.00 tok/s.
- Baseline is 7.50 tok/s, so this is a 2.00x gate.

## Self Verify

- Measure prefix-cache hit rate, B-1 KL near zero, and latency reduction on turns 2-N.
- Do not modify quality fixtures, gate runners, Track B ledgers, or controller files from a candidate checkout.
