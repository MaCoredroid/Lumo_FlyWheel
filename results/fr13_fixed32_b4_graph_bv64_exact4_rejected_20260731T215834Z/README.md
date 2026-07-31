# Rejected B4 graph-byte gate: 2026-07-31T21:58:34Z

This is a capacity rejection from the canonical real SWE-Verified exact4 B4
graph-byte campaign. It is not a completed task campaign, kernel byte PASS,
timing result, throughput result, or floor-acceptance result.

## Result

- Source: `c22d23c9c04993fc9891e640c6e7a15fc9dce9c6`
- Workload: canonical exact4, requested B4, concurrency 4
- Manual KV cache: `21,474,836,480` bytes
- KV cache created: `76,800` tokens
- PIECEWISE graphs completed: 8 of 8
- FULL graphs completed: 4 of 4
- FULL graph memory: `5.55 GiB`
- Process/runtime attestation: passed
- Canonical tasks started: 4 of 4
- Complete real decode events before interruption: 434
- Physical B4 replay observed: 0
- BV64 graph-byte records: 0

During real task traffic, vLLM remained at two running requests with two waiting
for capacity. KV utilization rose as high as 86.3%. The 20 GiB allocation
therefore could not admit an authenticated physical B4 decode replay, which the
gate requires before comparing the 48 captured layers.

The operator terminated the non-timing gate after this capacity condition was
stable from 22:07:20Z through 22:16:50Z. The interrupted arm failed its terminal
flush and did not complete task or ingress-ledger finalization.

## Correction

Commit `be2aa3032` raises only the exact B4 KV pin and its process contract from
20 GiB to 40 GiB. At the observed conversion of 3,840 cache tokens per GiB,
that increases capacity from 76,800 to 153,600 tokens. B1, max context length,
the exact4 task set, and concurrency are unchanged.

The correction passed 112 focused tests and 741 broad fixed32/kernel tests, with
7 broad skips.
