# Fixed32 CFWD full-value-tile candidate

Status: **default off; source/static verified; pinned-image GPU compile,
resource inspection, real byte qualification, and timing pending**.

## Change

The direct-ring CFWD recurrence now assigns the complete pinned `V=128` value
state to one eight-warp Triton program per layer, request, and value head. The
prior BV64 candidate assigned two programs to the two 64-row halves. Both
halves consumed the same 128-element K vector and repeated its L2
normalization, accepted-path lookup, and scalar recurrence metadata loads.

The full tile retains the ordered root-plus-accepted recurrence, FP32 state,
direct ring inputs, final-state-only store, and one captured all-layer graph.
State and V element traffic are tiled differently but are not removed. The
observer fails closed unless the runtime contract reports a full 128-row tile,
one program per layer/request/value-head, zero duplicate value-tile K loads,
eight warps, and 64 FP32 state elements per thread before compiler effects.

## Static work model

These counts are logical source work, not DRAM or latency measurements. For 48
layers, 48 value heads, `K=V=128`, and eight warps per program:

| Occupancy | BV64 programs/event | BV128 programs/event | BV64 warps | BV128 warps |
| --- | ---: | ---: | ---: | ---: |
| B1 | 4,608 | 2,304 | 36,864 | 18,432 |
| B4 | 18,432 | 9,216 | 147,456 | 73,728 |

Per root-inclusive recurrence step, B1 removes 2,304 duplicate K-vector loads,
2,304 128-term K-norm reductions, 2,304 reciprocal square roots, and 589,824
FP32 K-norm multiplications. B4 removes four times those counts. With BF16 K
rings, the logical K-vector load reduction is 0.5625 MiB per step at B1 and
2.25 MiB per step at B4. At the prior Hydra27 mean of 5.753885 root-inclusive
steps, that is 3.236560 MiB/event at B1 and 12.946241 MiB/event at B4; at the
reachable maximum of 12 steps, it is 6.75 MiB/event and 27 MiB/event.

This trade increases the pre-compiler state allocation from 32 to 64 FP32
elements per thread. No speedup is claimed. The candidate is ineligible until
the pinned GPU compile establishes registers, stack, local memory, and spills;
the captured candidate then passes the authenticated real SWE-Verified raw-byte
coverage for every reachable accepted length 0 through 11; only then may B1 and
B4 real-task timing run under the standing task-set rules.

## Static verification

- Focused CFWD source/lifecycle suite: `85 passed`.
- Python byte compilation: pass.
- `git diff --check`: pass.
- No GPU command, synthetic performance probe, or task timing was run.

This directory contains source-level aggregate metadata only. It contains no
prompts, responses, patches, traces, raw logs, process or container identities,
credentials, or timing samples.
