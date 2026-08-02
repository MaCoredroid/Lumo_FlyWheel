# Fixed32 CFWD event-independent gate precompute

Status: **default off; source/static verified; pinned-image GPU compile,
resource inspection, real byte qualification, and timing pending**.

## Change

The direct-ring 48-layer CFWD candidate previously evaluated
`-exp(A_log)` once in every value-tile program on every event. This revision
materializes the event-independent `-exp(A_log)` and `dt_bias` values into one
contiguous FP32 `[48, value_heads, 2]` tensor before graph capture. Every B1
through B4 graph state retains the same tensor, and the hot kernel loads the
two precomputed FP32 coefficients.

The precompute cache is process-global and bound to the source tensors'
device, pointers, shapes, strides, and dtypes. A second distinct operand pair
fails closed. The production observer requires exactly one process launch and
zero gate exponentials per event, preventing an older realization from
satisfying the candidate contract.

The intended lifecycle uses private contiguous detached model-weight stacks
that are created once and not mutated. In-place mutation through the same
pointers is not dynamically detected; no such mutation path was found in the
pinned patcher lifecycle.

## Static work model

These are logical instruction counts, not latency, DRAM, or speed
measurements. With the existing `BV=64` geometry:

| Occupancy | Hot programs/event | Gate exponentials removed/event |
| --- | ---: | ---: |
| B1 | 4,608 | 4,608 |
| B4 | 18,432 | 18,432 |

The one-time precompute performs 2,304 exponentials for the pinned 48-layer,
48-value-head model and writes an 18 KiB coefficient tensor. It does not
remove the two scalar gate-coefficient loads per hot program. The candidate
therefore targets duplicated transcendental work and hot-kernel instruction
pressure, not mandatory model-weight traffic.

No speedup is claimed. The kernel must compile in the pinned SM121 vLLM image,
pass resource and spill checks, prove one shared precompute across B1-B4, pass
the real SWE-Verified accepted-depth raw-byte gate, and then be compared at the
same real-task timing boundary.

## Static verification

- Focused CFWD lifecycle/committer suite: `84 passed`.
- Independent source review: no correctness blocker for the immutable
  one-model-per-worker lifecycle.
- Python byte compilation: pass.
- `git diff --check`: pass.
- No GPU command or synthetic performance probe was run.

This directory contains source-level aggregate metadata only. It contains no
prompts, responses, patches, traces, raw logs, process/container identities,
credentials, or timing samples.
