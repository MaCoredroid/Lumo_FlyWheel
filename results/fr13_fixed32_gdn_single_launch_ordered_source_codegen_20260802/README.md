# Fixed32 K64/root1 GDN single-launch ordered-loop candidate

Verdict: **SOURCE_CODEGEN_READY_LIVE_UNQUALIFIED**.

This artifact records the current-main transplant of
`fixed32_gdn_single_launch_tree_v2` with its five-step outer root expansion
replaced by one ordered `tl.range` loop. The source is wired behind the new
default-off `FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE=1` selector for exact fixed32,
BV8, `FR13_DRAFT_VOCAB_K=65536`, and `FR13_DRAFT_VOCAB_ROOT=1` only.

The B1 specialization uses grid `[48, 16, 1]`; B4 folds request identity into
grid `[48, 16, 4]`. Both use one physical kernel launch per GDN layer and keep
all 32 node recurrences in one CTA program, eliminating the two-launch path's
five fp32 state-export writes and eleven parent-state reads per request/layer.
Every output and ring node still has exactly one writer.

## Offline SM121a codegen

Two fresh, no-GPU container builds returned identical compile and cubin hashes.

| Metric | B1 | B4 |
| --- | ---: | ---: |
| Physical launches/layer | 1 | 1 |
| CTAs/launch | 768 | 3,072 |
| SASS instructions | 1,592 | 1,592 |
| Primary text bytes | 25,472 | 25,472 |
| Registers/thread | 112 | 112 |
| Registers/CTA | 28,672 | 28,672 |
| Stack / local bytes | 0 / 0 | 0 / 0 |
| LDL / STL / calls | 0 / 0 / 0 | 0 / 0 / 0 |
| Global scratch / TMEM bytes | 0 / 0 | 0 / 0 |

The prior static-root expansion audit reported 7,232 instructions and 97
registers/thread. The ordered loop therefore retains the prior 77.99% static
instruction-footprint reduction, at the cost of 15 more registers/thread.
This comparison is codegen-only: the loop body executes five times, and these
instruction counts are not a latency or throughput model.

## Validation

- Focused selector/topology/kernel/dispatch suite: `6 passed`.
- Existing fixed32 GDN and final-preseed regression suite: `148 passed`.
- `py_compile` and `git diff --check`: pass.
- Container import: selector is off by default; the exact K64/root1 arm resolves
  B1 per-request and B4 folded dispatch as intended.

One separate historical K64 route suite had `64 passed, 1 failed` because its
checked-in result directory lacks `prepared_campaign.sh`. That missing fixture
predates and is unrelated to this change; it is not represented as a candidate
failure.

## Boundary

No GPU kernel, real SWE-Verified task, synthetic probe, Docker service, byte
parity gate, CUDA graph replay, timing run, or floor-acceptance campaign ran.
The candidate is not production-qualified. A real-task B1/B4 byte gate and
exact4/16-task timing remain mandatory before any performance or acceptance
claim.
