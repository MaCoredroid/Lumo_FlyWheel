# Fixed32 CFWD guarded scan bounds

Status: **default off; source/static verified; codegen, real byte gates, and
timing pending**.

This reduced bundle binds the source-only change to commit
`e69243aeb6888599930274f8885c1b8efc5112d9` on branch
`agent/fixed32-cfwd-physical32-metadata-20260802`.

## Change

The layer-batched committer already copies live accepted-path metadata, then
enqueues a scalar `torch._assert_async` before graph replay. That guard requires
accepted length in `[0,11]` and every active physical node in `[0,31]`.

The hot scan previously repeated two min/max clamps for accepted length in
every program and two min/max clamps for every active root/path step. On the
guarded domain each clamp is the identity. The candidate now consumes
`T = accepted + 1` and `node = path_node` directly and removes the unused
`RING_N` constexpr.

There is no new launch, buffer, load, store, or recurrent step. Root remains
step zero, accepted nodes remain in their original order, and the dynamic loop
still executes exactly `accepted + 1` steps. The ordered floating-point math
slice is byte-identical in source before and after the change, with SHA-256
`d16ad65fe4affb85a85051bf8dc7530c17a34dd85826c05d6bd8adec67b1ce22`.

Cross-CTA metadata packing and grouping were rejected: they require another
per-event preparation launch or reduce layer/value parallelism. Neither is a
defensible source-only trade without real timing.

## Static work model

This is a logical source-operation count, not a latency or SASS claim. With
48 layers, 48 value heads, and two value tiles, there are 4,608 programs per
request. For a request accepting `d` drafts, the removed integer bound work is
`4608 * (2 + 2 * (d + 1))` min/max operations. At the validated maximum
`d=11`, that is 119,808 operations for B1 and 479,232 when all four B4 requests
have the same maximum depth.

The recurrent critical path, launch count, program count, HBM recurrence
traffic, and arithmetic order are unchanged. No speedup is claimed until the
pinned image compiles the kernel and real SWE-Verified traffic measures it.

## Evidence boundary

Host-only syntax and focused static suites passed. No Triton/CUDA codegen,
SASS/resource inspection, Docker/GPU execution, real SWE-Verified byte gate,
B1/B4 timing campaign, hardware-floor measurement, U95 acceptance test, or
synthetic probe was run for this bundle.

This bundle contains no prompts, responses, traces, raw logs, task IDs,
container identities, process identities, credentials, or secrets.
