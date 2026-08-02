# Fixed32 B1-to-B4 launch-scaling audit

Status: **static source audit plus CPU byte gate; default OFF; GPU gate and
real SWE-Verified exact4 timing pending**.

No GPU, Docker, synthetic serving probe, timing, TPS, or acceptance command
was run for this artifact. Performance counts below are source-level operator
or launch-site counts, not measured latency.

Source base: `d42f4c7ba12a1e51f4e8fc8dc6846cd9ece0480d` on branch
`agent/fixed32-b4-streamk-preflight-09f`.

## Result

Tree attention is not the B-scaled launch-count regression. The fixed32 path
has 16 tree-attention layer call sites at both B1 and B4. Each call packs
`32 * B` query rows, so arithmetic and activation/KV traffic grow with real
requests, but the Python call count does not.

The dominant avoidable B scaling is in the 48 GDN layers:

1. The stock per-request GDN scan has two launches/request/layer: 96/event at
   B1 and 384/event at B4. The qualified fixed32 batched BV8 source route can
   hold this at two launches/layer, or 96/event, while scaling its grid from
   12 to 48 path programs. The current CUTLASS B4 live-gate and timing runners
   intentionally force that production selector off, so CUTLASS-isolation
   results still include the 384-launch route.
2. Before every scan, causal-conv emulation remains a per-request loop. Its
   source preparation alone issues one prior-window gather, one source cat,
   and one persistent-staging copy per request per layer. That is 144
   launch-capable CUDA-op sites/event at B1 and 576 at B4. Window selection,
   taps, activation, and output publication also remain inside this loop and
   are the next batching target.

The new `FR13_FIXED32_CONV_SOURCE_BATCH` candidate removes the pure
data-movement portion with the lowest numerical risk. Per layer it performs
one B-dimensional prior gather and one `torch.cat(..., out=staging)` before
the request loop. It changes the source-preparation count to 96 sites/event
for every B1-B4 and removes the staging copy entirely: 48 sites removed at
B1 and 480 at B4. Bytes and row work still scale with B.

## Phase audit

| Phase | B1 -> B4 source behavior | Avoidable launch scaling |
| --- | --- | --- |
| Target/drafter dense projections | Fixed operator call sequence; GEMM M/rows scale with B | No obvious B launch loop |
| Tree attention, 16 layers | One unified-attention call/layer; 32 -> 128 packed rows | No Python B launch loop |
| GDN causal-conv source prep, 48 layers | Incumbent 3 -> 12 CUDA-op sites/layer; candidate 2 -> 2 | Yes; this patch removes it |
| Remaining GDN causal-conv taps | Window gather, ordered taps, SiLU, and output copy remain per request | Yes; next kernel target |
| GDN scan, 48 layers | Active CUTLASS-isolation route 2 -> 8 launches/layer | Yes; qualified BV8 batch route is 2 -> 2 |
| GDN cross-level handoff | Five compact state rows/request; 12 -> 48 scan programs/layer | Data/program work only after batched scan |
| TAW sampling | Fixed 12-level tensor-call sequence; B-sized full-vocab rows and 12 programs/request | Calls are batched; work scales with B |
| Publish/path/request-key packs | 2 publish, 2 path-pack, and 6 request-key launches/event | Launch counts fixed; slots scale with B |
| KV remap and conv pregather | Fixed prepare/apply/stage call structure; rows/programs scale with B | No Python B launch loop |
| Committer | One graph replay and one direct conv-commit launch/event; native recurrence has 48 internal calls | No B launch scaling; default-off layer-batch candidate can reduce 48 -> 1 |

The cross-level GDN handoff is not removable work: level 1 depends on five
state rows per request produced by level 0. The batched scan keeps that
handoff in compact `[B, 5]` storage and removes request-wise launches without
pretending its `5 * B` data dependence is free.

## Candidate safety

- The flag defaults to `0` at patch time and in the fixed32 floor sequence.
- It is accepted only for fixed32 B1-B4 with exactly 32 physical rows.
- It requires the existing preseeded batched-writeback staging route.
- The legacy request loop remains the complete OFF arm.
- The operation is copy-only: request order, prior column order, zero-row
  bytes, dtype, and device are unchanged.
- CPU tests compare integer views against the incumbent B1-B4 loop in bf16
  and fp32 and verify direct staging alias plus untouched-capacity bytes.

Before enabling it for production, run a same-source B1 GPU CUDA-graph byte
A/B on a real SWE-Verified task, then the standing exact4 set at B4. Any
timing must use the standing real SWE-Verified task rule. No speed or
hardware-floor claim is made by this source artifact.
