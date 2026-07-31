# Fixed32 GDN coefficient-hoist candidate

This experimental branch adds an opt-in fixed32 GDN source candidate behind
`FR13_FIXED32_GDN_COEFF_HOIST=1`. It is not deploy-ready and has not been
compiled or run on the GPU.

The production fixed32 path kernel uses `BV=8`. For every physical node, its
grid therefore recomputes q/k L2 normalization in all 48 combinations of three
value heads per q/k head and sixteen V tiles. Raw-gating decay and beta are
recomputed in sixteen V tiles per value head. Those values do not depend on V
tile or path state.

The candidate adds one precompute kernel per GDN layer. It computes fp32
scaled-q, normalized-k, decay, and beta once per node/head. The two path
kernels load those fp32 values and retain the existing recurrent state update,
node order, path handoff, output stores, and raw replay-ring bytes. This changes
the GDN launch count from 96 to 144 kernels per event, but removes 48x duplicate
q/k normalization and 16x duplicate gating transcendental work.

The current branch owns one scratch row per sequential fixed32 call and gates
the request slot to B1-B4 (`h0_batch_index` 0-3). Stream order makes reuse safe
for the current B4 request loop: precompute, level 0, and level 1 all finish
before the next request overwrites scratch. The separate compact B4 two-launch
kernel from `agent/fixed32-gdn-batch2launch` is not modified here; combining the
branches requires a batch-indexed scratch port and a new byte gate.

No capture-time allocation is added. The exact fixed32 schedule exports only
nodes `{0, 1, 4, 9, 14}`. Terminal node 31 is never exported or used as a path
parent, so its existing fp32 `state_export` row is reused as scratch. The
coefficient payload is 134,144 fp32 elements (536,576 bytes) in a row with
786,432 elements (3,145,728 bytes).

The measured attribution basis is a real SWE-Verified fixed32 Tail6 B1 trace:
GDN path kernels consumed 14.0195 ms/event, from 95.923 launches/event. The
candidate's expected saving is 2-5 ms/event, with 14.0195 ms as the impossible
hard upper bound. This is an engineering estimate, not a measurement. Separate
kernel codegen, the additional 48 launches, and fp32 scratch traffic can reduce
or reverse the gain. This branch does not claim a performance win.

`FR13_FIXED32_GDN_COEFF_SELFCHECK=1` runs the stock path and hoisted path on the
same tensors, then requires byte-equal outputs, replay rings, flags, and a
single counter increment. The checked-in CUDA test exercises that gate when a
CUDA+Triton environment is available. It was skipped on the CPU-only source
host.

Required next gates:

1. Production Triton/CUDA compile for SM121 and codegen inspection proving the
   candidate path no longer emits q/k reductions or gating transcendental work.
2. Same-process byte gates on captured production fixed32 inputs, including all
   48 layers and both Tail6 and Hydra23 physical masks.
3. CUDA graph capture/replay at B1 and B4, with scratch ownership and raw replay
   rings byte-checked.
4. Real SWE-Verified exact4 B1 screening, then exact4 B4 and exact16 acceptance.
   No synthetic performance probe is admissible.
