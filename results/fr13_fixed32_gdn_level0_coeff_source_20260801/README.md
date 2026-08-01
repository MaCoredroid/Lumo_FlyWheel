# Fixed32 two-launch GDN coefficient staging

Status: source candidate only. No GPU compile, byte gate, or timing was run on
this host. The route is default-off behind
`FR13_FIXED32_GDN_LEVEL0_COEFF=1` (or its worker-visible sidecar).

## Change

The incumbent fixed32 path route visits 32 physical nodes in two launches per
layer. With production `BV=8`, every node's q/k normalization is repeated in
48 programs (three value heads per q/k head times sixteen V tiles), while raw
gate softplus/exp/sigmoid work is repeated in sixteen V tiles per value head.

This candidate keeps the same 32 nodes, path decomposition `[1, 11]`, node
order, recurrent state update, replay-ring bytes, flags, and two launches. At
the end of level 0, its existing 768 programs stage one FP32 scaled-q,
normalized-k, decay, and beta value for every node/head. Level 1 loads those
coefficients and skips its repeated reductions and gate transcendentals. There
is no third precompute launch.

The staging payload is 134,144 FP32 elements (536,576 bytes) per request. It
reuses graph-stable rows in the existing 32-row state-export allocation:

- sequential B1-B4 route: row 31, after each request's level 0 and before its
  level 1;
- compact B2/B3/B4 route: rows 30-31, 29-31, or 28-31 respectively.

At B4, compact parent state uses rows 0-19, so coefficient rows 28-31 are
disjoint. Stream order bounds the scratch lifetime to the two launches of one
layer. No capture-time allocation is added.

## Work model

At B1 BV8, the incumbent executes q/k normalization and raw-gate folding in
24,576 node-program instances per layer. The candidate leaves the five level-0
nodes inline and stages one coefficient set for all 32 nodes:

- q/k normalization instances: 24,576 -> 4,352, down 82.29%;
- raw-gate fold instances: 24,576 -> 5,376, down 78.12%;
- GDN launches/event: 96 -> 96;
- recurrent physical node visits: 32 -> 32 per request/layer.

The measured attribution basis is the real SWE-Verified Tail6 B1 trace where
the two GDN path kernels cost 14.019520 ms/event at 95.923 launches/event. The
planning estimate is 1.5-4.0 ms/event saved. This is not a measurement; the
level-0 resource increase and FP32 scratch/L2 traffic can reduce or reverse the
gain. The 14.019520 ms/event attribution is the impossible upper bound.

Even the high estimate does not close the hardware-floor goal by itself. On
the nearest valid exact4 B1 accounting (232.779790 ms full-step wall), it would
only project 228.779790-231.279790 ms before the separate full-vocabulary head
delta. The required one-sided floor cap remains 113.39 ms/step.

## Required GPU gate

1. Compile both B1 sequential and B4 compact specializations on production
   SM121. Record registers, spills/local memory, occupancy, and codegen proving
   level 1 has no q/k reductions or gate transcendentals. Reject any spill or
   compile failure.
2. B1 byte gate on one authenticated real SWE-Verified task, all 48 layers,
   with the incumbent always served. Compare output, parent exports
   `{0,1,4,9,14}`, raw K/V/A/B replay rings, flags, and invocation counter.
3. B1 same-source exact4 stock/candidate timing on the canonical four-task set.
   Capture full wall, SFWD GPU, `_tree_gdn_path_kernel` group time, launch count,
   L2/DRAM bytes, and level-0/level-1 kernel resources.
4. B4 byte and timing gates on the canonical exact4 tasks
   `astropy__astropy-{12907,13033,13236,13398}`, with rows 28-31 explicitly
   checked as private scratch and rows 0-19 byte-compared as served exports.
5. Only after B1 and B4 show a real full-wall win, run the standing exact16
   acceptance/floor campaign. Synthetic probes and one-task timing are not
   acceptance evidence.
