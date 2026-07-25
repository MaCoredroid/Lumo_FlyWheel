# FR13 beat-native headroom ledger (2026-07-25)

## The hardware frame (GB10, unified ~273GB/s)

Decode is weight-read bound: 27B fp8 weights ~= 27GB / 273GB/s ~= **98.6ms
per forward** — the measured floor (reference_decode_hbm_bound). A perfect
step costs ONE weight read regardless of co-resident requests; everything
above it is our overhead or extra reads.

**Where the tree step actually is (2i pair, per physical step):**

| span | ms | vs floor |
|---|---|---|
| sfwd (verify fwd, weight read INSIDE) | 219.9 | ~121ms non-weight (scan, attn rows, soup, norms) |
| dfwd (drafter) | 94.6 | measured 117/140 HOST-bound (S3 decomposition) |
| cfwd (committer) | 27.5 | floor ~4/event |
| **total step** | **~342** | **~3.4x off the 98.6 + drafter-weights floor** |

## Why the prize is beating native LARGELY, not matching

Tree already wins accept: comb ~4.6-4.7 vs native E5 ~3.2 basis. TPS =
committed/step / wall/step. At EQUAL wall/step, the tree beats native by
the accept ratio (~+40%). Native 50.99 x (4.6/3.2) ~= **~70 tok/s is the
accept-advantage ceiling** if the tree tax goes to ~0. Every ms of tax
removed converts the accept edge into wall TPS.

## Ranked headroom (design per bucket)

1. **R4: drafter CUDA-graph capture (~-50ms/step, THE largest single item)**
   dfwd ~93-95ms of which ~117/140 fraction is HOST-bound (S3 measured:
   launch glue, python, not compute). Design: capture the drafter forward
   (MTP head + merged suffix logic) as a CUDA graph per batch size, mirroring
   vLLM's decode capture; the merged drafter's host branches (fired/skip)
   become captured variants or device-side selects. Was task #50, blocked on
   R3 (verify trim) — R3 is effectively DONE by this queue. UNBLOCKED.

2. **Subtree-parallel scan (general-topology, ~-6 to -9ms/step + scaling)**
   The scan kernel runs ALL 22 nodes serially inside every program; the
   only parallel axes are (vh, v-tile). But a tree's independent subtrees
   after the shared prefix have NO data dependence: partition nodes at
   runtime from parent[] (general — NOT the parked shape-specialized
   chain+leaf) into prefix + K independent subtrees; phase 1 scans the
   prefix, phase 2 scans subtrees on an added program axis (each serial
   internally, concurrent across subtrees). tail6 critical path 22 ->
   ~5+11=16 now, and the lever GROWS with wider trees (accept>5 work).
   Byte risk: per-node math unchanged, order within each path unchanged =
   same FMA chains; gate = selfcheck + capture arm (the 2d pipeline).

3. **HC x PG compat fix (banked -9.6ms/step sitting idle)**
   Runtime slot-map: pass a device int8[N_SPAN] node->slot table (built at
   preseed); PARENT_GATHER's one-hot becomes offs_h == tl.load(slot_map +
   parent_i). One kernel edit + one gate cycle; returns HC to the stack.

4. **2g-named residuals (sampler 4.2 + norms 8.5 + soup tail, ~-10-15ms)**
   Torchprof arm (running) names the python/aten sites; build the named
   fusions only. Norms candidate design if naming confirms: fold the 22-row
   RMSNorm+quant chain into one fused kernel per layer group (row-count
   work is tiny; launch+glue dominates).

5. **Committer to floor (12.3 -> ~4/event, -8/event)**
   Post-wb residual = sampler slice + glue; same 2g naming feeds it.

6. **Effective-batch (events/step) — the multiplier nobody owns**
   "B=4" runs at effective ~1.3 because vLLM serializes the agentic queue.
   Every events/step point multiplies TPS at fixed wall/step. Config-space
   probes (scheduler knobs, chunked-prefill interleave, max_num_batched)
   were tried once (#39) — revisit AFTER tax removal, when wall/step is
   closer to the floor and sharing the weight read pays more.

7. **Attention rows (+8.2) — likely structural floor**
   M-tile already covers the 22 rows in one KV pass (GEMM delta ~0 proven);
   the residual is KV-length-bound softmax/probs work per row. No lossless
   design under the no-reshape constraint. Revisit only if 1-6 land and the
   gap is still positive.

## Sequencing
Bar confirm (current stack) runs FIRST — the honest checkpoint. Then
R4 -> subtree-scan -> hc-compat -> 2g builds, each through the standard
gate pipeline (offline byte gate -> capture-mode 4-task -> pair/bar).
