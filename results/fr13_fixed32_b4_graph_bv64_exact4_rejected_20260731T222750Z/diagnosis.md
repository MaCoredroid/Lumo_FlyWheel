# BV64 mismatch diagnosis

## Classification

High-confidence low-bit Triton codegen/reduction-layout drift, not an indexing
or semantic bug. The current comparison changes both the kernel structure
(per-request reference to batched candidate) and the width (BV8 to BV64), so it
does not isolate BV64 by itself.

## Byte evidence

- Graph replay baseline versus explicit BV8 reference: byte-identical.
- Reference snapshot restoration: byte-identical.
- K/V/A/B rings, flags, invocation counter, and untouched export tail:
  byte-identical.
- Compact FP32 state: 195,944 of 62,914,560 bytes differ (0.311444600423%).
- First state difference: byte offset 28,676, float element 7,169,
  `[export row 0, head 0, v 56, k 1]`, low byte `0x1c -> 0x1d`.
- Export row 0 is request 0, node 0 for export nodes `(0, 1, 4, 9, 14)`.
  This removes request-stride and parent-slot indexing from the likely cause.
- BF16 output: one of 1,572,864 bytes differs, byte offset 1,481,082,
  element 740,541, `[row 120, head 25, v 61]`, low byte `0xbd -> 0xbc`.
  The encodings are adjacent BF16 values.

The gate did not record the other bytes of the first FP32 state value, so an
exact FP32 ULP magnitude cannot be claimed.

## Source evidence

`_gdn_node_step` reduces K=128 at
`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:5620` and again at line
5623. The containing state has shape `[BLOCK_V, 128]`; changing BV8 to BV64 can
change Triton's thread layout and K-reduction tree.

The source explicitly warns at lines 5575-5585 that identical inlined source
does not guarantee codegen identity. Lines 7477-7487 document a prior alignment
change that reshaped every `tl.sum` reduction tree and caused measured 1-2 ULP
drift.

## Action

1. Reject BV64. Do not weaken exact-byte parity and do not time this candidate.
2. Gate the batched kernel at BV8 against the per-request BV8 reference on real
   exact4 B4.
3. If it passes, use batched BV8: it already emits two physical launches per
   layer independent of B.
4. Only if BV8 program count is measured as material, implement a wide CTA as
   repeated fixed-BV8 subtiles so each reduction retains a `[8, 128]` layout,
   then exact-byte gate it. BV16/BV32 cannot be presumed exact.
