# Physical32 tree-attention next-kernel audit

Status: **no new kernel is eligible for integration**.

This is a static, no-GPU audit of the fixed32 Tail23/Hydra27 tree-attention
path. It does not authorize a timing arm and does not alter the live B4
campaign worktree.

## Current production geometry

The target model has 16 tree-attention layers: model layers 3, 7, 11, ...,
63. Every fixed32 batch slot presents 32 root-inclusive BF16 query rows, 24
query heads, four KV heads, and head dimension 256. KV is paged in 1024-row
blocks. The call is noncausal, full-window, has no ALiBi, softcap, or appended
KV, and uses `num_splits=1`. The FP32 tree bias is 32 by 32 per batch slot.
Tail23 and Hydra27 change bias/mask values, not the physical launch shape.

- B1 uses the attested qrow16 specialization: block-M 16, block-N 64, one
  warp, two query CTAs per head. The grid is 48 CTAs per layer and 768 CTAs
  per fixed32 graph replay. There is no split-K or combine launch.
- B4 intentionally uses exact-safe stock FA2: block-M 64, block-N 64, four
  warps, one query CTA per batch/head. The grid is 96 CTAs per layer and 1,536
  CTAs per batched graph replay. Thirty-two of each tile's 64 query rows are
  outside the physical query extent. Qrow16 is disabled because it would
  double the CTA count and reread K/V for the two query halves.
- The tree-bias operation is fused into the attention kernel. The existing
  suffix-tile early-out skips its per-score walk on context-only K tiles.
  CUDA graph replay preserves 16 kernel launches; there is no per-event host
  dispatch or separate bias/combine kernel to remove.

## Measured B1 value and ceiling

The valid real SWE-Verified Hydra27 exact4 qrow16 arm improved full wall by
6.246844 ms/event and the GPU-component sum by 5.448386 ms/event versus its
exact-safe baseline. It still measured 232.779790 ms/event, 1.945376 times the
119.658015 ms mandatory-weight floor, and 95.173072 ms above the 1.15x cap.

The only real kernel-symbol attribution available is an older B1 trace. It
assigned 24.708601 ms/event to the whole FA2 tree-attention group. This is an
absolute, deliberately optimistic ceiling, not a current residual estimate.
Even deleting all 24.708601 ms from the current valid wall point would leave
208.071189 ms/event, 1.738882 times floor and 70.464471 ms above the cap.

B4 has no real symbol attribution or valid K64 timing point, so this audit does
not invent a B4 millisecond ceiling.

## Why nothing was integrated

The attested qrow16 binary is byte-exact on retained real paged operands, but
its exact production symbol reports 255 registers, a 120-byte stack frame, and
zero static local bytes. Its SASS contains 31 `LDL` and 24 `STL` instructions.
The stock block-M-64 exact symbol reports 255 registers, a 136-byte stack
frame, and zero static local bytes. Thus neither compiled specialization meets
the required zero-stack/zero-local-load-store admission rule.

The historical split-K four-way candidate is not eligible either: its real
exact4 campaign changed acceptance/task trajectory and was rejected. No built,
byte-attested qrow32 binary exists. Therefore this branch contains evidence
only and deliberately integrates no kernel or launcher change.

## Bounded next candidate

Keep the attested qrow16 route at B1. For B4, compile a private block-M-32,
block-N-64, two-warp specialization. It keeps one complete ordered K loop per
query row, one CTA per batch/head, the same 96-CTA grid, and the same number of
K/V passes as stock, while removing the 32 out-of-range query rows and halving
threads per CTA. This is query tiling, not split-K, and needs no combine
kernel. It must remain fixed32; Tail23 and Hydra27 may not select different
launch geometry.

Before integration, require all of the following:

1. ABI/ELF parity against the pinned exact-safe FA2 binary: zero differences
   in defined exports, undefined imports, and `DT_NEEDED`.
2. Exact target-symbol resources of `STACK=0`, `LOCAL=0`, zero ptxas spill
   loads/stores, and no SASS `LDL`/`STL`.
3. A same-EngineCore shadow gate on the canonical real SWE-Verified exact4 B4
   set for both Tail23 and Hydra27. Stock must serve requests; retained live
   paged operands must run stock and qrow32 back-to-back and compare every BF16
   output byte and FP32 LSE byte.
4. The gate must attest all 16 target layers, B4 FULL graph identity, query
   shape `[128,24,256]`, four sequence metadata rows, 1024-row paged KV,
   physical32 masks, and the observed suffix alignment classes. Any fallback,
   differing byte, missing layer, or resource drift rejects the candidate.
5. Only a source- and pass-bound candidate may enter the Tail23/Hydra27 exact4
   timing pair. No synthetic probe or geometry sweep is authorized.

See `audit.json` for the machine-readable evidence and arithmetic.
