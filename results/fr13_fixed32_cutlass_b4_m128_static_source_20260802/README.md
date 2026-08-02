# Fixed32 B4 M128 static-scheduler source candidate

Status: source-only, default off. No GPU build, real-task byte gate, timing, or
hardware-floor result is claimed.

## Candidate

`persistent_b4_m128_static` composes the existing B4 `128x128x128`
cooperative projection collective with CUTLASS
`StaticPersistentTileScheduler100`. Its diagnostic selector is
`persistent_b4_m128_static_byte_ab`; unrecognized and unset selectors still
dispatch stock.

The candidate is gated to physical `M=128` and the five real Qwen projection
shapes. That is the B4 representation of four requests times 32 physical rows.
It does not inspect or change the Tail23/Hydra27 active mask, node count, draft
vocabulary, root setting, weights, activations, scales, or output buffers.
K64/root1 therefore remain external fixed campaign inputs.

The inherited collective remains:

- tile `128x128x128`, cluster `1x1x1`;
- `KernelTmaWarpSpecializedBlockwiseCooperativeSm120`;
- block scales `(1,128,128)` and normal A/B layout;
- the same `CollectiveMainloop` and `CollectiveEpilogue` types;
- one complete, ordered K reduction per output tile with no split-K, Stream-K,
  reduction workspace, or fixup.

Only complete-output-tile assignment changes. The incumbent SM120 scheduler is
dynamic persistent and obtains work through Blackwell CLC. The candidate maps
`blockIdx` to an output tile and advances by physical grid size using CUTLASS's
static-persistent scheduler. This can remove CLC scheduler queries while
preserving each output element's K iteration order. It does not reduce the
mandatory projection weight bytes.

For the five admitted `(M,N,K)` shapes, the unchanged M128/N128 geometry has
`272`, `40`, `40`, `128`, and `112` complete output tiles respectively. The
candidate changes who claims those tiles, not how many tiles or K iterations
are executed.

## Why this candidate

- `128x256` cooperative could not satisfy CUTLASS's two-mainloop-stage minimum.
- `64x256` ping-pong compiled with a 488-byte per-thread stack frame.
- Wide256 Stream-K and data-parallel candidates changed real B1 bytes and were
  rejected.
- Stream-K repartitions K and therefore does not meet this bounded
  reduction-order requirement.
- PDL is a separate launch-admission lever; it is not mixed into this source
  change.

The existing stock-tile static-scheduler branch established that this CUTLASS
class can be instantiated on the pinned source. This branch applies it to the
M128 cooperative collective instead of the stock B4 M64 ping-pong collective.
That composition is new; compile resources remain unknown until a pinned SM121
build is performed.

## Verification

- 42 focused Python patch/selector/contract tests passed.
- Ruff and Python bytecode compilation passed.
- The patch applied to the exact pinned vLLM source and was idempotent.
- The CUTLASS checkout and the two static-scheduler headers are commit/hash
  pinned.

The next valid work is a pinned SM121 build and resource audit, followed by an
authenticated real SWE-Verified exact4 B4 raw-byte gate for both Tail23 and
Hydra27. Timing remains forbidden until those gates pass. B1 stays on its
separately qualified projection path because this selector fails closed unless
`M == 128`.

