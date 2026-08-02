# Fixed32 B4 M128 integrated source checkpoint

Status: pinned, build-ready source; default off and not acceptance-valid.

Branch: `agent/fixed32-sfwd-b4-m128-integrated-source-20260802`

Integrated source tip before this artifact:
`0b18380bdf964e2c0c2b483295b84902562eec3c`

## Integrated changes

The branch contains all three M128 source optimizations in one generated vLLM
dispatch:

1. The static scheduler maps each persistent worker index directly to
   `(M=0, N=linear_idx, L=0)`, removing generic batch and two-dimensional tile
   divmods from each work fetch.
2. The scheduler exposes CTA coordinates as compile-time
   `(M=0, N=work.N_idx, K=_, L=0)` for the initial tile and all four
   participant next-work paths.
3. The sourceless epilogue uses an empty host argument and a fixed runtime
   `Params{1.0f}` scalar with `Sm90Compute<cutlass::multiplies>`, removing
   generic alpha/beta pointer and stride state while retaining FP32 multiply
   then round-to-nearest output conversion.

The candidate is restricted to M=128, L=1, a 128x128x128 tile, a 1x1x1
cluster, and the five allowlisted real projection shapes. Their N tile counts
are 272, 40, 40, 128, and 112. The stock route remains the default.

## Interaction review

The SM100 kernel constructs the scheduler with the exact
`(CLCResponse*, Params const&, dim3)` overload implemented here. Both
next-work overloads and the one-argument CTA-coordinate conversion used by all
participants are present. CUTLASS's pinned persistent grid builder returns a
physical grid with `z=1`; combined with M/L tile counts of one, the planar
worker index and grid stride enumerate every N tile exactly once.

The custom epilogue is passed through CUTLASS's supported direct-callback
builder path. Its empty `Arguments` is converted unconditionally to one FP32
device scalar. `FusionCallbacksTraits` intentionally reports a void compute
type for a direct visitor, and the collective falls back to the float
accumulator type. The candidate uses `Base::CollectiveMainloop`, so stage count,
full K ownership, and K iteration order stay unchanged. Its epilogue builder
keeps the stock tile, C/D types, layouts, alignments, schedule, and TMA store
path.

No further source edit was needed to compose the layers.

## Compile gates

Compilation must remain gated on these checks:

- NVCC must instantiate the derived scheduler and custom visitor against the
  pinned CUTLASS headers. This checkpoint intentionally performed no C++
  compile.
- The derived object retains the base scheduler's private linear/grid state and
  adds three fields of its own. Optimizers may discard unused base state, but a
  post-build resource audit must reject register growth, stack/local memory, or
  spills. The existing pre-linear static kernel used 168 registers with zero
  stack/local/spills.
- SASS must retain the complete mandatory QMMA/FFMA pipeline, 64 output FP32
  multiplies, and 32 BF16 packs, with zero device calls. It must separately
  confirm whether divmod, coordinate, and scalar-resolution instructions were
  removed; no emitted delta is claimed from source alone.
- A real SWE-Verified exact4 raw-byte comparison must pass before any timing or
  production use. Timing must then use the standing real-task campaign rules.

## Verification

- 54 focused source, selector, qualification, and binary-helper unit tests
  passed.
- Python bytecode compilation and `git diff --check` passed.
- The patch applied to vLLM `fe9c3d6` with CUTLASS `da5e086`, and a second
  application was idempotent.
- Patch source SHA256:
  `5f6c8da9ce1b873c0917b43b736bca6d29c7bc3b856d80eecf7b643c14a0ae1c`.
- Generated dispatch SHA256:
  `319bda31b05222e17eedefa65dbe328e87a9afd335301a75cf4bbb150911aedc`.

No NVCC/C++ compile, link, SASS/resource audit, GPU execution, container run,
synthetic probe, or real-task run was performed.
