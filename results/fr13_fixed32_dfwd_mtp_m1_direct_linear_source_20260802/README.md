# FR13 fixed32 DFWD MTP M1 direct-linear scheduler checkpoint

Status: **source-only, default off, not compiled, not byte-qualified, and not
timing eligible**.

This reduced artifact records bounded kernel source work completed while the
canonical real B4 campaign owned the host. It contains no task prompts,
responses, patches, traces, raw logs, process or container IDs, environment
dumps, or secrets.

## Exact geometry

The vLLM SM120 dispatch changes the original GEMM shape `(M,N,K,L)` to
`(N,M,K,L)` when `swap_ab=true`. The real MTP candidate is restricted to
original `M=1`, tile `128x32x128`, cluster `1x1x1`, and `L=1`. Its tiled MNKL
geometry is therefore:

| Original `(N,K)` | Post-swap tiled `(M,N,K,L)` |
| --- | --- |
| `(34816,5120)` | `(272,1,40,1)` |
| `(5120,17408)` | `(40,1,136,1)` |
| `(5120,6144)` | `(40,1,48,1)` |
| `(16384,5120)` | `(128,1,40,1)` |
| `(14336,5120)` | `(112,1,40,1)` |

Thus tiled N and L are exactly one for all five real projection shapes. Every
N and K is also an exact multiple of its tile dimension.

## Generic mapping

Pinned CUTLASS `StaticPersistentTileScheduler100` first divides each linear
work index into batch and MN remainder, divides by the cluster-minor shape,
then applies cluster, raster, and swizzle divmods. For this geometry the
default heuristic selects `AlongN`, swizzle is zero, cluster shape is one, and
the generic result reduces algebraically to:

```text
tile_m = blockIdx.y + wave * gridDim.y
tile_n = 0
tile_l = 0
```

The previous static scheduler still carries runtime divisors, so source-level
constant folding is not guaranteed. The new
`Fr13Fixed32MtpM1DirectLinearScheduler` preserves the same initial linear
index and physical-grid stride, but maps a valid linear index directly to
`{linear_idx, 0, 0}`. It inherits the existing static scheduler's params, grid
shape, workspace, full-K, epilogue, and coordinate interfaces.

The direct path is fail-closed unless tiled N and L are one, cluster M and N
are one, swizzle is zero, and the total problem blocks equal tiled M. The
existing dispatch additionally restricts the candidate to exact original
`M=1` and the five listed real `(N,K)` pairs. Stock remains the default.

## Verification

- 25 focused patch tests passed.
- Python compile and `git diff --check` passed.
- The exact pinned vLLM source applies and is idempotent.
- The exact pinned CUTLASS commit and both static-scheduler header digests pass.
- Patched dispatch SHA256 is
  `0a95bafecd5c07324202e58e9ba529b98fc1bb6d46f39edd5fa830e034158f0d`.
- Ruff was unavailable on the protected host.

No nvcc, Triton compilation, C++ build, Docker, GPU, synthetic workload, or
real task ran. There is no candidate SASS/resource, byte equality, latency,
TPS, quality, acceptance, B4, or hardware-floor claim.

## Required closure

1. After host teardown, compile the pinned `sm_121a` candidate and reject any
   stack, local memory, spill, call, unexpected divmod, or math-order drift.
2. Run the authenticated real SWE-Verified B1 K64/root1 phase-bound raw-byte
   gate over all five shapes and all 20 MTP projection launches per event;
   stock must be served throughout.
3. Only after byte equality may matched real exact4 Tail23 and Hydra27 timing
   test the direct scheduler. Exact16 and one-sided U95 remain the formal floor
   gate.
