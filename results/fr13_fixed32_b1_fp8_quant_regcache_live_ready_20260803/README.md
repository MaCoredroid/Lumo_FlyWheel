# Fixed32 B1 FP8 quant regcache live-gate readiness

Status: **deployable SM121a runtime built; default off; real SWE-Verified byte
gate and timing pending**.

## What is ready

- `FR13_FIXED32_B1_FP8_QUANT_REGCACHE=byte_ab` serves the stock output while
  comparing every output and scale byte from every exact admitted
  BF16 `[32,5120]`, group-128 invocation.
- Diagnostic admission requires the authenticated real-task marker for
  `astropy__astropy-12907`, Hydra27 fixed32 physical32, K64/root1, B1, eager
  execution, and stock CUTLASS.
- `FR13_FIXED32_B1_FP8_QUANT_REGCACHE=1` is inaccessible until the real gate
  issues a binary-, source-, task-, and raw-record-bound PASS sidecar.
- Every install smoke-loads the complete stable-libtorch extension and checks
  both `per_token_group_fp8_quant` and `cutlass_scaled_mm` registrations before
  server startup.
- The timing runner accepts only the standing exact four-task SWE-Verified set,
  installs the same candidate ELF for both arms, and changes only selector
  `0` to `1`.

## Runtime binary

The deployable ELF is intentionally not committed because it is 110,685,496
bytes. It remains at:

`/home/mark/fr13_fp8_quant_regcache_live_build_20260803/runtime-v2/_C_stable_libtorch.fp8_quant_regcache.sm121a.abi3.so`

SHA-256:
`847599fc7e3250cd56963592d4786d5f32fe5a391da107b4a791198a7d59c110`.

The build used the pinned CUDA 13 image with `--network=none` and no GPU
mount. Seventeen authenticated full-extension objects were reused. The FP8
quant object and the CUTLASS blockwise object were rebuilt from authenticated
source, preventing an earlier CUTLASS experiment from entering the stock
baseline.

The first four-object prototype was rejected by a CPU-only load test for an
undefined stable-libtorch symbol. It was not admitted. The full binary loads
with `CUDA_VISIBLE_DEVICES` empty and registers both required operations.

## Static kernel result

For the exact candidate kernel, `cuobjdump` reports 26 registers, zero stack,
zero local memory, and a 1,024-byte static shared metadata record. The launch
uses zero dynamic shared memory. The original static audit showed no `BAR`,
`LDS`, or `STS` instructions in the candidate.

This removes 83,886,080 shared-memory bytes and 10,240 CTA barriers per known
128-call target forward, without changing HBM reads/writes or launch count.

## Qualification boundary

- Combined focused tests after merging current `main`: 100 passed.
- Bash syntax, Python byte compilation, `git diff --check`: passed.
- Full ELF CPU-only smoke load with CUDA unavailable: passed.
- GPU used by this build/validation: no.
- Real SWE-Verified tasks run for this candidate: zero.
- Raw-byte PASS sidecar: absent by design until the real B1 gate.
- B1 or B4 timing samples: zero.
- Hardware-floor claim: none.

The next operation is exactly one real K64/root1 B1 byte gate. Only a PASS may
unlock the exact-four full-step timing pair.
