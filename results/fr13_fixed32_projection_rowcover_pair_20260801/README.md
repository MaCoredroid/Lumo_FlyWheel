# FR13 fixed32 projection row-cover pair

Status: **source and SM121 static build pass; not gate/install runnable, not
timing eligible**.

This artifact composes the strongest exact-preserving projection candidates
already available for the fixed physical-32 K64 workload:

- B1, `M=32`: keep the stock swapped `128x32x128` cooperative kernel and full-K
  reduction, but replace Blackwell dynamic CLC tile allocation with
  `StaticPersistentTileScheduler100`.
- B4, `M=128`: keep the stock math and use the existing cooperative
  `128x128x128` kernel instead of the stock `64x128x128` ping-pong row tile.
  This halves the M-axis output-tile count and lets one CTA cover all 128 rows.

These are two batch-specific implementations of the row-cover objective, not
one common scheduler. The stock-tile static scheduler compiled with an 8-byte
stack frame at B4, so this composition deliberately dispatches static
persistence only at exact `M=32`; every other static-selector row falls back to
stock. The B4 selector remains exact `M=128` and falls back to stock otherwise.

## Static build

The current patch compiled against vLLM
`fe9c3d6c5f66c873d196800384ed6880687b9e52` and CUTLASS v4.4.2
(`da5e086dab31d63815acafdac9a9c5893b1c69e2`) without GPU execution or
Docker. The compile-only extension is:

```text
/home/mark/fr13_projection_rowcover_build/bin/_C_stable_libtorch.static_b1_m32_b4_m128.combined_compile_only.abi3.so
SHA256 af48592c748ba80b1c614dc7a96c8250ae3bcca4c185c92939b4d308f8ef31f6
bytes  113078080
mode   0555
```

Both FP16 and BF16 B1 static kernels are `REG=168`, `STACK=0`, `LOCAL=0`,
`SHARED=1024`, `CONSTANT[0]=2688`. Both B4 M128 kernels are `REG=168`,
`STACK=0`, `LOCAL=0`, `SHARED=1024`, `CONSTANT[0]=2560`.

All six stock CUTLASS kernel symbol/resource records exactly match the reviewed
B4 binary after whitespace normalization. The 873 strong dynamic exports also
match exactly. The full dynamic symbol set has two expected additional weak
`get_grid_shape` symbols for the FP16/BF16 static scheduler specializations, so
full-symbol equality is not claimed.

## Correctness status

The evidence belongs to the prior per-candidate binaries and does not qualify
this combined extension:

- B1 static persistence passed a one-task real SWE-Verified full-vocabulary
  diagnostic at commit `c810bd5f1`: 320/320 exact comparisons across all five
  `M=32` projection shapes, 311,951,360 compared bytes, zero differing bytes.
  It used root 0/K 0, not the current root 1/K64 profile, and is not acceptance
  or timing evidence.
- B4 M128 produced 320/320 exact K64 comparator records across all five
  `M=128` shapes and 1,436,811,264 compared bytes. The exact4 campaign was
  rejected at commit `7c6e7ce73` because one completed remote Qwen trace was
  truncated. No live PASS or production credential was issued and timing did
  not start.
- This combined binary has not run a real SWE-Verified byte gate. Its binary
  identity, runtime selector allowlists, and qualification hashes are not wired
  in this branch. Existing reducers correctly reject it.

## Projection roofline

The projection-only lower bound includes FP8 weights, scales, and minimum
input/output traffic. It uses the planning ceilings of 273 GB/s unified-memory
bandwidth and 125 TFLOP/s dense compute.

| Batch | Physical rows | Minimum bytes | Traffic floor | Compute floor | TFLOP/s needed at traffic floor | Class |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B1 | 32 | 24,144,101,376 | 88.439932 ms | 12.197707 ms | 17.240 | memory side |
| B4 | 128 | 25,088,016,384 | 91.897496 ms | 48.790828 ms | 66.366 | memory side |

The historical real B1 attribution measured 112.312954 ms/event for 256
projection launches. Its excess over the explicit B1 traffic floor is
23.873022 ms. Even subtracting that entire historical excess from the current
K64 B1 wall anchor is only an optimistic cross-run ceiling: wall would still be
208.906768 ms, `1.745865x` the 119.658015 ms floor and 71.300050 ms above the
137.606718 ms cap. Projection work alone therefore cannot close the full-step
goal.

At B4, the second stock M tile can in principle reread 23,829,463,040 bytes of
weights and scales. At 273 GB/s that is an 87.287410 ms *upper bound* on what
the M128 row tile could avoid if none of the second-tile traffic were cached.
No cache or DRAM counters exist for the B4 path, so this is not a hard saving
and no speedup is claimed. B4 full-wall distance from the floor remains
unknown because it has no valid matched timing.

## Current wall anchor

The valid real exact4 Hydra27 B1 K64 result is 232.779790 ms/step,
24.718147 full-wall TPS, and 4.753885 accepted drafts/event. Against the
119.658015 ms floor and 137.606718 ms cap, it is `1.945376x` floor and
95.173072 ms above the cap. Its phase split is 159.619263 ms SFWD,
36.813368 ms DFWD, 20.677391 ms CFWD, and 15.669768 ms other.

## Required next execution

1. Bind the combined binary identity, static selectors, K64 profile, and patch
   hash through the installer, launcher, runtime patcher, and qualification
   reducer. This is mission-critical because the current route rejects the
   candidate rather than silently testing an unbound binary.
2. Run one allowed real SWE-Verified B1 K64 byte diagnostic at exact `M=32`.
3. Rerun the canonical real SWE-Verified exact4 B4 K64 diagnostic with the
   corrected remote trace capture.
4. Only after both credentials pass, run matched real exact4 full-wall timing
   and report TPS, acceptance, and SFWD/DFWD/CFWD/other. No probe or synthetic
   timing is valid.
