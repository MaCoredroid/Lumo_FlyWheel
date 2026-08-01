# FR13 fixed32 B4 persistent-M128 CUTLASS build

This artifact records the SM120 build candidate for the real exact4 B4 screen.
It is build/static evidence only; no throughput or hardware-floor claim is made.

## Candidate

- Selector: `persistent_b4_m128`
- Diagnostic selector: `persistent_b4_m128_byte_ab`
- Exact dispatch: physical `M=128` and one of the five real projection `(N,K)` pairs
- CUTLASS tile: `128x128x128`
- Schedule: `KernelTmaWarpSpecializedBlockwiseCooperativeSm120`
- Fallback: the stock dispatcher for every non-exact shape
- B4 diagnostic comparison-call limit: `320` (B1 remains `256`)
- Binary SHA-256: `6988f6a994c29e9196b6addc039e1d63bf08c32f268f9be3d2f14c5d863be1de`
- Binary bytes: `112698512`

The stock B4 path uses a `64x128x128` ping-pong tile. At `M=128`, the candidate
halves the M-axis output-tile count while keeping the N tile at 128. The new
kernel has a distinct nested `GemmKernel` symbol, so existing stock symbols are
not re-instantiated or replaced.

## Static checks

`cuobjdump --dump-resource-usage` reports both FP16 and BF16 candidate kernels at
`REG=168`, `STACK=0`, `SHARED=1024`, and `CONSTANT[0]=2560`. All six stock
kernel symbol/resource records match the prior exact-stock baseline after
whitespace normalization. See `candidate_kernel_resources.tsv`,
`stock_kernel_resources.tsv`, and `stock_kernel_resource_equivalence.txt`.
The raw resource dump SHA-256 remains
`8dab744202393bdc26cc2d2aa622f6ac1eb492dd932861ff62f63daa1c6c9841`,
confirming that the host-side comparator-bound change did not alter cubins.
The dynamic defined-symbol set also exactly matches the prior 256-call
persistent-M128 candidate.

The earlier `64x256x128` ping-pong experiment compiled with `STACK=488` for
both output types and is rejected. Cooperative `64x256` is structurally invalid
on this CUTLASS path, while normal cooperative `128x256` admits only one
mainloop stage and fails the required two-stage minimum.

## Next measurement

Run `scripts/fr13_run_b4_cutlass_persistent_m128_live_gate.sh` on the canonical
real SWE-Verified exact4 B4 set. Only after it emits an authenticated byte PASS
should `scripts/fr13_run_b4_cutlass_persistent_m128_timing.sh` run the paired
stock/candidate full-wall screen.
