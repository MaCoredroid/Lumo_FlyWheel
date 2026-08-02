# Fixed32 B1 M32 static-linear scheduler source candidate

Status: source-only and default off. No host build, GPU execution, raw-byte
gate, timing result, TPS result, or hardware-floor improvement is claimed.

## Candidate

The production selector is `m32_static_linear`; the diagnostic selector is
`m32_static_linear_byte_ab`. Unset, unknown, non-M32, and non-allowlisted
shapes dispatch stock.

The candidate preserves the exact stock swapped cooperative projection:

- tile `128x32x128`, cluster `1x1x1`;
- `KernelTmaWarpSpecializedBlockwiseCooperativeSm120`;
- block scales `(128,1,128)` and swapped A/B layout;
- `Base::CollectiveMainloop` and `Base::CollectiveEpilogue` without rebuilding
  either collective;
- one complete ordered K reduction per output tile, with no split-K, Stream-K,
  reduction workspace, stage override, fixup, or epilogue substitution.

Only complete-output-tile assignment and CTA-coordinate formation change.
`Fr13Fixed32M32LinearScheduler100` retains CUTLASS static persistence, asserts
one N tile, one L tile, a `1x1x1` cluster, no swizzle, and one block per M tile,
then maps `linear_idx` directly to `(M=linear_idx,N=0,L=0)`. Its CTA coordinate
is `(M,0,_,0)`, and successive work advances by physical grid size.

## Exact geometry

The selector is admitted only for these physical `(M,N,K)` shapes:

| M | N | K | swapped M tiles | swapped N tiles | full-K tiles |
|---:|---:|---:|---:|---:|---:|
| 32 | 34816 | 5120 | 272 | 1 | 40 |
| 32 | 5120 | 17408 | 40 | 1 | 136 |
| 32 | 5120 | 6144 | 40 | 1 | 48 |
| 32 | 16384 | 5120 | 128 | 1 | 40 |
| 32 | 14336 | 5120 | 112 | 1 | 40 |

Every admitted M, N, and K dimension is exact under the stock tile; no residue
or predicated partial output tile is introduced. K64/root1, Tail23/Hydra27,
node masks, drafter vocabulary, and the task set remain external campaign
inputs and are not changed by this source patch.

## Comparator

The lower-risk comparison candidate remains separate on
`agent/fixed32-cutlass-static-scheduler-fb35`: source commit
`0adf68ed969bf488f8a050d2d3ea17573d847898`, artifact commit
`668c06e0d1a3736266c7745adebdf9ac86ea94ae`. It uses unmodified CUTLASS
`StaticPersistentTileScheduler100` with the same stock M32 collective.

That comparator already compiled at 168 registers, zero stack/local memory,
1,024 bytes static shared memory, and 2,688 bytes `CONSTANT[0]` for both BF16
and FP16 M32 bodies. Offline disassembly of the current comparator rebuild
shows 1,176 stock instructions versus 936 generic-static instructions, a
240-instruction or 20.408% reduction. Both retain 32 QMMA, 32 FFMA, 24 FMUL,
24 LDSM, 4 STSM, and 8 output-pack instructions. The generic-static body still
contains CUTLASS batch/raster/swizzle/divmod and coordinate machinery; this
candidate removes that remaining mapping for the asserted one-dimensional
geometry.

The comparator is the lower-risk gate-first option because it uses CUTLASS's
published scheduler unchanged. This direct scheduler is the higher-upside
successor. Neither has passed a real SWE-Verified raw-byte gate.

## Source scope

The patch is additive to the existing candidate framework. A pinned host build
is expected to add exactly two device bodies: BF16 and FP16 swapped cooperative
`128x32x128` M32-static-linear kernels. No stock, generic-static, Stream-K,
wide256, or M128 device body is expected to change. This is a prediction to be
checked by cubin/function-body comparison after build, not a compiled fact.

## Verification

- 48 focused patch and existing candidate-framework contract tests passed in
  0.17 seconds.
- Ruff, Python bytecode compilation, exact pinned vLLM patch application, patch
  idempotency, and pinned CUTLASS header/commit validation passed.
- Generated dispatch SHA256:
  `85800af2eaaac712a0cb8371c942b067e032e51cf7d06efb43c06ea75369a435`.
- Source commit: `246ea98e7671d787079940fee9c958b4ded9642f`.

The filesystem had 4.4 GiB available at source closeout. No build was started.
Recheck disk and source review before creating an isolated build tree.

## Required gates

1. Perform the pinned SM121 host build and reject register count above 168,
   any stack/local memory or spills, shared memory other than 1,024 bytes, or a
   change in the mandatory math/memory-pipeline instruction counts.
2. Prove that only the predicted BF16/FP16 candidate bodies are additive and
   inspect their static scheduler/coordinate opcode delta against the separate
   generic-static comparator.
3. Wire the two selectors into the pinned loader and launcher allowlists, then
   run authenticated real SWE-Verified B1 raw-byte gates on the established
   task set at K64/root1. Synthetic probes and one-task diagnostics do not
   qualify for acceptance.
4. Time full-step B1 only after every armed comparison row is byte equal and
   both real task arms resolve cleanly. The full-step one-sided U95 requirement
   remains at most `1.15x` the hardware floor.

See `BUILD_AND_BYTE_GATE.md` for the exact handoff and the included format
patch for the complete source diff.
