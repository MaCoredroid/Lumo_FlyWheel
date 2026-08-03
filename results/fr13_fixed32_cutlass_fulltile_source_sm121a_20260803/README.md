# Fixed32 target-GEMM full-tile source candidate

Status: **SM121a compile, link, CPU import, resource audit, and fail-closed
binary admission passed; default off; real SWE-Verified qualification pending**.

## Kernel change

This source candidate attacks target-model CUTLASS replay without changing the
number of GEMM launches, arithmetic, reduction order, or admitted projection
shapes.

- B1 physical M32 keeps the two `N=5120` projections on the existing exact
  40-CTA `128x32x128` kernel. The three wider projections use a cooperative
  `256x32x128` identity-epilogue StageCount2 kernel and the complete full-grid
  static scheduler. All three expose at least 56 complete output tiles, enough
  to fill the 48-SM B1 device.
- B4 physical M128 preserves the existing exact 40-CTA single-tile scheduler
  for both `N=5120` projections. The three wide projections use the existing
  cooperative `128x128x128` identity-epilogue StageCount2 static kernel. This
  replaces the current hybrid's two 64-row tiles on each wide projection with
  one full-M tile without regressing its N5120 path.
- Both paths keep a single ordered full-K reduction, allocate no split-K
  workspace, and fall back to stock for every out-of-contract shape.

The direct selectors remain non-installable. Only
`identity_wide256_fullgrid_b1_byte_ab` and `identity_fullm_b4_byte_ab` can be
installed, and both require an explicit `k64_root` qualification profile.

## Modeled ceiling

The fixed target layer census is the 16x repetition of the five-shape call
histogram recorded in `tile_traffic_model.tsv`, or 256 projection launches per
full target step. Launch count and target weight bytes are unchanged.

- B1 removes 12,672 complete output tiles per target step (41.597%) and
  2,076,180,480 bytes of requested activation-panel replay (34.138%). Dividing
  that requested traffic by 273 GB/s gives a 7.605 ms/step optimistic ceiling.
- B4 removes 25,344 complete output tiles per target step (45.413%). The wide
  M64 geometry requests 16,609,443,840 bytes of second-M-tile weight-panel
  traffic that the full-M geometry does not request. At 273 GB/s that is a
  60.840 ms/step request-traffic ceiling.

Those byte ceilings are not HBM savings or speed measurements. Cache reuse,
TMA scheduling, occupancy, and instruction overlap can make the realized gain
much smaller. No performance claim is made.

## Host evidence

The complete pinned vLLM stable-libtorch target compiled for SM121a and linked
as an AArch64 shared object. CPU import with `CUDA_VISIBLE_DEVICES` empty
passed. The new FP16 and BF16 B1 kernels each use 168 registers, zero stack,
zero local memory, 1,024 bytes static shared memory, and contain no `LDL`,
`STL`, or `CALL`. The B4 selector reuses the existing audited exact-N5120 and
wide full-M kernels, which have the same resource limits and no local-memory
instructions.

## Required live gates

Before any timing run, the diagnostic selectors must be added to the existing
B1 and dual-topology B4 real-task gate allowlists and source-binding schemas
without weakening the combined Qrow32, SFWD, and CFWD launcher guards.

The next valid executions are authenticated real SWE-Verified raw-byte gates:
B1 at physical32/K64/root1, and B4 on the canonical exact-four set for both
Tail23 and Hydra27. Only zero-byte-difference candidates may proceed to clean
exact-four or exact-16 full-step and component timing. The hardware-floor
acceptance remains the one-sided U95 gate; this host artifact is not acceptance
evidence.
