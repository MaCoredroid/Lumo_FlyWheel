# Fixed32 B1 N5120 single-scheduler launch readiness

Status: **full shared library linked and CPU-imported; K64/root1 byte and
exact-four timing admission wired; default off; GPU qualification pending**.

## Deliverable

The retained `identity_onen_n5120_single_b1` candidate is now a complete
AArch64 vLLM stable-libtorch extension, not a translation-unit-only object.
The binary registry pins it to:

- SHA-256 `876a3d6a0c972926131b1e447ffba80e345979f2d6de3bfa7bf083e862469367`
- 118,468,696 bytes
- selector `identity_onen_n5120_single_b1`
- stock-serving diagnostic selector `identity_onen_n5120_single_b1_byte_ab`

The host build output is intentionally not committed because the shared
library exceeds GitHub's single-file limit. The registry identity, source
binding, build manifest, and sanitized verification evidence are committed.

## Kernel and build result

The candidate changes only the two physical `N=5120` B1 projections. Their
40-tile `(1,40,1)` launch maps one complete output tile to each CTA and returns
no next work. The other three admitted B1 projections retain the incumbent
one-N path. The collective remains `128x32x128`, cluster `(1,1,1)`, cooperative
SM120, StageCount2, identity epilogue, and full-K accumulation.

The pinned vLLM translation unit compiled for SM121a, linked against the
existing stable-libtorch object set, and imported with `CUDA_VISIBLE_DEVICES`
empty. The extension exports `PyInit__C_stable_libtorch`; its dynamic dependency
and undefined-symbol sets match the incumbent one-N library.

The two new FP16/BF16 device functions each have 520 encoded instructions, 27
NOPs, 493 operational instructions, 26 branches, 168 registers, zero stack,
zero local memory, zero `LDL`/`STL`/`CALL`, and 1,024 bytes static shared memory.

## Admission contract

- The new binary and old incumbent have separate selector-keyed source hashes.
- The new selector requires an explicitly selected `k64_root` profile before
  file, GPU, or Docker work.
- The byte gate runs the pinned one-task real SWE-Verified diagnostic, serves
  stock, and admits only exact output bytes across the five projection shapes.
- Timing defaults to the standing-rule exact-four real SWE-Verified task set
  and requires the authenticated byte-gate sidecar and the exact candidate
  SHA/size.
- Production remains default off and requires the K64 sidecar.

## Verification boundary

- 197 adjacent CUTLASS, B1 diagnostic, and process-attestation tests passed.
- Python byte compilation, shell syntax, `git diff --check`, source binding,
  full-TU SM121a compilation, linking, CPU import, ELF, resource, and SASS
  checks passed.
- No GPU kernel launch, Docker run, real SWE-Verified execution, raw-byte
  comparison, full-step timing, or hardware-floor acceptance run was performed.

The next valid operation is the authenticated one-task K64/root1 byte gate on
the real shared library. Exact-four full-step timing is allowed only after that
gate passes.
