# FR13 fixed32 native GDN warp-group source candidate

Status: **source-only, default OFF, no GPU measurement, no pinned-image
compile result, and no production authorization**.

## Candidate

The candidate is a native CUDA `_C` op for the exact root-inclusive fixed32
tree. One CTA owns one `(request, value-head, BV8 tile)` unit. Warp 0 executes
the root path `0 -> 1 -> 4 -> 9 -> 14` and places those five FP32 state tiles
in 20,480 bytes of shared memory. A block barrier then releases five resident
four-warp groups. Eleven member warps execute the eleven original child paths;
nine padded member warps are inactive. Each path preserves its original node
order and owns every output/ring row it visits exactly once.

The critical recurrence is root depth 5 plus maximum branch depth 7, or 12.
There is one kernel launch per layer and no HBM parent-state export or reload.
The launch grid is 768 CTAs/layer at B1 and 3,072 CTAs/layer at B4.

## Native hook

The serving image pins vLLM source commit
`fe9c3d6c5f66c873d196800384ed6880687b9e52` and retains its `_C` build
toolchain. The patch adds one `.cu` file to the existing `VLLM_EXT_SRC`, one
declaration to `csrc/ops.h`, and one schema/implementation registration to
`csrc/torch_bindings.cpp`. It does not introduce another extension loader or
build system. Exact source hashes and one-count anchors make the patch
fail-closed and idempotent.

## Resource model

- Fixed geometry: 32 rows, 16 key heads, 48 value heads, DK=DV=128, BV=8.
- Threads: 20 warps / 640 threads per CTA, with `__launch_bounds__(640, 1)`.
- Shared memory: five `8 x 128 x fp32` tiles = 20,480 bytes per CTA.
- Explicit live state per active thread: 32 FP32 state scalars; q/k add eight
  FP32 scalars before compiler temporaries and address registers.
- Planning register ceiling: 102 registers/thread if a 65,536-register block
  allocation is the limiting SM121 resource. This is not a compile result.
- Required pinned-image gate: ptxas registers <=102, zero local bytes, zero
  spill loads/stores, legal 640-thread launch, and 20,480-byte static shared
  allocation.

Relative to the two-launch parent-group route, the static descriptor removes
1,509,949,440 bytes/event at B1 and 6,039,797,760 bytes/event at B4 of FP32
parent export-plus-read traffic. Relative to the original eleven-parent-read
path route, it removes 2,415,919,104 bytes/event at B1 and 9,663,676,416
bytes/event at B4. These are logical byte counts, not observed DRAM traffic or
timing.

## Selection and qualification

`FR13_FIXED32_GDN_WARPGROUP_CUDA=diagnostic` is the only accepted non-OFF
selector. It requires Tail23 or Hydra27 fixed32 mode, B1-B4, exact 32-row
geometry, BV8, Q/K normalization, SCAN_ALIGN, in-kernel K/V/A/B ring export,
in-kernel flags, col-0 h0 initialization, and the patched op. Any drift raises;
there is no fallback and no production selector.

Qualification must run an explicit incumbent BV8 arm and candidate arm inside
the same authenticated real SWE-Verified bracket. Restore all mutable surfaces
before the candidate, compare raw bytes for output, K/V/A/B rings, flags, and
invocation counter across all 48 layers, restore the incumbent result, and
always serve the incumbent during qualification. B1 uses a real one-task
bracket; B4 uses the canonical exact4 campaign bracket. No durable production
credential is emitted by this source candidate.

The CUDA reduction partition and transcendental code generation differ from
Triton, so raw-byte equality is unresolved and may fail. Exact recurrence/path
ordering is source-proven; numerical and byte equivalence are not claimed.

## Verification boundary

The focused candidate tests and adjacent parent-group tests pass 39/39. The
patch applies and reapplies idempotently to the exact pinned vLLM sources.
Python syntax checks pass. A host-only nvcc attempt could not compile because
the host torch package lacks its generated CUDA CMake header; the pinned image
build remains mandatory. Ruff is not installed in this host environment.

No GPU kernel, performance probe, synthetic timing, or real SWE-Verified timing
was run. Full-step TPS, SFWD phase time, register count, spills, graph capture,
byte equivalence, and B1/B4 acceptance remain unknown.
