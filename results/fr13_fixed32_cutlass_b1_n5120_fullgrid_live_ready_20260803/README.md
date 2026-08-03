# Fixed32 B1 N5120 plus wide full-grid CUTLASS readiness

Status: **SM121a build, link, CPU import, and fail-closed host admission
passed; default off; real-task byte and timing qualification pending**.

## Kernel change

The candidate preserves the previously admitted exact 40-CTA single-tile
scheduler for the two physical `N=5120` projections. For the three wider B1
projections it retains the audited one-N device coordinate mapping and
ping-pong collective, but replaces the divisor-balanced 28--34 CTA host grid
with CUTLASS's full static persistent grid. On the 48-SM B1 device this makes
all 48 SMs available to the wide weight-streaming GEMMs.

The collective remains `128x32x128`, cluster `(1,1,1)`, StageCount2,
identity epilogue, swap-AB layout, and full-K accumulation. Unset, unknown,
and out-of-contract shapes remain stock.

## Static evidence

The complete pinned vLLM translation unit compiled for SM121a and linked into
an AArch64 stable-libtorch extension. CPU import with `CUDA_VISIBLE_DEVICES`
empty passed. Dynamic dependencies and the 183-entry undefined-symbol set
match the incumbent N5120 library.

The FP16 and BF16 full-grid kernels each use 168 registers, zero stack, zero
local memory, 1,024 bytes static shared memory, and contain no `LDL`, `STL`,
or `CALL`. Their complete instruction streams are byte-identical to the
corresponding incumbent one-N ping-pong device kernels. The change is therefore
isolated to host grid selection; it does not alter device math or mapping.

## Qualification boundary

- 122 focused CUTLASS, source-binding, gate, and timing-contract tests passed.
- Python byte compilation, shell syntax, `git diff --check`, SM121a compile,
  full shared-library link, CPU import, ELF, ABI, resource, and SASS checks
  passed.
- The exact binary SHA and size are pinned in the verifier.
- The K64/root1 gate requires the exact committed source identity and serves
  stock while comparing candidate bytes on one real SWE-Verified task.
- No GPU launch, synthetic/probe timing, real-task byte result, full-step
  timing, or hardware-floor acceptance claim is included here.

The next valid operation is the authenticated one-task real SWE-Verified B1
K64/root1 byte gate. Exact-four full-step timing is permitted only after that
gate passes.
