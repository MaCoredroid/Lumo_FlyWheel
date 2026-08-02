# Verification

- Source commit: `1a638c4d39577a9c84f6df265b2a0cce3d5e2585`.
- Source file SHA-256:
  `f296133d946ba83d6b04add2938c5869461bce9a917a09bc9cbd790e17e52b90`.
- Candidate B1 and B4 emit the same cubin, PTX, and SASS hashes.
- A second compile from a fresh cache reproduces both batch builds exactly.
- `BAR`, `LDS`, `STS`, launch shared, and ELF shared are all zero.
- Reported registers equal the 64-register ceiling; stack, local, LDL, STL,
  spills, and calls are absent.
- AST-derived tap tuples equal every source row from the canonical 32-node
  topology for taps 0, 1, and 2.
- Source coverage requires 32 independent one-dimensional C64 row loads and
  rejects gather, join, permute, reshape, and split operations.
- Product and accumulation order is tap 0, tap 1, tap 2, then current row, with
  every product rounded to BF16 before the FP32 add.
- The descriptorless source and launcher suites pass: 26 focused tests.
- Ruff, Python compilation, and Git whitespace validation pass.
- The candidate is intentionally unbound. Runtime byte equivalence, real-task
  correctness, full-step TPS, and hardware-floor acceptance remain unmeasured.
