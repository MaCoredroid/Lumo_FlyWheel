# Verification

- Focused SFWD launcher and descriptorless-kernel tests: `26 passed`.
- The structural tests recover every tap source from the AST, prove the exact
  node 17-24 reload map, and preserve the ordered four-product accumulation.
- Ruff, Python compilation, shell syntax validation, and `git diff --check`:
  passed.
- Fresh exact-HEAD offline SM121a B1 C128/W2 and B4 C256/W4 codegen: passed.
- Both schedules report 44 registers and zero stack, local, shared, spill,
  barrier, and call resources.
- CUDA visibility was explicitly empty; no Docker or GPU work ran.
- No live correctness, timing, throughput, acceptance, or floor claim is made.
