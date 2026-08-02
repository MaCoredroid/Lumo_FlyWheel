# Verification

- `26 passed` in the focused SFWD launcher and descriptorless-kernel tests.
- The structural tests prove first-use ordering for all 32 current rows and
  exactly two loads of row 4.
- Python compilation passed for the launcher, kernel, and strict gate.
- Shell syntax validation passed for the real B1 byte-gate runner.
- Ruff and `git diff --check` passed on the source commit.
- Offline SM121a B1 and B4 codegen produced 48 registers/thread and zero
  stack, local, shared, spill, barrier, and call resources.
- The codegen run did not execute the kernel on a GPU.

The source remains default-off. No live correctness, timing, throughput,
acceptance, hardware-floor, or production claim is made.
