# Verification

- `133 passed` across the focused SFWD, ingress, wall-timer, and final-preseed
  unit tests.
- Python compilation completed for the launcher, kernel source, and gate.
- Shell syntax validation completed for the B1 real-task runner.
- `git diff --check` completed without errors.
- Two independent empty-cache SM121a builds produced identical summaries and
  identical cubin, SASS, PTX, and kernel-function hashes.
- Both B1 and B4 specializations contain zero `BAR`, `LDS`, `STS`, `LDL`,
  `STL`, calls, local memory, stack, and launch/ELF shared memory.
- `CUDA_VISIBLE_DEVICES` was empty for all offline compiler invocations.

This package records no live throughput, acceptance, or hardware-floor claim.
