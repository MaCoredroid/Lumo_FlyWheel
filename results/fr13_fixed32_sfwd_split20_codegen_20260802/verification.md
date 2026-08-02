# Verification

- `133 passed` across the focused SFWD, ingress, wall-timer, and final-preseed
  unit tests after the split20 source and topology-test changes.
- The structural topology test checks all 32 nodes and four operand columns
  against the fixed descriptorless source table.
- The launcher fails closed unless the two 32-bit weight loads are 4-byte
  aligned; the post-guard focused kernel suite reports `26 passed`.
- Python compilation completed for the launcher, kernel source, and gate.
- Shell syntax validation completed for the real B1 runner.
- `git diff --check` completed without errors.
- Two independent empty-cache SM121a builds for B1 and B4 reproduced complete
  summaries and every binary/text hash.
- Both schedules contain zero `BAR`, `LDS`, `STS`, `LDL`, `STL`, calls, local
  memory, stack, and launch/ELF shared memory.
- `CUDA_VISIBLE_DEVICES` was empty for every compiler invocation.

The source remains default-off. No live correctness, timing, throughput,
acceptance, hardware-floor, or production claim is made.
