# Verification

- Source commit: `9b54beec6cb4e196cdfde7d9daa1e58af94da64e`
- Full SFWD source suite: 68 passed, 0 failed.
- Ruff checks: passed.
- Python syntax checks: passed.
- B1 gate runner shell syntax: passed.
- Git whitespace/error check: passed.
- AST source mapping: all 128 tap products retain the fixed source rows.
- Arithmetic source order: 32 bias starts and 96 ordered FP32 adds retained.
- Activation structure: 16 saved first accumulators and 16 adjacent SiLU pairs.
- Store structure: each pair performs both activations before its four stores.
- Memory structure: 32 global x loads, zero x reloads, 32 output stores, and
  32 current-row source-stage stores.
- Liveness: deferred-stage peak remains five; its live-x sum is 125 and the
  activation-accumulator peak is two.

No kernel codegen, GPU byte comparison, real-task execution, performance
measurement, or hardware-floor acceptance claim is included.
