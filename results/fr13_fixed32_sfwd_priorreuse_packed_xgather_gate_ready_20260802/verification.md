# Verification

- The packed source decoder matches all 32 established topology rows.
- The x-gather kernel has exactly one global `x` tile load in source and reuses
  it for all historical taps.
- The focused kernel and gate suite passes 19 tests.
- The final-full-preseed and ingress suites pass 68 tests.
- Python compilation, shell syntax, Ruff, and `git diff --check` pass.
- The source manifest is bound to source commit
  `eb1a69de3dc180bd29b4488e834c60a3db7bca88`.
- Offline B1 SM121a codegen at row32/C64/W16 reports 55 registers, 4,096
  launch-shared bytes, 408 static and 424 encoded SASS instructions, and zero
  stack, local memory, spills, or calls.

The codegen result is not runtime evidence. A real K64/root1 B1 byte gate on
both surfaces and all 48 layers remains mandatory.
