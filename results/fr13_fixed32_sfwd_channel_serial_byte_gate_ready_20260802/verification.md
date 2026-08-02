# Verification

- The source manifest generator accepts the exact integration commit and binds
  all 14 required files plus the unchanged fixed reference GDN source.
- The wrapper launches `_fr13_fixed32_sfwd_channel_serial_kernel` on the
  `(batch, channel_block)` grid with C64 and two warps.
- The channel-serial kernel receives no rowgroup argument and performs no local
  gather or layout conversion.
- Candidate identity and two-warp geometry agree across wrapper, runner, gate,
  emitted records, and final validation.
- The byte gate remains default-off, requires an authenticated real task, and
  compares both convolution output and commit source-stage bytes for all 48
  layers while returning only the reference result.
- Descriptorless source, wrapper/gate, final preseed, and ingress suites pass:
  94 focused tests.
- Ruff, Python compilation, shell syntax, and Git whitespace validation pass.
- Runtime byte equivalence, full-step TPS, and hardware-floor acceptance remain
  unmeasured.
