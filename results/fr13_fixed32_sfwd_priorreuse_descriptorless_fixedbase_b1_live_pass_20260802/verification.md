# Verification

- One authenticated real SWE-Verified task ran at B1 with K64/root1.
- The task resolved and its tests passed.
- The candidate executed on all 48 layers.
- 21,504 layer invocations produced 43,008 surface comparisons.
- 29,947,330,560 bytes compared equal with zero shape, dtype, or byte
  differences.
- Both `conv_out` and `commit_source_stage` were checked while the reference
  remained the only served result.
- Source, runtime, and external launch/end manifests matched exactly.
- Docker and GPU process counts were zero after teardown.

This is a correctness gate only. It is not timing, B4, production, or
hardware-floor acceptance evidence.
