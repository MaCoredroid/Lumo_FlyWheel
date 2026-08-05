# Fixed32 B4 exact-shape scheduler SM121a codegen

This artifact records the offline SM121a gate for the fixed32 B4 projection
scheduler specialization. The retained selector is still default-off and is
restricted to physical `M=128`, `K=5120`, and the three non-`N=5120`
projection shapes in the real 16-call projection histogram.

## Result

The exact-shape scheduler moves the logical tile bound, launch width, and N
stride from runtime scheduler state to compile-time constants. It retains one
32-bit N cursor and the incumbent tile order. Across all three shapes and both
output dtypes, it compiles with 168 registers, zero stack, zero local memory,
and no `LDL`, `STL`, or `CALL` instructions.

Each exact kernel has 1,024 SASS slots versus 1,032 for the in-object generic
two-M scheduler. It removes six `LDCU` instructions and one `LDC` instruction,
while branch count, QMMA, FFMA, LDSM, and STSM counts are unchanged. The new
route covers eight of the 16 real B4 projection calls; the other eight already
use the retained `N=5120` single-tile scheduler.

Two earlier linear-cursor forms were rejected during development because each
introduced an 8-byte per-thread stack frame. They are not retained in source.

This is an offline code-generation win, not a performance or acceptance
measurement. No GPU kernel, Docker service, synthetic workload, or real task
was run. A fresh real SWE-Verified exact4 byte gate and full-step B4 timing are
required before promotion.

## Files

- `manifest.json`: pinned inputs, source identity, contract, and verdict.
- `projection_schedule.tsv`: exact real-shape schedule and call coverage.
- `kernel_resources.tsv`: candidate and in-object baseline resources.
- `sass_summary.tsv`: reduced instruction counts.
- `source_checksums.sha256`: retained patch-source identity.
- `verification.txt`: completed checks and explicit runtime boundary.
- `SHA256SUMS`: reduced artifact integrity.
