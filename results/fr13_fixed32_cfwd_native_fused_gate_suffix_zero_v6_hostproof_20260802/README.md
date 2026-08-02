# FR13 fixed32 CFWD native fused-gate suffix-zero v6 host proof

## Classification

`READY_AFTER_V5_FOR_REAL_B1_BYTE_GATE`

This is a host-only source, compile, and regression receipt. It is not a
correctness, timing, B1, B4, exact4, exact16, or hardware-floor result.

## Change

V6 keeps the v5 final FP32 `state + +0` suffix collapse and restores the gate
scalar work to the first active threads of the existing native key-group CTA.
It removes the separate per-event Triton scalar launch, its persistent output
buffer, and the global write/read round trip.

The fused layout computes one scalar pair for each active
`(layer, request, step, value_head)`. One CTA owns a key head and its three
value heads, so the fusion does not duplicate scalar work across CTAs.

The prior v3 fused binary and v4 split binary both omitted the final fixed-16
suffix normalization. Their SM121 arithmetic delta was exactly the expected
per-scalar gate footprint: three `EX2`, one additional reciprocal, and twelve
FFMA log-polynomial instructions. V6 combines that already spill-free fused
shape with the v5 suffix repair.

For the selected softplus log arm, `1 + exp(x)` is at least one. The CUDA and
Triton log polynomials have the same constants and operation order; their FTZ
opcode distinction is outside this normal-valued input domain. The real byte
gate remains authoritative.

## Static work delta from v5

| Surface | B1 | B4 |
| --- | ---: | ---: |
| Captured candidate launches per event | 2 to 1 | 2 to 1 |
| Event-scalar buffer elements | 55,296 to 0 | 221,184 to 0 |
| Event-scalar buffer bytes | 221,184 to 0 | 884,736 to 0 |
| Worst-case scalar global write plus read bytes | 442,368 to 0 | 1,769,472 to 0 |

Only the active prefix was read and written by v5, so the worst-case traffic
row is a capacity bound, not a measured event. Gate arithmetic is fused, not
eliminated. No latency or speedup is claimed.

## Verification

- Source commit: `02593f63589619fd919bae968586dae34779af1b`
- CUDA source SHA-256:
  `6d75ac0685641f5433592c9a39ce6584f676980cc472c5a39e6a23d4a6d5e814`
- Focused candidate, binary, gate, committer, arm, and boundary tests:
  169 passed.
- Ruff, Python byte compilation, JSON binding hashes, and diff checks: pass.
- Pinned target: SM121a, CUDA 13.0.
- Registers per thread: 64; stack, local memory, spills, and calls: zero.
- Cuobjdump shared bytes: 7,592.
- Signed-zero `FADD register, RZ, register` instructions: 64.
- No GPU execution and no container launch were used for this receipt.

## Gate order

Run the frozen v5 real SWE-Verified B1 raw-bank byte gate first to isolate the
suffix repair. If v5 passes, rebuild the source-bound v6 binary and run the
same B1 byte gate. Only a clean v6 byte result permits real-task timing or the
formal exact4 B4 campaign.
