# FR13 fixed32 CFWD native warp-step v7 host proof

## Classification

`READY_AFTER_V5_FOR_REAL_B1_BYTE_GATE`

This is a host-only source, compile, and regression receipt. It is not a
correctness, timing, B1, B4, exact4, exact16, or hardware-floor result.

## Change

V7 retains the v6 fused gate and v5 fixed-16 suffix-zero repair. It replaces
the three-wave K-normalization schedule with one warp per active root-inclusive
step. The first 12 of 16 warps cover the 12-step capacity in one wave.

Each lane loads four K values, one from each 32-element quad. Four butterfly
reductions produce the same quad partials as the prior four-warp group. Lane
zero then combines `quad0 + quad2`, `quad1 + quad3`, and those two partials in
the incumbent order before reciprocal square root. A warp broadcast publishes
the inverse norm, and one CTA barrier publishes all normalized rows to the
recurrence warps.

The transformation preserves K loads, products, normalization count, and
recurrence math. It removes cross-warp norm scratch, inverse-norm scratch, and
six CTA barriers. It trades quad-level warp parallelism for fewer barriers, so
the direction of the wall-time change must be measured on real tasks.

## Static delta from v6

| Surface | V6 | V7 |
| --- | ---: | ---: |
| SM121 CTA barrier instructions | 9 | 3 |
| Source shared bytes | 6,568 | 6,488 |
| Cuobjdump shared bytes | 7,592 | 7,512 |
| Registers per thread | 64 | 64 |
| Spill loads / stores | 0 / 0 | 0 / 0 |
| Stack / local bytes | 0 / 0 | 0 / 0 |
| Signed-zero normalization FADDs | 64 | 64 |

Static SASS copies are not dynamic instruction counts. In particular, V7 has
one static RSQ site executed by each active step warp; V6 had three unrolled
wave sites. No latency or speedup is claimed.

## Verification

- Source commit: `5457657d97bf8ccbd5360b243b27b908659ecd76`
- CUDA source SHA-256:
  `e022acdefdd045fe08407c222a3b8b56eb6caac7f5e929e3ce1190dbbda3fc9d`
- Host object SHA-256:
  `152c55b3f950a0a9d163e8a7f61e73dc20f4cd915927435e89b6fce446fed88f`
- Focused candidate, binary, gate, committer, arm, and boundary tests:
  169 passed.
- Ruff, Python byte compilation, patched-source hashes, codegen checker, and
  diff checks: pass.
- No GPU execution and no container launch were used for this receipt.

## Gate order

Run the frozen v5 real SWE-Verified B1 raw-bank byte gate first. If v5 passes,
the fastest route is to rebuild and byte-gate v7 directly. If v7 rejects, run
v6 as the localization fallback to distinguish fused-gate risk from the
warp-step normalization change. Only a byte-clean candidate may enter
real-task timing or exact4 B4 qualification.
