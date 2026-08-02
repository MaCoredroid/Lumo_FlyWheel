# Fixed32 FA2 qrow16 RU3/R216 static admission

Status: **spill-free 216-register SM121a object built; real B1 byte gate and
timing remain pending**.

This candidate combines two changes that are required as a pair:

1. The private fixed-page qrow16 trait compiles out dynamic paged/nonpaged
   routing while stock FA2 traits retain the original runtime path.
2. The private translation unit is compiled with ptxas register-usage level 3
   and its kernel has `__maxnreg__(216)`.

The boundary is material. The 216-register cap alone produced a 16-byte stack
with 24 bytes of spill stores and 20 bytes of spill loads. RU3 alone compiled
without spills at 218 registers. RU3 plus the cap compiled at 216 registers,
zero stack/local/spills/calls, and a peak live-GPR count of 208.

| Variant | Registers | Peak live GPR | Stack | Spill stores/loads | Text slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| prior division-free qrow16 | 224 | 221 | 0 | 0 / 0 | 5,040 |
| static-paged path, default scheduler | 224 | 222 | 0 | 0 / 0 | 4,928 |
| static-paged path, RU3 only | 218 | 212 | 0 | 0 / 0 | 5,056 |
| static-paged path, RU3 + R216 | 216 | 208 | 0 | 0 / 0 | 5,064 |

The final SASS retains 512 BF16 HMMAs, 132 FFMAs, 264 FMULs, 336 asynchronous
global-to-shared copies, and 288 shared-matrix loads. Source-level row mapping,
the ordered K-block loop, QK/PV accumulation order, masking, paged-KV address
semantics, output stores, and `num_splits=0` behavior are unchanged. The
private static trait returns early if its required block table is absent.

Pinned external binaries are intentionally not committed:

- object SHA-256: `67a47c406cda887b8bd1686095315431a447a3dd6db0db62ab32fc6a1ec6a452`
- cubin SHA-256: `1b80aaafb69d0caf730c14de373a15df377b8341ab1e21166350361ea593e2cd`
- normalized SASS SHA-256: `351234296985a77c98d707c54a30be65c89c5303329d411a2fdaebfca4d498d2`

No GPU, container, synthetic timing probe, or real task was used for this
checkpoint. The candidate is not timing-eligible or production-authorized
until the standing real SWE-Verified B1 K64/root1 byte gate passes. Only then
may full-step timing be compared on the standing real task set.
