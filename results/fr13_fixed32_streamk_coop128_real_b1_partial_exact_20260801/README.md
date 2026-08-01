# Fixed32 stock-tile Stream-K scheduler real B1 partial byte pass

Status: `PARTIAL_BYTE_PASS_NOT_QUALIFIED`. This is a one-task
kernel-correctness diagnostic, not acceptance or timing evidence.

## Run identity

- Source commit: `fb35badb44fe344def7c2cd5874c5fb44f4426d9`
- Real SWE-Verified task: `astropy__astropy-12907`
- Physical rows: `32`
- Draft vocabulary: full (`ROOT=0`, `K=0`)
- Candidate: `streamk_coop128_byte_ab`
- Candidate binary SHA-256:
  `f9bbbb8dc4ffc2227a71d2bc7b260e586ffbdc0fd946749e4f69e322c46a362d`
- Raw comparator SHA-256:
  `61f23193df979858c116bb1a93f9e62b8a77c50af33bee0bdc8922bbe3a7234b`

## Byte result

| M | N | K | Comparisons | Differing bytes |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 5,120 | 6,144 | 64 | 0 |
| 32 | 5,120 | 17,408 | 64 | 0 |
| 32 | 16,384 | 5,120 | 48 | 0 |
| 32 | 34,816 | 5,120 | 64 | 0 |

All 240 authenticated target-projection comparisons passed: zero differing
bytes among 234,881,024 compared bytes. This candidate retains the stock
`128x32x128` tile and uses CUTLASS's Stream-K-capable scheduler in heuristic
mode. This run does not prove that heuristic mode selected a K-split
decomposition.

After the complete 64-layer target forward, the existing eager diagnostic
lifecycle raised `FR13 fixed32 KV16 compact/full row-map drift`. The engine
therefore stopped before the `8192x5120` MTP projection, the formal live-gate
reducer did not run, and no task result was produced. The candidate remains
ineligible for production or timing until a repaired real-task run covers all
five projection shapes and completes the gate.
