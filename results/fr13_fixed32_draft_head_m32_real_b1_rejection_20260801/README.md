# FR13 fixed32 full-head M32 real-B1 rejection

Status: `REJECTED_BYTE_INEXACT`. This was a one-task kernel-correctness
diagnostic, not a performance measurement or hardware-floor acceptance run.

## Result

The pinned real SWE-Verified task `astropy__astropy-12907` resolved cleanly
because every request continued to serve the incumbent BF16 logits. Across
909 complete fixed32 B1 events, the shadow candidate executed exactly five
full-vocabulary head comparisons per event:

- comparisons: 4,545
- BF16 elements compared: 1,128,614,400
- raw BF16 element mismatches: 3,259,820
- mismatch rate: 2,888.338125 ppm (0.288833813%)
- candidate GEMM: `M=32, N=248320, K=5120`
- physical rows: 32
- served rows: 1
- draft vocabulary: full (`root=0`, `K=0`)

The terminal live result is `FAIL`. The gate exited 13 after its fail-closed
final flush rejected the mismatch census. M32 must not be timed, promoted, or
included in a B1/B4 acceptance stack.

## Identity

- source commit: `12fd8212733417894c1c7e6383f399ba888695e0`
- candidate source SHA-256:
  `6f1a70b1d0aa88005a0dac434f4f2a4196251db356041e940d0c2140444f84b1`
- runroot:
  `output/fr13_draft_head_m32_b1_20260801T104104Z`
- raw live result SHA-256:
  `e5827885778ee27bb63fe2a6fc1833307e04fc543b8b35a355d9b8d3fe1d3a5a`
- authenticated work census SHA-256:
  `ee2f02307a37ead96075062b3a27dcb14ae6734ef282da1945bfa4234de96f7a`
- real-task campaign summary SHA-256:
  `467d0d6c1d7f83c48dacb15746e0c7c989ccce0307d00d364428ec6e8dd8cfdf`

The failed container was deliberately preserved by the launcher, inspected
through host-mounted evidence, then removed after the rejection was confirmed.
No latency, TPS, saving, confidence bound, or hardware-floor ratio is derived
from this shadow run.
