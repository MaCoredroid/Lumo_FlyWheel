# Fixed32 wide-256 Stream-K real B1 rejection

Status: `REJECTED_BYTE_INEXACT`. This is a one-task kernel-correctness
diagnostic, not acceptance or timing evidence.

## Run identity

- Source commit: `fb35badb44fe344def7c2cd5874c5fb44f4426d9`
- Real SWE-Verified task: `astropy__astropy-12907`
- Physical rows: `32`
- Draft vocabulary: full (`ROOT=0`, `K=0`)
- Candidate:
  `streamk_force_wide256_byte_ab`
- Candidate binary SHA-256:
  `f7d5c01ca79829fbfff4c93949d057bd740905165b0b6793b3c0007629add962`
- Raw comparator SHA-256:
  `eef1ae90dfe0bb3377f9e32dd3c4b7b2029b192f4f86c6fdc274d6ec54634233`

The authenticated engine ledger contains one accepted and one completed
request for the canonical task. After one complete 64-layer target forward,
the engine raised `FR13 fixed32 KV16 compact/full row-map drift` before the MTP
projection. The outer SWE runner subsequently returned 15 because its
completion metrics could not reconcile after the engine failure, so the formal
live-gate reducer did not run. That does not weaken the kernel rejection: all
comparator records carry the authenticated real-task marker, and every
observed stock/candidate output differs.

## Byte result

| M | N | K | Comparisons | Differing bytes |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 5,120 | 6,144 | 64 | 1,314 |
| 32 | 5,120 | 17,408 | 64 | 1,892 |
| 32 | 16,384 | 5,120 | 48 | 2,164 |
| 32 | 34,816 | 5,120 | 64 | 6,141 |

All 240 comparisons failed byte equality. There were 11,511 differing bytes
among 234,881,024 compared bytes, or 49.008 parts per million. Per-output
mismatch counts ranged from 11 to 123 bytes, and every first mismatch offset
was even, consistent with a changed low byte in a BF16 element. The comparator
did not retain raw tensors, so exact ULP deltas cannot be reconstructed. The
fifth expected projection shape was not reached before the outer runner
stopped, but one mismatch is sufficient to reject the candidate.

## Kernel diagnosis

The candidate changes the stock small-row tile from `128x32x128` to
`256x32x128` and forces Stream-K decomposition. CUTLASS deterministic reduction
makes the split partial-sum order repeatable; it does not preserve stock's
single-CTA FP32 accumulation order. The sparse BF16 differences are consistent
with that reassociation. This binary must not be timed or promoted.

The next kernel gate is a no-K-split persistent/data-parallel candidate,
followed by the same real-task byte comparison. A genuine K-split Stream-K
candidate cannot satisfy the strict stock-byte contract in general.
