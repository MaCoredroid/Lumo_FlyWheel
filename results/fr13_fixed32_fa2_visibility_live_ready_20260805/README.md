# Fixed32 FA2 visibility live-gate readiness

Status: `READY_FOR_REAL_BYTE_GATE`, not byte-qualified, not timing-qualified,
and not production-enabled.

Source commit `154d6877b9e33d71b7c07a543d3cf339b448a307` adds an explicit,
default-off `visibility` selector to the existing real SWE-Verified B1 and B4
byte gates. The live gates serve the incumbent result and compare candidate
BF16 output and FP32 LSE bytes across all 16 tree-attention layers.

## Candidate identity

| Route | Final SO SHA-256 | Bytes | FA2 source closure |
| --- | --- | ---: | --- |
| Hydra27 B1 physical32 K64/root1 | `c5ab32a6ae4e615f1e77a4997db5429152053c549e761fb11d90b33bb3959a79` | 300200192 | `a30eca031cd5067133e6278527787c5987635670930e5840ac983f66b088e4fc` |
| Hydra27 B4 physical32 K64/root1 | `805635d6881dbf73287d66c10541880b7cf93bcb6bf7b04e50efd3d32728b0aa` | 299810632 | `1dac8f7fd910a564c5c3b792770029f0013e2df48c25c89376e4d5e7da949ced` |

Both closures are rooted at FA2 commit
`29210221863736a08f71a866459e368ad1ac4a95`. The final SOs passed the
existing host-only ABI finalizer with 856 dynamic symbols and 10 needed
libraries. No GPU or Docker command was used for this readiness step.

## Scope invariants

- B4 defaults to the existing `qrow32` arm. B1 defaults to existing `split2`.
- B1 production remains restricted to the existing `nosplit` arm and binary.
- qrow16 and B1 split2 source and dispatch are unchanged.
- Visibility is admitted only for Hydra27, physical32, K64, root reduction on.
- B1 visibility uses `num_splits=0`; B4 visibility uses `num_splits=0`.
- The real byte gates are diagnostic only and always serve incumbent output.

## Static result

The GCC11 SM121a visibility objects use 252 registers, zero stack, zero local
memory, zero spills, and one barrier. Relative to the dense tree-bias load
kernel, exact global loads fall from 68 to 4 while constant loads rise from 68
to 72. B1 SASS size falls from 4008 to 3640 instructions; B4 falls from 3992
to 3632 instructions.

The next admissible evidence is the real SWE-Verified B1 byte gate followed by
the real exact4 B4 byte gate. Their commands are pinned in `commands.txt`.
