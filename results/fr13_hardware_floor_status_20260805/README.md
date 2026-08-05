# Hydra27 hardware-floor status

Snapshot date: 2026-08-05

Authoritative code snapshot: `a61594d530c88129e94aebbb9b66012621066fda`
on pushed `main`.

This document separates code that is landed, host/offline verification, real
SWE-Verified evidence, and work that is still pending. A merged selector or a
passing unit test is not a performance result.

## Fixed target

- Hydra27, K64 drafter vocabulary, root enabled.
- 32 physical rows per request, with the Hydra27 mask `0x7abdffff`.
- B1 is concurrency one. B4 is four distinct canonical SWE-Verified tasks at
  concurrency four.
- Corrected mandatory-weight floor: `119.658015414 ms/step`.
- Acceptance cap: one-sided U95 at or below `1.15x` the floor,
  `137.6067177261 ms/step`.
- Exact4 is the screening gate. Exact16 is required only after exact4 clears
  the cap.

## Status by evidence class

| Item | Status | What the status proves |
| --- | --- | --- |
| Five-kernel B1 composition | `MERGED` | Qrow32 split2, GQA-group3, mapped K64 top3, wide256 full-grid target GEMM, and SFWD conv/post-prep are wired behind fail-closed production contracts. |
| Direct CFWD six-way composition | `MERGED` | TAW production and logit-direct CFWD can join the exact five-kernel tuple; near-miss tuples remain rejected. |
| Host/static verification on merged tree | `PASS` | 52 focused composition/CFWD tests passed; Python compilation and shell syntax passed. |
| CFWD offline code generation | `PASS, HISTORICAL SOURCE ARTIFACT` | Both SM121a kernels compiled without spills in the published source artifact. This does not prove live engagement or speed. |
| New composed real-task byte gate | `PENDING` | No final-HEAD one-task credential exists for the merged stack. |
| New composed B1 exact4 timing | `PENDING` | No valid full-step wall/TPS or phase breakdown exists for the merged stack. |
| New composed B1 exact16 acceptance | `BLOCKED BY EXACT4` | It must not run until exact4 U95 clears the cap. |
| Current-stack B4 byte gate and timing | `PENDING` | No B4 hardware-floor claim is valid for this merged stack. |
| Hardware-floor acceptance | `NOT MET` | Neither B1 nor B4 has current-stack evidence at or below the cap. |

## Latest valid B1 measurement

The latest valid point remains the earlier qrow16 Hydra canonical exact4 arm,
not the newly merged stack:

| Metric | Value |
| --- | ---: |
| Full-step wall | `232.779790071 ms/step` |
| Full-step wall TPS | `24.718146718` |
| Accepted drafts per event | `4.753885004` |
| Committed tokens per event | `5.753885004` |
| SFWD GPU | `159.619263244 ms/step` |
| DFWD GPU | `36.813368134 ms/step` |
| CFWD GPU | `20.677390557 ms/step` |
| Host and unattributed residual | `15.669768137 ms/step` |
| Floor ratio | `1.945375655x` |
| Gap to acceptance cap | `95.173072345 ms/step` |

Closing the cap from this point requires a `40.885453%` wall-latency
reduction. The point is `113.121775 ms/step` above the weight-read floor, so a
single small launch reduction cannot close the goal.

The source artifact is
`results/fr13_fixed32_qrow16_prod_exact4_b1_20260731T182827Z/`.
Its Tail arm was invalid and is not acceptance evidence.

## What is landed but not yet real-task verified

The merged B1 path now has:

1. A combined FULL-graph Gate A for Qrow32 split2, GQA-group3, and mapped K64
   top3.
2. A combined eager Gate B for the wide256 full-grid target GEMM and SFWD
   conv/post-prep fusion.
3. Source-bound TAW and logit-direct CFWD credential validation.
4. A one-task FULL-graph six-way production smoke wrapper.
5. A candidate-only canonical exact4 wrapper with wall TPS and
   SFWD/DFWD/CFWD timing reduction.

These are execution paths, not completed gates. The earlier Gate A attempts
did not reach a real task: two stopped at patch-time environment validation,
and the final attempt was interrupted during model loading. They produced no
health record, credential, or timing and are not evidence. The missing Docker
environment forwarding was fixed in `51ff48f9a` and `aacc361a3` before the
six-way merge.

## Next valid campaign

All steps must use one clean, pushed, frozen source commit:

1. Run combined Gate A and issue Qrow32, GQA-group3, and top3 credentials.
2. Run combined Gate B and issue target-GEMM and SFWD credentials.
3. Run the Hydra27 source-v7 TAW B1 gate, the required reviewed B4 TAW gate,
   and merge their source-bound production credential.
4. Run the real one-task CFWD logit-direct byte/product gate.
5. Run the real one-task FULL-graph six-way production smoke and prove every
   candidate engagement marker and the CFWD served return.
6. Run canonical B1 exact4 and report full-step wall TPS, GPU-component TPS,
   acceptance, SFWD, DFWD, CFWD, residual wall time, point floor ratio, and
   one-sided U95.
7. Run B1 exact16 only if exact4 U95 is at or below
   `137.6067177261 ms/step`.
8. Repeat the byte, exact4, and conditional exact16 ladder for B4 using four
   distinct tasks at concurrency four.

No raw prompts, responses, patches, benchmark workspaces, credentials, or
unsanitized run logs should be committed with the resulting evidence.
