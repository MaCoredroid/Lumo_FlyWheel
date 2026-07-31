# FR13 fixed32 qrow16 num_splits=0 live-paged PASS

This artifact packages the completed real SWE-Verified B1 correctness run at
`output/fr13_b1_qrow16_num_splits0_live_gate_20260731T172755Z`.

The qrow16 candidate SO is
`1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86`
(299,507,792 bytes), built from commit
`f5c4d0b2d841328f010dbfc652f1708955756d45`. The repair changes the exact
qrow16 dispatch guard to the value actually left by FA2 varlen setup for
32 query rows: `params.num_splits == 0`. The rejected predecessor incorrectly
required `params.num_splits == 1`.

## Live result

The same EngineCore and CUDA boot that served the real task retained the first
observed fixed32 FULL tree-attention operands and recalled stock and candidate
FA2 on them. The exact live geometry was:

- query `[32, 24, 256]` BF16;
- paged K/V `[639, 1024, 4, 256]`;
- block table `[1, 128]`, sequence length 22,901;
- FP32 tree bias `[32, 32]`.

Every one of 393,216 BF16 output bytes and 3,072 FP32 LSE bytes matched stock.
The candidate dispatch was required to engage; geometry drift would have
raised. The captured stock output was returned to the request, so this run did
not serve candidate output and is not a candidate-performance measurement.

`fr13_fa2_qrow16_production_pass.json` is the canonical production sidecar
issued from the exact live-result digest and exact candidate-SO digest. Its
scope is only the qrow16 exact-target tree-attention path in fixed32 B1 FULL.

## Real-task lifecycle

The single real task `astropy__astropy-12907` resolved with harness exit 0 and
tests passed. The engine finalized 810 complete fixed32 decode events covering
forward steps 0 through 809. Final flush reported zero pending DFWD, CFWD, and
SFWD work. Engine and proxy ingress each completed all 13 accepted requests
with no campaign rejects, aborted requests, or failed attempts.

The launcher recorded `ARM_DONE ... swerc=0`, removed the run container, and
the post-teardown GPU compute-process query was empty.

## Scope

This is a real B1 SWE-Verified correctness result and the qrow16 live gate is
PASS. It is still a one-task `b1_diagnostic` run with
`gate_eligible=false` and `floor_acceptance_eligible=false`. It contains no
TPS, kernel-time, speedup, acceptance, or hardware-floor claim. The next gate
is a real B1 run with the attested production selector serving this exact
candidate, followed by the standing real-task performance campaign.
