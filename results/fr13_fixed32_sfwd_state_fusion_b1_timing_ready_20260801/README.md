# FR13 fixed32 SFWD B1 timing readiness

Status: **source-ready; waiting for a new real-task live PASS**.

The source-gated timing runner is
`scripts/fr13_run_b1_sfwd_state_fusion_timing.sh`. It is pinned to the real
SWE-Verified task `astropy__astropy-12907`, B1/concurrency 1, K=0/root=0, and
32 physical rows. It runs the stock arm first and the SFWD state-fusion arm
second with the same FA2 binary and records full-wall deploy-speed plus raw
SFWD, DFWD, and CFWD timers.

The candidate arm cannot boot without a regular live PASS whose raw SHA-256 is
provided by the caller and whose `source_sha256` equals the running
`fr10_gdn_tree_kernel.py`. That qualified kernel file remains byte-identical to
the stable gate branch; production selection lives in a separate module. The
runtime validates the installed PASS again. The candidate then reads the col-0
prior directly, writes the served conv output
and the persistent 36-row accepted-path commit source in one launch, and skips
both the prior-window pregather and incumbent per-request SFWD loop. A separate
attestation must observe 48 unique served layers before the pair can reduce.

This is a one-task diagnostic. Every output remains
`timing_eligible=false`, `floor_acceptance_eligible=false`, and
`production_eligible=false`. The recorded 153.938384645 ms/step bound is the
optimistic full-vocabulary mandatory-weight-read floor, not a measured
full-step hardware floor. Exact4 or exact16 real-task timing is still required
for any acceptance conclusion.

No GPU, Docker container, probe workload, exact4 campaign, or exact16 campaign
was run while preparing this artifact. The focused source suite completed 10
tests successfully; Python compilation, shell parsing, and `git diff --check`
also passed.
