# FR13 fixed32 SFWD B1 gate readiness

Status: **source-ready; real B1 run not started**.

Commit `9f178d14ee83b407e78c0fce41624513981c6e58` wires the default-off
SFWD state-fusion candidate to a real SWE-Verified, full-vocabulary B1
diagnostic. The runner is
`scripts/fr13_run_b1_sfwd_state_fusion_gate.sh`.

The route is pinned to `astropy__astropy-12907`, `MAX_NUM_SEQS=1`, one
32-row physical tree, `FR13_DRAFT_VOCAB_K=0`, and
`FR13_DRAFT_VOCAB_ROOT=0`. It disables every other diagnostic candidate and
requires the incumbent conv writeback, tree-conv, ring-export, freshness-flag,
and two-launch GDN contracts.

The engine ingress middleware creates the read-only real-event marker only
after the request identity passes fixed32 authentication. Rejected traffic
cannot arm the marker. The fused kernel runs in shadow, compares `conv_out`
and `commit_source_stage` bytes across all 48 layers, and always returns the
incumbent tensors. Its one conv/state launch does not alter the existing GDN
program schedule `[1, 11]` or its two physical launches per layer.

All B1 outputs explicitly carry `timing_eligible=false` and
`floor_acceptance_eligible=false`. The full-vocabulary mandatory-weight floor
bound is 153.938384645 ms/step and the recorded 1.15x cap is
177.02914234175 ms/step, but this source-readiness artifact contains no timing
sample and cannot be used for acceptance.

No GPU was used while preparing this route. No synthetic/probe workload and no
exact4 or exact16 campaign was launched. The focused CPU/source suite completed
197 tests successfully.
