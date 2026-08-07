# B1 K64 static-FP8 real-task gate PASS (recovered 2026-08-07)

This bundle records the passed integrity gate for the 2026-08-05 real
SWE-Verified B1 diagnostic of the static-I/O block-FP8 K64 draft head at
source commit `3967fb29cc69335ac930b522f2a5dea20ce11aa0`.

The run executed 2026-08-05 22:19:13Z to 22:32:05Z in
`output/fr13_b1_k64_fp8_static_3967fb29c_20260805a`, arm
`hydra27_fixed32_k64_root_3967fb29c_20260805a`. The real
`astropy__astropy-12907` task resolved (harness exit 0, agent exit 0, no
timeout, no network drop) and the deployment measurement completed and was
written to `draft_head_fp8_acceptance.json`.

The original post-run gate nevertheless crashed. It did not fail on evidence:
`scripts/fr13_draft_head_fp8_gate.py` imports its traffic validation from
`scripts/fr13_draft_head_m32_pass.py`, which was still pinned to the v2 chat
traffic audit contract while the writer `scripts/fr13_floor_gate.py` had moved
to v3. The validator was reconciled with the v3 writer in commit
`8d89e064781a9afcf0362592854bc18f5764947c` on 2026-08-07 (check-key rename,
three new `successful_engine_requests_*pure_decode*` census keys, the
`-audit-v3` schema pin, and the now-required `concurrency=1` argument).

The gate was then re-run on 2026-08-07 over the **unmodified** 2026-08-05
evidence tree and returned `status: PASS` (schema
`fr13.fixed32.draft_head_fp8_real_b1_gate.v2`). No measurement was repeated
and no evidence file was rewritten; only the gate result was added to the
arm directory and the resolution lines were appended to
`fp8_real_task_scope.txt`. The `gate_completed_by` line in that file records
that the verdict was produced by this salvage, not by the original runner
process.

The gate confirms the FP8 candidate served the exact Hydra27 physical32
K64/root1 B1 route with one selected root call, four captured loop calls,
direct FP8 proposal logits, zero BF16 shadow calls, zero fallback calls, zero
steady-state synchronizations, static preallocated raw-out activation I/O, and
unchanged verifier/committer rejection-sampling authority.

The run measured, over 849 events:

- 216.7480 ms full-step wall time
- 28.0233 ms drafter GPU time per step
- 152.7879 ms verifier/SFWD GPU time per step
- 19.7542 ms committer GPU time per step
- 16.1826 ms other overhead per event
- 23.1878 full-step wall TPS
- 4.0259 accepted drafts per event

Against the matched exact4 reference slice, DFWD improved by 7.0739 ms
(20.16%) and the full step improved by 7.2236 ms (3.23%). The candidate's
216.7480 ms step still sits 86.2069 ms above its 130.5411 ms one-sided u95
acceptance cap, a floor ratio of 1.9094 against the 113.5140 ms mandatory
weight-read lower bound.

This is an integrity gate
(`classification: one_real_swe_verified_b1_integrity_gate`). It attests that
the candidate genuinely served the route on a real resolved task with
authenticated request provenance. It is explicitly **not** timing acceptance:
`performance_tuning_eligible` is false and `floor_acceptance_eligible` is
false. The timing numbers above are directional evidence only, and the
candidate remains far above its acceptance cap.
