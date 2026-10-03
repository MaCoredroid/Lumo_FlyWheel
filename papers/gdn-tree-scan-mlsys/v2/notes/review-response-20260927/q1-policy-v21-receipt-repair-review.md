# Q1 policy v2.1 receipt-mask repair: bounded closure

Reviewed 2026-09-27 local time against immutable snapshot `v2/p0/monitor/review-response-20260927/repair-source-snapshot-20260928T000400Z/`.

**Verdict: the previously reproduced altered-mask false PASS is closed in this exact evaluator snapshot.** This closes the standalone-evaluator finding in `q1-policy-v21-repair-review.md`; it is not final freeze or GPU launch approval.

| Inspected artifact | SHA-256 |
| --- | --- |
| Snapshot `tools/q1_policy_evaluator_v2_1.py` | `2bdf5b6bda6e470820d7e779fe0217915f67fdd048fa8fef221ff186810190a5` |
| Snapshot imported `tools/q1_policy_evaluator_v2.py` | `a65f492b0b576221de328852c84b5b367a89dca2d542b9dde7062c410ed3ff0f` |
| Campaign `policy/q1_component_numerical_policy.v2.1.json` | `4a103df013439ae02977d10a42c3cf2eeb916a7dcb1a61c19e6b929da620f7b3` |
| Campaign `fixtures/q1_2b/expected_observations.json` | `c5df3aefd4b838ef2098b03cfc54775b1cb467510c5956edb9e868c725ef7448` |

The evaluator now reconstructs candidate-blind reference eligibility from the final record's reference projection and compares its mask and digest with the supplied receipt before applying the paired rules (`tools/q1_policy_evaluator_v2_1.py:363-368`).

I reran the same three standard-library-only local CPU probes from the preceding review against the actual bound ordinary-random L0/node0 expected case. Approval was supplied only in memory for the test; no policy or gate file changed.

| Probe | Result |
| --- | --- |
| Finite native/C2 reference; candidate metrics within paired bounds | `EVALUATED / PASS`, 48 PASS heads |
| Native RMS and max-error arrays set to positive infinity; unmodified ineligible receipt | `EVALUATED / UNCOVERED`, 48 UNCOVERED heads |
| Same nonfinite reference; change only receipt mask from REF_INVALID to OK | `REFUSED_NO_MATCHING_ELIGIBILITY`, no numerical verdict |

The refusal reports: `eligibility receipt mask/digest does not match a fresh reference-only recomputation`.

The actual policy's constants and rules remain equal to the reviewed v2 policy, including kappa `1.10`, `u32=2^-24`, `eta32=2^-149`, both RMS and max criteria, and no added bf16 allowance. The stored policy remains `approved: false`; immutable external approval semantics are unchanged.

No remote access, GPU execution, new experiment, implementation edit or numerical-scope expansion was performed. The parent still must bind and approve the final complete freeze and its test/runner/reducer evidence.
