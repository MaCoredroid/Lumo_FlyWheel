# Q1 numerical policy v2.1 repair review

Date: 2026-09-27. Scope: local CPU review of the repaired policy builder/evaluator, existing tests and the narrow reducer call site needed to assess reference-receipt handling. This is an **unfinished-package source review, not freeze or launch approval**. No GPU work, remote mutation, implementation edits, policy approval or gate changes were performed. Synthetic metric records used below are evaluator test inputs, not candidate measurements.

**Bounded verdict:** the prior v2 R1–R4 problems are repaired for the inspected pipeline: candidate-blind reference evidence, mandatory reference/specification digests, a prospective exact Q1.2b manifest, and external approval without changing policy bytes. The numerical equations and constants remain unchanged. One remaining standalone-evaluator receipt-consistency weakness is reproduced below; the inspected reducer recomputes the eligibility mask and does not permit that weakness to produce a numerical PASS. Final approval still requires the completed immutable freeze, its exact source/test bindings and the separate runner/reducer review.

## Snapshot

Paths are relative to `v2/experiments/review-response-20260927/`.

| File | SHA-256 |
| --- | --- |
| `tools/q1_policy_evaluator_v2_1.py` | `a53a5c482d831bed75a3b7b776a2bfb8cdb22461d84e4215253632815ea16a22` |
| `tools/q1_policy_build_v2_1.py` | `754ddef4775ef300d05883433a8a4cf65e4ad45bdb494772f088e421fd302f08` |
| `policy/q1_component_numerical_policy.v2.1.json` | `4a103df013439ae02977d10a42c3cf2eeb916a7dcb1a61c19e6b929da620f7b3` |
| `tools/tests/test_q1_2b_v2_pipeline.py` | `1233569d8753182974e575e148fa23e38cd392580ac5e6300e8d2a39e9876d75` |
| `tools/q1_2b_reduce_v2.py` (call-site check only) | `68aac40e04362095435c23c8a658e51c1b732dfc8b0a393437747c9211966b0f` |
| `fixtures/q1_2b/manifest.json` | `2dc0a42921e6e9eda9d5b636060a246b6d4af98b88a3b2f78c9a0dbc61197522` |
| `fixtures/q1_2b/expected_observations.json` | `c5df3aefd4b838ef2098b03cfc54775b1cb467510c5956edb9e868c725ef7448` |

The reviewed base v2 bytes remain `ce307eed20369c8e7789b9d2b704d68b228595bd2fc7ae1c2fb454dcaa47f61d`. Actual base, fixture-manifest and expected-observation file hashes match v2.1's bindings. The v2.1 canonical policy hash recomputes correctly. Fixture-manifest canonical payload independently recomputes to `8b1912905a112e7c96f3ffe40a26e1fb79e0646b0b6e0b14c7f4c36cfea787b9`; expected-observation canonical payload is verified by `bind_expected_manifest`.

The manifest contains two calibration and two evaluation fixtures, each with 48 synthetic operator instances, 32 physical rows, 16 key heads and 48 value heads of dimension 128. The expected manifest declares 5,760 calibration cases and 5,760 evaluation cases; these counts recompute from its case list. Reading evaluation metadata here does not inspect held-out numerical outcomes.

## Repairs verified

- **Reference-only phase:** evaluator lines 183–251 define and validate a C1/C2 projection; lines 254–286 issue a receipt without C0. Candidate fields in the reference phase are rejected. Explicit boolean `reference_valid`, native SHA and C2 SHA are required. The C2 SHA must match the expected case. Later C0 insertion preserves the reference digest; a changed native hash/reference metric invalidates it.
- **Mandatory bindings:** lines 321–342 require a trusted expected entry, its specification digest, policy SHA and non-null reference digest for a usable receipt. The old null-digest bypass is closed. A missing or altered reference/specification digest refuses evaluation.
- **Strict domain:** lines 97–179 validate canonical integer depths, exactly 48 heads, known surfaces and required case fields, then bind the expected manifest's actual/canonical identity to the prospective policy. Mixed-stress is now explicitly declared, and an unlisted case is refused. Valid single-surface/single-depth cases remain supported.
- **External approval:** lines 51–93 separate integrity from authorization. Missing approval, a wrong policy SHA or wrong stage refuses authorization. A valid external policy-SHA/stage record authorizes the in-memory loader while the on-disk policy's `approved` remains false. This function is not the one-run launcher gate; the completed package must retain the independent run/block/gate bindings.
- **Fixed aggregation:** lines 366–379 reject duplicate, missing and unexpected case IDs against a fixed expected set.
- **Equations unchanged:** lines 304–305 retain both paired RMS and max criteria with kappa 1.10 and floors `2^-24 * scale + 2^-149`, including no extra bf16 allowance. All ten sections in builder `IDENTICAL_SECTIONS` compare equal to the reviewed v2 policy: constants, rules, depth definition, evidence contract, identities, contract, power witnesses, observation product, claim and diagnostics.

The builder verifies a supplied base SHA, preserves the base, adds actual new manifest file hashes and constructs a new canonical hash. For this concrete package, its copied manifest relationships were independently verified above. The builder itself checks stored manifest canonical labels for equality rather than recomputing both; mandatory base SHA and canonical recomputation would strengthen future regeneration, but this is not a demonstrated defect in the inspected, independently verified bytes. No new numerical parameters or calibration were requested.

## CPU probes and tests

Direct local CPU probes loaded the actual v2.1 policy and actual expected manifest, granted approval **only in memory**, selected its ordinary-random L0/node0 case, and supplied handcrafted finite metric arrays. Results:

| Probe | Outcome |
| --- | --- |
| C1/C2-only reference record, then insert C0 | Eligible; numerical PASS |
| Change C0 alone within the fixed paired bounds | Same reference eligibility; PASS |
| Delete reference digest or specification digest | Refused, no verdict |
| Include C0 fields in reference phase | Malformed, no verdict |
| Omit reference validity or native SHA | Malformed, no verdict |
| False reference validity | Ineligible; no PASS |
| Negative, boolean or truncated reference metrics | Malformed, no verdict |
| Boolean heads, string depth, fake surface, changed specification, unlisted fixture | Malformed, no verdict |
| Change bound native SHA or reference metric after preflight | Malformed, no verdict |
| Negative/boolean candidate metric or omitted `metrics_valid` | Malformed, no verdict |
| NaN candidate metric | Numerical FAIL |
| Bound mixed-stress case | Eligible with valid synthetic reference evidence |
| Nonfinite native metrics with unmodified receipt | Ineligible; UNCOVERED |
| Duplicate result IDs in fixed aggregate | Refused/malformed |
| Missing/wrong-SHA/wrong-stage approval | Unauthorized |

The unchanged v1/v2 policy CPU suites were rerun with bytecode/cache writing disabled: **42 passed in 3.93 seconds**. These are regression checks, not a claim that v2.1's complete pipeline suite ran locally. The new `test_q1_2b_v2_pipeline.py` imports torch and creates runner/reducer fixtures; local torch is unavailable. Its policy tests were inspected, and the direct v2.1 probes above independently exercise the repaired branches. Full pipeline test execution belongs to the package's final source-bound test receipt.

## Remaining standalone receipt-consistency weakness, with scope

`evaluate_case_v2_1` does not require `eligibility.eligible is True` or independently reconstruct its `per_observation` mask before `_paired_rules` uses it (lines 296–305). Changing only the mask of an ineligible receipt can make a nonfinite native reference reach an infinite comparison bound and return PASS. The unchanged reference projection digest does not cover that derived mask.

A minimal reproduction using the actual bound case is:

```python
# E = q1_policy_evaluator_v2_1; P is the actual hash-verified policy,
# approved only in memory; B comes from bind_expected_manifest.
c = B['calibration/fx0_ordinary-random|L0|out|node00']
r = dict(case_id=c['case_id'], stratum=c['stratum'],
         surface='output', depth=1, reference_valid=True,
         native_sha256='a' * 64, c2_sha256=c['reference_sha256'],
         rms_C1_C2=[0.01] * 48, maxabs_C1_C2=[0.02] * 48,
         rms_C2=[1.0] * 48, maxabs_C2=[2.0] * 48)
def final(r):
    return dict(r, metrics_valid=True,
                rms_C0_C2=[0.01] * 48, maxabs_C0_C2=[0.02] * 48)
el = E.reference_only_eligibility(P, c, [r], bound_cases=B)
assert el['eligible']
assert E.evaluate_case_v2_1(P, c, [final(r)], el,
                          bound_cases=B)['verdict'] == 'PASS'
r.update(rms_C1_C2=[float('inf')] * 48,
         maxabs_C1_C2=[float('inf')] * 48)
el = E.reference_only_eligibility(P, c, [r], bound_cases=B)
assert el['eligible'] is False
assert E.evaluate_case_v2_1(P, c, [final(r)], el,
                          bound_cases=B)['verdict'] == 'UNCOVERED'
el['per_observation'] = {'output|d1': ['OK'] * 48}
assert E.evaluate_case_v2_1(P, c, [final(r)], el,
                          bound_cases=B)['verdict'] == 'PASS'  # weakness
```

**The inspected reducer guards the scientific verdict.** `q1_2b_reduce_v2.py:396-405` recomputes reference eligibility from the sealed reference record, compares the recomputed evidence digest and `eligible` flag with the stored receipt, and passes the **freshly recomputed** receipt to evaluation. Altering only the stored mask therefore cannot make this reducer emit a numerical PASS: it uses the fresh invalid mask and produces UNCOVERED. If the stored `eligible` flag is also changed to true, its comparison fails. The stored mask is not checked for consistency, but is also not used for the result. This is not an end-to-end qualification bypass reproduced against the current pipeline.

Recommended narrow repair: recompute or validate the receipt's derived eligibility mask/status from its bound reference projection inside the evaluator, and directly prevent nonfinite C1/C2 metrics from reaching the paired-rule comparison. Reject contradictory receipt fields; retain UNCOVERED for invalid references. Add the finite control plus altered-mask regression above. No bound or tolerance changes are needed. Comparing the full derived receipt fields in the reducer would also detect malformed archived masks.

The evaluator's `fixture_sha256` receipt field is likewise not independently checked there. The inspected reducer checks the execution fixture SHA against its manifest at line 271 and requires matching fixture bytes for C2 regeneration at lines 459–470. This is a layered input binding, not evidence that wrong fixture bytes can pass the current recomputing pipeline. Preserve those checks in the final freeze.

## Final-freeze boundary

This source review does not authorize a GPU job. Bind the finished source snapshot, immutable test log, policy, expected manifest, fixture manifest and external one-run gate in the final freeze. Keep raw-tensor metric recomputation enabled and reference eligibility recomputed in the reducer. The parent and separate runner/reducer reviewer must confirm actual C1/C2-before-C0 execution and complete process/repeat/negative coverage. The policy evidence supports only the predeclared finite-case paired non-regression criterion; no candidate result or broader Q1 claim follows from these CPU checks.
