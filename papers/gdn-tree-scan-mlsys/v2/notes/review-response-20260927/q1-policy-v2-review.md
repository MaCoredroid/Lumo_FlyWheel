# Paired numerical policy v2: independent executable review

Reviewed 2026-09-27. Scope: CPU-only audit of the frozen policy, builder, evaluator, accepted-native provenance, power witnesses and the reference-eligibility integration. No GPU execution, remote file changes, source edits, gate changes or candidate outcomes were used in this review.

**Verdict: the paired numerical criterion is acceptable for its stated finite-case non-regression claim, but the frozen executable package is not ready to authorize Q1.2b.** Repair the reference-only phase, bind the stage-specific expected/fixture manifest, and resolve the immutable approval/freeze flow. These are implementation/provenance repairs; do not change kappa, floors or the two required metrics. The component reducer has separate blocking findings owned by the independent reducer reviewer.

## Reviewed snapshot

Paths below are relative to `v2/experiments/review-response-20260927/` unless noted.

| Artifact | SHA-256 |
| --- | --- |
| `FREEZE-Q1-POLICY-v2.json` | `eabe768a8c3197ff0672dcde9fafa5881e5f36fe2b34ea14840b4073eaf2b75a` |
| `policy/q1_component_numerical_policy.v2.json` | `ce307eed20369c8e7789b9d2b704d68b228595bd2fc7ae1c2fb454dcaa47f61d` |
| `policy/power_witnesses.v2.json` | `3486dc271f62756d86765fed43ca4104d6a60b54fbb26cba7fc5135d7f708f6e` |
| `policy/accepted_native_run.v1.json` | `4dbd9c9ae1f94b28ae09f76cec0a5cfb0fe865cbba6d092e99b62173b7c79ced` |
| `tools/q1_policy_build_v2.py` | `606abdf2c6538a429117cee271c73bd64ff81c05464655f66d7aeb675273b834` |
| `tools/q1_policy_evaluator_v2.py` | `a65f492b0b576221de328852c84b5b367a89dca2d542b9dde7062c410ed3ff0f` |
| `tools/q1_power_witnesses_v2.py` | `e1226ab12ea456e34c2a527819c53554c737e7c06217771ac8fc207df1f71103` |
| `tools/q1_bf16_ulp.py` | `b6158c4de483dc014c52df8b1a9ad9158aa8a25a5810c749a4a172325a12e00c` |
| `tools/q1_oracle.py` | `645944992d116115976eea5a925f1ebf9a23562d8d8a3ebd693116daa0ec5d82` |
| `tools/q1_policy_build.py` | `39283016f2d56c0a88bff541df5f2ae137b11806e0564677f11ce1ae3f41bc61` |
| `tools/q1_policy_evaluator.py` | `84830b2a9f4bcacc6344c2762af027a178bd3267e369d41fde06cd3a08f332d6` |

The prospective contract at `v2/p0/monitor/review-response-20260927/Q1-PAIRED-NUMERICAL-CONTRACT-v2.json` has SHA `3e4fd09b165f8d1f1a122a441ae74ebcb6dfce206bc655659b90266b5b6a3b50`. The policy's stored canonical hash excluding its timestamp is `365c8953b9d9c7d45a59d28985e4e6eef773824636781c1e03eeea751b74cfbe`.

The freeze is timestamped `2026-09-27T22:10:09.540431+00:00`. Its other inspected local file bindings matched, but its test-log binding does not: frozen `tools/test_log.txt` SHA is `62f7c374be7a186416c76726fdc00733efc8080a27bc135bbe5749a28c31fbca`; current bytes hash to `4e0126a208dee4f4e6a8452cf0842abaee501f1423138da29637832eabd6930a`. Preserve the prior snapshot and use an immutable version-specific test log and new freeze for the repair.

## Accepted criterion and improvements

For each required case, state/output surface, head and materialized depth, let R be the independent float64 C2 reference, N the specified native C1 on the same pristine operands/path, and Y the candidate C0. Require both:

```
RMS(Y-R)    <= 1.10 * RMS(N-R)    + 2^-24 * RMS(R)    + 2^-149
maxabs(Y-R) <= 1.10 * maxabs(N-R) + 2^-24 * maxabs(R) + 2^-149
```

The evaluator implements these equations at `tools/q1_policy_evaluator_v2.py:167-168`, without pooling heads or using an extra bf16 ULP allowance. Applying the same fp32 arithmetic floor to bf16-stored outputs is a defensible predeclared operational test: native error already contains native bf16 storage rounding. This is not a universal error bound, proof of mathematical equivalence, population statement or full-model logit tolerance. Keep raw native-versus-C2 errors visible. The old two-sample absolute caps and scale ranges are correctly diagnostic only.

The v2 builder meaningfully improves source integrity. `tools/q1_policy_build_v2.py:28-68` verifies the accepted native run's actual A/B result, summary, run receipt and fixture-manifest bytes, runtime/operator/oracle identities, calibration block and fixture IDs. The repaired v1 builder invoked there validates the observation product and metric arrays. This closes the prior practice of merely recording asserted source hashes. The actual frozen contract and accepted native receipt are consistent with the intended source. The builder embeds witness data at line 111; that embedding alone is not validation, so the independent witness checks below remain material.

## Required repairs

### R1. Make eligibility genuinely reference-only and execute it before C0

`reference_eligibility` says it never reads C0 (`tools/q1_policy_evaluator_v2.py:111-112`), but line 122 calls the common validator, whose lines 85-103 require and inspect all six metric arrays, including C0. Line 135 then hashes the entire candidate-containing record. Consequently:

- Removing the two C0 arrays from otherwise valid reference evidence returns `MALFORMED_EVIDENCE`.
- Establishing eligibility with placeholder candidate arrays and then inserting actual C0 arrays returns `REFUSED_NO_MATCHING_ELIGIBILITY`.
- Removing `reference_valid` still allows `PASS`, because it is optional at line 82 and defaults true at line 131.

The existing integration does not supply the promised phase ordering: `tools/q1_component_runner.py:36` imports only the policy loader, and `tools/q1_2b_reduce.py:117-119` invokes eligibility on records already containing candidate results. A post-hoc reducer check is not the executable pre-C0 gate.

**Minimal repair:** introduce a strict reference schema and reference-only digest. Bind case/fixture/input/path/source identities, the trusted expected-case specification, explicit boolean reference validity, the four C1/C2 metric arrays, and their raw/reference hashes. Exclude all C0 arrays and candidate validity fields. Construct and validate C1/C2 first, save the reference eligibility receipt, and only then execute C0 for eligible cases. Later candidate evaluation must require the same reference digest and case specification. The reducer independently recomputes this binding. Missing or malformed reference validity must not default to success.

Tests must demonstrate that a record containing no C0 fields can become reference-eligible; inserting valid C0 fields preserves that eligibility; altering any bound reference/input/path/specification field invalidates it; and an ineligible reference prevents the candidate function from being called.

### R2. Bind and strictly validate the expected-observation manifest

`tools/q1_policy_evaluator_v2.py:142-143` permits an eligibility digest of `None`. Deleting the digest from a usable eligibility receipt and changing the candidate record can still produce `PASS`. Require a digest for every usable eligibility receipt; an unsupported-domain result may report `UNCOVERED`, but must not provide a reusable unbound authorization.

The trusted manifest must also be validated rather than treated as arbitrary caller input. Current lines 62-64 accept boolean heads and coerce depth strings; lines 79-81 make record identities optional. The expected-case specification itself is not digest-bound. Executed CPU reproductions found:

| Modified input | Current outcome |
| --- | --- |
| `heads=True`, one-element arrays | `PASS`, two head/surface outcomes |
| Expected depth `"03"`, record depth `3` | `PASS` |
| Expected and observed surface `madeup_surface` | `PASS`, 48 outcomes |
| Undeclared fixture ID and wrong input SHA under a recognized stratum | `PASS` |

**Minimal repair:** load a hash-bound stage-specific manifest and validate its schema, exact case set and identities. Allow only the declared surface names; require the current 48 value heads and declared geometry/dtypes; require real, canonical, positive integer depths within the declared maximum; reject booleans and coercions. Bind the full expected-case specification plus fixture/input/path hashes and runtime/reference identities into eligibility. Compare records against it explicitly.

Do **not** require every case to contain both surfaces or all 12 depths. A case may legitimately request one output or one accepted-state depth; its exact required subset must come from the trusted manifest. Missing required observations remain `UNCOVERED`, malformed or duplicate observations produce no valid verdict, and extras cannot silently replace required observations.

### R3. Declare the Q1.2b input domain prospectively; do not use old stratum names as membership

The frozen policy binds the Q1.2a sequential fixture manifest and IDs. `tools/q1_2b_fixtures.py:37-39` introduces a different seed block and `ordinary-random`/`mixed-stress` fixtures with 32 physical rows and the tree paths. These are not the old fixtures. The evaluator instead treats recognized old stratum strings as domain membership (`tools/q1_policy_evaluator_v2.py:116-121`). This has two opposite effects: the new `mixed-stress` case returns `UNCOVERED`, while an arbitrary fixture/input identity with an old recognized stratum can pass.

**Minimal repair:** freeze an explicit prospective Q1.2b domain/expected-manifest extension bound to the new fixture bytes, generator, topology/path mapping, geometry, dtypes and source identities before any C0 outputs. Resolve eligibility from that exact manifest, not inferred string membership. Preserve unsupported-domain behavior and test it. Renaming `mixed-stress` to an old label is not a repair. No numerical parameter changes or expansion of empirical maxima are needed under the paired criterion.

### R4. Preserve immutable policy approval and freeze provenance

The frozen policy's `approved` flag is false. The loader requires it to be true (`tools/q1_policy_evaluator_v2.py:51`), while the review request asks the parent to flip it. Flipping it changes the policy file SHA, invalidates the freeze binding and changes the canonical payload (which includes `approved`; builder lines 113-114). Updating only the raw gate hash leaves the stored canonical identity stale.

**Preferred repair:** keep policy bytes immutable and separate integrity validation from authorization. An external parent-controlled approval/gate binds the exact policy SHA, expected-manifest SHA, stage and one-run identity; loader/launcher/reducer check those bindings. A separately named approved-copy artifact with its own recomputed raw/canonical hashes and explicit link to the reviewed semantic payload is an alternative. Preserve the old unapproved snapshot and do not call the old freeze valid for changed bytes. Repair the overwritten test-log binding with a new immutable log/freeze.

## Independent CPU verification

Local tests were run with bytecode and pytest cache writing disabled:

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/tests/test_q1_policy_v2.py \
  papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/tests/test_q1_policy.py
```

**42 tests passed in 3.91 s.** Local torch is unavailable, so the four bf16 ULP tests were separately executed with read-only remote CPU Python (`-B`, no output files): **4 passed**. This review does not claim to have independently rerun the archive's entire 97-test suite.

The earlier false-pass metric cases now fail closed: empty evidence yields `UNCOVERED`; missing output yields partial `UNCOVERED`; truncated arrays, missing/nonboolean `metrics_valid`, negative or boolean metrics, and noncanonical observed depths yield `MALFORMED_EVIDENCE`. The remaining adverse cases are listed in R1-R3. In-memory test helper policies were used for these evaluator probes; no stored approval or gate was modified.

The imported aggregate function can still report `PASS` for duplicate passed case IDs without an expected case set. The actual component reducer's denominator, process coverage, raw-witness and native/source binding audit belongs to the separate `q1-2b-reducer-review.md`; its findings must be resolved before launch. This report does not duplicate that review.

### Power witness checks and their limits

The frozen witness file's 20 calibration fixture IDs and file hashes match the accepted native fixture manifest. Its summaries recompute from its recorded cells as follows. Each reported cell has all 48 heads above the declared 10x power requirement in both metrics.

| CPU state corruption | Cells | Minimum max-error/bound | Minimum RMS-error/bound |
| --- | ---: | ---: | ---: |
| No-op publication | 240 | 1406.1659 | 1614.1657 |
| Previous-depth stale state | 220 | 329.1968 | 405.7090 |
| Transposed state | 240 | 269200.5218 | 293655.7615 |
| Wrong-head rotation | 240 | 221507.0938 | 227130.2725 |
| Other-token operand substitution | 240 | 28.1631 | 34.2928 |

I independently recomputed the weakest recorded witness from raw tensors on CPU: `calibration/gqa-head-asymmetry_1`, depth 9, replacing that update's q/k/v/a/b with the specified next token. The minimum ratios exactly reproduced **28.163063651662725** and **34.29282496529892**, and all 48 heads exceeded 10x for both metrics.

These are powered **state-corruption witnesses on the old synthetic sequential calibration fixtures**. They do not establish power for bf16 outputs, execution of GPU fault controls, or causal off-path independence in the new tree fixtures. New candidate-route structural negatives must retain their own source engagement and paired control checks. In particular, poisoned-versus-unpoisoned C0 is the causal off-path comparison; ordinary C1/C2 numerical qualification remains a separate comparison.

Minor test repair: `tools/tests/test_q1_bf16_ulp.py:40-44` labels `1+2^-9` as a bf16 halfway case. The spacing at 1 is `2^-7`; those existing values are not the claimed ties. Actual tie cases `1+2^-8` and `1+3*2^-8` independently rounded to `1` and `1+2^-6` on CPU. Replace or supplement the misleading test with these cases. The ULP diagnostic does not affect acceptance thresholds.

## Minimal re-review package

Provide the repaired reference-only evaluator and executable runner phase ordering; a hash-bound, strict Q1.2b fixture/expected-case manifest; external immutable policy authorization; regression tests for every R1/R2 reproduction; the unsupported-domain test; an immutable test log and replacement freeze. Keep both equations and all numerical constants unchanged. Include the separately repaired reducer in that package. This is a bounded component-policy readiness check; broad Q1 and full-model/serving qualification remain outside this verdict.
