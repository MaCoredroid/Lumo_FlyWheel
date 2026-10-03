# Q1 executable numerical policy: snapshot review

27 September 2026. CPU-only independent review. No GPU launch, remote mutation, source edits, gate changes, or inspection of held-out evaluation outputs. Worker edits were ongoing; findings below bind the exact snapshot hashes rather than claiming to cover later revisions.

## Verdict

**Require fixes before approving the policy/evaluator or a Q1.2b launch.** The saved numerical cap calculations reproduce correctly from the accepted native calibration. The evaluator and builder do not yet fail closed on incomplete, malformed, or mismatched evidence. The sampled reference-scale intervals also need an applicability check before C0; fixing software validation alone does not establish useful coverage.

The policy currently has `approved=false`. The false-pass reproductions below deliberately call the numeric core with `approved_ok=True`, as its tests do, to expose behavior that would occur after approval. They are not existing authorized qualification results.

```text
48cf162e67afabd64d31cfb21d1480d4b86acd96efce76d8b5c0bd0e6e153341 tools/q1_policy_build.py
73a533702029825d579ea7f1bbcc9c18f512a19ea4921775f4067ca29e7abb0f tools/q1_policy_evaluator.py
4c1378928d60569dcc5128035df332769b4d55fb382eb8d188db09000f9c1a13 tools/tests/test_q1_policy.py
fbc401d06693675d77f4d4a7713ddfd51082e9d6a892e2de0dbf8c6e8d40d8d2 policy/q1_component_numerical_policy.v1.json
```

## Correct calculations in the present artifact

Independently deriving the table from accepted `procA/result.json` SHA `84771344c6b4c3504e698f1f491cc1ce2b4879199954e8c47bcfe8c5d2aa63bf` reproduces all **11,520 caps, 11,520 scale-domain entries, and 23,040 calibration provenance contributions** exactly. Each of the 10 strata has 48 heads × 12 depths × 2 surfaces; two calibration fixtures contribute to each entry. The builder uses:

```text
cap = 1.10 * maxabs(C1-C2) + 2^-24 * maxabs(C2) + 2^-149
```

The evaluator's scalar matched-RMS formula at lines 29–30 is also the adopted equation. State/output caps are separate and head/depth identities are retained. No numerical equation change is needed to repair the software failures below.

## Reproduced evaluator failures

Starting with an otherwise in-domain synthetic case built from native calibration provenance, with required state/output surfaces at depth 3 and 48 heads, the following isolated mutations return the wrong result:

| Mutation | Observed result |
| --- | --- |
| `surfaces={}` | PASS with zero outcomes |
| Omit output | PASS with 48 outcomes |
| Omit state depth | PASS with 48 outcomes |
| Keep only one of 48 heads | PASS with two outcomes |
| Empty candidate RMS arrays; leave other arrays populated | PASS with zero outcomes |
| Omit `metrics_valid` | PASS with 96 outcomes |
| `metrics_valid=[False]*48` | PASS with 96 outcomes |
| Set one native max error to `1e6`, above its cap | PASS with 96 outcomes |
| Negative candidate RMS/max errors | PASS with 96 outcomes |
| Boolean candidate RMS/max errors | PASS with 96 outcomes |
| Supply both depth keys `"3"` and `"03"` | PASS with 192 counted checks but only 96 distinct outcome keys |

These are CPU executions of the hashed evaluator, not inferred edge cases.

### 1. Bind coverage to an independent manifest

`q1_policy_evaluator.py:40–45` iterates only supplied surfaces/depths and takes head count from the candidate RMS array. Line 60 treats zero failures as a pass even when no comparisons occurred. A producer can therefore omit precisely the evidence needed to establish the claim. Integer conversion at line 43 aliases `"3"` and `"03"`; outcomes overwrite each other while counts increase.

Require a trusted, hash-bound case specification containing case ID, raw-input identity, stratum, geometry/dtypes and the exact required surface/depth/head set. **Do not require all 12 depths for every case**: a legitimate depth-3-only case needs exactly its declared comparisons, while another scenario may require every prefix. For example, a manifest requiring state and output at depth 3 with 48 heads defines 96 comparisons independently of the candidate arrays.

Before numerical evaluation, require equality of expected and observed surface/depth/head key sets, canonical depth representation, equal lengths for all six metric arrays, and a nonempty expected comparison set for a numerical case. Reject missing, extra, duplicate/aliased, and truncated evidence as malformed. The aggregate reducer must enforce the run's required case/process/repetition set and compare expected count with **unique** evaluated outcome count. A zero-update identity case needs an explicit structural specification; it must not become an empty numerical PASS.

### 2. Validate types and values before arithmetic

Line 49 defaults `metrics_valid` to true, accepts truthy non-booleans, and treats Python booleans as numbers via `math.isfinite`. It does not reject negative error norms. Require the exact schema type: a real boolean `True` if validity is per tensor, or an explicitly declared per-head boolean vector with all required entries checked. Do not support both implicitly through truthiness.

Require numeric metric values of the declared type, excluding booleans, all finite and nonnegative; reject `None`, strings, NaN and infinity with a structured invalid-evidence verdict. Validate numerical consistency such as RMS not exceeding its corresponding maximum under the metric producer's specified arithmetic convention. Validate policy constants, caps and domain endpoints as finite, nonnegative and ordered too. Invalid evidence must never fall through to PASS or be silently truncated.

### 3. Check native-reference eligibility before a candidate verdict

Line 51 unpacks `mx1`, but line 56 never checks it against the absolute cap. A huge native error can both enlarge the matched RMS allowance and remain ineligible for the calibration domain without being detected. Require `maxabs(C1-C2) <= cap` for every required comparison; otherwise emit UNCOVERED/reference-domain failure, preserving all records and blocking aggregate qualification.

The API should expose a **reference-only eligibility pass** over C1/C2 metrics and the manifest, before C0 executes. It must check declared input/scale domain, reference precision/operator identity, native cap and coverage without reading C0 metrics. The later numerical evaluation consumes a verified eligibility record bound to the same case/input/policy hashes. Invalid/nonfinite evidence remains an integrity failure; it is not a reason to run C0 or widen the domain.

## Reproduced builder failures

`q1_policy_build.py:34–67` records expected identity arguments but does not enforce them. Calling the real builder with wrong native-module and fixture-manifest SHA arguments succeeds and writes both mismatches into its returned policy. In-memory mutations of the source record (without changing any files) also show:

| Source mutation | Observed builder behavior |
| --- | --- |
| Empty fixture list | Accepts a policy with zero caps |
| `block=evaluation`, `eligible=false`, `integrity_ok=false` | Accepts 11,520 caps |
| Duplicate one calibration fixture | Accepts it and records three contributors for affected entries |
| Truncate every reference-RMS array to one head | `zip` silently reduces the policy to 240 caps |
| A NaN calibration max error with `metrics_valid=true` | Accepts a NaN cap |
| An infinite calibration max error | Accepts an infinite cap |

### 4. Validate the accepted source and full calibration denominator

The builder should consume an accepted native-run specification/receipt with expected result, fixture/execution manifest, image, loaded native function/module, oracle and runner identities. Verify the actual result bytes against the trusted expected SHA, and reject native/fixture identity mismatches rather than merely recording them. Bind run identity to that receipt, not an arbitrary command-line string. Check calibration block, eligible/complete/integrity status, required two-process/reducer acceptance, and expected fixture IDs/unique seeds/strata. It is acceptable to derive from A after proving accepted A/B equivalence; it is not acceptable to accept an arbitrary single-process JSON as equivalent evidence.

Lines 43–46 must validate exact required depth/head lengths before iteration and never use truncating `zip` as a validator. Require every expected contributor exactly once, all required finite/nonnegative metric arrays and validity/nonfinite flags, and the complete expected cap/domain/provenance key set. Derive the expected 11,520-entry denominator from the bound geometry/scenario manifest; do not let observed data choose it. A future scope with fewer declared depths can have a different legitimate denominator, but it must be independently declared.

Write JSON with nonfinite serialization disabled (`allow_nan=False`) as a final backstop. Retain source/build hashes and immutable versioning. At load, read policy bytes once for hashing and parsing, enforce the supported schema/constants, and carry that verified artifact into evaluation; a free-standing `approved_ok` boolean should not be the only link between the policy object being evaluated and the gate's reviewed SHA. Parent approval remains external authority, not something this builder grants itself.

## Applicability limit: sample minima/maxima are not an input-domain specification

The present scale domain at builder lines 53–55 is the literal min/max of only two calibration values for every stratum/head/depth. That accurately describes the sample, but it does not establish coverage of a generator's declared parameter domain, a new random fixture, or a model layer. Requiring all heads/depths to fit can make a nominally valid new case UNCOVERED. Matching the two training fixtures is not a coverage validation.

Without inspecting any evaluation fixtures, I performed a **calibration-only leave-one-fixture-out diagnostic**. For each stratum, one fixture supplies the policy and the other is checked, then their roles reverse. Each one-fixture scale interval is necessarily a singleton. None of the 20 left-out fixtures fits: **0/23,040 surface/head/depth cells** match both reference-scale endpoints. Separately, **14,268/23,040 cells** satisfy the cap derived from the other fixture and **8,772 do not**. This is a stress diagnostic for a one-fixture fit; it is **not an estimate of coverage for the actual two-fixture policy**, and should not be reported as such.

Before candidate output is opened, distinguish:

1. A **declared input domain**, grounded in the generator's raw-operand/gate/state bounds, geometry, dtype and prefix construction, rather than inferred solely from two reference output scales.
2. A **sampled calibration envelope** and absolute caps, with their finite-sample provenance.
3. An independent **baseline-only coverage validation**, reporting whole-case and surface/head/depth coverage and all uncovered reasons.

A concrete bounded next calibration check is to predeclare four additional calibration-validation fixture seeds per synthetic stratum, disjoint from both fitting fixtures and held-out evaluation, and hash their generator/input manifest before examining their references. First run CPU C2-only scale/domain checks; if a case is outside the existing domain, record UNCOVERED without C0. For any covered cases, a separately approved native-only check can test C1 cap coverage. Freeze the success condition and full denominator before that check. This is a proposed next stage, **not GPU authorization**.

If this validation shows inadequate applicability, preserve the failed policy/validation and explicitly revise the calibration/domain plan in a new version with fresh validation seeds. Do not pad min/max intervals to fit those observed failures, reassign strata, drop uncovered heads/cases, loosen 1.10 or the fp32 floors, or use held-out evaluation/candidate outcomes to repair the cap. The adopted operational equations remain unchanged unless a separately reviewed new protocol explicitly changes them.

## Tests and conditions for revision review

The existing local CPU suite for this snapshot returned **3 passed, 1 failed**. The failure at `test_q1_policy.py:90` is a test error: multiplying native RMS by 1.15 need not exceed `1.10*native_RMS + fp32_floor`. In the chosen low-error stratum, the additive floor allows it. Fix the test by constructing a value strictly above the **complete** prescribed bound while keeping the independent max cap satisfied; do not modify the formula to satisfy the mistaken multiplier assumption.

Add explicit tests for the reproductions above, source hash/status/denominator enforcement, malformed policy values, and the separation of reference eligibility from candidate evaluation. Include a legitimate partial-depth case whose manifest requires only that depth, an omitted required depth that fails, and exact expected-versus-observed unique outcome counts. Include zero and small reference values, storage-boundary disagreement without ULP exemption, and separate state/output cap selection. Unknown but manifest-declared strata return UNCOVERED; a case that changes its stratum/input/shape against its manifest is malformed.

After those checks pass, review a new immutable policy/builder/evaluator/test snapshot and the candidate runner's actual use of the manifest and reference-only eligibility phase. No candidate launch is approved by this snapshot review. The accepted native v4 evidence and its original failed first attempt remain untouched.

## Prospective alternative requested after the snapshot review

The parent proposed narrowing the claim, before any C0 data, to **matched-input numerical non-regression against the specified native implementation**, with both per-head/per-depth RMS and L-infinity tests. I recommend this as more defensible than treating two-sample maxima as population absolute bounds or repeatedly enlarging calibration until fresh fixtures are covered. This is a prospective policy revision, not approval of the v1 implementation above.

For each manifest-required case, surface, materialized depth and head, use:

```text
R = independent pure-C2 result on the same recorded input/path
N = independently prepared result of the named native C1 on that input/path
Y = candidate C0 result

RMS(Y-R) <= 1.10 * RMS(N-R) + 2^-24 * RMS(R) + 2^-149
maxabs(Y-R) <= 1.10 * maxabs(N-R) + 2^-24 * maxabs(R) + 2^-149
```

Apply both inequalities separately to fp32 state and bf16-stored output. The output's native/C2 term already includes its bf16 storage rounding; the additional term is the declared fp32 arithmetic allowance. No pooled head/depth envelope, population cap, extra bf16-ULP allowance, or rounding exception is needed. The equations avoid division, retain near-zero behavior, and detect a localized error through the maximum test even when its contribution to RMS is small.

This supports the precise claim that candidate error on the tested same-input comparisons stays within the declared native-relative tolerance. The word “non-regression” must retain that **1.10 multiplier plus floor**, rather than imply strict equality or zero increase. It does not prove the recurrence mathematically exact, establish an absolute error theorem, show a numerically healthy baseline by itself, or transfer to full-model logits/task quality. A large native error permits a correspondingly larger candidate error; that is a visible limitation of the stated comparison, not something an observed-max table can repair. Publish the raw C1/C2 and C0/C2 errors, reference scales, differences and per-surface/depth worst cases, so a weak native baseline cannot be hidden by a pass label.

Conditions before adopting/freezing this alternative:

1. **Version it explicitly before C0.** Preserve the unapproved v1 artifact/review and write a new policy with both paired equations and precise claim scope. Remove absolute-cap rejection from the revised acceptance equation deliberately; do not merely leave that branch accidentally unenforced. Store calibration sample caps/ranges as diagnostics, including out-of-envelope labels. Missing diagnostic calibration coverage must not masquerade as either a pass or an acceptance-domain failure under the new rule.
2. **Retain a real declared input domain.** Specify fixture IDs/generator version, raw operands, topology/path authority, heads/dimensions, storage dtypes, update counts and native dispatch. The empirical scale range of two reference outputs is no longer that domain. New topology, precision, kernel, real model layers or prefix captures still require their appropriate scoped protocol; the paired criterion does not authorize arbitrary transfer.
3. **Keep independent matching references.** Native C1 and C2 must use the same original raw operands and independently derived path as C0. Candidate rings/selected metadata cannot become the reference source of truth. Preflight complete reference provenance, finite metrics and supported input identity before C0. The old v1 native-absolute-cap eligibility check is replaced prospectively by these reference-integrity/domain checks; sample-envelope excursions are reported rather than used to cherry-pick cases.
4. **Keep the executable fail-closed fixes.** Manifest-owned coverage, exact metric schemas, full source validation, unique denominators, same-shape repeatability, nonfinite rejection, and structural invariants remain mandatory. Require every declared case to be accounted for; no omitted head/depth/process may yield a numerical pass.
5. **Power negatives against the actual revised rule.** Designated calibration-only numerical corruptions must violate at least one of the paired bounds at their declared observation point; structural corruptions must fail their independent exact invariant. A one-ULP key perturbation remains a sensitivity diagnostic. Inspect negative-control power before candidate outcomes; if meaningful wrong-state controls pass, revisit the protocol and claim before opening C0 rather than weaken controls or silently tune on evaluation.

If the parent adopts this narrowing, the extra sampled-cap coverage calibration proposed for v1 above is **not a prerequisite merely to make v2's paired criterion evaluable**. Calibration remains useful for precision characterization and negative-control design, but no loop should expand its maximum until a held-out case passes. Review the new policy and executable implementation directly, with the preserved v1 failure modes covered by CPU tests. This recommendation changes neither the native evidence nor any gate and authorizes no GPU launch.
