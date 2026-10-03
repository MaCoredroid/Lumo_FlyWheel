# Q1 numerical-policy proposal: independent review

27 September 2026. CPU-only review of the proposal, source snapshots, and already-audited native result records. No GPU execution, remote mutation, gate change, or candidate measurements.

## Verdict

**Accept the archived numerical observations; require policy revisions before freezing a Q1.2b candidate gate.** The draft correctly labels itself unfrozen and separates state/output precision and native dispatches. Its S1–S5 rules are not yet an executable acceptance criterion: S1 replaces the intended matched-input comparison with a pooled envelope, S2 has an unbounded exception, and S5 both contaminates the holdout and demands detection of a perturbation that may be mathematically unobservable. These are fixable CPU protocol/implementation tasks. The accepted Q1.2a characterization remains valid.

Reviewed versions:

```text
8b92331aa1daeb1e4600b75c7347cbe5e32679311e344b011faab8808fda1d72 NUMERICAL-POLICY-PROPOSAL-Q1.md
3c25f9e72dc9b90d4eb70b5bedf5172d9eb25ac23087c11b2bf303ce5c8ebe65 numerical-policy-proposal-tables.json
84771344c6b4c3504e698f1f491cc1ce2b4879199954e8c47bcfe8c5d2aa63bf source procA/result.json
87f46775b8973580718bd368b990e65b48e8ed36842ead3eeba099f274691fd8 unchanged native freeze v4
```

## What reproduces, and what the tables do not contain

I independently derived all **6,000 stored observation fields** from the SHA-bound A-side result using standard-library arithmetic: 1,200 stratum/depth summary values and 4,800 final-state per-head values. Every value agrees (comparison tolerance `1e-14` relative / `1e-18` absolute for re-evaluating ratios/medians). The source-result hash agrees. This is a check of descriptive arithmetic, not approval of any tolerance. The earlier raw audit established that A and B agree and that their retained tensors match the records.

The depth cells contain state absolute maximum, relative RMS maximum/median, reference RMS minimum/maximum, and analogous output summaries. **They do not contain `M_state(s,t)`**, the depth-indexed normalized max-abs quantity invoked in S1. `final_state_per_head` supplies head identity only at depth 12. The proposed numerical caps need the original per-token/head arrays, not the Markdown table's pooled/final-only columns. Add a versioned CPU policy builder that binds the source hashes and writes every per-stratum, per-head, per-depth cap and its contributing calibration fixture; freeze that builder together with its output.

There are also three concrete presentation corrections:

- Proposal line 8 / JSON `constants.bf16_half_ulp_rel`: bf16 has seven fraction bits. Spacing above 1 is `2^-7`, and round-to-nearest unit roundoff is `2^-8`. `2^-9` is not the general bf16 half-ULP relative bound. A local ULP depends on magnitude, binade and subnormal handling. The observed RMS errors remain correct; their interpretation is mislabeled.
- Line 10 combines a peak error with an unrelated maximum reference scale. The `large-state-magnitude_0` peak is at **depth 10 / zero-based token 9 / head 22**, max error `0.00285361943974749`, same-head reference RMS `104.51925084298199`, hence normalized max error `2.7302333462325024e-5`. Dividing by the stratum's maximum RMS near 263 produces a misleading `~1.1e-5`. The smallest observed state-reference RMS is `0.00019695944672876097`, not 0.006.
- S4 line 56 abbreviates seeds incorrectly. The manifest block seeds are `202609271` and `202609272`; actual fixture seeds start `202609271000` and `202609272000`. Use manifest IDs/hashes as authority rather than an informal numeric prefix.

## Required S1 replacement: compare matching cases and retain absolute caps

S1 lines 48–50 would grant every head the largest relative native error observed in any calibration head in that stratum/depth, then scale that allowance with an unrestricted evaluation RMS. This is not the previously proposed matched-case `C0/C2` versus `C1/C2` comparison and is not an absolute cap. For a concrete example, the large-state depth-10 envelope is `4.262217307752814e-6`, whereas `large-state-magnitude_1`, head 25 has native relative RMS error `6.384234072906378e-8`: **66.76 times smaller**, before the proposed multiplier and floor. Pooling hides precisely the localized discrepancies the head-wise test should expose.

Use the already-proposed strict rule in absolute units. For each eligible evaluation case `x`, surface, materialized update count `d`, and head `h`, let `R = C2(x,d,h)`, `N = C1(x,d,h)`, and `Y = C0(x,d,h)`. C1 must independently use the same frozen raw operands/path as C2, not accept candidate-produced rings as its source of truth.

For fp32 recurrent state:

```text
u32 = 2^-24
eta32 = 2^-149
floor_rms(x,d,h) = u32 * RMS(R) + eta32
floor_max(x,d,h) = u32 * maxabs(R) + eta32

RMS(Y-R) <= 1.10 * RMS(N-R) + floor_rms(x,d,h)

cap_state[s,d,h] = max over calibration fixtures c in stratum s of
    (1.10 * maxabs(C1(c,d,h)-C2(c,d,h))
     + u32 * maxabs(C2(c,d,h)) + eta32)

maxabs(Y-R) <= cap_state[s,d,h]
```

This is an explicit deployment criterion, not a proved error bound. It preserves the previously proposed 1.10 multiplier and does not infer a larger allowance from observed candidate behavior. The comparison has **no division**, so exact/near-zero reference RMS needs no arbitrary normalized acceptance floor. Relative RMS remains a diagnostic: report its denominator and mark a zero denominator explicitly; never turn an undefined ratio into a pass. The fp32 cache must not be rounded to bf16 for floor construction.

Additional implementation requirements:

1. Use **depth-indexed caps** for this first bounded stage. Define depth as the number of actual recurrent updates, including any materialized root; bind the topology-to-depth mapping explicitly. Root-only is one update when that is the harness convention; accepted-draft count zero must not silently skip the test. A true zero-update case is exact identity. The observed table covers 1–12 updates, not arbitrary rollout length.
2. Preserve head and stratum identity; do not pool across heads, depths, synthetic strata, or native operators. Store the calibration input/reference scale domain with each cap. The implementation must define those fields and their coverage check before execution. This two-fixture-per-stratum sample is a bounded calibration set, not a distribution-free bound.
3. If an evaluation case is outside the declared scale/domain, or its matching native C1 itself exceeds the calibrated cap, return **UNCOVERED / reference-domain failure**, retain the case and all errors, and prevent an aggregate qualification pass. Do not discard it, reassign it to a looser stratum, rescale the cap, or retune on held-out data. This rule must be prespecified, not introduced after a candidate fails.
4. Captured-prefix/model-layer tensors need their own matched native calibration. Their prompt lengths do not make them members of any synthetic stratum. Do not choose a pooled synthetic envelope for them. Likewise, these caps do not transfer to changed input dtype, state precision, GQA geometry, normalization, gate order, or native kernel.

## Required S2 replacement: no automatic ULP exemption

S2 line 52 says output differences below one ULP cannot be distinguished. Values on opposite sides of a rounding boundary can differ by much less than one ULP before storage and produce distinct bf16 outputs. Conversely, equal stored outputs can hide upstream differences. ULP categories are useful diagnostics; they do not establish correctness by themselves.

The exception “unless the same element also differs in the native output” is unbounded: it does not constrain how much worse C0 may be at that element. Remove it. Use a matched-input per-token/head RMS test plus a separately frozen absolute per-token/head cap, in the same form as S1, with an explicitly declared output floor. Do not reuse state caps or add a blanket extra bf16-ULP allowance: the measured C1/C2 error already includes bf16 storage rounding.

The parent's proposed output floor is acceptable for CPU implementation and subsequent frozen-policy review: use a **fp32 arithmetic allowance**, while C1's measured error already accounts for its bf16 storage. Keep separate output caps:

```text
floor_out_rms = 2^-24 * RMS(R_out) + 2^-149
floor_out_max = 2^-24 * maxabs(R_out) + 2^-149
RMS(Y_out-R_out) <= 1.10 * RMS(N_out-R_out) + floor_out_rms
cap_out[s,d,h] = max_cal (1.10 * maxabs(N_out-R_out) + floor_out_max)
maxabs(Y_out-R_out) <= cap_out[s,d,h]
```

This floor deliberately does **not** use bf16 unit roundoff: it represents a small fp32 arithmetic allowance, not the target storage's full rounding budget. The measured native/C2 term already includes output quantization. The same-case comparison plus independent output cap therefore remains defensible as a predeclared operational criterion; it is not a universal correctness theorem or a promise that every mathematically reasonable rounding order will pass. Correctly rounded values near a storage boundary can still yield different bf16 results and fail this strict rule; report that as a failure of the chosen deployment criterion, not proof that the recurrence is mathematically wrong. If C2 and C1 are exactly zero, the `2^-149` endpoint may reject a nonzero bf16 subnormal; that is a consequence of this strict criterion, not grounds for an after-the-fact exception. Record normal/subnormal/underflow scope explicitly.

Before approving a candidate freeze/launch, verify that the policy builder implements these exact output formulas separately from state formulas, uses all matched per-head/per-token calibration records, freezes the constants and coverage checks, and tests both a baseline identity pass and above-bound rejection. Tests must cover zero reference, small nonzero reference, nonfinite input/reference/difference, a storage-boundary disagreement without an exemption, separate state/output caps, domain failure, and missing/duplicate evidence. No output ULP-count exception may override an RMS/max-cap failure. The candidate runtime must actually use the declared bf16 output and fp32 arithmetic/state route; a dtype or native dispatch change invalidates the transfer.

Report bitwise equality and a precisely implemented ULP-distance histogram as diagnostics. For native-versus-candidate bf16 distance, use a sign-aware ordered bf16 representation, canonicalize signed zero if numerical equality is intended, and reject nonfinites separately. If reporting distance to C2, explicitly distinguish unrounded float64 absolute error from distance to a declared round-to-nearest bf16 projection of C2. Test binade transitions, negative values, signed zero and subnormals on CPU. “One ULP” without those definitions is not executable.

## Required S3–S5 changes: structural power without holdout leakage

Keep candidate same-shape determinism and finite-input/nonfinite-output rejection. Expand the abbreviated S3 list back to the actual serving-boundary contract: independently derived ring operand bytes, path/parent/slot/token/position integers, staging and neutral-tail bytes, untouched bank/KV/conv locations, and lifetime/ownership metadata. Compare pointer identity to its own preseeded process-local binding, **not numeric addresses across fresh processes**. Compare logical KV/conv content through independent maps where physical layouts differ. Native cross-shape bitwise equality observed in Q1.2a remains a finding; exact same-shape independent replay is the structural control.

S5 line 58 must not mutate an evaluation fixture before freezing the policy. Use dedicated, hashed **calibration-only power fixtures**. A one-ULP change to a single key component is a sensitivity probe, not a guaranteed corruption witness: normalization, gates, recurrence decay, cancellation, and output projection can remove or attenuate its effect, including at later depths. Failure to detect it at every subsequent depth does not prove a cap is too loose. Report its sensitivity without a blanket must-fail claim.

Use two explicit kinds of controls:

- **Structural controls:** wrong sibling/path source, stale owner/generation, wrong accepted length, wrong slot/head mapping, or non-neutral staged tail must fail their corresponding exact independent invariant when injected. They need not also exceed a numerical cap. The original input/path oracle stays unchanged.
- **Numerical power controls:** choose nondegenerate calibration-only witnesses for transpose/wrong-head/no-op or stale-state publication. Before C0, record the intended observation point and prove from independently computed C2/CPU diagnostics that the changed state differs from the correct state by a stated margin beyond the frozen cap. Compare the corrupted result to the **original correct** reference, not a recomputed reference for the corrupted operands. Do not require an effect at later depths where the recurrence legitimately erases it.

An invalid-input NaN case is a separate failure-path test: it must be explicitly rejected or flagged invalid, and never contribute a finite error or calibration cap. It is not an ordinary finite-valid qualification fixture.

The CPU reducer/metric control tests can be completed before a GPU gate. Actual stale-metadata, lifetime, and graph fault injections require the corresponding route harness; include them in that separately reviewed executable scope after the policy is frozen. Do not describe those route-dependent controls as already demonstrated by a native-only operator test. If a designated powered control is admitted, investigate the test/policy before candidate evaluation and preserve the failed version; do not tune against evaluation or candidate outputs.

## Reference-dispatch boundary and minimal next step

Retain the distinction in section 4. The archived `gdn_linear_attn.py:806–818` checks `enable_packed_recurrent_decode`, absence of speculation masks, zero prefills and positive decodes, then selects `_forward_core_decode_non_spec`; its body calls the packed operator at line 1085. In archived `fla_ops__fused_recurrent.py:314–325`, the packed kernel uses division by `sqrt` and casts sigmoid beta through `b.dtype` before returning to fp32. Those differ from the characterized fused-sigmoid committer operator. The actual model C1 dispatch/dtypes must be attested; the flag/default inference alone does not constitute a runtime qualification result. Keep the known committer operator as the named reference for this component stage; a packed-decode or full-model comparison needs separately bound evidence.

The minimal next deliverable is a CPU-tested policy builder and frozen JSON containing the matched-input equations, constants, absolute head/depth caps, coverage rules, full source/result provenance, and calibration-only power fixture IDs. The candidate runner must consume that hash-bound artifact and fail closed on missing/extra/UNCOVERED records. Review that concrete policy and runner before opening Q1.2b. No additional native GPU run is needed merely to derive these tables; no new GPU action is authorized by this review.
