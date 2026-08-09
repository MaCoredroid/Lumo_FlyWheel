# FR13 Tier-B bounded-drift kernel qualification policy (2026-08-09)

Offline policy document. **No GPU was touched. No serving code was changed by
this artifact.** It defines an admission procedure; it is not a performance
result, not a qualification, and not an acceptance run.

Greenlit by Mark on 2026-08-09 under one condition — *"correct rejection
sampling"* — which is codified below as the hard invariant of §2 and made
mechanical by `scripts/fr13_tier_b_sampler_pin.py`.

Baseline: `origin/main` at `c6cd0e92c`.

---

## 0. The problem Tier-B exists to solve

Every kernel candidate in this campaign is admitted on **byte identity**: the
candidate's output tensors must be bit-for-bit equal to the incumbent's on a
real authenticated event. Call that **Tier-A**. It is the default, it is always
preferred, and nothing here weakens it.

Tier-A has a structural blind spot. Some kernel changes are byte-impossible
*by construction*, not by defect — they reassociate a floating-point reduction,
and floating-point addition is not associative. The campaign has already
measured this and rejected on it:

> The captured reference uses `VLLM_BATCH_INVARIANT=0`, hence `num_splits=0`
> and one ordered online softmax reduction. The candidate forces
> `num_splits=2`, writes two independent FP32 O/LSE partials, and invokes
> FA2's split-K combine. That reassociates the softmax and output reductions.
> **It cannot satisfy a raw-byte contract against the incumbent one-part
> reduction in general, even when both paths implement the same real-valued
> attention formula.**
>
> — `results/fr13_b1_qrow32_split2_byte_rejection_20260805/README.md`

That artifact is the correct Tier-A verdict and stands. But the sentence
"cannot in general, even when both implement the same formula" is the whole
motivation for a second tier: Tier-A cannot distinguish *a kernel that computes
something else* from *a kernel that computes the same thing in a different
order*. It rejects both. The attack ladder puts up to **17.0 ms/step (≤7.2% of
envelope)** behind that single wall
(`results/fr13_attack_ladder_analysis_20260808/README.md`, rows 12 and the FA2
tile-geometry row, both marked BLOCKED on exact-math).

**Tier-B** admits a candidate whose only deviation is reduction order, on
evidence that the deviation is *behaviorally immaterial*. It is a strictly
weaker claim than Tier-A and must never be reported as one.

---

## 1. Scope

**Tier-B is admissible for VERIFIER-FORWARD kernels only** — everything
upstream of the sampler:

| in scope | out of scope |
|---|---|
| full attention (FA2 geometry, split-K, GQA pairing, tile shape) | the rejection sampler and everything it calls |
| target GEMMs (SFWD projections, MLP, o_proj) | the committer's integer decision path |
| norms, activations, and elementwise fused into the verify forward | the drafter (see §7) |
| GDN scan / conv on the verify side | anything that changes a *served integer* rather than a *verify float* |

The unifying test: **a Tier-B candidate may perturb the verifier's logits; it
may not perturb the mechanism that consumes them.**

Tier-A remains the default. A candidate that *can* be byte-identical **must**
be byte-identical — Tier-B is not an escape hatch from a failing byte gate. A
Tier-B qualification is only opened after a byte gate has failed *and the
failure has been attributed at source to a reduction-order change*, in the
manner of the split2 rejection above. An unexplained byte failure is a defect,
not a Tier-B candidate.

---

## 2. The hard invariant: the sampler stays byte-identical

### 2.1 Why

Speculative decoding is lossless because of the rejection sampler, not because
of the drafter. Given target distribution `p` and any draft proposal `q`, the
rejection-sampling mechanism emits samples distributed exactly as `p`,
*whatever `q` is*. This is why drafter changes are cheap here (§7): the sampler
absorbs them.

Tier-B does not touch `q`. It perturbs `p` itself, into `p_ε`. The sampler then
emits samples distributed exactly as `p_ε`. So:

```
Tier-A:  served distribution  ==  p                 (exact, by byte identity)
Tier-B:  served distribution  ==  p_ε               (exact, by the sampler)
                             and  ||p_ε - p||  is bounded and measured
```

The served distribution remains **exactly lossless with respect to the
(ε-perturbed) verifier**. The three gates of §3 exist to prove that the
perturbation from `p` to `p_ε` is behaviorally immaterial. State the claim in
exactly those two halves; do not collapse them into "lossless".

If the sampler itself drifts, the first half evaporates. The served
distribution is then neither `p` nor `p_ε` but an unbounded third thing, and
two uncontrolled errors compound with no instrument that can separate them.
That is why the invariant is hard and not a gate: it is not traded off against
anything.

### 2.2 What is pinned

**Every Tier-B qualification must carry a mechanical assertion that the sampler
source is unchanged versus main**, recorded as a sha over the sampler patch
region in the qualification artifact.

The region is the set of top-level functions in
`scripts/fr10_phase4_patch_vllm_tree_gdn.py` that write
`vllm/v1/sample/rejection_sampler.py`. Between them they carry the injected
sampler body — `_lumo_tree_canonical_multidraft_sample`, the LCP committer —
and the `apply_sampling_constraints` path:

| function | what it injects |
|---|---|
| `_patch_rejection_sampler_tree_lcp` | the sampler body, the `apply_sampling_constraints` splice, the double-temperature guard, the `rejection_sample()` signature and greedy branch |
| `_patch_rejection_sampler_bonus_handoff` | `_FR13_SG_BONUS_OUT` bonus-token handoff |
| `_patch_rejection_sampler_target_logits_handoff` | `_FR13_SG_TL_QUEUE` target-logits handoff |

`scripts/fr13_tier_b_sampler_pin.py` locates them by **AST**, not by line
number, and hashes their concatenated source under a per-function name banner.
Consequences, all intended:

- an edit anywhere else in the 41,690-line patcher does **not** move the pin;
- an edit *inside* any of the three does;
- renaming, deleting, or reordering one of them does;
- the patcher growing a **new** `_patch_rejection_sampler_*` function fails the
  extractor closed, rather than silently leaving the new code unpinned.

### 2.3 The pin at `c6cd0e92c`

```
sampler_region_sha256 = d93df2c2eaa2f88fe4db2c5939b5b8c9df6bd328fcf6500a3c551f8041bbb785
```

| function | lines | bytes | sha256 |
|---|---:|---:|---|
| `_patch_rejection_sampler_tree_lcp` | 17962–21387 | 164,899 | `65522b35481c6fd6fc339ce9eb4a3d7392da465d1a9718d22071523706adb2e7` |
| `_patch_rejection_sampler_bonus_handoff` | 38380–38457 | 3,334 | `d5da80d71ecf9d9d21ab28f5f64a45ebab082cbf4470eef2f42f9f332762ecb3` |
| `_patch_rejection_sampler_target_logits_handoff` | 38460–38495 | 2,083 | `686dab2ff89960c82f45438231fb57988861f64275dd6720ff3b71719d68c77f` |

Whole patcher at the same commit:
`0696bfc5458d01cca72767c6b54432490463722840153ffbab3b732bf09235c2`
(1,980,561 B). The full record is banked beside this file as
`sampler_region_pin.at_c6cd0e92c.json`.

Usage:

```sh
python3 scripts/fr13_tier_b_sampler_pin.py emit
python3 scripts/fr13_tier_b_sampler_pin.py assert --expect <sha256>
python3 scripts/fr13_tier_b_sampler_pin.py assert --qualification <artifact.json>
```

`assert` exits `2` and prints `SAMPLER REGION DRIFTED -- Tier-B qualification is
void` on any mismatch. A qualification run that cannot produce a matching pin
is not a failed Tier-B candidate; it is **not a Tier-B run at all** and must not
be recorded as one.

---

## 3. The three gates

All three are **required**. There is no partial Tier-B and no "two of three
plus judgement". A candidate that clears (a) and (b) but not (c) is rejected.

### (a) Shadow logit-ε bound

The candidate runs **shadow alongside the incumbent** on a real authenticated
task, and the maximum logit deviation is measured and declared.

- **Pattern**: the SFWD byte-A/B harness
  (`scripts/fr13_sfwd_prior_reuse_gate.py`, schema
  `fr13.fixed32.sfwd_prior_reuse.byte_ab.v1`) with the `byte_equal` /
  `zero_diff` predicate replaced by a deviation bound. The deviation-reporting
  half already exists in `scripts/fr13_apc_shadow_reduce.py` (`max_abs`,
  `max_abs_max`, `argmax_flips`, `n_value_compared`, floor via
  `--max-abs-floor` / `FR13_APC_SHADOW_FLOOR`) and in
  `scripts/p5b_fp8_kv_purity_attestation.py` (`--atol-logit` / `--rtol-logit`,
  `overshoot`). Tier-B needs the union: same-boot paired capture, non-zero
  tolerance, both absolute and relative.
- **Recorded**: `max_abs_logit_delta`, `max_rel_logit_delta`,
  `argmax_flips`, `n_values_compared`, over every compared verify-forward
  surface.
- **The bound is declared per candidate, in that candidate's qualification
  artifact. There is no global ε.** A geometry change that moves logits by
  2⁻¹⁰ and one that moves them by 2⁻⁴ are not the same risk and must not share
  a threshold. The reviewer approves the number alongside the candidate.
- **`argmax_flips` is reported, not gated.** A flip at a near-tie position is
  the expected consequence of any reduction-order change and is exactly what
  gates (b) and (c) exist to price. Gating on zero flips would make Tier-B
  equivalent to Tier-A and pointless.

### (b) Acceptance parity

Accepted drafts per event must be within CI of the incumbent on **paired
exact4 arms**.

- **Metric**: `accept_per_event` = accepted tokens / draft events, as emitted
  by `scripts/fr13_measure.py` `deploy-speed` (per task and aggregate). The
  artifact-side name is `accepted_drafts_per_event`.
- **Arms**: the canonical hash-pinned exact4 subset —
  `config/fr13_fixed32/subset_b4_four.json`, `EXACT4_SUBSET_SHA256 =
  0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5`. Paired
  means incumbent and candidate on the same four tasks in the same session; a
  cross-session comparison is not admissible.
- **Interval**: the 95% interval on the paired per-task delta must contain
  zero, and it is reported as `accept_parity_ci = [lo, hi]` regardless of
  verdict.
- **Binding floor**: the interval on four pairs is wide, so it is not by
  itself a strong test. The existing one-sided slack from
  `scripts/fr13_corruption_gate.py` therefore also binds:
  `ACCEPT_PER_EVENT_SLACK = 0.05`, i.e. `candidate − incumbent ≥ −0.05`. That
  threshold is the campaign's established multi-spine-signature detector; a
  Tier-B candidate must not trip it.
- **Why this gate exists at all**: acceptance is the one behavioral quantity
  that reads the perturbed logits *through the sampler*. If `p_ε` has drifted
  somewhere that matters, the draft acceptance rate is where it shows up first
  and cheapest.

### (c) Behavioral band

A full **16-task campaign** — `config/fr13_fixed32/subset_b4_sixteen.json`,
B=4, concurrency 4, temperature 0.6 — must land inside the historical band on
three axes.

**c1. Resolve rate in band: 8–11 of 16.**

| edge | value | source |
|---|---|---|
| floor | `~8/16` — user-mandated resolve gate, 2026-07-18; *"drifting below = issue signal"* | `FR13_BEAT_NATIVE_LADDER.md:33` |
| typical | `8–9/16` | `FR13_BEAT_NATIVE_LADDER.md:113`, `FR13_BEAT_NATIVE_HEADROOM.md:83` |
| high water | `11/16` — *"best resolve ever on this subset"* (t33333 + SLOT_REORDER) | `FR13_B4_CACHE_MATRIX_RESULTS.md` |

The band's floor is the gate. **Its ceiling is not a failure condition**: a
candidate resolving 12/16 passes c1. The upper edge is recorded so that a
suspiciously high number is read as "replicate before claiming", which is the
campaign's own standing caution on this table (*"n=16 caveat: 11-vs-9 within
variance; replicate before claiming"*).

**c2. Zero garble regression** on the undefined-name gate.

`scripts/fr13_garble_gate.py` scores identifier-bait generations at temp 0.6 by
parsing each generated function with `ast` and counting name loads that resolve
to nothing (`undefined_rate`, `samples_with_undef`, `syntax_err`). Its criterion
is explicitly **relative**:

> `Tree FAILS iff its undefined-name rate > native's (both should ≈ no-spec's).`

Tier-B inherits that form: the candidate's `undefined_rate` must not exceed the
**incumbent's, measured in the same session on identical prompts and seeds**.
An absolute-zero reading is not the right bar and is not satisfiable — the
tree-default arm itself sits at 8–11% undefined-name rate against native's 0%
(`scripts/fr13_conv_geom_garble_gate.sh:9`,
`scripts/fr13_flag_garble_gate.sh:40`: `default=9.62% native=0%`). Tier-B is not
the workstream that closes that gap; it is the workstream that must not widen
it. Additionally, `scripts/fr13_garble_watch.py` must report zero **strong**
flags (`NEAR-NEIGHBOR`, `ERROR-LOOP`) that the incumbent does not also produce.

*(Do not confuse this 8–11% with the 8–11/16 of c1. They are different
quantities that happen to share digits and a temperature; conflating them has
already happened once.)*

**c3. Zero give-up anomalies.**

A give-up is an episode that terminates with an empty patch for behavioral
reasons: `empty_patch_retry.cause == "agent_gave_up"` in
`scripts/run_swe_bench_q36_a.py`, counted campaign-side as `giveups`
(`patch_bytes == 0`) by `scripts/fr13_slot_reorder_verdict.py`. The two
non-behavioral causes are explicitly carved out and are **not** give-ups:
`setup_loop` (≥3 identical failing commands — the pip/conda/build loop) and
`infra_stall_suspect`.

The bar is `giveups == 0`, which every modern arm meets — the four-arm table in
`FR13_B4_CACHE_MATRIX_RESULTS.md` records `give-ups = 0` for cat8+SLOT_REORDER,
native MTP-5, cat6root+SLOT_REORDER and t33333+SLOT_REORDER alike. A non-zero
count is therefore a genuine anomaly rather than background, and it fails c3
even when c1 passes. The failure signature to look for is documented in
`research/fr13_workflows/FR13_GIVEUP_AUTOPSY.md`: exit 0, no timeout, no
network drop, all HTTP 200, and a last substantive turn that emits no
`tool_use` — *"give-up = coherent off-task"*, which is precisely the failure
mode that a logit perturbation could plausibly induce and that c2's token-level
garble metric cannot see.

---

## 4. Sticky re-qualification

**A Tier-B qualification is bound to an exact triple and expires on any change
to it:**

1. the candidate binary — `candidate_so_sha256` (and `candidate_source_sha256`);
2. the sampler region — `sampler_region_sha256` (§2);
3. the serving source commit — `source_commit`.

Any change to any of the three voids the qualification and requires all three
gates of §3 to be re-run in full. There is no incremental re-qualification, no
"gate (a) only" refresh, and no inheritance from a sibling candidate in the
same family: a GQA-pair qualification does not qualify a kBlockM-32 candidate
even though both are FA2 geometry.

Stickiness is the price of the weaker claim. A Tier-A candidate re-proves
itself with one byte comparison; a Tier-B candidate cannot, because its
admission rests on behavioral evidence that does not transfer across binaries.

---

## 5. The artifact: `fr13.tier_b.qualification.v1`

One JSON record per qualified candidate, written under `results/`, canonical
JSON (sorted keys), 0400, never overwritten in place.

| field | type | meaning |
|---|---|---|
| `schema` | str | `fr13.tier_b.qualification.v1` |
| `candidate_id` | str | stable name, e.g. `fr13_fa2_qrow32_gqa_pair` |
| `candidate_family` | str | e.g. `fa2_geometry` |
| `tier` | str | `B` |
| `deviation_class` | str | why byte identity is impossible, at source — e.g. `reduction_order:split_k_combine` |
| `source_commit` | str | serving source commit the qualification is bound to |
| `candidate_so_sha256` | str | the `.so` this qualification admits |
| `candidate_so_bytes` | int | |
| `candidate_source_sha256` | str | |
| `patcher_sha256` | str | whole-file digest of `fr10_phase4_patch_vllm_tree_gdn.py` |
| **`sampler_region_sha256`** | str | **§2 invariant — asserted, not merely recorded** |
| `sampler_region_functions` | list | per-function `{name, lineno, end_lineno, bytes, sha256}` |
| `logit_bound` | obj | `{max_abs, max_rel, declared_max_abs, declared_max_rel, argmax_flips, n_values_compared, shadow_artifact_sha256}` — gate (a) |
| `acceptance_parity` | obj | `{incumbent, candidate, delta, ci_low, ci_high, slack, subset_sha256, paired_arms, deploy_speed_sha256}` — gate (b) |
| `behavioral_band` | obj | `{resolved, instances_total, band_low, band_high, undefined_rate_candidate, undefined_rate_incumbent, strong_garble_flags, giveups, campaign_artifact_sha256}` — gate (c) |
| `verdicts` | obj | `{sampler_region_unchanged, logit_bound_declared, acceptance_parity, behavioral_band, tier_b_qualified}` — booleans, all must be `true` |
| `reason` | str | why, in words |
| `qualified_at` | str | UTC ISO-8601 |

`tier_b_qualified` is the conjunction. `sampler_region_unchanged` is produced by
`scripts/fr13_tier_b_sampler_pin.py assert`, not by hand.

**Every arm that serves a Tier-B kernel must reference its qualification.** The
field spec, in the existing `launcher_meta.txt` `key=value` format:

```
kernel_tier=B
tier_b_qualification=results/<dir>/<candidate_id>.qualification.json
tier_b_qualification_sha256=<sha256 of that file>
tier_b_sampler_region_sha256=<the §2 pin, re-asserted at launch>
```

Arms serving only Tier-A kernels record `kernel_tier=A` and omit the rest. The
absence of `kernel_tier` means Tier-A by default, so existing artifacts stay
valid.

**Wiring is a follow-up.** This document specs the fields; it does not emit
them. Until the launchers are wired, a Tier-B arm's qualification reference is
carried by hand in the arm's results README, and no Tier-B kernel may be
default-enabled.

---

## 6. First candidate: the FA2 geometry family

The first Tier-B candidate is **FA2 geometry**, taken in this order:

1. **GQA-pair** (`scripts/fr13_fa2_qrow32_gqa_pair_gate.py`, schema
   `fr13.fixed32.fa2_qrow32_gqa_pair_b4_live_verification.v1`) — **if its byte
   gate drifts.** If GQA-pair turns out to be byte-identical, it is a Tier-A
   candidate and Tier-B does not apply to it.
2. Otherwise the next geometry candidate in the family: kBlockM 16→32, or the
   `num_splits=2` split-K shape already characterised in
   `results/fr13_b1_qrow32_split2_byte_rejection_20260805/`.

Rationale: this family is where the exact-math wall is most expensive and the
deviation is most cleanly attributable. The attack ladder scores FA2 tile
geometry and the FA2 KV L2-persistence window at up to **17.0 ms/step (≤7.2%)**
each, both blocked on exact-math, and the split2 rejection already contains the
source-level attribution that §1 requires before a Tier-B qualification may be
opened.

Standing caveat from the same ladder: the full legal ladder totals ~19.4 ms/step
against a 237 ms/step envelope and **cannot close the acceptance gap to the
137.607 ms/step cap**. Tier-B does not change that arithmetic. It unblocks a
class of levers; it does not make the campaign's headline target reachable.

---

## 7. What Tier-B is *not* about

**Drafter changes are not Tier-B's concern.** They remain governed by the
existing drafter exemption established at `83ceeb0db` and carried in artifacts
as `qualification_policy: "lossless_deterministic_proposal_v1"`
(`scripts/fr13_dfwd_k64_m1_r64_u8_gate.py`, `..._production_credential.py`,
`scripts/fr13_cfwd_dfwd_u8_composed_gate.py`). The distinction is not a matter
of degree: a drafter change perturbs `q`, which the rejection sampler corrects
*exactly and by construction*, so it needs no ε bound at all. A verifier change
perturbs `p`, which nothing corrects. Do not route a drafter candidate through
Tier-B, and do not cite the drafter exemption to justify a verifier candidate.

**KV FP8 would be Tier-B if unparked.** `FR13_FULL_ATTN_KV_FP8` is gated OFF at
`scripts/fr13_launch_forked_fa2_tree_server.sh` and is currently a
*discriminator* for the APC tree residual locus, not a ship candidate — its
own comment says *"validate losslessness before any ship"*. It changes full
attention's KV storage precision and never touches the mamba/GDN recurrent
state, so it is verifier-forward and byte-impossible: exactly Tier-B's shape.
If it is ever unparked for serving, it enters through §3, not through the
discriminator's existing `--rtol-logit` / `--atol-logit` attestation
(`scripts/p5b_fp8_kv_purity_attestation.py`), which is a probe rather than a
qualification.

**The topk/top-p sampler op is outside the pinned region.**
`_patch_topk_topp_rng_predraw` writes `vllm/v1/sample/ops/topk_topp_sampler.py`
and is behavior-inert unless the S1-full wrapper sets the `q` handoff, so it is
deliberately not in §2's pin. If that path is ever armed by default, it must be
added to `SAMPLER_REGION_FUNCTIONS` — which will, correctly, void every
outstanding Tier-B qualification.

---

## 8. Scope and limits

**The behavioral band is a degeneracy detector, not an equivalence test.** At
n=16 the 95% Wilson interval on 8/16 spans **[0.280, 0.720]**, i.e. roughly
4.5–11.5 resolves; on 11/16 it spans [0.444, 0.858]. The entire 8–11/16 band
sits inside the interval of its own floor. Gate (c1) can therefore detect a
candidate that has broken the agent — 3/16, 0/16 — and cannot detect a
candidate that has cost one or two resolves. That is a real limitation and it
is why (a) and (b) are required rather than optional supporting evidence: the
logit bound is the only gate with tight resolution, and acceptance parity is
the only one that reads the perturbation through the sampler. Do not report a
Tier-B pass as "no behavioral difference"; report it as "no behavioral
difference detectable at n=16".

**Tier-B does not certify composition.** Two independently qualified Tier-B
kernels in the same serving stack have not been qualified together, and their
ε bounds do not add in any way this policy models. A stack serving more than
one Tier-B kernel requires its own qualification, run on the composed stack.

**The ε bound is measured, not proven.** Gate (a) bounds the deviation *on the
captured task*. It is not a worst-case bound over inputs, and no claim of one
should be made from it. A candidate whose deviation is input-dependent in a way
the exact4/exact16 tasks do not exercise will pass (a) and still be wrong; the
only defence this policy offers against that is the breadth of gate (c) and
the stickiness of §4.

**Nothing here is retroactive.** No existing arm becomes Tier-B by this
document. Every arm on main today is Tier-A or is already covered by the
drafter exemption, and stays that way.

---

## 9. Identity

- policy commit baseline: `origin/main` at `c6cd0e92c`;
- sampler-region pin at that commit:
  `d93df2c2eaa2f88fe4db2c5939b5b8c9df6bd328fcf6500a3c551f8041bbb785`;
- patcher whole-file sha256:
  `0696bfc5458d01cca72767c6b54432490463722840153ffbab3b732bf09235c2`;
- helper: `scripts/fr13_tier_b_sampler_pin.py`;
- tests: `tests/test_fr13_tier_b_sampler_pin.py` (12 passed).

No GPU, Docker, real SWE-Verified task, correctness gate, timing, acceptance,
or candidate-production claim is included in this artifact.
