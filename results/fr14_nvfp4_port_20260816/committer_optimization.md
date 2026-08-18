# FR14 lane 3 — committer optimizations: what the cfwd profile actually supports

**Scope:** the two greenlit sub-levers (A: commit-sequence fusion; B: spine-state
checkpointing), measured before being built.

`analysis_only=true`, `acceptance_valid=false`, `citable=false`. No TPS, floor or
acceptance claim is made or moved. Timings are `step_wall_ms` / phase spans per the
instrument doctrine; no TPS number appears in this note.

---

## 0. Headline

Both sub-levers were briefed against a description of `cfwd` that its own profile does
not support. Measured at HEAD:

| briefed | briefed value | measured status | what actually remains |
|---|---:|---|---:|
| **A** — fuse/CUDA-graph the 387 "elementwise/glue" launches; "LAUNCH-BOUND" | −5..6 ms | **premise refuted.** `cfwd` is 84.5 % GPU-**busy** (18.066 ms busy inside a 21.383 ms span). The commit sequence is *already* CUDA-graphed and required-on. The 387 launches are not KV slot-scatter/state-copy glue — they are the exact-float tree rejection walk, and they are real bandwidth. | **3.316 ms** of total GPU gap, nsys-inflated; prior banked host-idle **0.809 ms** |
| **B** — checkpoint spine states in `sfwd`, skip the accepted-path replay | −3 ms | **STOP.** Net **+3.5 to +5.1 ms regression** by measurement, and the bitwise gate has no candidate. | — |

**What is real instead, and is new:** inside `cfwd`'s 18.066 ms of GPU-busy time there is
exactly one op class that is *not* bandwidth-bound. The tree-accept walk issues **24
full-vocabulary softmaxes per step, each over a single row**, and ATen's
`cunn_SoftMaxForward` gives one row **one block** — so each call occupies **one SM** and
runs at **35.8 GB/s**, ~8× off the ~273 GB/s the rest of `cfwd` achieves.

Batching them into one softmax per family is **bitwise identical** (measured, 400
comparisons, zero differing elements) and worth **−1.01 to −1.05 ms/step**. It is
implemented, gated, **wired and default-OFF** — see §4, including the coordinator-sanctioned
`_FR13_FIXED32_TAW_SOURCE_SHA256` re-attestation and the attribution (§3.4) that had to
precede it.

A second measured result, incidental to the lever but larger in consequence than it
(§3.5): **at the served B=1 the walk's own `cumsum` is run-to-run non-deterministic** —
20/20 repeats differ, up to 2 ULP — while **every batched width 2/4/12/31 is perfectly
deterministic, 0/20**. The deployed width is the only non-reproducible one.

---

## 1. Method

Two independent instruments, neither of which required taking the GPU from another lane:

1. **The banked nsys capture** `output/fr13_fixed32_b1_nsys_20260818T001018Z/…/logs/fr13_fixed32_b1_real_swe.sqlite`
   (759 MB, 5.09 M kernel rows, 1 388 `fr13.fixed32.cfwd` NVTX ranges). Per range, the
   launching `CUPTI_ACTIVITY_KIND_RUNTIME` rows were joined to kernels by `correlationId`,
   then GPU **span** (first start → last end), GPU **busy** (union of kernel intervals) and
   **gap** (span − busy) were computed. Reported as the median over a 200-range
   steady-state sample.
2. **A standalone probe**, `scripts/fr14_cfwd_softmax_batching_probe.py`, run in the pinned
   engine image (`vllm/vllm-openai@sha256:3dbe092e…`, torch 2.11.0+cu130, GB10, sm_121).
   Output banked at `results/fr14_nvfp4_port_20260816/committer_softmax_probe.json`.

Denominators are the banked non-nsys phase budget (maxstack arm, 32 809 decode steps,
`host_dfwd_characterization.md` §1): **step 210.700 · sfwd 130.437 · dfwd 52.235 ·
cfwd 20.604 · other 7.424 ms**. nsys inflates host launch cost heavily (the `cfwd` *host*
NVTX range measures 143 ms against 21.4 ms of GPU span), so **gap figures from the capture
are upper bounds** and are labelled as such throughout.

---

## 2. What `cfwd` is, measured

Median per `cfwd` invocation, 200-range sample:

| quantity | value |
|---|---:|
| GPU span | **21.383 ms** |
| GPU busy (union) | **18.066 ms** |
| GPU gap (span − busy) | **3.316 ms** |
| kernels launched | **1 189** |
| …of which already CUDA-graph nodes | **78** (4.420 ms busy) |

The 78 graph nodes are the committer: `graphId = 809`, containing all 48
`fused_sigmoid_gating_delta_rule_update_kernel` instances, whose summed preceding-gap
across all 48 launches is **0.005 ms**. The GDN commit sequence is already one graph
replay — `committer.route = fixed16_device_fill_graph`, `graph_replays = 1` in the live
work census — and `FR13_COMMITTER_GRAPH` / `FR13_COMMITTER_BATCHED` are *required-on*, not
optional.

Where the 18.066 ms of busy time sits, and where the 3.316 ms of gap sits:

| kernel class | n/step | busy ms | gap ms |
|---|---:|---:|---:|
| `fused_sigmoid_gating_delta_rule_update` (graphed) | 48 | 4.066 | 0.005 |
| `vectorized_elementwise_kernel` | 480 | 3.683 | 1.346 |
| `elementwise_kernel` | 149 | 3.578 | 0.329 |
| **`cunn_SoftMaxForward`** | **26** | **2.261** | 0.051 |
| `_topk_topp_kernel` | 2 | 1.197 | 0.006 |
| `unrolled_elementwise_kernel` | 131 | 0.613 | 0.316 |
| `_scatter_gather_elementwise_kernel` | 56 | 0.609 | 0.148 |
| `apply_repetition_penalties_kernel` | 3 | 0.464 | 0.007 |
| `index_elementwise_kernel` | 73 | 0.460 | 0.458 |
| `reduce_kernel` | 104 | 0.417 | 0.356 |
| remainder | ~143 | ~0.72 | ~0.29 |

The live work census identifies the source:
`taw.route = fixed32_pytorch_exact_float_triton_integer_commit`, `walk_levels = 12`,
`vocab_size = 248 320`, `self_shape = target_shape = [31, 248 320]` fp32,
`full_vocab_softmax_calls = 24`. The "387 glue launches" are the exact-float tree
rejection walk over a 248 320-entry vocabulary — **arithmetic on 4–5 GB/step of logits**,
not bookkeeping.

### 2.1 Sub-lever A: verdict

The brief priced fusing/graphing the glue at −5..6 ms on the premise that it was
launch-overhead-dominated. The entire gap budget in `cfwd` is **3.316 ms**, that figure is
an nsys-inflated upper bound, and the previously banked measurement of the same quantity
under a clean serve is **0.809 ms** (`results/fr13_host_residual_20260811/design.md`
segment E, "sub-noise"). The largest single gap line is 480 `vectorized_elementwise`
launches at 2.8 µs each — i.e. already at the launch-latency floor.

**There is no 5–6 ms of launch overhead in `cfwd` to recover.** `FR13_FIXED32_CONV_COMMIT_BATCHED_SLOTS`
(`src/lumo_flywheel_serving/fr13_fixed32_commit_slot_scatter.py`), the KV slot-scatter
batching the brief pointed at, is by its own arithmetic worth **0 launches at B=1** — it is
a B4 lever and is documented as one. The deployed serve is B=1.

Sub-lever A as briefed is **closed**. What replaces it is §3.

---

## 3. The lever the profile does support: batch the single-row softmax

### 3.1 The defect

Every walk level, for each of two families (self, target), the reference walk runs:

```python
self_indices = starts + current.clamp(...)                         # shape [B]
self_prob = torch.softmax(X[self_indices].to(torch.float32), dim=-1)   # [B, V]
self_prob = self_prob / self_prob.sum(dim=-1, keepdim=True)
```

At the served **B = 1** the softmax input is `[1, 248320]`. ATen dispatches
`cunn_SoftMaxForward` with **grid = one block per row**, so a one-row softmax is a
**one-block, one-SM** kernel. nsys confirms grid `(1,1,1)` on every instance.

Best-controlled run (GPU verified **empty immediately before and after**, atomically):

| | measured |
|---|---:|
| single-row softmax | **55.5 µs** |
| …effective bandwidth | **35.8 GB/s** |
| 24 of them, per step | **1.546 ms** |
| the same rows, one batched call | **0.541 ms** (all 31 rows) / **0.498 ms** (24 gathered) |
| **saving** | **1.006 – 1.048 ms/step** |

That is **4.9–5.1 % of the 20.604 ms `cfwd`** and **0.48–0.50 % of the 210.700 ms step**.
Two earlier runs read a larger incumbent (1.726 / 2.260 ms) and hence a larger saving
(1.16–1.22 / 1.67–1.72); the run quoted here is the only one with a verified-empty GPU on
both sides, so **−1.0 ms is the floor of the claim, not its centre**. Full three-run table
in §6.
It is a reduction in GPU-**busy** time, not in launch overhead, so it is not subject to the
0.809 ms host-idle ceiling that closes sub-lever A.

### 3.2 The claim that was blocking it, and its measurement

`fr13_device_multidraft_kernel.py` states, in the byte-identity contract of
`_fr13_dm_depthsync_walk`:

> single-row softmax (**never batched: a stacked [A,V] softmax could shift p by 1 ULP**)

and sets `_FR13_FIXED32_TAW_PINNED_MIN_BATCH = 2`, refusing the whole batched family at
B=1. "Could" is a hypothesis. It was measured.

| gate | what it compares | result |
|---|---|---|
| **G1** | `torch.softmax(X[rows])[i]` vs `torch.softmax(X[rows[i]])`, V=248 320 fp32, widths 2/3/4/8/12/16/24/31 × 5 logit regimes (normal, ×40 dynamic range, single dominant logit, massive ties, bf16→fp32) | **PASS** — 160/160 cases bitwise equal, **0 differing elements** |
| **G3** | the **whole per-level expression at exactly B=1**: reference `softmax([1,V]) → /sum` vs cache `softmax([12,V]) → index → /sum` | **PASS** — 240/240 comparisons bitwise equal |
| **G4** | `softmax([V])` vs `softmax([1,V])` — that the two ranks dispatch identically, so the other gates compare what they claim to | **PASS** — 60/60 |
| **G2** | *diagnostic:* batched row-**sum** vs per-row sum | **differs**, up to **2 ULP**, 77/160 cases |

Comparisons are on `int32` bit patterns, not `allclose`.

G2 is the load-bearing negative result: **the shape sensitivity is real, but it lives in
the reductions, not in the softmax.** That is exactly why `_FR13_FIXED32_TAW_PINNED_MIN_BATCH = 2`
exists — the all-parent candidate batches the row sums and the cumsum across parents. This
lever crosses none of it: it batches only the softmax and leaves every reduction at the
`[B, V]` shape it has today, textually unchanged in both branches. **The B≥2 floor is
correct for the all-parent candidate and over-broad for the softmax alone.**

### 3.3 What landed

In `scripts/fr13_device_multidraft_kernel.py` (the re-attestation this required is §4):

* `FR13_FIXED32_TAW_SOFTMAX_CACHE`, strict `0`/`1`, **default OFF**, env or sidecar
  (`/logs/fr13_fixed32_taw_softmax_cache.arm`), malformed values raise;
* `_fr13_fixed32_taw_softmax_cache_requested()` — deliberately *not* routed through
  `_fr13_fixed32_taw_native_selector`, so it cannot re-impose the all-parent B≥2 floor and
  the two levers can ship and be reverted independently;
* `_FR13_FIXED32_TAW_SOFTMAX_CACHE_TENSOR_CALL_CENSUS` — the census published when the
  lever is armed. It records that row gathers **rise** 24 → 26 (the per-level lookups still
  happen, they just read the cache; building the cache costs one more full-vocab gather per
  family). The lever is a trade and the census says so;
* `_fr13_fixed32_taw_probability_cache_requested()` — the shared resolver now read by all
  three walk arm points. `OR`, not `AND`: either the native selector or this flag is
  sufficient to want the cache, and with the flag clear it reduces to exactly the
  pre-lane-3 expression.

Tests: `tests/test_fr14_cfwd_softmax_cache.py`, 18 cases, all pass — strictness, sidecar
precedence, default-OFF, unarmed reduction to the incumbent expression in **both** states of
the native selector, census-follows-the-arm, all three arm points wired, the re-attested
digest matching what the wired module computes, and the native selector unmoved at every
batch width.

---

### 3.3.1 G5 — the deployed B=1 walk is the only non-deterministic width

`fr13_device_multidraft_kernel.py` justifies `_FR13_FIXED32_TAW_PINNED_MIN_BATCH = 2` with:

> At B=1 the reference operator itself is not reproducible (cumsum over a single [1, V] row
> is run-to-run non-deterministic on this device), so no byte-exact batched candidate can
> exist there and the walk stays unbatched.

Measured — same `cumsum` run 20× on identical input, bit-compared:

| width | non-deterministic repeats | max ULP |
|---:|---:|---:|
| **1 (the served width)** | **20 / 20** | **2** |
| 2 | 0 / 20 | 0 |
| 4 | 0 / 20 | 0 |
| 12 | 0 / 20 | 0 |
| 31 | 0 / 20 | 0 |

**The claim is correct, and sharper than stated: B=1 is not merely "not reproducible", it is
the *only* width that is not.** Every batched width is bit-stable across repeats. (Plausibly
a dispatch change — a decoupled-lookback / multi-block scan for a single long row versus a
block-per-row scan once there are rows to spread — but the mechanism is not established
here, only the behaviour.)

Three consequences, in order of importance:

1. **The deployed hydra27_fixed32 B=1 serve runs a sampler whose CDF inversion is not
   run-to-run reproducible, at up to 2 ULP.** No end-to-end byte gate of the B=1 walk can
   exist — not for this lever, not for any other. Anything in TAW territory that claims
   end-to-end byte identity at B=1 is claiming something the operator cannot deliver. This
   is for whoever owns the TAW credential; it is flagged, not acted on, by this lane.
2. **It does not touch this lever.** The softmax cache does not go near the `cumsum`, which
   stays at the same `[B, V]` shape and the same call site in both arms. The correct gate is
   the stage-wise one (G1/G3/G4), which is what was run.
3. Worth noting for calibration: the shape-pinning discipline that produced
   `_FR13_FIXED32_TAW_PINNED_MIN_BATCH = 2` guards against a **2 ULP** batched-reduction
   difference (G2), while the deployed path already carries **2 ULP** of run-to-run noise
   from its own `cumsum`. These are not the same thing — a deterministic candidate is worth
   having even against a noisy reference, and a candidate must not *add* error — but the
   comparison is the right one to have in hand when pricing how much the B≥2 floor is
   buying.

---

### 3.4 Attribution of the pre-existing HEAD credential drift

Ordered ahead of the wiring by the coordinator, so that a sanctioned digest drift could not
launder an unexplained one. **Both failures trace to a single FR13-era commit; none of
tonight's five lanes are implicated, and no code landed without its re-gate.**

Failing tests, in `tests/test_fr13_taw_b1_diagnostic_pass_artifact.py`, both reading the
banked artifact `results/fr13_fixed32_taw_b1_diagnostic_pass_20260731T162536Z/`
(published once, in `c8d8bda91`, 2026‑07‑31, **never refreshed since**):

| # | test | assertion | expected (banked) | observed (HEAD, pre-lane-3) |
|---|---|---|---|---|
| 1 | `test_diagnostic_pass_cannot_arm_current_production` (:110) | `current_production_requirement.json → required_payload.source_contract_sha256` == `_FR13_FIXED32_TAW_SOURCE_SHA256` | `42b92d872d2324bf618b35fdd71c22d0e68e5c00e25ad2a43ae553c8ab1f92da` | `484babd7a883c81c7317ef23862940143c248dcbc1b66c9d4ac6775ff5a0fa93` |
| 2 | `test_candidate_math_projection_is_identical_but_control_source_is_not` (:134) | AST projection of HEAD's TAW source == `source_equivalence.json → candidate_math_projection.current_sha256` | `56c51ada155df8bea5d67a2af4d4a9744b999068f15bb27e0ca0c81327993763` | `7a67a3b0a8a732e04fdd9099993282ee5fee60e00c4172de6a49e83132bfcaed` |

**Culprit: `a5110fe71` — 2026‑08‑01, "FR13: fuse fixed32 all-parent committer walk".**
Verified by recomputing both quantities at every commit touching the TAW source since the
artifact's own reference tip (`1f5b63c16`):

* **Failure 1.** `a5110fe71` re-attested `_FR13_FIXED32_TAW_SOURCE_SHA256`
  `42b92d87… → 51541928…` without refreshing the banked
  `current_production_requirement.json`, which is still frozen at `42b92d87…`. The constant
  has since been legitimately re-attested **seven more times**
  (`0c91a7503`, `f989ad3b2`, `c83b9639a` 08‑01; `87344abdc` 08‑05; `b48dfe124`, `4a3e55851`,
  `0adbfb6e4` 08‑15), each carrying the stale fixture forward.
* **Failure 2.** The projection was `56c51ada…` at `a5110fe71~1` and `e50bea3e…` at
  `a5110fe71` — the same commit — via `_fr13_fixed32_layout_contract` and
  `_FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_TENSOR_CALL_CENSUS`, both inside the pinned
  projection set. It moved once more, to `7a67a3b0…`, at `b48dfe124` (2026‑08‑15,
  "widen the TAW byte gate and shape-pin the batched candidate") via
  `_fr13_fixed32_taw_execute{_torch,_exact_cuda}`, `_fr13_fixed32_taw_execute` and
  `_fr13_taw_inv_cdf`.

**This is a stale-fixture failure, not an ungated landing.** The re-attestation mechanism
ran correctly on all eight commits — the constant was updated every time. What was never
updated is one banked 2026‑07‑31 artifact that the test compares against **HEAD**.

Design comment, since it will recur: `source_equivalence.json`'s `current_source` block and
`current_production_requirement.json` are **snapshots of the tip as it stood on 2026‑07‑31**.
Asserting them against live HEAD guarantees a failure the next time anything in the
projection set legitimately changes — which is exactly what happened, twice. Either the
artifact's `current_*` fields should be regenerated as part of every TAW re-attestation, or
the test should assert the historical *relationship* the artifact was banked to prove
(`run_sha256 == current_sha256` at bank time, plus `diagnostic_pass ≠ current requirement`)
rather than re-deriving `current` from today's source. `run_projection == run_sha256` still
passes, so the historical half of the credential is intact. **Recommend the second fix**;
it keeps the credential meaningful without requiring a fixture rewrite on every legitimate
source change.

---

## 4. The wiring, and the sanctioned re-attestation

The walk chain — `_fr13_fixed32_taw_execute{,_torch,_exact_cuda}`,
`_fr13_fixed32_taw_probability_caches`, `fr13_fixed32_taw_commit`,
`_fr13_taw_inv_cdf{,_parts}`, `_fr13_fixed32_taw_pinned_*` — is **source-digest pinned**
(`_FR13_FIXED32_TAW_SOURCE_SHA256`, plus banked B1/B4 PASS artifacts under `results/`).

Those functions **already contain** the cache branch this lever needs
(`self_prob = self_prob_cache[self_indices]`); it is selected by how `native_precompute`
resolves *inside* them. So the wiring is three identical lines:

```diff
-        native_precompute = _fr13_fixed32_taw_native_precompute_enabled()
+        native_precompute = (
+            _fr13_fixed32_taw_native_precompute_enabled()
+            or _fr13_fixed32_taw_softmax_cache_requested()
+        )
```

at the three sites in `_fr13_fixed32_taw_probability_caches`,
`_fr13_fixed32_taw_execute_torch` and `_fr13_fixed32_taw_execute_exact_cuda`, plus routing
`_fr13_fixed32_taw_tensor_call_census` to the recorded census when armed.

There is no wiring point outside the digest set — every caller in the chain is itself
pinned — so wiring necessarily re-attests the constant. Applied without re-attestation it
fails **24 tests whose entire job is to notice exactly that** (19 failed + 7 errors, against
the 2-failure baseline of §3.4).

**Sanctioned and landed 2026‑08‑18**, after §3.4's attribution, so that a sanctioned drift
could not launder an unexplained one:

| | value |
|---|---|
| prior `_FR13_FIXED32_TAW_SOURCE_SHA256` | `484babd7a883c81c7317ef23862940143c248dcbc1b66c9d4ac6775ff5a0fa93` |
| re-attested to | `68b289aee5773edf1134f184c37551a90ec8543430d768a05066bc1341473c6d` |
| candidate-math projection | `7a67a3b0…` → `319ca0bd61fc0ca9ecc314bcc22a5c968ac638907e155e58263daf8f491ad63a` |

The prior value is the digest the **2026‑08‑18 B1 nsys serve actually ran under**
(`taw.source_contract_sha256` in that run's work census), which is what makes it the right
thing to name as the predecessor.

**The only source change under this re-attestation is how `native_precompute` resolves at
the three arm points.** No sampler arithmetic moved. With the flag clear,
`_fr13_fixed32_taw_probability_cache_requested()` reduces exactly to
`_fr13_fixed32_taw_native_precompute_enabled()` — the pre-lane-3 expression — which is
asserted directly, in both states of the native selector, by
`test_unarmed_resolution_is_identical_to_the_pre_lane3_expression`.

Post-landing test state: **78 passed, 3 skipped, 2 failed** — the two failures being exactly
the §3.4 stale-fixture pair, unchanged in kind and now showing this lane's digests as the
observed side. All 24 digest-drift failures cleared.

**Deferred, and owed:** re-earning the banked *live* TAW PASS artifacts (which need a serve,
not a unit test) is folded into the next TAW gate re-earn per the coordinator's ruling. The
constant is re-attested here; the artifacts are not.

---

## 4.1 Mirror inventory — the re-attestation missed eleven more sites

The promotion campaign's gate re-earn was refused by
`scripts/fr13_fixed32_work_census.py:157`, which still carried `484babd7…` while the
emitter published `68b289ae…`. The census **self-asserts on every boot**, so serves
completed and their *terminal audits* died; every fixed32 credential re-earn at this HEAD
was refused and arms ran the qrow16 incumbent.

A sweep found the campaign had hit **the mirror that fires first, not the only one.**

### Class A — `_FR13_FIXED32_TAW_SOURCE_SHA256` retyped (12 literals, 11 files) — **FIXED**

Every one is bound to the live `scripts/fr13_device_multidraft_kernel.py`, not to a banked
artifact, so every one must track HEAD. All updated to `68b289ae…`:

| file | symbol |
|---|---|
| `scripts/fr13_fixed32_work_census.py:157` | `TAW_SOURCE_CONTRACT_SHA256` *(the blocker)* |
| `scripts/fr13_taw_b1_credential.py:21` | `SOURCE_CONTRACT_SHA256` (compares `module._FR13_FIXED32_TAW_SOURCE_SHA256`) |
| `scripts/fr13_dfwd_k64_m4_r64_u8_gate.py:39` | `TAW_SOURCE_CONTRACT_SHA256` |
| `scripts/fr13_dfwd_k64_m1_r64_u8_gate.py:46` | `EXPECTED_TAW_SOURCE_CONTRACT_SHA256` |
| `scripts/fr13_run_b1_k64_taw_source_v7_gate.sh:25` | `TAW_SOURCE_CONTRACT_SHA256` |
| `scripts/fr13_run_b4_tail23_all_parent_live_gate.sh:23` | `TAW_SOURCE_SHA256` |
| `scripts/fr13_run_b1_k64_physical32_fullstack_pair.sh:36` | `TAW_SOURCE_CONTRACT_SHA256` |
| `scripts/fr13_run_b4_taw_width4_timing.sh:103` | `TAW_SOURCE_SHA256` |
| `tests/test_fr13_fixed32_cfwd_logit_direct_runners.py:155` | inline literal |
| `tests/test_fr13_fixed32_cfwd_logit_direct_decision.py:707,755` | inline literals |
| `tests/test_fr13_fixed32_cfwd_logit_direct_live_gate.py:224` | inline literal |

Deliberately **not** changed: `fr13_device_multidraft_kernel.py:1662`, which is prose
recording the prior value, and the historical records in `results/` and the refusal log.

### Class B — whole-file SHA of the kernel (4 sites) — **NOT FIXED, NOT OURS**

| file | symbol | pinned |
|---|---|---|
| `scripts/fr13_device_multidraft_cfwd_packed_v3.py:14` | `BASE_SHA256` | `8dbb0bd0…` |
| `scripts/fr13_cfwd_logit_direct_packed_runtime_overlay.py:14` | `BASE_SOURCE_SHA256` | `8dbb0bd0…` |
| `scripts/fr13_generate_cfwd_packed_runtime_overlay.py:156` | `BASE_SOURCE_SHA256` | `8dbb0bd0…` |
| `tests/test_fr13_fixed32_cfwd_logit_direct_decision.py:828` | inline literal | `8dbb0bd0…` |

`8dbb0bd0…` was the kernel's file SHA at **`0adbfb6e4` (2026‑08‑15)**. The file SHA at
lane 3's parent `263a12f79` was already `deb5c9da…` — **so these were stale before this
lane touched anything**, broken by whichever commit changed the kernel after `0adbfb6e4`.
Lane 3 compounds it (now `4dd05fbd…`) but did not cause it. They break 10
`cfwd_logit_direct` tests with `credential-bound device module identity drifted` and would
refuse the packed CFWD runtime; **none of the 10 failures is digest-related** (verified
individually).

**Left for the `cfwd_logit_direct` owner deliberately.** Re-pointing `BASE_SHA256` is not
mechanical: the pin's meaning is *"the overlay was reviewed against this base"*, and the
overlay installs over the very walk functions this lane changed. Re-pinning would assert a
compatibility this lane has not established — which is precisely the "don't launder one
drift under another" discipline the census defect illustrates.

### Class C — the armed-lever census route — **latent, must fix before the live A/B**

`fr13_fixed32_work_census.py` mirrors the tensor-call census too, and dispatches expected
counts by `taw.route`. With `FR13_FIXED32_TAW_SOFTMAX_CACHE` armed the emitter publishes
`full_vocab_softmax_calls: 2` / `row_gathers: 26` under the **unchanged** default route
string, so the validator would compare against `TAW_TENSOR_CALL_CENSUS` (24/24) and refuse.
Blocks nothing today — the flag is default-OFF — but it **will** kill the live A/B
recommended in §6. The fix is a distinct route name plus a matching validator arm, which
re-attests the emitter a third time; deliberately not done under time pressure while gate
re-earns were blocked.

### The structural fix

Per the test-the-contract doctrine, retyping twelve literals correctly once is not a fix for
a defect class. `tests/test_fr14_cfwd_softmax_cache.py` now carries
`test_every_taw_source_digest_mirror_matches_the_emitter`, which scans every `.py`/`.sh`
under `scripts/` and `tests/` for `*TAW_SOURCE[_CONTRACT]_SHA256` assignments and requires
each to equal what the emitter publishes, plus
`test_work_census_mirror_tracks_the_emitter`, which asserts the boot-critical mirror against
the emitter's constant directly rather than a retyped literal. The scan self-checks that it
matched at least 8 files, so a regex that silently stops matching cannot make it a no-op.

**Verified by reintroducing the exact reported defect:** reverting only
`fr13_fixed32_work_census.py` to `484babd7…` makes both tests fail; restoring it makes them
pass. A future re-attestation that misses a mirror now fails in a unit test rather than in a
terminal audit after a completed serve.

Importing the digest from the emitter instead of retyping it was considered and rejected for
the census: it is a standalone fail-closed validator that runs without torch/triton, and
importing the kernel would drag the whole CUDA stack into offline census validation. The
contract test gets the same guarantee at no runtime cost.

---

## 5. Sub-lever B — spine-state checkpointing: **STOP**

### 5.1 The replay is state-traffic-bound, not token-bound

The accepted-path replay is 48 `fused_sigmoid_gating_delta_rule_update` launches,
**4.420 ms/step**, grid `(1,4,48)`, all inside graph 809. Over 66 624 instances the
duration is **median 82.2 µs, p10 81.0, p90 85.2** — a 5 % spread across accepted lengths
that vary 1..12.

That flatness is the decisive measurement. The replay costs one read and one write of the
recurrent state (48 layers × 48 v-heads × 128 × 128 × fp32 = **144 MiB**) regardless of how
many tokens it replays. **Therefore starting a replay from a checkpoint saves essentially
nothing** — only skipping the replay entirely does.

### 5.2 The economics, with measured constants

Measured state-copy floor (probe B1, GB10): 144 MiB copied in **1.288 ms** at
**234.4 GB/s**; 48 per-layer copies **1.398 ms**.

| leg | value |
|---|---:|
| replay avoided, per firing | 4.420 ms |
| commit by copy → best-case saving | 4.420 − 1.288 = **3.13 ms** |
| commit by bank-row index swap → best-case saving | **4.42 ms** |
| **export cost, one spine depth** (144 MiB written in `sfwd`, every step) | **≈ 0.54–0.64 ms** |
| **export cost, all 11–12 spine depths** (1.55–1.73 GiB) | **≈ 5.9–7.4 ms** |
| P(spine ⏐ accepted), measured | **53.2 %** (790 spine / 695 branch / 515 empty, 2 000 commits) |

Full-coverage net: saving `0.532 × 4.42 = 2.35 ms` against `5.9–7.4 ms` paid every step ⇒
**net +3.5 to +5.1 ms/step REGRESSION.**

Single-depth variant break-even: it pays only if
`P(spine ∧ accepted length == d) > 0.54/4.42 ≈ **12.2 %**` (row-swap commit) or
`> 0.54/3.13 ≈ **17.3 %**` (copy commit) — for one fixed depth `d`. That distribution is
**not measured anywhere at FR14**: `seam_move_economics.md` §2 records that no per-step
accepted-length records exist across 22 runs / 228 036 steps. The 53.2 % figure itself is
FR13-era (tail6, B=4, temp 0.6; `git show HEAD:FR13_SPINE_COMMIT_DESIGN.md`) and is an
upper bound on firing rate, not the per-depth number the single-depth variant needs.

### 5.3 The bitwise gate has no candidate

Independently of the economics, the brief's own stop condition applies. The checkpoint
would be produced by the **`sfwd` tree-GDN Triton kernel**; the committed state is produced
today by **FLA's `fused_sigmoid_gating_delta_rule_update`** (`FR13_COMMITTER_NATIVE`,
default-on, sidecar-armed at `fr14_armb_leg3_launch_nomiddleware.sh:4734`). These are two
different kernels with different tiling and accumulation order. Bitwise equality between
them is not something a gate can be expected to find, and the brief is explicit that
approximate state equality must not ship.

The only construction that avoids this is running the FLA kernel during `sfwd` to make the
checkpoint — which is the replay, moved earlier, at the same cost.

**Sub-lever B is stopped on three independent grounds:** the replay is length-invariant so
checkpoint-and-resume saves nothing; full-coverage export is a measured net regression of
+3.5..5.1 ms/step; and the bitwise gate has no candidate by construction. Not banked as
"refuted pending re-measurement" — banked as **stopped**, with §5.2's break-even
inequality as the falsifiable condition under which it could be reopened.

---

## 6. What remains for serve-promotion

| item | state | next step |
|---|---|---|
| softmax cache — arithmetic | **gates G1/G3/G4 PASS**, 460 bitwise comparisons, 0 differing elements | none |
| softmax cache — saving | **−1.01 to −1.05 ms/step** (best-controlled run), isolated microbench | confirm in-walk — the microbench models softmax + gather, not the whole level |
| softmax cache — wiring | **landed**, default OFF, digest re-attested `484babd7…` → `68b289ae…` | — |
| softmax cache — unarmed safety | **asserted**: resolver reduces to the pre-lane-3 expression in both selector states | — |
| softmax cache — live A/B | **not run** | arm the sidecar, then `cfwd` span A/B at B=1 with `s_per_fwd_gpu` held flat; engagement needle is work-census `full_vocab_softmax_calls` 24 → 2 |
| TAW banked PASS artifacts | **owed** — re-attestation covered the constant, not the live artifacts | fold into the next TAW gate re-earn (needs a serve) |
| class-A digest mirrors (§4.1) | **fixed**, 12 literals / 11 files, guarded by a contract test | — |
| class-B whole-file SHA pins (§4.1) | **stale since before this lane**, 4 sites, 10 tests red | `cfwd_logit_direct` owner: re-pin `8dbb0bd0…` → `4dd05fbd…` **only** with an overlay-compatibility judgement |
| class-C armed-route census (§4.1) | **latent** — default-OFF so blocks nothing today | needs a route name + validator arm **before** the live A/B, or the A/B serve's audit dies |
| stale 2026-07-31 fixture (§3.4) | attributed to `a5110fe71`, **not fixed** | adopt §3.4's recommended test fix, or regenerate `current_*` on every re-attestation |
| B=1 cumsum non-determinism (§3.3.1) | **measured**, 20/20 at width 1, 0/20 at widths 2–31 | TAW credential owner: no end-to-end byte gate at B=1 is achievable |
| sub-lever A | closed | — |
| sub-lever B | stopped | reopen only if `P(spine ∧ len == d) > 12.2 %` for some fixed `d` is measured |

**Caveat on the timing numbers, and on what is banked.** The probe was run three times. The
banked `committer_softmax_probe.json` holds **run 3**, the only one with the GPU verified
empty *both before and after* (the check and the launch were made atomic after run 2 lost a
race to another lane). Every timing quoted in this note is run 3.

| | run 1 (0 containers at check) | run 2 (**one foreign container**) | **run 3 (verified empty both sides)** |
|---|---:|---:|---:|
| incumbent, 24 single-row softmax | 1.726 ms | 2.260 ms | **1.546 ms** |
| batched, all 31 rows | 0.570 ms | 0.588 ms | **0.541 ms** |
| batched, 24 gathered rows | 0.505 ms | 0.537 ms | **0.498 ms** |
| single row | 55.4 µs (35.8 GB/s) | 66.6 µs (29.8 GB/s) | **55.5 µs (35.8 GB/s)** |
| implied saving | 1.16–1.22 ms | 1.67–1.72 ms | **1.01–1.05 ms** |

The spread across runs is in the *incumbent* leg, not the batched one — 24 sequential
one-block kernels are exactly the shape most sensitive to any other work on the device.
**Treat −1.0 ms/step as the floor of the claim rather than its centre**, and note that all
three runs agree the lever is worth ≥1 ms. G1 passed identically in every run (160/160,
160/160, 120/120 by trial count); gates are bitwise and contention cannot move them.

## 7. Evidence index

| artifact | what it carries |
|---|---|
| `scripts/fr14_cfwd_softmax_batching_probe.py` | gates G1–G5, benches T1/B1 |
| `results/fr14_nvfp4_port_20260816/committer_softmax_probe.json` | banked probe output |
| `tests/test_fr14_cfwd_softmax_cache.py` | 18 cases: strictness, unarmed-reduction, census-follows-arm, arm points wired, digest re-attested |
| `scripts/fr13_device_multidraft_kernel.py` | flag, shared resolver, census, three wired arm points, re-attested `_FR13_FIXED32_TAW_SOURCE_SHA256` |
| `tests/test_fr13_taw_b1_diagnostic_pass_artifact.py` + `results/fr13_fixed32_taw_b1_diagnostic_pass_20260731T162536Z/` | the §3.4 stale fixture, attributed to `a5110fe71` |
| `output/fr13_fixed32_b1_nsys_20260818T001018Z/…/fr13_fixed32_b1_real_swe.sqlite` | the span/busy/gap re-derivation of §2 |
| `output/…/logs/fr13_fixed32_work_census.jsonl` | `taw.*`, `committer.*` live route identity |
| `results/fr14_nvfp4_port_20260816/host_dfwd_characterization.md` | §1 phase budget, §4 prior cfwd ceiling |
| `git show HEAD:FR13_SPINE_COMMIT_DESIGN.md` | the 53.2 % spine-accept measurement |
