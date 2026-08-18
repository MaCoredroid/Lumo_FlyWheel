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
comparisons, zero differing elements) and worth **−1.16 to −1.22 ms/step**. It is
implemented as a flag, gated, and **held one three-line diff short of wired** by a
campaign credential — see §4.

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

| | measured |
|---|---:|
| single-row softmax, clean microbench | **55.4 µs** |
| …effective bandwidth | **35.8 GB/s** |
| 24 of them, per step | **1.726 ms** |
| the same rows, one batched call | **0.570 ms** (all 31 rows) / **0.505 ms** (24 gathered) |
| **saving** | **1.156 – 1.222 ms/step** |

That is **5.6–5.9 % of the 20.604 ms `cfwd`** and **0.55–0.58 % of the 210.700 ms step**.
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

In `scripts/fr13_device_multidraft_kernel.py` (digest-neutral — see §4):

* `FR13_FIXED32_TAW_SOFTMAX_CACHE`, strict `0`/`1`, **default OFF**, env or sidecar
  (`/logs/fr13_fixed32_taw_softmax_cache.arm`), malformed values raise;
* `_fr13_fixed32_taw_softmax_cache_requested()` — deliberately *not* routed through
  `_fr13_fixed32_taw_native_selector`, so it cannot re-impose the all-parent B≥2 floor and
  the two levers can ship and be reverted independently;
* `_FR13_FIXED32_TAW_SOFTMAX_CACHE_TENSOR_CALL_CENSUS` — the census the wired lever would
  publish. It records that row gathers **rise** 24 → 26 (the per-level lookups still
  happen, they just read the cache; building the cache costs one more full-vocab gather per
  family). The lever is a trade and the census says so;
* `assert_softmax_cache_not_armed()` — **raises** if the flag is armed, because the walk is
  not wired and a lever that arms into a no-op is worse than one that refuses.

Tests: `tests/test_fr14_cfwd_softmax_cache.py`, 18 cases, all pass — strictness, sidecar
precedence, default-OFF, the refusal message, census honesty, and that the native selector
is unmoved at every batch width.

---

## 4. Why it is not wired, and the exact diff that wires it

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

**This was applied and reverted.** Applied, it drifts the TAW source digest and fails **24
existing tests whose entire job is to notice exactly that** (verified: 19 failed + 7 errors
with the diff, against a 2-failure pre-existing baseline). There is no wiring point outside
the digest set — every caller in the chain is itself pinned.

Re-attesting `_FR13_FIXED32_TAW_SOURCE_SHA256` invalidates banked TAW PASS credentials.
That is a campaign decision, not a lane decision, and it is not one to take unilaterally in
a shared tree mid-campaign. **The lever is measured, gated and staged; promoting it is one
credential decision away.**

> Noted in passing, not fixed by this lane: two digest assertions in
> `tests/test_fr13_taw_b1_diagnostic_pass_artifact.py` **already fail at HEAD** before any
> lane-3 change (`42b92d87… ≠ 484babd7…`, `7a67a3b0… ≠ 56c51ada…`). Whoever owns the TAW
> credential should know the baseline is not clean.

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
| softmax cache — arithmetic | **gates G1/G3/G4 PASS**, 400 bitwise comparisons | none |
| softmax cache — saving | **−1.16 to −1.22 ms/step**, isolated microbench | confirm in-walk (the microbench models softmax + gather, not the full level) |
| softmax cache — wiring | staged, three-line diff in §4 | **needs `_FR13_FIXED32_TAW_SOURCE_SHA256` re-attestation — Mark's call** |
| softmax cache — live A/B | not run | after wiring: `cfwd` span A/B at B=1, `s_per_fwd_gpu` flat, work-census `full_vocab_softmax_calls` 24→2 as the engagement needle |
| sub-lever A | closed | — |
| sub-lever B | stopped | reopen only if `P(spine ∧ len == d) > 12.2 %` for some fixed `d` is measured |

**Caveat on the timing numbers, and on what is banked.** The probe was run twice. Run 1
was taken against a verified **zero-container** GPU and is the source of every timing
quoted in this note:

| | run 1 (uncontended) | run 2 (one foreign container up) |
|---|---:|---:|
| incumbent, 24 single-row softmax | **1.726 ms** | 2.260 ms |
| batched, all 31 rows | **0.570 ms** | 0.588 ms |
| batched, 24 gathered rows | **0.505 ms** | 0.537 ms |
| single row | **55.4 µs** (35.8 GB/s) | 66.6 µs (29.8 GB/s) |

`committer_softmax_probe.json` holds **run 2** (it overwrote run 1 at the same path) plus
the composite gates G3/G4, which were added between the runs. Read its `G1`/`G3`/`G4`
verdicts and its `B1` copy floor; read its `T1` as the **contended** column above. Gates
are bitwise and unaffected by contention — G1 passed identically in both runs (160 and 120
cases).

**G5 is implemented but has not been run.** It tests the module's own stated justification
for `_FR13_FIXED32_TAW_PINNED_MIN_BATCH = 2` — *"cumsum over a single [1, V] row is
run-to-run non-deterministic on this device"* — which, if true, would mean the **deployed
B=1 walk is already non-reproducible step to step**. That is worth knowing independently of
this lane. It was not run because two foreign containers held the GPU at the end of this
lane's window and a third process could have contaminated another lane's measurement.
Nothing in this note depends on its outcome: the softmax lever does not touch the cumsum,
which stays at the same `[B, V]` shape in both branches. Run it with
`--skip-bench` (~2 s, ~250 MB) on an empty GPU.

---

## 7. Evidence index

| artifact | what it carries |
|---|---|
| `scripts/fr14_cfwd_softmax_batching_probe.py` | gates G1–G5, benches T1/B1 |
| `results/fr14_nvfp4_port_20260816/committer_softmax_probe.json` | banked probe output |
| `tests/test_fr14_cfwd_softmax_cache.py` | 18 cases: flag strictness, refusal, census honesty |
| `scripts/fr13_device_multidraft_kernel.py` | flag, resolver, recorded census, arm guard |
| `output/fr13_fixed32_b1_nsys_20260818T001018Z/…/fr13_fixed32_b1_real_swe.sqlite` | the span/busy/gap re-derivation of §2 |
| `output/…/logs/fr13_fixed32_work_census.jsonl` | `taw.*`, `committer.*` live route identity |
| `results/fr14_nvfp4_port_20260816/host_dfwd_characterization.md` | §1 phase budget, §4 prior cfwd ceiling |
| `git show HEAD:FR13_SPINE_COMMIT_DESIGN.md` | the 53.2 % spine-accept measurement |
