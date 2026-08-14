# FR13 B4 — the honest width-4 floor, and what is actually inside the last 17.9%

**ANALYSIS ONLY. NOT CITABLE. `acceptance_valid=false`, `gpu_touched=false`.**
No arm was run. Nothing here changes a measured wall. It changes the
**denominator** those walls are divided by, and it opens the one kernel family
nobody had ever named.

* tools — `scripts/fr13_b4_honest_floor.py`, `scripts/fr13_b4_other_bucket_reduce.py`
* artifacts — `floor.json`, `other_bucket.json` (this directory)
* evidence — the 539-step width-4 nsys capture
  (`results/fr13_b4_width4_nsys_20260813`, raw sqlite on `/home/mark/shared`),
  `results/fr13_b4_prefill_gaps_20260813`, `results/fr13_attack_ladder_analysis_20260808`,
  `results/fr13_hardware_floor_correction_20260731`,
  `results/fr13_b4_padded_b4_derisk_20260813` (wave-model control),
  `output/fr13_b4_hydra27_sealing_campaign_20260814T011514Z` (the sealed post-lever point)

---

## 0. THE QUESTION

> "B4 sits at 3.0× its hardware floor and B1 sits at 1.94×. Does that mean B4
> has lots of room left?"

Three separate things are wrong with the comparison that produces 3.0 vs 1.94,
and they do not point the same way.

---

## 1. THE DENOMINATOR IS NOT THE SAME DENOMINATOR (this one makes B4 look *worse*)

`results/fr13_hardware_floor_correction_20260731/floor_ledger.json` publishes
three weight-only scenarios. Two are in live use:

| scenario | component formula | bytes | floor |
|---|---|---:|---:|
| `root_64k_five_64k_draft_heads` | target + verifier head + MTP×5 + **5 × 64K** draft heads | 32,666,638,208 | **119.658015414 ms** |
| `current_one_full_plus_four_64k_draft_heads` | target + verifier head + MTP×5 + **1 full-vocab root** + 4 × 64K | 34,538,346,368 | **126.51408926 ms** |

B1's 1.9419 divides by **119.658**. The width-4 artifacts divide by **126.514**
(`tests/test_fr13_b4_width4_window.py:60 FLOOR_MS = 126.514089260`).

But the sealed B4 arms launch with `FR13_DRAFT_VOCAB_ROOT=1`,
`FR13_DRAFT_VOCAB_K=65536` — verbatim from
`output/fr13_b4_hydra27_sealing_campaign_20260814T011514Z/run_00/launcher_meta.txt`:

```
draft_vocab_root=1
draft_vocab_k=65536
mandatory_weight_floor_ms=119.658015414
```

The root drafter head **is** the 64K subset head. B4 reads 32.667 GB of weights,
not 34.538 GB. The published width-4 floor_ratio is being divided by a floor for
weights the arm never reads.

| | wall | floor used | ratio |
|---|---:|---:|---:|
| B1 sealed (gqa_pair production default) | 232.360 | 119.658 | **1.9419** |
| B4 sealed post-lever, **as published** | 381.284 | 126.514 | 3.0138 |
| B4 sealed post-lever, **on B1's basis** | 381.284 | 119.658 | **3.1864** |
| B4 sealed stock (pre-lever), on B1's basis | 408.313 | 119.658 | 3.4123 |

**The honest weight-only gap is 3.19 vs 1.94 = 1.64×, not 3.01 vs 1.94 = 1.55×.**
Fixing the basis makes B4 look 6% *worse*. That correction has to be taken
before the one that makes it look better, or the second is not credible.

---

## 2. A WEIGHT-ONLY FLOOR IS STRUCTURALLY UNFAIR TO A WIDE BATCH

The floor ledger says so itself: `"nonweight_costs_included": false`, and
`results/fr13_fixed32_dfwd_k64_tc_real_b1_20260805/deploy_speed_fullwall.json:47`
spells out that "KV/state/activation traffic … are excluded."

Weight traffic is **batch-invariant** — the same 32.667 GB moves whether the step
serves one request or four. That is confirmed by the capture, not assumed: the
fp8 GEMM is 1.08× from width 1 to width 4 and the bf16 head GEMM is flat above
width 2. **Every other mandatory byte is per-request and is therefore 4× larger
at width 4.** A denominator built only from the invariant term flatters B1 and
penalises B4 by construction. The ratio 3.0 vs 1.94 is partly just an artifact of
what was left out.

### The honest floor

Same rule the B1 FA2 roofline used — *mandatory unique bytes / 273 GB/s* —
applied to every term, with the same per-request context for both batches.
Geometry is asserted against `scripts/fr13_fixed32_topology.py` and the served
config; `fr13_b4_honest_floor.py` raises rather than emitting if anything
mismatches. The check that matters: this geometry re-derives the published B1
FA2 floor to **4.3285 ms** against the ledger's 4.33.

Per request, per speculative step:

| term | bytes | what it is |
|---|---:|---|
| target tree-attention KV read | 16 × C × 4096 | 16 full-attn layers re-read the whole context |
| MTP drafter KV read | 5 × C × 4096 | 5 drafting passes over the 17th KV tensor |
| KV write, this step's tree | 17 × 32 × 4096 = 2,228,224 | 32 root-inclusive rows into all 17 cache tensors |
| **GDN recurrent carry** | 48 × (3,145,728 + 61,440) × 2 = **307,888,128** | read committed state, write new committed state |
| LM-head logits write + read | 32 × 248,320 × 2 × 2 = 31,784,960 | logits must be materialised then verified |
| draft-head logits write + read | 31 × 65,536 × 2 × 2 = 8,126,464 | |
| residual stream round trip | 64 × 32 × 5120 × 2 × 2 = 41,943,040 | cannot stay resident across 64 layers |

The GDN term is the **post-`single_launch` minimum**: one read and one write of
the committed state per request per GDN layer. The deletable per-node handoff
that `single_launch` removes is *not* counted, and neither are the 31
speculative rows of the shipped 34-row conv staging buffer — only the true
recurrent carry, `conv_kernel − 1 = 3` rows. Counting the shipped 34 rows instead
would add 0.89 ms/step at width 4; it is in `floor.json` as a sensitivity, not in
the headline.

At **C = 18,031 tokens/request** — the attack-ladder implied context measured on
the *same* SWE-Verified agent workload, used for both batches so the comparison
is apples-to-apples:

| | weights | non-weight | **honest floor** | wall | **honest ratio** |
|---|---:|---:|---:|---:|---:|
| **B1** | 119.658 | 7.117 | **126.775** | 232.360 | **1.8329** |
| **B4 (post-lever)** | 119.658 | 28.468 | **148.126** | 381.284 | **2.5741** |
| B4 at the 384.02 reading¹ | 119.658 | 28.468 | 148.126 | 384.021 | 2.5925 |
| B4 stock, pre-lever | 119.658 | 28.468 | 148.126 | 408.313 | 2.7565 |

¹ 411.05 sealed unprofiled width-4 wall − 27.029 sealed campaign gain.

**THE ANSWER TO THE QUESTION AS ASKED: 2.57 against 1.83, not 3.0 against 1.94.**
The gap between the two operating points is **1.40×**, not the 1.55× the
published pair implies and not the 1.64× the basis correction alone would give.

### Context sensitivity, stated honestly

C could not be re-measured at width 4 — §5 explains why the capture actively
rejects the inversion. So it is bounded instead:

| C (tokens/request) | provenance | B1 floor | B1 ratio | B4 floor | B4 ratio | gap |
|---:|---|---:|---:|---:|---:|---:|
| 12,000 | low band | 124.87 | 1.861 | 140.52 | 2.713 | 1.458 |
| **18,031** | **B1 attack-ladder measurement (central)** | **126.77** | **1.833** | **148.13** | **2.574** | **1.404** |
| 24,531 | token-flow: 137,128 prefill + 10,058 decode tokens entered KV in the 360.19 s window at ≈3 admissions ≈3 completions ⇒ ~49.1 k terminal context/task, mean resident ~24.5 k | 128.82 | 1.804 | 156.32 | 2.439 | 1.352 |
| 44,288 | **hard cap**: KV pool 177,152 tokens ÷ 4 slots | 135.05 | 1.721 | 181.22 | 2.104 | 1.223 |

**The direction is monotone and it is the whole point: every honest choice of C
shrinks the apparent B4 penalty, and the more context there is the more it
shrinks.** The 3.0-vs-1.94 headline is the *most* pessimistic reading available
and it is the one that was published.

Two further sensitivities are in `floor.json`: `mamba_ssm_dtype=float32` is a
PARKED losslessness contract, not physics — bf16 state would remove 2.21 ms/step
of floor at width 4 (and roughly that much of real traffic); and the conv-state
basis is worth 0.89 ms/step.

---

## 3. WHERE THE ROOM ACTUALLY LIVES

Per-width in-step kernel time, plain sum, from the 225 width-4 steps of the
capture (PROFILED — upper bounds). Plain sum equals the union on this trace
(gaps.json: 140.01 s vs 139.89 s), so the sums are honest.

| family | w1 | w2 | w3 | **w4** | w4/w1 |
|---|---:|---:|---:|---:|---:|
| GEMM fp8 blockwise | 123.37 | 126.30 | 130.80 | **133.01** | 1.08 |
| **other (unnamed until now)** | 43.97 | 48.46 | 66.42 | **83.92** | 1.91 |
| FA2 tree attention | 32.19 | 41.22 | 69.51 | **80.20** | 2.49 |
| GDN tree scan | 12.38 | 25.05 | 37.92 | **50.04** | 4.04 |
| bf16 GEMM (LM + draft heads) | 15.27 | 29.99 | 30.14 | **30.37** | 1.99 |
| unified attention (MTP) | 11.32 | 14.51 | 15.16 | **16.54** | 1.46 |
| GDN delta-rule | 4.00 | 7.42 | 10.24 | **12.98** | 3.25 |
| FA2 causal (prefill spill) | 1.86 | 2.54 | 4.30 | **4.95** | 2.66 |
| **in-step total** | 244.35 | 295.49 | 364.47 | **412.00** | |
| step wall (profiled) | 646.44 | 316.45 | 384.58 | **429.33** | |

*(w1 is 4 steps and is noise; w2 is the honest base. The 646 ms w1 wall is a
transition artifact, not a decode point.)*

Against the honest floor, component by component:

| component | measured w4 | floor | headroom | ×floor |
|---|---:|---:|---:|---:|
| GEMM fp8 blockwise | 133.01 | 94.72 | 38.29 | 1.40 |
| **other bucket** | 83.92 | ~0 | **83.92** | — |
| **FA2 tree attention** | 80.20 | 17.31 | **62.88** | 4.63 |
| **GDN scan + delta-rule** | 63.03 | 4.51 | **58.51** | 13.97 |
| bf16 GEMM (LM + draft heads) | 30.36 | 22.07 | 8.29 | 1.38 |
| unified attention (MTP) | 16.54 | 5.41 | 11.13 | 3.06 |
| FA2 causal (prefill spill) | 4.95 | ~0 | 4.95 | — |
| GPU idle / host inside the step | 17.33 | ~0 | 17.33 | — |
| **total** | **429.33** | **144.03** | **285.31** | |

The component floors sum to **144.03 ms** against the analytic honest floor of
**148.13 ms** — 2.8% apart, from two independent constructions. That agreement is
the load-bearing cross-check on §2.

Two things to read off this table. First, the FA2 ratio to floor lands at
**4.63×**, against the B1 roofline's independently measured **4.93×** — the
kernel's inefficiency is essentially batch-invariant, which is why width buys so
little. Second, **the GDN scan is 13.97× its state-traffic floor**, by far the
worst ratio on the board and the largest *relative* target in the step.

---

## 4. WHAT IS IN THE OTHER BUCKET

`gaps.json` left 64.43 s (17.9% of window; 38.38 s in-step, 26.05 s out) in a
family called "other". Opened: **184 distinct kernels, ~3.06 M instances,
83.92 ms/step at width 4** — 19.5% of the profiled width-4 step, the
**second-largest item in the step after the GEMM**, and larger than FA2.

Because the bucket is where ~3.06 M tiny launches live, it is also where CUPTI's
per-launch cost concentrates. The whole width-4 profiler inflation is 18.28 ms/step
(429.33 profiled − 411.05 sealed), so the bucket is bounded
**65.6 … 83.9 ms/step**, central (pro-rata by busy share) **80.2 ms/step**.

| class | ms/step w4 | w4/w2 | scaling | inst/step | reducible band | reducible ms/step |
|---|---:|---:|---|---:|---:|---:|
| elementwise math (add/mul/div/compare/where) | 29.30 | 1.89 | width-scaling | 2,066 | 45–80% | 13.2–23.4 |
| sampling & verification | 16.20 | **1.22** | **width-invariant** | 83 | 35–70% | 5.7–11.3 |
| copies / fills / cats | 15.94 | 1.89 | width-scaling | 907 | 55–90% | 8.8–14.3 |
| gather / scatter / index | 11.45 | 2.12 | width-scaling | 629 | 40–75% | 4.6–8.6 |
| mandatory fused pipeline (silu+quant, conv, rms_norm) | 6.61 | 1.92 | width-scaling | 585 | 0–25% | 0.0–1.7 |
| fr13 conv staging (`_fr13_*`) | 3.06 | 2.04 | width-scaling | 207 | 20–60% | 0.6–1.8 |
| reductions / norms | 0.92 | 1.53 | width-scaling | 173 | 30–65% | 0.3–0.6 |
| unclassified | 0.44 | 1.36 | sub-linear | 127 | 0–50% | 0.0–0.2 |
| **total** | **83.92** | **1.73** | | **4,777** | | **33.1 – 62.0** |

The individual kernels that carry it:

| ms/step w2 | ms/step w4 | w4/w2 | inst/step | kernel |
|---:|---:|---:|---:|---|
| 6.93 | **6.93** | **1.00** | 37 | `tensor_kernel_scan_innermost_dim<float, plus<float>>` — torch's generic cumsum |
| 3.25 | 6.67 | 2.05 | 18 | `vectorized_elementwise_kernel<4, CUDAFunctor_add<float>>` |
| 2.80 | 5.76 | 2.06 | 19 | `vectorized_elementwise_kernel<2, FillFunctor<long>>` — buffer zero-fills |
| 1.31 | 5.23 | 3.99 | 6 | `_scatter_gather_elementwise_kernel … ReduceAdd` |
| 2.59 | 5.22 | 2.02 | 196 | `unrolled_elementwise_kernel<direct_copy_kernel_cuda…>` |
| 2.58 | 4.93 | 1.91 | 6 | `elementwise_kernel<128,4, compare_scalar_kernel<long>>` |
| 2.36 | 4.67 | 1.98 | 6 | `elementwise_kernel<128,4, MulFunctor<float>>` |
| 2.57 | 4.66 | 1.81 | 2 | `_topk_topp_kernel` — vLLM's *fused* sampler |
| 2.36 | 4.21 | 1.78 | **768** | `elementwise_kernel<128,2, CUDAFunctor_add<float>>` — 5.5 µs each |
| 1.44 | 3.70 | 2.57 | 226 | `vectorized_gather_kernel<16, long>` |
| 1.59 | 3.00 | 1.88 | 69 | `silu_and_mul_per_block_quant_kernel` — mandatory |
| 2.27 | **2.29** | **1.01** | 26 | `cunn_SoftMaxForward<float>` |

### The two findings inside the bucket

**(a) The sampler runs on padded max-width buffers.** `tensor_kernel_scan_innermost_dim`
is **6.93 ms/step at width 2 and 6.93 ms/step at width 4** — bit-identical
width-invariance, and it does not appear at width 1 at all. `cunn_SoftMaxForward`
is likewise 2.27 → 2.29. Together **9.2 ms/step of sampler math is done on a
fixed 4×32-row buffer regardless of how many requests are resident.** At width 3
— 44% of captured steps — a quarter of it is provably wasted. And it sits *beside*
`_topk_topp_kernel`, vLLM's already-fused top-k/top-p path, which costs 4.66 ms
for the same population: torch's generic cumsum+softmax route is running in
parallel with the fused route that exists to replace it.

**(b) 768 launches per step of one 5.5 µs kernel.** `elementwise_kernel<128,2,
CUDAFunctor_add<float>>` fires 768 times per step for 4.21 ms. Adding
`vectorized_gather` (226), `direct_copy` (196+82), `CatArrayBatchedCopy` (192),
`indexSelect` (196) and the rest, the bucket launches **4,777 kernels per step**
at width 4. This is the same shape of problem `single_launch` was built to solve
for the GDN scan — and the GDN precedent (`results/fr13_gdn_scan_b4_probe_20260814`:
two-launch 41.35 ms → single-launch 32.37 ms, −8.98 ms at 48 layers) says
launch-count consolidation is worth roughly what it looks like it is worth.

**Reducible estimate: 33 – 62 ms/step at width 4, i.e. 8.0% – 15.1% of the
411.05 ms width-4 wall**, central ~47 ms (11.5%). The bands are engineering
judgement per class, stated per class so they can be argued with; they are *not*
measured, and no candidate has been built. Applying `gaps.json`'s
`base_dilution` of 0.5934, that is **4.8% – 9.0% of total window wall** — which
is 2–4× the entire ranked lever ladder in
`results/fr13_b4_width4_nsys_20260813` (14–21 ms/step, 3.4–5.1%).

---

## 5. A FINDING THAT KILLS AN ASSUMPTION: FA2 IS NOT KV-BANDWIDTH-BOUND AT WIDTH 4

The width-4 attribution extrapolated the FA2 floor as "~17.3 ms **if KV traffic
scales with batch**". Testing the antecedent directly:

For every >1 s prefill gap with 5 clean width-4 step-groups on each side, regress
the change in mean `flash_fwd_splitkv_kernel(grid=(1,4,24))` launch time on the
KV tokens the gap admitted (tokens read off `silu_and_mul_per_block_quant_kernel`
gridX — the same source `gaps.json` uses; verified: gridX is 128 on a pure
width-4 decode step = 4 requests × 32 rows).

```
isolated gaps                       12
KV tokens admitted              79,859
net change in launch time       -0.1315 ms
slope                            -3.52 ns / token / launch
slope 95% CI            [-7.36, +0.32] ns / token / launch
DRAM floor for one KV token       15.01 ns   (4096 B / 273 GB/s)
```

**The marginal KV token costs ≤ 0.32 ns of FA2 time against a 15.0 ns DRAM
floor — 47× cheaper than physics allows if the kernel were reading that KV from
DRAM.** Proportionality between FA2 time and KV bytes is rejected at width 4.

The grid semantics are proven, not assumed: there is **no
`flash_fwd_splitkv_combine_kernel` anywhere in the trace**, so `num_splits == 1`
and the grid is `(m_blocks, batch, heads)`; and the `gridY=4` population is
exactly 3,616 = 226 × 16 launches against 225 (+1 boundary) width-4 steps × 16
attention layers.

The mechanism follows from the wave-model control
(`results/fr13_b4_padded_b4_derisk_20260813`: FA2 uses 102,400 B of shared
memory, opt-in max ⇒ **1 CTA/SM, wave = 48 CTAs**). At width 4 the launch is 96
CTAs = exactly 2 waves, and each wave contains all 4 requests. **A wave finishes
when its longest request finishes.** FA2's cost is set by `max(ctx)` and by
cross-request load imbalance, not by `Σ ctx`.

That is why the ~63 ms of FA2 headroom in §3 is real but is **not** a
bandwidth-efficiency problem, and it is why the rank-1 `gqa_pair` re-test — which
changes the KV *layout* — should be expected to move less than a byte-traffic
argument predicts. The lever that matches this diagnosis is
**request-balanced work assignment / split-KV at width 4**, which has never been
on the ladder.

---

## 6. WHAT THIS DOES TO THE LADDER

| claim | before | after |
|---|---|---|
| B4 floor ratio | 3.0× | **2.57×** honest, 3.19× weight-only-on-B1-basis |
| B1 floor ratio | 1.94× | **1.83×** honest |
| gap B4 : B1 | 1.55× | **1.40×** |
| B4 excess over floor | 261 ms (vs 126.5) | **233 ms**, = **58.3 ms per request** vs B1's **105.6 ms** |
| largest unaddressed item | FA2 (69.75 blended) | **the "other" bucket, 83.9 ms/step at width 4** |
| FA2 diagnosis | 52 ms of bandwidth headroom | **load imbalance across requests**; KV bytes are not the binding constraint |

**Per request, B4 is already 1.81× closer to the floor than B1 is** (58.3 ms of
excess per request against 105.6). The step wall looks 1.4× worse only because
one step now does four requests' worth of mandatory work.

## RECOMMENDED NEXT ACTION

1. **Retire the 126.514 denominator for the B4 arms.** It is the wrong scenario
   for a `FR13_DRAFT_VOCAB_ROOT=1` launch. `tests/test_fr13_b4_width4_window.py:60`
   should read the launcher's own `mandatory_weight_floor_ms`, not a literal.
2. **Delete the torch cumsum/softmax sampler path.** 9.2 ms/step, width-invariant,
   running beside the fused `_topk_topp_kernel` that already exists. Offline,
   source-only to establish; cheapest item on the whole board.
3. **Price CFWD launch consolidation.** 4,777 launches/step in the other bucket;
   the GDN `single_launch` probe is the precedent and the measured template.
4. **Re-aim the FA2 work at load imbalance, not layout.** §5 falsifies the
   byte-traffic premise the `gqa_pair` re-test was ranked on.

## DOES NOT CLAIM

* No timing or acceptance reading. No GPU was touched; no arm was run.
* The floor is a **lower bound on time**, not an achievable target: it assumes
  every mandatory byte moves once, at 273 GB/s, with zero latency, zero launch
  cost, zero recompute and perfect overlap. Nothing reaches it.
* C = 18,031 tokens/request is imported from the B1 attack ladder, not
  re-measured at width 4 — §5 is the reason it could not be. The sensitivity
  table is the honest bound, and every entry in it moves the conclusion the
  same way.
* Width-4 component ms/step are CUPTI-profiled and are **upper bounds**; the
  other-bucket band in §4 carries the profiler correction explicitly.
* Reducibility bands in §4 are per-class engineering judgement, not measurement.
* A saving of X ms/step inside a decode step moves total window wall by ~0.59·X
  (`gaps.json` `base_dilution`).
