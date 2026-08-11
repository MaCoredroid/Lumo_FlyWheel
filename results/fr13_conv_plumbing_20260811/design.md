# FR13 conv/plumbing rung — launch census, and why the rung is already closed

Offline, read-only analysis over banked evidence, plus one default-off code
change. **No GPU was touched by the analysis. No serving arithmetic was
changed.** `attribution_only=true`, `acceptance_valid=false`,
`citable=false`. This is a design document and a negative result, not a
performance result.

- Branch: `codex/fr13-conv-plumbing-20260811`, from `origin/main` `1e0158bf2`.
- Evidence: `results/fr13_fixed32_b1_nsys_attribution_20260808T212056Z/`
  (curated reduction of the 218,692,330 B post-Qrow capture, 1146 complete
  `fr13.fixed32.step` NVTX instances) and
  `results/fr13_attack_ladder_analysis_20260808/` @ `7263c134d`.
- Constraint honoured throughout: **exact-math, Tier-A**. Every transformation
  below is byte-identical or it is not proposed.

---

## 0. Headline

**The conv-commit rung as costed no longer exists.** The "~340 per-step kernel
launches in the fixed32 conv-commit path" is a pre-`direct-leaf-fix` number.
On the merged stack:

| claim in the rung brief | measured on the merged stack |
|---|---|
| conv commit is ~340 small launches | conv commit is **1** Triton launch/event (`commit_direct_launches_per_event: 1`, gather 0, scatter 0) plus **1** row-guard launch |
| conv is a meaningful phase | conv is **0.542 ms/step = 0.2%** of the 237.248 ms envelope |
| batching the launches buys ms | the whole conv phase is 0.542 ms and its two kernels run **1.0** and **48.0** inst/step |

The ~340 figure is nevertheless *real* — it just does not live where the brief
puts it. It is the **per-GDN-layer SFWD plumbing**: eleven distinct kernels at
48.03 inst/step each, 528.3 launches/step, 3.167 ms/step (§2). That block has
already been attacked: the SFWD conv/post-prep fusion candidate collapses five
of those launches per layer into one (−192 launches/step) and has been sitting
**default-off, source-complete, byte-gate-pending since 2026-08-03** (§3).
It is not a new lever; it is a queued one, and it is queued behind alienware.

What is actually left of the rung on the merged stack is **one** transformation
worth **6 ATen launches per event at B4 and 0 at B1** (§4). It is implemented
here behind a new default-off flag, unit-tested to byte identity, and honestly
labelled as the last scraping of a closed rung.

**Recommendation: close the conv-plumbing rung and move to the GDN fused scan
(3.45 ms/step, ladder item 2), which is the next rung in Mark's order and is
worth 575x more than what remains here.**

---

## 1. What the conv-commit path costs today

Route contract, read out of the merged source
(`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`,
`preseed_fixed32_conv_col0_pregather`):

```
commit_route                             fixed32_direct_source_col0
commit_launches_per_event                1
commit_direct_launches_per_event         1
commit_gather_launches_per_event         0
commit_scatter_launches_per_event        0
commit_row_guard_kernel_launches_per_event  1
commit_full_node_writebacks              0
commit_conv_remaps                       0
commit_row_guard_torch_index_transforms  0
```

Corroborated independently by the trace, not taken on the contract's word:

| phase | kernel | inst/step | ms/step |
|---|---|---:|---:|
| cfwd | `_fr13_fixed32_conv_direct_col0_kernel` | **1.00** | 0.168 |
| sfwd | `_fr13_conv_col0_pregather_kernel` | **1.00** | 0.431 |
| sfwd | `_fused_post_conv_kernel` (stock FLA) | 48.03 | 0.111 |

The 48-layer Python committer the brief names, `_fr13_conv_commit_to_col0`
(`scripts/fr10_phase4_patch_vllm_tree_gdn.py`), is ~13 ATen ops × 48 layers
≈ **624 launches** — and it is **not on the fixed32 served path**. Its three
call sites are the staged `FR13_REPLAY_ROUTE` tail and the two S1 step-graph
routes. Under fixed32 the sampler returns through
`_fr13_fixed32_device_commit_route`, which calls `_fixed_conv_commit`
(the single Triton launch) and returns before the generic committer is
reachable. `tests/test_fr13_fixed32_conv_commit_wiring.py` already pins that
(`"_fr13_conv_commit_to_col0" not in route_calls`). **That is where the ~340
went: it was deleted, not merely reduced.**

Consequence for this rung: a "vectorize the conv commit" lever gated on fixed32
would be an **unsatisfiable precondition** — the code it vectorizes cannot run
in the configuration the gate demands. It is not proposed.

---

## 2. Where ~340 launches actually are: per-layer SFWD plumbing

From `fr13_fixed32_b1_nsys_attribution.json`, every SFWD kernel at 48.03
instances/step (= once per GDN layer):

| ms/step | inst/step | kernel |
|---:|---:|---|
| 0.803 | 48.03 | `unrolled_elementwise_kernel<direct_copy_kernel_cuda>` |
| 0.486 | 48.03 | `elementwise_kernel<128,4, gpu_kernel_impl_noc>` |
| 0.390 | 48.03 | `nvjet_sm121_tst_mma_192x16x64_..._splitK_TNNN` |
| 0.262 | 48.03 | `indexSelectSmallIndex<BFloat16,long,uint>` |
| 0.255 | 48.03 | `CatArrayBatchedCopy<OpaqueType>` |
| 0.230 | 48.03 | `vectorized_gather_kernel<16,long>` |
| 0.220 | 48.03 | `vectorized_elementwise_kernel<4, FillFunctor<float>>` |
| 0.160 | 48.03 | `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mul_...` |
| 0.153 | 48.03 | `triton_red_fused__to_copy_add_per_token_group_fp8_quant_..._rms_norm_2` |
| 0.111 | 48.03 | `_fused_post_conv_kernel` |
| 0.097 | 48.03 | `triton_per_fused__to_copy_mean_pow_view_0` |
| **3.167** | **528.3** | **total** |

Plus 192.12/step and 144.09/step families (4/layer and 3/layer respectively).
Seven of the eleven rows above sum to **336 launches/step**, which is almost
certainly the origin of the "~340" in circulation.

### 2.1 Why launch *count* is the wrong lever here

SFWD is **one CUDA graph** (`graphId=812`, 1 replay/step, 1894 kernel nodes;
`results/fr13_attack_ladder_analysis_20260808` §0). Every one of these 528
launches is a **graph node**, not a `cudaLaunchKernel`. The total inter-node gap
across all 1894 nodes is **0.404 ms/step**, i.e. **213 ns/node** — the
graph-replay floor. Collapsing all 528 nodes into one would therefore recover
at most

```
528 × 213 ns = 0.112 ms/step   (0.047% of the 237.248 ms envelope)
```

and the analysis already books the SFWD intra-graph gap as **AT FLOOR**
(ladder item 6). **Launch-latency batching inside SFWD is dead.** What is not
dead is the 3.167 ms of *kernel execution* those tiny ops spend — 0.803 ms for
48 instances of a copy is 16.7 µs each — but recovering that means **fusing the
work**, which is a codegen change with an arithmetic surface, not plumbing.

---

## 3. The SFWD conv/post-prep fusion: already built, already parked

`src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion.py` (+ its generated
kernel) is exactly that codegen change, and it is merged:

- flag `FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION`, strict 0/1, **default 0**;
  variants `FR13_FIXED32_SFWD_NODEGROUP8_DIRECT`,
  `FR13_FIXED32_SFWD_EMBED_GATE_CTA`; byte-A/B arm
  `FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB`.
- static result, from
  `results/fr13_fixed32_sfwd_conv_postprep_fusion_20260803/README.md`:
  it replaces *"the incumbent conv launch, three `rearrange_mixed_qkv`
  contiguous-copy launches, and post-prep launch with one launch per layer:
  **5 to 1**, removing **192 launches** across the model"*, and removes
  106,954,752 logical global bytes/step at B1.
- status in that same README: *"default-off eager and FULL-capture wiring
  complete; served arm blocked pending a real-task byte-gate credential;
  **not GPU measured**"*.

**Modelled saving.** The five folded launches are the conv, the three
`rearrange_mixed_qkv` copies and post-prep. Attributing them to the census
above (`direct_copy` 0.803 + the 144-inst 3/layer family 0.268 +
`_fused_post_conv_kernel` 0.111) gives an upper bound of **≈1.18 ms/step** of
GPU execution, before whatever the fused kernel itself costs. Nothing in the
banked evidence measures the fused kernel, so **the honest number is a bound,
not an estimate**, and it is exactly what the pending byte gate + timing pair
would settle.

**This is the conv-plumbing rung's real remaining value, and it is blocked on
alienware, not on design.** Nothing new needs to be written for it.

---

## 4. What is left, and what this branch implements

The one conv-adjacent block on the merged fixed32 served path that is (a) still
a per-row Python launch loop, (b) provably byte-safe to batch, and (c) not
already covered by a parked candidate, is the **committer path publish** in
`_fr13_fixed32_device_commit_route`, immediately upstream of the conv commit:

```python
for compact_row, slot_row in enumerate(slot_indices):
    slot_paths[slot_row].copy_(device_paths[compact_row])   # 16 int32
    slot_lens[slot_row].copy_(device_lens[compact_row])     # 1 int32
spec_paths[:batch].copy_(device_paths)
spec_lens[:batch].copy_(device_lens)
```

`spec_paths`/`spec_lens` are the buffers `launch_fixed32_conv_commit_to_col0`
and `launch_tree_gdn_replay_all_layers` read. `slot_paths`/`slot_lens` are the
sampler-slot-ordered mirror, written sparsely. This block runs **eagerly in
CFWD** (CFWD is 1111 eager ops/step against 78 graph nodes), so unlike §2 its
launches *do* pay the full eager bubble — the analysis measures CFWD's eager
overhead at **2.73 µs per op**.

### 4.1 Transformation

Publish the compact rows first, then scatter them into the slot family with two
`index_copy_` calls off a cached index tensor.

| batch | incumbent launches | batched launches | saved |
|---:|---:|---:|---:|
| 1 | 4 | 4 | **0** |
| 2 | 6 | 4 | 2 |
| 3 | 8 | 4 | 4 |
| 4 | **10** | **4** | **6** |

`census: 2 + 2B  →  2 + 2`. The arithmetic is generated by
`batched_slot_launch_census()` and the code is measured against it under a
`TorchDispatchMode` in
`tests/test_fr13_fixed32_conv_commit_batched_slots.py::test_measured_launch_counts_match_the_census`,
so the table cannot rot away from the implementation.

### 4.2 Expected step-wall saving

At CFWD's measured 2.73 µs/eager-op:

```
B4:  6 launches × 2.73 µs  =  0.016 ms/step   (0.006% of a 277 ms B4 step)
B1:  0 launches            =  0.000 ms/step
```

**This is below every noise floor the campaign has measured** (stock-vs-stock
B4: 6.5% per-request, 3.8% aggregate). It is shipped because it is free, byte-
safe and provable, **not** because it is expected to move a gate. Any report
that quotes it as a speed win is misreading it.

### 4.3 Byte-safety argument, per transformation

1. **Source substitution.** The slot rows are now sourced from
   `spec_paths[:batch]` instead of `device_paths`. The two statements that
   publish `spec_paths[:batch]` are `copy_` of the same `device_paths`, so
   `spec_paths[:batch] ≡ cast(device_paths, int32)` elementwise. The incumbent
   loop applies the *same* cast through `slot_paths[slot_row].copy_(...)`.
   int64→int32 narrowing is deterministic and elementwise, so the written bytes
   are identical. **Enforced:** dtype equality of both families is checked per
   event.
2. **Reorder.** Compact publish now precedes slot publish (it must — the slot
   publish reads it). This is observation-free iff the families are disjoint
   storage and nothing reads between them. Nothing reads between them (the next
   statement is a Python attribute assignment). **Enforced:** untyped-storage
   disjointness of `slot_* vs spec_* vs device_*` is checked per event and
   raises; it is not inferred from "they are different globals".
3. **Scatter uniqueness.** `index_copy_` is order-dependent under duplicate
   indices. `slot_indices` is derived from `sampler_req_ids.index(req_id)` over
   ids the route already proves unique. **Enforced:** re-proved locally, so the
   guarantee does not depend on a check three frames away.
4. **Sparsity.** Slot rows not in `slot_indices` must keep their previous
   content in both forms. `index_copy_` writes exactly the indexed rows.
   **Enforced:** the byte test poisons the untouched rows and asserts the poison
   survives in both forms.
5. **Order where order matters.** Within a layer the conv committer's
   read-then-write is untouched; this change is entirely upstream of the
   committer launch, and the committer still runs before the replay
   (`test_route_calls_the_module_and_keeps_no_per_row_loop` pins
   publish → conv commit → replay).
6. **Narrowing interaction.** The publish block never reads
   `spec_state_indices`, so it is orthogonal to `FR13_MAMBA_SPEC_BLOCKS_CDIV`.
   That orthogonality is a claim, so it is tested: every byte test runs under
   both the OFF table (private per-column physical rows) and the ON table (one
   scratch page aliased across logical columns 1..num_spec), and the
   committed-conv-state test drives a CPU model of
   `_fr13_fixed32_conv_direct_col0_kernel`'s addressing off each publish.

### 4.4 Fail-closed structure

- `FR13_FIXED32_CONV_COMMIT_BATCHED_SLOTS`, strict `"0"`/`"1"`, default `"0"`.
  A typo raises; it is never read as OFF. (The campaign has shipped "candidate"
  arms that were silently the stock path; strict parsing is the fix.)
- **Fixed32-only, fail-loud**, mirroring
  `_fr13_assert_mamba_spec_blocks_cdiv_requires_fixed32`. The assert runs at
  `main()` preflight *and* at the rejection-sampler patch site; the patch site
  early-returns on an already-patched image, so the preflight is the copy that
  always runs.
- The flag is **baked as a literal into the injected source** by the patcher
  prelude (pid 1, where the FR13_* master env exists). The mp/spawn EngineCore
  worker's curated env drops bare masters, so an `os.environ` read in the worker
  would silently disarm the candidate — the exact failure mode the RDAB flag
  documents.
- **Satisfiable by construction:** the flag's precondition is
  `FR13_FIXED32_MODE ∈ {tail6_fixed32, hydra27_fixed32}`, which is the *only*
  configuration in which the code it modifies executes. There is no artifact,
  sidecar or peer flag demanded that this change does not itself produce.
- No silent fallback: any precondition failure **raises and aborts the event**.
  A batched publish that cannot prove itself must not degrade to the loop,
  because a wrong publish corrupts both the conv commit and the GDN replay for
  the step.
- The index cache refuses to build during CUDA-graph capture (an H2D inside a
  capture bakes one event's slot ordering into every replay), re-validates every
  hit, and is bounded at 512 entries.

---

## 5. Files

| file | what |
|---|---|
| `src/lumo_flywheel_serving/fr13_fixed32_commit_slot_scatter.py` | the lever: flag resolver, fixed32 assert, cached scatter index, both publish forms, launch census |
| `scripts/fr10_phase4_patch_vllm_tree_gdn.py` | flag declaration, strict resolver, fixed32 assert, prelude bake, route call site, `main()` preflight |
| `tests/test_fr13_fixed32_conv_commit_batched_slots.py` | 33 tests: byte identity (4 batches × 2 narrowing regimes × all slot permutations), committed-conv-state identity, measured launch counts, guards, patcher threading |

---

## 6. What remains for the byte gate + timing pair (alienware)

1. **This lever.** Local boot diagnostic with the flag ON (boot + FULL capture +
   smoke serve) — runnable locally, no offload proxy. A B4 timing pair is
   *not* worth scheduling: 0.016 ms/step is 400x below the measured B4 noise
   floor. If it ever ships it should ride along with a batch of levers, not pay
   for its own pair.
2. **The real rung: the SFWD conv/post-prep fusion.** Needs (a) the real-task
   byte-gate credential its own README names as the blocker, (b) a B1 diagnostic
   with `FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=1`, (c) the standing exact4
   timing pair. Bound: ≤1.18 ms/step of folded GPU execution; the fused kernel's
   own cost is unmeasured, and the prior `rowgroup8` result (−2.28 ms/step,
   i.e. a **regression**) is the standing warning that more parallelism is not
   automatically faster.
3. **Retire the rung.** Conv is 0.2% of the envelope. Per Mark's order the next
   rung is the GDN fused single-launch scan (3.45 ms/step, ladder item 2), which
   already has `FR13_SUBTREE_PARALLEL_SELFCHECK` and a byte gate in tree.
