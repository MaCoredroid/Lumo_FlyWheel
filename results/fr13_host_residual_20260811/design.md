# FR13 host-residual rung — the tail is 3.46 ms, not 10.08

**Rung:** host residual, ladder item 1 (`results/fr13_attack_ladder_analysis_20260808`
@ `7263c134d`, §5.2 and §6).
**Branch:** `codex/fr13-host-residual-20260811`, from `origin/main` `1e0158bf2`.
**Prior art on this rung:** `f7ff87262` (`FR13_HOST_TAIL_NVTX` /
`FR13_HOST_TAIL_DEFER`, both default-off, merged 2026-08-09).

**Verdict: the rung as costed does not exist.** The ladder prices the
post-DFWD host tail at **11.969 ms/step with 10.080 ms of GPU idle and
~7.8 ms of pure Python**, and calls it *"the single largest addressable block
in the trace."* Re-derived from the same 218,692,330 B capture, the tail's
**median is 3.588 ms** and its **mean is 11.977 ms** — the mean is carried by
**15 steps out of 1146 (1.3%)** whose "tail" is a **chunked-prefill forward**
(up to 3.06 s, 3608 eager launches, 2340 `cuTensorMapEncodeTiled`), because
the fixed32 NVTX step markers wrap only the spec-decode path and a prefill
forward therefore lands in the window between one decode step's `dfwd` and
the next decode step's `sfwd`. On decode-cadence steps the tail is
**3.667 ms host wall / 3.458 ms GPU idle / 2.857 ms of host CPU inside no
CUDA call.** The ladder's *realistic* band (5–7 ms) was closer than its
ceiling; its ceiling was inflated **3.3x**.

Second refutation, from the offline probe: the **1.945 ms/step of GPU idle
that the capture attributes to `cudaGraphLaunch`** is not a serving cost.
Measured on the same GB10, launching a **1894-node** CUDA graph costs
**3.15 µs** of host time (fit: 0.44 ns/node + 2.31 µs fixed, max residual
0.04 µs). The capture's figure is **617x** that: it is CUPTI instrumenting
1894 graph-node activities per replay. An unprofiled step does not pay it.

Evidence: `results/fr13_host_residual_20260811/host_cost_probe.json`
(`scripts/fr13_host_tail_cost_probe.py`, GB10, pinned image, one container,
CPU-launched, docker 0 before and after) and the banked
`fr13_fixed32_b1_real_swe.sqlite` export of the post-Qrow capture.
`analysis_only=true`, `acceptance_valid=false`, `citable=false`. No TPS,
floor, or acceptance claim. No GPU arithmetic was changed.

---

## 1. Method, and why it disagrees with the ladder

The ladder projects NVTX host ranges onto GPU ops through `correlationId` and
reports `first_to_last` spans. That framing is correct and this document
**reproduces it exactly**: step GPU span **237.248 ms** (banked: 237.248),
`sfwd` 155.829 (155.807), `postprocess` 12.349 (12.339), `cfwd` 20.705
(20.686), `dfwd` 35.131 (35.107), boundary `cfwd_gpu_end → dfwd_gpu_start`
**1.224** (1.224), tail `dfwd_gpu_end → step_gpu_end` **11.977** (11.969).

Two things were added:

1. **A host timeline.** The NVTX ranges are host push/pop pairs, so they also
   measure what the *host* was doing. Nobody had read them that way.
2. **Per-step distributions instead of means**, and a decode-cadence filter
   (host step wall in 150–400 ms) that separates decode steps from steps whose
   window contains a prefill forward.

Tail distribution over all 1146 steps: p5 3.116, p25 3.320, **p50 3.588**,
p75 3.882, p95 4.631, **p99 355.0**, max 1239.5 ms. The 15 steps above 10 ms
contribute **8.354 ms/step** of the 11.977 ms mean; the other 1131 average
**3.671 ms**.

Everything below is over the **1130 decode-cadence steps** unless stated.

## 2. The host timeline of a step (1130 steps, host wall 230.964 ms)

GPU idle is computed as the complement of the merged busy union over all
kernels, memcpys and memsets on every stream.

| # | host segment | wall ms | GPU idle ms | GPU busy ms |
|---|---|---:|---:|---:|
| A | `sfwd` range (graph enqueue) | 3.674 | **1.999** | 1.675 |
| B | gap `sfwd`→`postprocess` | 0.071 | 0.000 | 0.071 |
| C | `postprocess` range | 0.177 | 0.000 | 0.177 |
| D | gap `postprocess`→`cfwd` | 0.181 | 0.000 | 0.181 |
| E | `cfwd` range | 176.585 | 0.809 | 175.776 |
| F | gap `cfwd`→`dfwd` (unlabelled) | 11.397 | **2.911** | 8.486 |
| G | `dfwd` range | 35.211 | 1.243 | 33.967 |
| H | gap `dfwd`→next `sfwd` (**the tail**) | 3.667 | **3.458** | 0.209 |
| | **total** | **230.964** | **10.421** | |

The host is *blocked* for 176.6 ms in `cfwd` — it is waiting for the GPU to
finish 168 ms of `sfwd`+`postprocess` before it can read a sampled token.
The CUDA driver agrees: **63,922 `Command buffer full` records**, 165.5
ms/step, land in E and **essentially none anywhere else** (F: 0.0003 ms/step;
G, H, A: zero). That is the validity argument for the rest of the table: in
the windows this rung is about, the queue is empty and the host really is the
critical path.

**Total host-attributable GPU idle: 10.421 ms/step (4.5% of the step).** The
ladder's 15.81 ms is the same quantity computed on the mean-inflated framing.

## 3. Inside the tail (H): 3.458 ms of idle, 0.209 ms of GPU work

| quantity | value |
|---|---:|
| host wall | 3.667 ms/step |
| GPU idle | 3.458 ms/step |
| GPU work performed | **0.209 ms/step** |
| time inside CUDA APIs | 0.641 ms/step |
| **host CPU inside no CUDA call** | **3.026 ms/step** (2.857 of it while the GPU is idle) |
| `cudaLaunchKernel` | 92.0 /step |
| `cudaMemcpyAsync` | 62.1 /step (H2D 37 × 5,090 B; D2D 23 × 656 KB; D2H 2 × 376 B) |
| `cudaStreamSynchronize` | 1.03 /step |

Python-gap structure (gaps between consecutive CUDA API calls):

| gap class | n/step | ms/step |
|---|---:|---:|
| < 1 µs | 164.0 | 0.028 |
| 1–10 µs | 152.5 | 0.691 |
| 10–100 µs | 39.3 | 1.151 |
| 100–1000 µs | 5.7 | 1.152 |

So the tail is **154 CUDA ops that buy 0.209 ms of GPU work**, plus ~45
Python blocks of 10 µs–1.45 ms. It is **not** a sync-bound window (one
`cudaStreamSynchronize`), and it is **not** dominated by per-op dispatch:
the probe prices the tail's own op mix at **3.87 µs/op**, so all 154 ops
account for **0.596 ms** — 17% of the idle. The other ~2.9 ms is framework
Python.

**What that Python is.** The `fr13.fixed32.step` NVTX range closes
immediately before `set_forward_context`, so window H is: the remainder of
step N's `sample_tokens`, the executor/EngineCore hand-off,
`Scheduler.update_from_output` + `schedule`, and then **all of step N+1's
`_update_states` + `_prepare_inputs` + attention-metadata build**. The GPU
work in H confirms it — `index_elementwise` 15/step,
`_compute_slot_mapping_kernel` 4/step, `arange` 6/step, `scatter` 6/step,
`_zero_kv_blocks_kernel`,
`triton_poi_fused_add_bitwise_and_clamp_copy__ge_gt_index_where_0` — that is
input preparation, not bookkeeping.

**This is the finding that closes the ladder's step 1.** Lever 1 was
*"move the post-DFWD host tail off the step critical path… nothing in the
retire path may be a precondition for the next submit."* Measured, the tail
is **mostly precondition**: it computes the slot mapping, block tables,
positions and spec metadata that the next forward consumes. It cannot be
retired. What can be attacked is its *cost*, not its *position*.

## 4. The other three idle items

**F — `cfwd`→`dfwd`, 2.911 ms idle inside an 11.397 ms host gap.** One
object dominates: a **single 4-byte D2H `cudaMemcpyAsync` that blocks the
host for 9.54 ms** (p50; max 10.95 ms), while the GPU drains 8.486 ms of
committer work (`fused_sigmoid_gating_delta_rule_update_kernel` 48/step =
4.03 ms, softmax 1.72 ms, ~900 small elementwise). The other 35 memcpys in
the window are D2D and cost 0.114 ms of host time in total. This is the
ladder's item 4 (1.224 ms on the GPU projection) and it is a **genuine data
dependency**: a scalar read of committer output that gates the drafter.
Removing it is the ladder's "keep the accepted-token count on device" and it
is **not** host bookkeeping — it changes what the drafter is fed.
**Flagged for Mark, not touched here.**

**A — `sfwd` enqueue, 1.999 ms idle, 1.945 ms of it inside
`cudaGraphLaunch`.** Refuted by the probe (headline). Not a serving cost.
The 1.622 ms Python block also in window A runs *after* the launch, while the
GPU is already busy, and is free.

**G — `dfwd`, 1.243 ms idle** (0.799 ms Python between drafter iterations,
0.336 ms in `cudaMemcpyAsync`). Five drafter iterations, each with a
blocking readback; this is the drafter's own structure and the drafter is
locked per Mark's 2026-08-08 directive.

**E — `cfwd`, 0.809 ms idle.** Sub-noise.

## 5. Attribution table (item → ms/step → disposition)

| item | ms/step idle | safe for this rung? | disposition |
|---|---:|---|---|
| Tail: framework Python in next-step input prep | 2.86 | partly | genuine precondition; only its cost is attackable |
| — of which: tree depth-position re-derivation | **0.242** | **YES, byte-safe** | **shipped**, `FR13_HOST_TAIL_PREP_BAKE` |
| — of which: capture-manifest `json.loads` (census open) | 0.003–0.025 | YES | priced, not shipped (§7) |
| — of which: Arctic cached-id set rebuild | 0.000–0.203 | YES | **unpriced in deployment**; curve measured, occupancy unknown |
| Tail: eager dispatch of 154 CUDA ops | 0.60 | no | needs graph capture of `_prepare_inputs`; data-dependent shapes |
| Tail: small H2D copies, pageable→pinned | ≤0.108 | YES | 2.93 µs/copy × ≤37; pageable share unmeasured |
| F: blocking 4 B D2H committer→drafter handoff | 2.91 | **NO** | touches what the drafter is fed — **Mark's call** |
| A: `cudaGraphLaunch` | 1.945 | n/a | **CUPTI artifact**, real cost 3.15 µs |
| G: drafter inter-iteration Python | 1.24 | no | drafter is locked (2026-08-08 directive) |
| E: `cfwd` | 0.81 | no | sub-noise |
| "tail" of the 15 prefill steps | 8.35 (of the mean) | n/a | **not a tail**; prefill forward, belongs to the APC/prefill campaign |

Anything that changes sampler inputs, ordering, or what the drafter/verifier
consumes is **out of scope by construction** and appears above only as a flag
for Mark.

## 6. What this branch ships

### 6.1 `FR13_HOST_TAIL_PREP_BAKE` — the one measured reduction

Under fixed32 the patcher **already** rewrites `_fr10_tree_src` to the baked
`_FR13_FIXED32_TREE_SOURCE` literal and then asserts the derived choices
equal `_FR13_FIXED32_CHOICES`. So `# FR10_TREE_DEPTH_POSITIONS` runs
`ast.literal_eval` + `sorted` + four comprehensions + two `np.array` builds
**on a compile-time constant, every decode step**, and the topology assertion
that follows is already a tautology. The flag replaces the whole span with
the literals it can only produce.

- **Measured cost removed: 242 µs/step** at the deployed 31 paths
  (probe p50; 34 µs at 9 paths, 921 µs at 63 — the derivation is superlinear
  in path count). That is **7.0% of the 3.458 ms tail idle**, and it is below
  every noise floor the campaign has measured (B4 stock-vs-stock: 6.5%
  per-request, 3.8% aggregate). **It is shipped because it is provable and
  free, not because it will move a gate.**
- **Byte-safety.** Nothing here touches a tensor, an ordering, or a sampler
  input. `derive_tree_depth_plan` is the single reference implementation used
  both to bake the literals and to check them, and
  `tests/test_fr13_host_tail_prep_bake.py` execs **both forms** over seven
  tree sources (including the deployed 31-path tree) requiring identical
  values, dtypes, shapes **and object freshness** — the bake hands back a
  fresh `list` and fresh `np.ndarray` every call, so no cached object can be
  aliased or mutated across steps. Patch time additionally refuses to emit
  the bake unless the plan reproduces `_FR13_FIXED32_CHOICES`.
- **Fail-closed.** Strict `"0"`/`"1"`, default `"0"`, read exactly once at
  patch time; empty raises. Fixed32-only and fail-loud, because outside
  fixed32 `_fr10_tree_src` is a runtime value and baked offsets would feed
  wrong RoPE positions into `positions[...] = base + depth_offsets` with
  nothing downstream to catch it — the guard message names that mechanism and
  both ways out. Asserted at the patch site **and** at `main()` preflight
  (the patch site early-returns on an already-patched image). Launcher
  validates `0|1` and forwards it. Satisfiable by construction: the only
  precondition is a fixed32 mode, which is the only configuration in which
  the code it edits runs.

### 6.2 Two instrumentation defects fixed

1. **`fr13.fixed32.sched_next` could never appear.** `f7ff87262` injected the
   `_fr13_sched_next_nvtx` *definition* into `Scheduler` and no call to it, so
   the range the reducer reserves was unreachable at any flag setting — an
   unsatisfiable measurement precondition of exactly the class this campaign
   keeps finding. The patch now renames the stock method to
   `_fr13_update_from_output_inner` and wraps it, popping in a `finally` so
   the range survives the error path.
2. **No sub-range covered the measured time.** `sample_readback` is the only
   tail sub-range shipped, and it sits in step N; §3 shows the host time is in
   step N+1's input prep. This branch adds **`fr13.fixed32.prep_next`**,
   spanning the preprocess block to the existing step-NVTX close — i.e.
   exactly the unlabelled remainder — and registers it in the reducer's
   optional `HOST_TAIL_RANGES`. It also fires on prefill forwards, which is
   deliberate: that is the confound of §1, and the next capture will separate
   it without arithmetic.

### 6.3 The host-tail flags made fail-closed

`FR13_HOST_TAIL_NVTX` / `FR13_HOST_TAIL_DEFER` were permissive `== "1"`
comparisons evaluated **at runtime inside the served process**, with no
strict parsing, no preflight and no launcher validation. The served process
is the mp/spawn EngineCore worker, whose curated env drops bare `FR13_*`
masters — the exact failure mode the prelude bake exists to prevent. Both are
now resolved once at patch time, strictly, with the fixed32 precondition
enforced, and **baked as literals** into the injected source; the helpers
return the literal and the injected code reads no environment at all.

### 6.4 The offline probe

`scripts/fr13_host_tail_cost_probe.py` — no credential, no served model, no
offload host. Measures on GB10:

| quantity | measured |
|---|---:|
| CUDA graph replay, host cost | 0.443 ns/node + 2.31 µs (max residual 0.04 µs) |
| — at 1894 nodes (the SFWD graph) | **3.15 µs** vs 1.945 ms attributed in the capture |
| graph re-upload after interleaving another graph | +0.16 µs (no eviction effect) |
| eager ATen dispatch, the tail's op mix | **3.87 µs/op** |
| small H2D copy, pageable | 5.39–5.57 µs |
| small H2D copy, pinned | 2.46–2.51 µs |
| tree depth-position derivation, 9 / 31 / 63 paths | 34 / **242** / 921 µs |
| capture-manifest `json.loads`, 2–10 KB × 1–4 | 2.6 – 25.1 µs |
| Arctic cached-id set, n = 10 / 100 / 1k / 10k | 0.27 / 1.84 / 16.2 / 203 µs |

The graph-launch row is the load-bearing one: it converts a 1.945 ms line in
the ladder into a profiling artifact, and it prices **every** future
graph-node reduction (e.g. the parked SFWD conv/post-prep fusion's −192
nodes) at **0.085 µs/step of host time** — i.e. graph node count is *not* a
host-side lever either.

## 7. Deliberately NOT implemented

- **The submit/retire split and the on-device accepted-token handoff**
  (ladder lever 1 steps 1–2). §3 shows the tail is a precondition for the next
  forward, and §4 shows the F-window bubble is a genuine scalar dependency on
  committer output. Both need Mark's ruling because they change what the
  drafter is fed, not merely when bookkeeping runs.
- **Memoizing the capture-manifest `json.loads`** (2.6–25 µs/step) and the
  Arctic `cached_ids` rebuild. The first is real but ~10-100x smaller than the
  lever shipped. The second has an **unmeasured deployed size** — the cache is
  capped at 10,000 but its steady-state occupancy on exact4 is not in any
  banked artifact, so its cost is somewhere between 0.3 µs and 203 µs/step and
  claiming a number would be exactly the kind of unpriced estimate that killed
  the previous two rungs. Both are priced in §6.4 and left for a pass that can
  measure occupancy.
- **Pinning the tail's small H2D copies.** ≤0.108 ms/step, and the pageable
  share is unmeasured; it needs a live census, not a guess.
- **CUDA-graph capture of `_prepare_inputs`** (≤0.60 ms/step of dispatch).
  Data-dependent shapes, stock vLLM ownership, and it would not remove the
  ~2.9 ms of Python that computes the metadata.

## 8. Files

| file | what |
|---|---|
| `src/lumo_flywheel_serving/fr13_host_tail_prep.py` | strict flag parser, fixed32 fail-loud assert, the single reference derivation, the bake emitter, census |
| `scripts/fr10_phase4_patch_vllm_tree_gdn.py` | flag declarations + strict resolver + `main()` preflight; baked flags in the injected module block; the `prep_next` range; the `sched_next` call sites; the depth-position bake |
| `scripts/fr13_fixed32_nsys_reduce.py` | `prep_next` registered as an optional tail range |
| `scripts/fr13_launch_forked_fa2_tree_server.sh` | `0\|1` validation for all three host-tail flags; forwards `FR13_HOST_TAIL_PREP_BAKE` |
| `scripts/fr13_host_tail_cost_probe.py` | the offline host-cost probe |
| `scripts/fr13_host_tail_attribution.py` | the offline host-timeline reduction that produces every table in this document (`fr13.host_tail_attribution.v1`) |
| `results/fr13_host_residual_20260811/tail_attribution.json` / `.txt` | its output over the banked 20260808T212056Z capture |
| `results/fr13_host_residual_20260811/host_cost_probe.json` | its output (`fr13.host_tail_cost_probe.v1`) |
| `tests/test_fr13_host_tail_prep_bake.py` | 77 tests: flag hygiene, strict parsing, fixed32 guards, preflight ordering, both-form equivalence over 7 trees, object freshness, patch anchors, idempotency, reducer + launcher contracts |
| `tests/test_fr13_host_tail.py` | updated for the baked-flag contract; new regression test that `sched_next` is actually opened |

### 6.5 The offline attribution reduction

Every table above is produced by `scripts/fr13_host_tail_attribution.py`, so
this document cannot drift from its evidence. It is read-only, needs no GPU,
and takes ~13 s over the banked 640 MB sqlite export. Reproduce:

```
python3 scripts/fr13_host_tail_attribution.py \
  --sqlite <runroot>/tail6_fixed32_b1_nsys_f32_20260808T212056Z/logs/\
fr13_fixed32_b1_real_swe.sqlite \
  --out results/fr13_host_residual_20260811/tail_attribution.json \
  --text-out results/fr13_host_residual_20260811/tail_attribution.txt
```

## 9. Validation

**CPU tests.** `TMPDIR=/home/mark/shared/tmp-scratch`, `--basetemp` under it,
`--ignore=tests/test_codex_long_assets.py`, `PYTHONPATH=src`.
`tests/test_fr13_attn_kv_remap.py` is additionally ignored: this host's venv
has no `triton`, so the module cannot import. That is an environment limit,
not a result.

| run | failed | passed | skipped |
|---|---:|---:|---:|
| `origin/main` `1e0158bf2` (baseline worktree) | 153 | 3785 | 55 |
| this branch | 153 | 3863 | 55 |

`comm` over the sorted FAILED node-id sets: **0 new failures, 0 fixed**.
`+78` passing tests, all in the two host-tail modules. The 153 pre-existing
failures are inherited from main and untouched.

**Local GPU validation.** The probe ran on GB10 in the pinned image
(`lumo-flywheel-vllm:26.01-py3-v0.19.0`), docker verified at 0 containers
before and after; the two runs reproduce every headline figure to <2%.

**Not reachable locally:** arming `FR13_HOST_TAIL_PREP_BAKE` in a booted
server. There is no credential-free fixed32 boot entry point in tree — the
boot diagnostic hard-requires SFWD gate credentials — so the flag's
served-path arming is offline-verified only: the patch is executed against
stub modules carrying the real anchors, the emitted source is compiled, and
both derivation forms are exec'd and compared.

## 10. For the byte gate + timing when alienware returns

1. **`FR13_HOST_TAIL_PREP_BAKE`**: boot diagnostic with the flag on, then the
   standing exact4 byte gate. Byte drift would mean the bake stopped
   reproducing the expression, which the CPU tests already forbid — so a
   drift is a bug report, not a tuning result. **Do not schedule a timing
   pair for it alone**: 0.242 ms/step is ~27x below the measured B4
   per-request noise floor. Ride it along with a batch.
2. **`FR13_HOST_TAIL_NVTX=1` + `FR13_FIXED32_NVTX_PROFILE=1` capture**, then
   `scripts/fr13_fixed32_nsys_reduce.py` unchanged. This is the first capture
   in which `sched_next` and `prep_next` can actually appear, and it is what
   splits the remaining 2.86 ms. Report the tail's **median**, not its mean,
   and bin the prefill-carrying steps separately.
3. **Mark's ruling needed** on the F-window bubble: the single blocking 4-byte
   D2H between committer and drafter costs 2.91 ms/step of GPU idle. Keeping
   the count on device is the largest remaining host item on the whole
   critical path, and it is not this rung's to take.
4. **Retire the rung.** After the bake, the host residual on decode-cadence
   steps is ~10.2 ms/step of GPU idle, of which ~2.9 ms is Mark's call, ~1.9 ms
   is a profiling artifact that does not exist unprofiled, ~1.2 ms is the
   locked drafter, and the remainder is next-step input preparation that the
   forward genuinely depends on. There is no 5–7 ms of movable bookkeeping.
