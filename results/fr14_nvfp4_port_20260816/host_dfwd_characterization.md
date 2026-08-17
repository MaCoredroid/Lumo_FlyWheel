# FR14 host/scheduling domain — where the milliseconds are at 4-bit

**Deliverable 1** of the FR14 host/scheduling rung (DFWD pass-scheduling consolidation +
host-tail remainder). Characterization **only**: no GPU was taken, no arithmetic changed,
no flag added. Every number below is read out of an already-banked sidecar, an already-banked
census, or an already-pinned constant; nothing is derived from a model of the code.

`analysis_only=true`, `acceptance_valid=false`, `citable=false`. No TPS, floor or acceptance
claim is made or moved.

---

## 0. Headline: the rung as briefed does not survive its own evidence

The brief prices this rung at **≈ −6 ms** (DFWD per-pass host scheduling consolidation)
**+ ≈ −5..7 ms** (post-DFWD Python tail + CFWD eager-launch graph capture). Measured at
4-bit on the live serves:

| briefed lever | briefed | measured status at HEAD / 4-bit | ceiling that actually remains |
|---|---:|---|---:|
| DFWD per-pass host scheduling ("5 MTP passes each with host round-trips") | −6 | **already consolidated.** 4 of the 5 MTP passes run inside **one** CUDA-graph replay per step (`drafter_runtime.graph_replays = 1`, `graph_captures = 0`, `mtp_observation = capture_manifest_bound_replay`, on **20 579/20 579** live steps). `FR13_DRAFTER_GRAPH` is a *required* member of the fixed32 hardware-floor runtime, not an optional lever. | ≤ **1.24 ms** (FR13's whole measured `dfwd` host-idle window, pre-graph) |
| CFWD eager-launch graph capture | −3 | **already harvested.** The ladder's 2.99 ms came from *1111 eager ops/step* in the 2026‑07‑31 capture. `FR13_COMMITTER_GRAPH` + `FR13_COMMITTER_BATCHED` are now required-on (`committer.route = fixed16_device_fill_graph`, `graph_replays = 1`, `pointer_table_rebuilds = 0`, `host_flag_readbacks = 0` on all 20 579 steps), and the 2026‑08‑08 capture measures the surviving `cfwd` host-idle at **0.809 ms**. | ≈ **0.81 ms** |
| post-DFWD Python tail | −5..7 | already refuted **once** by `c78b3ad41` (5–7 was the ladder's *realistic* band around a 10.08 ms ceiling; the decode-cadence median is 3.46 ms of GPU idle / 2.86 ms of pure Python), and 0.242 ms of it is already shipped as `FR13_HOST_TAIL_PREP_BAKE` — **armed in tonight's maxstack serve**. | ≈ **2.6 ms**, no single item > 1 ms |

**The whole host-attributable, byte-safe envelope at 4-bit is ≈ 4.1–4.7 ms/step — about
2.0–2.2 % of the 210.5 ms step — and it is spread across ~45 Python blocks of 10 µs–1.45 ms.**
The two multi-millisecond items adjacent to it are both out of bounds: the 2.91 ms blocking
4-byte D2H committer→drafter handoff is Mark's call (it changes what the drafter is fed), and
the 1.945 ms `cudaGraphLaunch` line is a CUPTI artifact worth 3.15 µs in reality
(`results/fr13_host_residual_20260811/design.md` §4, §6.4).

What the 4-bit evidence *does* newly reveal is in §3: **`overhead_other` is not host time.**
Most of it is a weight-bound GPU term — the verifier `lm_head` projection — that sits outside
every existing span timer.

---

## 1. The 4-bit phase budget (measured, no nsys required)

Instrument: `_Fr13SfwdGpuTimer` (pure-decode model-forward, async cuda events, idle-capped
start-to-start wall) + `_Fr13SpanTimer` ×2 (drafter = `propose_draft_token_ids`,
committer = `rejection_sampler` dispatch). `overhead_other_ms_per_event` is the harness's own
first-class residual, defined in `scripts/fr13_measure.py:2009` as
`wall_s_per_event − (s_fwd_gpu + (dfwd + cfwd)/events_per_step)` — i.e. exactly
`step_wall − sfwd − dfwd − cfwd`. B = 1, so `events_per_step = 1` and the identity is exact.

| arm | checkpoint / draft-head config | step ms | sfwd ms | dfwd ms | cfwd ms | **other ms** | decode steps |
|---|---|---:|---:|---:|---:|---:|---:|
| `armb_b1_partial_n3` | arm B, `K=65536 ROOT=1` | 199.403 | 125.806 | 45.900 | 20.341 | **7.356** | 12 587 |
| `b1_stock` 0816T200746Z | unsloth, `K=65536 ROOT=1` | 206.310 | 125.881 | 44.173 | 21.301 | **14.956** | 2 368 |
| `maxstack` 0817T210423Z *(live)* | radixark, `K=0` | 210.463 | 130.296 | 52.175 | 20.656 | **7.336** | 20 481 |
| `b1radix` 0817T031507Z | radixark, `K=0` | 211.465 | 134.250 | 49.226 | 20.531 | **7.458** | 23 496 |
| `b1radix` k0-aggressive n4 | radixark, `K=0` | 215.311 | 134.552 | 52.674 | 20.642 | **7.443** | 21 487 |
| `stock_leverpair` 0817T130251Z | radixark, `K=0` | 215.899 | 134.793 | 53.069 | 20.541 | **7.497** | 18 507 |
| `b1_stock_tail6` | unsloth, `K=65536 ROOT=1` | 218.764 | 134.671 | 47.622 | 20.117 | **16.354** | 12 791 |

Draft-head config read from each arm's `container_env.txt`
(`FR13_DRAFT_VOCAB_K` / `FR13_DRAFT_VOCAB_ROOT` / `FR13_MANDATORY_WEIGHT_BYTES`), which the
reducer cross-checks against the pinned ledger and refuses to run on a mismatch
(`fr13_measure.py:1915-1985`). The radixark arms serve `K=0` — the 64K subset draft head is
**retired**, so all five drafter passes read the full NVFP4 head (`FR13_MANDATORY_WEIGHT_BYTES
= 25 430 574 256`, floor 93.152 ms). The unsloth arms serve the FR13 `K=65536 ROOT=1` ledger
(27 977 022 848 B, floor 102.480 ms).

Share of step on the live maxstack arm: **sfwd 61.9 % · dfwd 24.8 % · cfwd 9.8 % · other 3.5 %.**

Sources: `output/fr13_sfwd_sidecar/*.json.{154,155,156}` (+ `_dfwd`/`_cfwd` twins, schema
`fr13.span_gpu_timer.v1` / `fr13.sfwd_gpu_timer.v2`) and the banked reducer outputs
`results/fr14_nvfp4_port_20260816/{b1_stock_tail6,b1_k0_aggressive_n4,armb_b1_partial_n3}_deploy_speed.json`.
The maxstack row is read live off the running serve's sidecar and will move; it is quoted as
the *shape* of the budget, not as a banked number.

### 1.1 Cross-precision anchor (same instrument family, FP8 era)

FR13's 2026‑08‑08 nsys capture, GPU-projected first-to-last spans
(`results/fr13_host_residual_20260811/design.md` §1):
`step 237.248 · sfwd 155.829 · cfwd 20.705 · dfwd 35.131`.

| phase | FP8 (3.6, nsys projection) | NVFP4 radixark (cuda events) | delta |
|---|---:|---:|---:|
| sfwd | 155.829 | 130.296 – 134.793 | **−21 to −25 ms** |
| dfwd | 35.131 | 49.226 – 53.069 | **+14 to +18 ms** |
| cfwd | 20.705 | 20.117 – 21.301 | **≈ 0** |

Caveat, stated once and load-bearing: the FP8 column carries CUPTI overhead and a different
model, tree arm and lever set; it is a *shape* anchor, not an A/B. The two conclusions it
supports are robust to that caveat because they are sign-and-order arguments, not deltas:

1. **Quantization moved the budget toward the drafter and the host.** sfwd fell ~16 %, dfwd
   rose ~45 %, cfwd did not move. The verifier forward is no longer 66 % of the step; it is
   62 %, and the non-verifier remainder (dfwd + cfwd + other) is now **38 %** of a decode step.
2. **cfwd is precision-invariant.** 20.7 ms at FP8, 20.1 ms at unsloth-NVFP4, 20.5–20.7 ms at
   radixark-NVFP4. It contains **no weight bytes at all** (§4) — it is fixed sampler work, and
   it is the phase whose *share* grows fastest as the weights shrink (8.7 % → 9.8 %).

---

## 2. Every per-step work shape is invariant over 20 579 consecutive live decode steps

Parsed the live maxstack `logs/fr13_fixed32_work_census.jsonl` (schema
`fr13-fixed32-work-census-v12`), all 20 579 records. **Every** shape counter is single-valued:

| counter | value | distinct values in 20 579 steps |
|---|---:|---:|
| `batch_size` / `physical_drafts` / `verify_rows` / `active_nodes` | 1 / 31 / 32 / 27 | 1 each |
| `drafter.mtp_forward_calls` | 4 | 1 |
| `drafter_runtime.graph_replays` / `graph_captures` | 1 / 0 | 1 each |
| `drafter.arctic_lookup_calls` / `main_tail_length` | 3 / 6 | 1 each |
| `committer.graph_replays` / `fused_layer_calls` / `ring_gather_ops` / `neutralize_ops` | 1 / 48 / 4 / 5 | 1 each |
| `committer.route` | `fixed16_device_fill_graph` | 1 |
| `taw.loop_iterations` / `walk_levels` | 12 / 12 | 1 each |
| `taw.full_vocab_softmax_calls` / `_fp32_casts` / `_row_gathers` / `_normalizations` | 24 / 24 / 24 / 36 | 1 each |
| `taw.exact_commit_launches` / `row_scatter_slots` | 12 / 24 | 1 each |
| `kv_remap.apply_cache_calls` | 17 | 1 |
| `gdn.launches` / `scan_calls` | 96 / 48 | 1 each |
| `tree_attn.calls` | 16 | 1 |

That is the fixed32 design intent discharged: **the per-step host work shape is
data-independent in deployment.** Any graph capture proposed on this path can therefore be
shape-checked fail-closed against a compile-time literal rather than a runtime guess — and,
symmetrically, any lever justified by "the shape varies" has no evidence behind it here.

---

## 3. `overhead_other` is mostly a weight-bound GPU term, not host time

The `other` column in §1 splits cleanly by **checkpoint**, not by arm or lever:

- radixark arms (n = 4, 83 971 decode steps): **7.336, 7.443, 7.458, 7.497** → mean **7.433 ms**
- unsloth arms (n = 2, 15 159 decode steps): **14.956, 16.354** → mean **15.655 ms**
- difference: **8.222 ms/step**

The `FR13_*=1` lever sets of the two `tail6_fixed32_b1{stock,radix}` arms differ by exactly
**one** entry (`FR13_DRAFT_VOCAB_ROOT`, a *drafter* flag, which cannot move a post-drafter
window), so the split is not a lever effect. The mechanism is byte-level and pinned:

- `_Fr13SfwdGpuTimer` brackets **`self._model_forward` only** (patch docstring,
  `scripts/fr10_phase4_patch_vllm_tree_gdn.py:34389`). vLLM v1 calls `compute_logits` — the
  verifier `lm_head` projection — **after** `_model_forward` returns. It is therefore in
  *none* of the three spans and lands entirely in `other`.
- Pinned head bytes (`scripts/fr13_hardware_floor_ledger.py:181`, derived from the shipped
  tensors): radixark NVFP4 head **715 161 608 B = 2.620 ms** of floor; unsloth's BF16 head
  **2 542 796 800 B = 9.314 ms**. Delta **1 827 635 192 B = 6.695 ms**.
- Measured delta 8.222 ms ÷ floor delta 6.695 ms ⇒ the head GEMM is running at **81.4 % of
  roofline** — squarely inside the 82–86 % band the campaign has measured for every other
  weight-bound GEMM on this box. The mechanism is confirmed by an independent efficiency
  cross-check, not asserted.

**Consequence — the number every later host claim must be built on:**

```
other (radixark, mean)                     7.433 ms
  − lm_head floor                          2.620 ms   (81.4% roofline ⇒ ~3.22 ms as executed)
  ------------------------------------------------
  host-attributable remainder      4.21 – 4.81 ms/step
```

That is an **upper bound** — it still contains any other off-span GPU work — and it agrees
with a completely independent instrument: FR13's decode-cadence nsys host timeline
(`design.md` §2) puts the non-span host windows at
`B 0.071 + C 0.177 + D 0.181 + H 3.667 = 4.096 ms` of host wall. Two instruments, two eras,
two checkpoints: **≈ 4.1–4.8 ms.**

Two immediate consequences beyond this rung, flagged, not claimed:

1. **`other` must stop being reported as host overhead.** `overhead_other_note` in
   `fr13_measure.py` calls it "host glue, sampler, packer, scheduler gap". At floor prices the
   verifier head is **35 %** of radixark's `other` and **59 %** of unsloth's; at the 81.4 %
   roofline measured above, **43 %** and **73 %**. Any arm-vs-arm comparison that reads `other`
   as host cost across checkpoints with different heads is reading a GEMM.
2. **The 6.695 ms head-byte saving is real and already banked in arm B's floor** — it is the
   single largest term the NVFP4 port moved, and it shows up in `other`, which is why no
   phase table has been crediting it.

---

## 4. cfwd: 20.66 ms/step of zero-weight-byte work, and what is left to graph

cfwd reads **no** model weights (the floor ledger assigns it none), and it is
precision-invariant across three checkpoints (§1.1). At 9.8 % of the step it is now larger
than the entire host tail by 4×. The census says what it is doing:

`taw.route = fixed32_pytorch_exact_float_triton_integer_commit` — an exact-float tree
rejection sampler walking **12 levels** (`loop_iterations = walk_levels = 12`, invariant),
each level issuing full-vocabulary (`vocab_size = 248 320`) ops:

| op class (per step) | count |
|---|---:|
| `full_vocab_softmax_calls` | 24 |
| `full_vocab_fp32_casts` | 24 |
| `full_vocab_row_gathers` | 24 |
| `full_vocab_normalizations` | 36 |
| `source_cdf_calls` / `residual_where_calls` | 12 / 24 |
| `residual_clamp` / `residual_subtract` / `qmix_scatter_add` / `qmix_zero_fill` | 12 each |
| `exact_commit_launches` (triton) | 12 |
| **total full-vocab tensor ops** | **≈ 192** |

At the 12-row × 248 320 × fp32 working set this is O(4–5 GB) of logits traffic per step — the
right order for 20.66 ms at 273 GB/s. **cfwd is bandwidth-bound on logits, not launch-bound.**

Graph capture of the surviving eager ops is therefore worth the *launch bubble only*:
FR13 measures the whole `cfwd` host-idle at **0.809 ms/step** (`tail_attribution.json`,
segment E), which at the campaign's measured 2.73 µs/eager-op is ~296 ops' worth — consistent
with the ~192 counted here plus the untimed remainder. **Ceiling ≈ 0.81 ms, not 2.99 ms.**
The ladder's 2.99 ms was priced against 1111 eager ops in the 2026‑07‑31 capture, before
`FR13_COMMITTER_GRAPH` / `FR13_COMMITTER_BATCHED` became required-on.

The 4–5 GB of logits traffic *is* the large addressable object here, but reducing it means
changing what the sampler computes — arithmetic, not scheduling. Out of scope for this rung
and named only so it is not lost.

---

## 5. dfwd: 52.2 ms/step against a 28.0 ms byte floor — and the host is not the gap

Pinned floor (`fr13_hardware_floor_ledger.py`, 273 GB/s):

| term (radixark `K=0`, the served config) | bytes | ms |
|---|---:|---:|
| MTP block × 5 passes (`1 initial + 4 post-root-graph`) | 4 246 993 920 | 15.557 |
| drafter head: 5 × full NVFP4 head (subset head retired) | 3 575 808 040 | 13.098 |
| **dfwd floor** | **7 822 801 960** | **28.655** |
| measured dfwd (radixark, 4 arms) | — | **49.2 – 53.1** |
| **above floor** | — | **20.6 – 24.4** |

The DFWD span brackets `propose_draft_token_ids` in full (patch site
`fr10_phase4_patch_vllm_tree_gdn.py:23750`), so all five passes are inside it. Of the 20.6–24.4 ms
above floor, the **host** share is bounded by FR13's `dfwd` window measurement of
**1.243 ms/step** of GPU idle (0.799 ms Python between drafter iterations + 0.336 ms in
`cudaMemcpyAsync`) — and that was measured **before** the drafter graph, when the drafter ran
five eager iterations each with a blocking readback. Today 4 of 5 passes are one graph replay
(§2), so the true figure is at or below 1.243 ms.

**≥ 94 % of dfwd's 20.6–24.4 ms overhead is GPU, not host scheduling.** Consolidating per-pass
host work cannot reach it. The residual belongs to the drafter-kernel / draft-head domain.

### 5.1 A flagged observation for the drafter domain (not this rung's to take)

The two checkpoints' drafter-head floors are within 6.5 % of each other — radixark `K=0`
reads 5 × the full NVFP4 head (13.098 ms), unsloth `K=65536 ROOT=1` reads 5 × the 64K BF16
slice (12.291 ms), a **0.807 ms** floor difference — and their MTP blocks are
**byte-identical** (849 398 784 B/pass, 15 BF16 tensors, both repacks, verified by summing the
shipped tensor spans). Yet measured dfwd differs by **≈ 4.5 ms** (unsloth 44.2–47.6 vs
radixark 49.2–53.1), i.e. **5.6× the byte difference**.

Bytes do not explain it. The obvious candidate is *output* width, not input: under `K=0` every
one of the five drafter passes materialises and top-k's **248 320** logits instead of
**65 536** — 3.79× the write traffic and 3.79× the reduction, five times per step. K0 is
nearly free in weight bytes precisely because the NVFP4 head is small; that is what made it
attractive, and it is exactly why the byte ledger cannot see its cost.

Confounded by checkpoint ⊗ flag, so this is **not a claim** — it is a one-pair A/B
(`FR13_DRAFT_VOCAB_K` 0 vs 65536 on radixark, same checkpoint, same everything else) that
would settle a ~4 ms/step question, and it belongs to the drafter/DVK domain, not this rung.

### 5.2 DEFECT: the instrument that would settle §5.1 is unreachable in deployment

`FR13_DFWD_SPLIT` (`fr10_phase4_patch_vllm_tree_gdn.py:36972`) already exists and does exactly
the needed job: a 3-way cuda-event split of the drafter into **model** (draft forward),
**head** (`compute_logits` + top-k, i.e. the full-vocab lm_head read per level) and **other**
(metadata rebuild, repeat/cat, slot math, buffer copies), dumped to a
`fr13.dfwd_split.v1` sidecar. Its docstring says it exists to decide
"FR-Spec-vocab vs level-fusion vs shape-depth as the drafter attack" — the §5.1 question,
verbatim.

**It has never engaged.** The arming path is the proven worker-env-drop-proof pattern: the
patcher's `main()` runs at container pid 1, reads `FR13_DFWD_SPLIT` and writes
`/logs/fr13_dfwd_split.flag`, which the EngineCore worker reads
(`fr10_phase4_patch_vllm_tree_gdn.py:42878`, `:36993`). But **neither launcher forwards
`FR13_DFWD_SPLIT` into the container** — `scripts/fr13_launch_forked_fa2_tree_server.sh` and
`scripts/fr14_armb_leg3_launch_nomiddleware.sh` forward only `FR13_DFWD_SPLIT_NEEDLE`
(a *different*, unrelated host-wall probe at `:29522`). Verified end to end:

- the live serve's `container_env.txt` contains `FR13_DFWD_SPLIT_NEEDLE=0` and **no**
  `FR13_DFWD_SPLIT` entry at all;
- `logs/fr13_dfwd_split.flag` reads `0` in **35 of 35** runroots on this box;
- **zero** `fr13_dfwd_split.json.*` sidecars and zero `fr13_dfwd_split.err` files exist
  anywhere in `output/` or `results/`.

So the flag is unsatisfiable at any host-side setting — the same class of defect as
`fr13.fixed32.sched_next` (injected definition, no call site) and the pre-`c78b3ad41`
`FR13_HOST_TAIL_*` (read inside the worker whose curated env drops bare `FR13_*` masters),
both of which this campaign has already had to find and fix.

The repair is one forwarding line per launcher plus strict `0|1` validation, default `0`,
byte-identical when off. `FR13_DFWD_SPLIT_JSON` needs nothing — it already defaults to
`/logs/fr13_dfwd_split.json`, inside the bind-mounted logs dir. **This is the single
highest-value change available to this rung**, because it converts the largest unexplained
block in the 4-bit budget (dfwd's 20.6–24.4 ms above floor) from an argument into a
measurement. It is deliberately NOT taken here: it edits launcher files during a live serve,
which is a freeze-window decision, not a characterization one.

---

## 6. The host tail, carried forward and re-priced at 4-bit

From `results/fr13_host_residual_20260811/` (1130 decode-cadence steps, GPU idle computed as
the complement of the merged busy union over all streams), with this rung's 4-bit disposition:

| item | ms/step | disposition at HEAD, 4-bit |
|---|---:|---|
| tail: framework Python in next-step input prep | 2.86 | **live**; genuine precondition (it builds slot mapping / block tables / positions the next forward consumes) so only its *cost* is attackable |
| — tree depth-position re-derivation | 0.242 | **shipped** (`FR13_HOST_TAIL_PREP_BAKE`) and **armed in tonight's maxstack serve** |
| — capture-manifest `json.loads` | 0.003–0.025 | priced, unshipped |
| — Arctic cached-id set rebuild | 0.000–0.203 | occupancy still unmeasured ⇒ still unpriceable |
| tail: eager dispatch of 154 CUDA ops | 0.60 | needs graph capture of `_prepare_inputs`; shapes now provably invariant (§2), so the classic blocker is gone |
| tail: small H2D copies pageable→pinned | ≤ 0.108 | pageable share still unmeasured |
| F: blocking 4-byte D2H committer→drafter | 2.91 | **OFF-LIMITS** — changes what the drafter is fed (Mark's call) |
| A: `cudaGraphLaunch` | 1.945 | **CUPTI artifact**; real cost 3.15 µs |
| G: drafter inter-iteration Python | 1.24 | now largely inside the drafter graph (§5) |
| E: cfwd eager dispatch | 0.81 | §4 |

Sum of the byte-safe, in-bounds items: **≈ 2.6 ms (tail Python less the shipped bake) + 0.60
(tail dispatch) + 0.81 (cfwd dispatch) ≈ 4.0 ms/step**, which is the same envelope §3 reaches
from the opposite direction. **1.9 % of the step**, no single item above 1 ms, against a B4
stock-vs-stock noise floor of 3.8 % aggregate / 6.5 % per-request.

`FR13_HOST_TAIL_DEFER` is **pulled** and stays pulled: REDTEAM passes 27–28 record a
first-request engine-fatal with DEFER + PREP_BAKE together that neither reproduces alone. Its
ordering semantics are suspect and nothing here is built on it.

---

## 7. What this means for the rung

1. **Nothing in the host/scheduling domain is worth ≥ 1 ms/step at 4-bit.** The two
   multi-millisecond items are a CUPTI artifact and an off-limits data dependency. The
   remaining ≈ 4 ms is ~45 Python blocks and ~250 eager launches.
2. **Both graph-capture levers in the brief are already shipped** — as *required* members of
   the fixed32 hardware-floor runtime, not as flags anyone can turn on. Re-implementing them
   would measure nothing.
3. **Two instrument defects, both one-line fixes, both worth more than the levers.**
   (a) §3: `overhead_other` is 43–73 % verifier-head GEMM (as executed) depending on
   checkpoint — the number the campaign has been reading as "host overhead" in every arm
   comparison. (b) §5.2: `FR13_DFWD_SPLIT`, the existing 3-way drafter split timer, is not
   forwarded by either launcher and has engaged **zero** times in 35 runroots, so the largest
   unexplained block at 4-bit has an instrument written for it that nobody can turn on.
4. The largest addressable objects the 4-bit budget exposes are **not** host: cfwd's 4–5 GB of
   full-vocab logits traffic (§4, arithmetic-changing), dfwd's 20.6–24.4 ms above its byte
   floor (§5, kernel-domain), and the `K=0` output-width question in §5.1 (~4 ms, one A/B).
   All three are named here so the host rung's null result does not bury them.

**Recommended revision to deliverables 2–4**, for the parent to rule on before any code:

- **(2) CFWD graph capture → REFUTED, do not build.** Already required-on; residual 0.81 ms.
- **(3) Python-tail reduction → build, but scoped honestly.** The two byte-safe unshipped
  items are the capture-manifest `json.loads` memo (0.003–0.025 ms) and the tail's eager
  dispatch (0.60 ms, needs a `_prepare_inputs` capture that §2's invariance now makes
  shape-checkable). Combined ceiling ~0.63 ms = 0.3 % of step, ~12× below the noise floor.
  Worth shipping only on the `PREP_BAKE` precedent — provable and free — never as a gate mover.
- **(4) DFWD scheduling consolidation → REFUTED as briefed.** ≥ 94 % of the gap is GPU. What
  should replace it is **instrument plumbing**, and both halves are one-line changes with no
  new machinery to design:
  1. **Forward `FR13_DFWD_SPLIT` in both launchers** (§5.2) — an existing, default-off,
     already-written 3-way drafter split timer that has never engaged in 35 runroots. Turns
     dfwd's 20.6–24.4 ms into model / head / other.
  2. **Bracket `compute_logits`** with a fourth span timer (§3) so the verifier head stops
     being reported as host overhead.

  Both are byte-identical when off, need no GPU to write, and are worth more than every
  millisecond deliverables 2–4 were briefed to chase, because they are the difference between
  a measured budget and an argued one.

---

## 8. Evidence index

| claim | file |
|---|---|
| 4-bit phase budget, all arms | `output/fr13_sfwd_sidecar/*.json.{153..156}` (+`_dfwd`,`_cfwd`); `results/fr14_nvfp4_port_20260816/{b1_stock_tail6,b1_k0_aggressive_n4,armb_b1_partial_n3}_deploy_speed.json` |
| `overhead_other` definition | `scripts/fr13_measure.py:2009,2271` |
| sfwd span brackets `_model_forward` only | `scripts/fr10_phase4_patch_vllm_tree_gdn.py:34389` (docstring) |
| dfwd span brackets `propose_draft_token_ids` | `scripts/fr10_phase4_patch_vllm_tree_gdn.py:23750` |
| cfwd span brackets `rejection_sampler` dispatch | `scripts/fr10_phase4_patch_vllm_tree_gdn.py:35504` |
| span timer = event-elapsed, not kernel-busy | `scripts/fr10_phase4_patch_vllm_tree_gdn.py:35127` (`_Fr13SpanTimer`) |
| per-step shape invariance, 20 579 steps | `output/fr14_maxstack_20260817T210423Z/hydra27_fixed32_maxstack_maxstack/logs/fr13_fixed32_work_census.jsonl` |
| drafter graph = 4 MTP calls, 1 replay, manifest-bound | `scripts/fr10_phase4_patch_vllm_tree_gdn.py:6661` (`_fr13_fixed32_drafter_graph_replay`) |
| committer graph/batched required-on | `scripts/fr10_phase4_patch_vllm_tree_gdn.py:10115` (`required_one`); arm files `logs/fr13_committer_{graph,batched}.arm` |
| pinned head / MTP / target byte ledger | `scripts/fr13_hardware_floor_ledger.py:60,123,181,201` |
| per-arm draft-head config + declared weight bytes | each runroot's `container_env.txt` (`FR13_DRAFT_VOCAB_K`, `FR13_DRAFT_VOCAB_ROOT`, `FR13_MANDATORY_WEIGHT_BYTES`, `FR13_WEIGHT_FLOOR_MS`); enforced at `scripts/fr13_measure.py:1915-1985` |
| MTP blocks byte-identical across both NVFP4 repacks | safetensors header sum, `/models/qwen3.8-27b-nvfp4{,-radixark}`: 15 BF16 tensors, 849 398 784 B each |
| FP8-era host timeline + refutations | `results/fr13_host_residual_20260811/design.md`, `tail_attribution.json` |
| ladder's 1111-eager-op CFWD pricing (superseded) | `results/fr13_attack_ladder_analysis_20260808/README.md:64,404` |
| DEFER pulled, engine-fatal pair | `results/fr14_nvfp4_port_20260816/REDTEAM_20260816.md` passes 27–28 |
| `FR13_DFWD_SPLIT` exists / arms via flag file / never engaged | `scripts/fr10_phase4_patch_vllm_tree_gdn.py:36972,36993,42878`; launcher forwarding lists `fr13_launch_forked_fa2_tree_server.sh:3677,6288` and `fr14_armb_leg3_launch_nomiddleware.sh:3468,6085`; `logs/fr13_dfwd_split.flag` = `0` in 35/35 runroots; no `fr13_dfwd_split.json.*` anywhere |
