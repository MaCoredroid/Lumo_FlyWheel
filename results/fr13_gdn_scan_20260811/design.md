# FR13 GDN fused scan — design note and verdict

**Rung:** attack-ladder item #2, *"GDN two-level scan → fused single launch,
3.45 ms/step, LEGAL"*
(`results/fr13_attack_ladder_analysis_20260808/README.md` §3, @ `7263c134d`).

**Verdict: the lever as specified is worth ~0 ms/step. Do not build it.**
The 3.45 ms/step figure is an artefact of a work-accounting error. The deployed
two-launch schedule is already **critical-path-optimal** for the fixed32 tree,
and the mechanism the ladder prescribes (per-node ready flags, one launch,
11-way path parallelism preserved) cannot change either term of the cost model.
This is a verified null and it closes the rung.

**Second finding, not in the ladder:** the GDN scan's dominant cost is not
occupancy at all — it is the **fp32 state handoff traffic**, measured at
**56% of the two-launch kernel cost**. A candidate that removes it already
exists in the tree, parked and never timed
(`_tree_gdn_kernel_fixed32_single_launch`). This note times it: it is
**byte-identical and 40.1 µs/layer faster ⇒ −1.92 ms/step** at the probe's
scale. That is a real, already-built lever that the ladder's model missed
because it never priced bytes.

Evidence: `results/fr13_gdn_scan_20260811/cost_probe.json`, produced by
`scripts/fr13_gdn_scan_fusion_cost_probe.py` on GB10.
`analysis_only=true`, `acceptance_valid=false`. Kernel microbenchmark on
synthetic tensors — **no TPS, floor, or acceptance claim**.

---

## 1. What runs today, per GDN layer, per step

`_tree_gdn_path_kernel`
(`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:10021`), two launches,
same stream, no events. Armed by `FR13_SUBTREE_PARALLEL=1` (baked default-on;
boot aborts if != 1).

Deployed geometry: `NUM_VH=48, NUM_KH=16, DIM_K=DIM_V=128, BLOCK_V=8`
(via `FR13_TREE_GDN_GEOM_OVERRIDE=BV=8`), `num_warps=8`.
Grid `(NUM_VH, cdiv(DIM_V, BLOCK_V), n_paths)` = `(48, 16, n_paths)`
⇒ **768 CTAs per path**.

The fixed32 tree (`_FR13_FIXED32_PARENT`, `:2572`) decomposes into
(`_FR13_FIXED32_SUBTREE_LEVELS`, `:2576`):

| level | n_paths | grid | paths (length) | inst/step | nsys ms/step |
|---|---:|---|---|---:|---:|
| 0 | 1 | (48,16,1) | `(0,1,4,9,14)` — 5 | 48 | 3.992 |
| 1 | 11 | (48,16,11) | `(19,24,26,28,29,30,31)` — 7; `(2,7,12,17,22)` — 5; `(3,8,13,18,23,25,27)` — 7; and 8 singletons | 48 | 8.355 |
| | | | | **96** | **12.347** |

Level 0 reads `h0` and exports all 5 of its node states to a fp32 HBM buffer;
level 1 reads its path's parent state back and exports nothing
(`STATE_SOURCE`/`EXPORT_MODE` specialisation, `:16334-16405`).

**State handoff traffic.** One node's state tile is `48 × 128 × 128 × 4 B` =
**3.0 MiB**:

| | tiles | bytes/layer | ×48 layers |
|---|---:|---:|---:|
| level-0 export writes | 5 | 15.0 MiB | 720 MiB |
| level-1 parent reads | 11 | 33.0 MiB | 1.55 GiB |
| **total handoff** | | **48.0 MiB** | **2.25 GiB/step** |

---

## 2. What a fused scan would fuse — and why it saves nothing

### 2.1 The tree's dependency depth is 12, and the deployed schedule is 12

From the parent vector alone, the deepest chain is

```
0 → 1 → 4 → 9 → 14 → 19 → 24 → 26 → 28 → 29 → 30 → 31      (12 nodes)
```

No schedule of any kind can retire this tree in fewer than **12 serial
node-steps**. The deployed schedule's makespan is
`len(level0) + max(len(level1 paths))` = `5 + 7` = **12**.

**The deployed two-launch schedule is already critical-path-optimal.**

This is not luck. The split point is node 14, and the deepest level-1 path
(`19,24,26,28,29,30,31`, length 7) hangs off node 14 — the *last* node of the
root path. So the level barrier delays nothing that is on the critical path.
Every path that *could* have started earlier (the two rooted at node 0, lengths
5 and 7) is short enough to finish by wave 8 even in a fused schedule. Pinned by
`tests/test_fr13_gdn_scan_fusion_schedule.py::test_two_launch_schedule_is_critical_path_optimal`
and `::test_level1_deepest_path_roots_at_last_level0_node`.

### 2.2 Both terms of the cost model are invariant under fusion

Write any schedule's cost as `T = Σ_waves f(CTAs_in_wave)`.

| | deployed (2 launches) | fused w/ per-node ready flags |
|---|---:|---:|
| serial waves | 5 + 7 = **12** | **12** (= DAG depth) |
| executed node-steps | **32** | **32** |
| CTA-steps | 32 × 768 = **24,576** | **24,576** |
| handoff traffic | 48.0 MiB | **48.0 MiB** (unchanged) |

Ready flags let a consumer *start earlier*; they do not remove the producer's
store or the consumer's load, because the two live in different CTAs and the
only channel between different CTAs is global memory. So the fused design keeps
100% of the handoff bytes, keeps the same wave count, and keeps the same work.
Under any model of that form the saving is **exactly zero**.

Fitting the linear form `t(C) = a + b·C` to the two nsys levels gives
`a = 15.82 µs`, `b = 1.072e-3 µs/CTA`, and both schedules evaluate to
`12a + b·24,576` = **257.3 µs/layer** — identical to 0.01 µs. A
`max(latency, work/rate)` model gives the same answer for the same reason.

### 2.3 Where the ladder's 3.45 ms came from

The ladder priced the fused kernel as *"all 82 padded slots × 768 CTA-groups =
62,976 units at level 1's demonstrated 2.944 ns/unit = 185.4 µs"*.

Both halves of that are wrong, and they are wrong in the same way:

1. **The 82 padded slots are a descriptor property, never executed.** The
   kernel's trip count is `path_len = tl.load(path_lengths + pid_path)`
   (`:10130-10131`) — a *per-path* device-loaded bound, not `MAX_PATH_LEN`. The
   eight singleton paths run exactly one iteration. Only **32** of the 82 slots
   ever execute.
2. **The 2.944 ns/unit rate divides real time by phantom work.** It is
   `174.10 µs / (768 × 77)`, but only `768 × 27` CTA-steps run. The true rate is
   `158.30 µs / 20,736` = **7.63 ns/CTA-step** — 2.6× higher.

**Measured directly.** Forcing all 11 level-1 paths to actually execute 7 slots
each (`L1_len_full_padded`, `path_lengths := 7`) costs **401.89 µs**, versus
**158.30 µs** for the deployed lengths. Executing the ladder's 82-slot workload
is 2.5× *more* expensive than what runs today — it is not a target to aim at.
Applying the corrected rate to the ladder's own model gives
`185.4 µs × 2.6 ≈ 480 µs/layer`, i.e. a large regression, not a 3.45 ms win.

### 2.4 Two further reasons not to build it

- **Forward progress.** 12 programs × 768 CTAs = 9,216 CTAs must be co-resident
  for a naive cross-CTA spin-wait to be deadlock-free. At 8 warps and 48-112
  regs/thread, GB10's 48 SMs hold far fewer. It would need a persistent
  work-queue kernel — a large rewrite whose modelled payoff is zero.
- **Spin-wait contention.** Waiting CTAs poll global memory, stealing bandwidth
  from the critical path — exactly the resource the scan is short of (§3). The
  realistic sign of this lever is **negative**.

This is the same failure mode that killed the conv rung: a throughput model
applied to a latency/dependency-bound structure. Here it is worse, because the
throughput rate itself was computed over work that does not run.

---

## 3. What actually costs the time: the fp32 state handoff

Measured on GB10 (`cost_probe.json`, p50 of 150 reps, synthetic tensors):

| probe | µs/layer | what it isolates |
|---|---:|---|
| `L0_deployed` | 91.62 | level 0 as shipped |
| `L0_no_export` | 45.63 | ⇒ **export writes cost 46.0 µs** (50% of L0) |
| `L0_len1_no_export` | 29.06 | h0 read + launch floor |
| `L1_deployed` | 158.30 | level 1 as shipped |
| `L1_len1` | 82.27 | ⇒ **parent reads + 1 step = 82.3 µs** (52% of L1) |
| `L1_len_full_padded` | 401.89 | the ladder's 82-slot workload, actually executed |
| `two_launch_total` | **228.74** | the deployed route |
| `single_launch` | **188.64** | the parked one-launch candidate |

**Handoff = 46.0 + 82.3 = 128.3 µs/layer = 56.1% of the deployed kernel cost.**

The GDN scan is **bandwidth-bound on state movement**, not occupancy-bound. The
ladder's framing — *"level 0 starves the machine, 7.36× less efficient per unit
of work"* — measures level 0 against a work metric (padded slots) that does not
track bytes. Level 0 is not idling; it is writing 15 MiB.

---

## 4. Byte safety (Tier-A analysis)

The recurrence body `_gdn_node_step` (`:9582`) is the same function on every
route. Its only reductions are `tl.sum(..., axis=1)` over `DIM_K=128` and the
q/k L2-norm `tl.sum` — and **`DIM_K` is never split across programs**
(`offs_k = tl.arange(0, DIM_K)` is not a grid axis). There is no `tl.dot`, no
split-K, no atomic accumulation, and every output element has exactly one
writer. Grid axes 0/1/2 (`vh`, `v`-block, path) partition *independent* state
rows, so repartitioning them changes no summation order.

⇒ **Fusing levels does not, by construction, change float accumulation order.**
The handoff it removes is an fp32 store followed by an fp32 load of the same
tile, which is bit-exact in both directions.

The residual Tier-A risk is **codegen**, not algebra: Triton infers the
reduction tree from the tile layout, and register pressure differs
(48/40 regs on the path kernel vs 112 on the one-launch kernel). The repo has
prior art of a layout change moving results by 1-2 ULP (`:12021-12036`). That is
why this class must be *gated*, never asserted — which is what the campaign
already does.

**Measured:** across 4 seeds × 5 input regimes (nominal, zero state, large
gating, tiny gating, all-zero), the one-launch kernel is **byte-identical** to
the deployed two-launch route on `out` **and** all four ring surfaces —
20/20 cases, `max_abs = 0.0`. This corroborates the campaign's own real-SWE B1
gate for the GQA-group3 sibling
(`results/fr13_b1_gate_a_nosplit_attempt14_pass_20260805/`,
`raw_byte_equal: true`).

---

## 5. Node-floor honesty

The ladder's own measurement is that the L0→L1 gap is **−0.128 µs**: the two
launches are already back-to-back inside CUDA graph 812. At the graph-replay
floor of **213 ns/node**, removing one graph node per layer is worth
`48 × 213 ns = 0.010 ms/step`. **Launch-count reduction contributes essentially
nothing.** Any win must come from bytes or from waves — and waves are provably
fixed (§2.2), so it must come from bytes (§3).

---

## 6. Recommendation

1. **Close ladder item #2 as a verified null.** The fused-scan-with-ready-flags
   mechanism is worth ~0 ms/step and is more likely negative. Nothing to build,
   nothing to flag. **No new env flag is introduced by this rung** — adding a
   default-off flag for a kernel whose modelled and measured payoff is zero
   would be dead plumbing.

2. **Re-price the parked one-launch candidate on the ladder.** It is built,
   byte-gated, and default-off behind `FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE`.
   Measured here at **−40.1 µs/layer ⇒ −1.92 ms/step** (probe scale;
   ≈ **−2.2 ms/step** rescaled to the nsys envelope, since the probe's
   two-launch baseline runs 11% faster than in-serving). It buys this by
   deleting the handoff (56% of cost) while paying a 32-deep instead of 12-deep
   critical path. It had **never been timed until now** — the campaign parked it
   on a codegen artefact alone.

3. **The unbuilt design worth considering next** — *not* the ladder's. Get
   12 waves **and** zero handoff by **replicating the 5-node root path** across
   the three long-branch programs, so each branch inherits its parent state in
   registers instead of through HBM. Costs 2 extra copies of a 5-node, 768-CTA
   path; saves all 48 MiB/layer. Bound from the measured numbers:
   `12 waves × 5.9-8.3 µs/wave ≈ 71-100 µs/layer` vs 228.74 today ⇒ a
   **modelled −6.2 to −7.6 ms/step**. Bit-identity is plausible (recomputing a
   node with identical operands and identical partitioning is deterministic;
   single-writer is preserved by masking the root-path `out` store to one
   program) but **unproven** — it needs the same byte gate. Flagged as a
   candidate, deliberately **not built** here.

   *Alternative mechanism for the same goal, unexplored:* Blackwell thread-block
   **clusters** would let the 12 path-programs sharing a `(vh, v-block)` slice
   form one cluster (12 ≤ the 16-CTA cluster limit) and hand the 4 KiB state
   tile over **distributed shared memory** instead of HBM — 12 waves and zero
   HBM handoff without recompute. This is the only version of "fuse the levels"
   that would actually pay, and note it is *not* what the ladder proposed:
   ready flags in global memory move no less traffic than the current export.
   Untested; cluster launch inside an existing CUDA graph is the open risk.

### Caveats on the probe

- Synthetic tensors, eager launches, back-to-back reps ⇒ the 5 exported state
  tiles (15.7 MiB) stay **hot in GB10's 24 MiB L2**. In serving, with 48 layers
  of other traffic interleaved, the handoff is likely **more** expensive, so
  §3's 56% is a *lower* bound and the one-launch win is likely *understated*.
- The probe's `two_launch_total` (10.98 ms/step at 48 layers) sits 11% under the
  nsys in-serving figure (12.347 ms/step). Deltas should be scaled by ~1.12
  before being quoted against the step envelope.
- No CUDA-graph capture, no batch > 1, no served model. The one-launch route
  additionally hard-requires B1 + BV8 + K64/root1 and rejects B4.

---

## 7. What remains for the byte gate and timing when alienware returns

The verdict above needs **no** further gate — a null ships nothing. What needs
gating is recommendation #2, if the campaign takes it:

1. **B1 byte gate on `single_launch` proper.** The runners exist and have never
   been executed: `scripts/fr13_run_b1_gdn_single_launch_live_gate.sh`. The
   GQA-group3 sibling passed the equivalent gate on a real SWE task; the plain
   `single_launch` arm has not.
2. **Timing pair.** No ms/step number exists for either one-launch candidate
   from a served run. The §3 numbers are kernel-level on synthetic tensors and
   must not be quoted as a step-envelope result.
3. **CUDA-graph capture/replay** with the one-launch route armed (the probe is
   eager-only), and confirmation that the 112-reg kernel does not perturb
   graph-812 capture.
4. **B4.** The one-launch route rejects batch 4 by contract; the deployed
   two-launch route would remain for B4 unless a batched variant is built.

## 8. Files

| file | what |
|---|---|
| `design.md` | this note |
| `cost_probe.json` | measured decomposition + byte A/B sweep |
| `../../scripts/fr13_gdn_scan_fusion_cost_probe.py` | the probe (offline, no credential) |
| `../../tests/test_fr13_gdn_scan_fusion_schedule.py` | schedule-invariance tests pinning §2 |

## 9. Reproduce

Docker must be at 0 first. One container, CPU-launched, no campaign credential,
no offload host:

```
docker run --rm --gpus all --ipc=host \
  -e TRITON_CACHE_DIR=/workspace/.triton-cache \
  -v $PWD:/workspace -w /workspace \
  lumo-flywheel-vllm:26.01-py3-v0.19.0 \
  python3 scripts/fr13_gdn_scan_fusion_cost_probe.py \
    --out /workspace/results/fr13_gdn_scan_20260811/cost_probe.json \
    --reps 150 --byte-seeds 4
```

Exits non-zero if any byte A/B case fails.
