# FR14 — GDN root-path replication: design note, and the measurement that killed it

**Rung:** the under-210 plan's kernel lane, *"GDN replication ~−7 ms"*
(`REDTEAM_20260816.md` pass 29). The design is FR13's own successor
recommendation, `results/fr13_gdn_scan_20260811/design.md` §6.3 @ `9d65d6ea8`:

> Get 12 waves **and** zero handoff by **replicating the 5-node root path**
> across the three long-branch programs, so each branch inherits its parent
> state in registers instead of through HBM. […] a **modelled −6.2 to −7.6
> ms/step**. Bit-identity is plausible […] but **unproven**.

**Status: MEASURED AND REFUTED — DO NOT BUILD.** Stage 0 ran 2026-08-17 on a
free GPU (`gdn_replication_stage0_b1.json`). The replicated schedule costs
**353.76 µs/layer** against `single_launch`'s **188.19** — **+165.57 µs/layer =
+7.95 ms/step**, which is **78× the run-to-run spread**. The pre-registered rule
in §7 says **REFUTED**, and the probe emits that verdict itself.

The sections below are kept as written *before* the run, with the measured
outcome added in **§9**. The prediction they make is +134..170 µs/layer worse
than `single_launch`; the measurement is +165.57, inside that range. The
load-bearing assumption — that a node-step costs the same wherever it sits, so
waves are nearly free and replication trades a free resource for an expensive
one — was tested directly and holds across a **12× width change**.

---

## 1. The exact design (R-A: full replication)

One launch. Every program starts from `h0` and replays the root-path **prefix**
it needs before running its own nodes, so no program ever reads or writes a
state tile in HBM.

The tree (`_FR13_FIXED32_PARENT` / `_FR13_FIXED32_SUBTREE_LEVELS`,
`fr10_gdn_tree_kernel.py:3002-3024`) gives exactly this schedule:

| z | replayed prefix (no `out`) | own nodes (`out` written) | chain |
|---:|---|---|---:|
| 0 | — | 0, 1, 4, 9, 14 | 5 |
| 1 | 0, 1, 4, 9, 14 | 19, 24, 26, 28, 29, 30, 31 | **12** |
| 2 | 0 | 2, 7, 12, 17, 22 | 6 |
| 3 | 0 | 3, 8, 13, 18, 23, 25, 27 | 8 |
| 4 | 0, 1 | 5 | 3 |
| 5 | 0, 1 | 6 | 3 |
| 6 | 0, 1, 4 | 10 | 4 |
| 7 | 0, 1, 4 | 11 | 4 |
| 8 | 0, 1, 4, 9 | 15 | 5 |
| 9 | 0, 1, 4, 9 | 16 | 5 |
| 10 | 0, 1, 4, 9, 14 | 20 | 6 |
| 11 | 0, 1, 4, 9, 14 | 21 | 6 |

- **Makespan 12** = the DAG depth (`0→1→4→9→14→19→24→26→28→29→30→31`). Optimal.
- **Handoff 0 MiB** (from 48.0 MiB/layer).
- **Node-steps 67**, from 32. **+35 (2.09×).**

**A correction to FR13's sketch.** It said *"replicate the 5-node root path
across the three long-branch programs; costs 2 extra copies of a 5-node path."*
That does not match the tree: only the branch rooted at node 14 needs all five
root nodes; the two branches rooted at node 0 need **one**. And the eight
singleton leaves cannot be ignored — they hold eight of the eleven parent
reads. Replicating only the three long branches (variant **R-C**, §5) leaves
25% of the handoff in place *and still needs two launches*, so it achieves
neither of the design's two goals.

**What has to be built** (beyond the descriptors, which are pure host-side
data): the path kernel's `out` store is unconditional on `n_ok`
(`:10911-10915`). Under replication, up to six programs compute node 0 and all
six would store `out[0]` — identical bytes, but a genuine multi-writer race and
a double-write of the ring surfaces under `RING_EXPORT`. A `REPLAY_STEPS`
per-path constexpr/descriptor is required to suppress `out`, ring, and flag
stores for the prefix. That is the whole kernel change; the recurrence body
`_gdn_node_step` is untouched.

---

## 2. Tier-A: the design does NOT force an accumulation-order change

This is the question the brief asked to answer before anything else, and the
answer is clean.

A replayed prefix step is *the same call to `_gdn_node_step` on the same
operands with the same partitioning* as the original. Specifically:

- the only reductions are `tl.sum(..., axis=1)` over `DIM_K=128` and the q/k
  L2-norm, and **`DIM_K` is never a grid axis** (`offs_k = tl.arange(0, DIM_K)`);
- grid axes `(vh, v-block, path)` partition independent state rows, and
  replication changes only the third;
- there is no `tl.dot`, no split-K, no atomic accumulation.

⇒ the state entering a branch is **bit-identical** to the state the branch reads
from HBM today, because it is the same arithmetic in the same order. The fp32
store/load pair that replication deletes is bit-exact in both directions.
**Tier-A is satisfiable by construction; no STOP for Mark is required on this
axis.**

Two residual risks, both codegen rather than algebra, both gate-able:

1. **Register pressure.** The path kernel runs 48/40 regs; folding a 12-step
   chain pushes toward the 112 regs the `single_launch` kernel needs. The repo
   has prior art of a layout change moving results by 1–2 ULP (`:12021-12036`).
2. **Store suppression.** `REPLAY_STEPS` must suppress `out` *and* all four ring
   surfaces *and* flags — a partial suppression would corrupt the ring
   comparison surfaces the byte gate reads, which is a correctness bug that
   looks like a byte-gate failure.

---

## 3. The arithmetic that says not to build it

All six inputs are FR13's banked probe (`cost_probe.json`, p50 of 150 reps, B1,
synthetic, eager, GB10). µs per GDN layer:

| probe | µs | node-steps | max width |
|---|---:|---:|---:|
| `L0_len1_no_export` | 29.06 | 1 | 1 |
| `L0_no_export` | 45.63 | 5 | 1 |
| `L0_deployed` | 91.62 | 5 | 1 |
| `L1_len1` | 82.27 | 11 | 11 |
| `L1_deployed` | 158.30 | 27 | 11 |
| `L1_len_full_padded` | 401.89 | 77 | 11 |
| `two_launch_total` | **228.74** | 32 | 11 |
| `single_launch` | **188.64** | 32 | 1 |

### 3.1 A node-step costs ~4.1–4.8 µs, and that does not depend on wave width

Three independent marginals, at three different widths:

| fit | from | width | µs/node-step |
|---|---|---:|---:|
| a | `(L0_no_export − L0_len1_no_export)/(5−1)` | 1 | **4.143** |
| b | `(L1_deployed − L1_len1)/(27−11)` | 2–3 | **4.752** |
| c | `(L1_len_full_padded − L1_len1)/(77−11)` | 11 | **4.843** |

If the kernel were latency-bound per wave, a node-step inside a width-11 wave
would cost ~1/11 of one inside a width-1 wave. It costs **within 17% of the
same**. The machine is throughput-saturated even at width 1 (768 CTAs × 8 warps
on 48 SMs is already ~2× the warp budget per SM), so **cost tracks total
node-steps, and waves are nearly free.**

### 3.2 The already-built kernel is the direct proof that waves are free

`single_launch` runs the identical 32 node-steps in **32 serial waves** — 2.67×
the deployed schedule's 12 — on a *more expensive* 112-register kernel
(5.116 µs/node-step vs 4.143), and it still **wins by 40.1 µs/layer**. It buys
that purely by deleting the handoff.

That single comparison prices both resources at once: **32 extra waves are worth
less than the handoff they cost nothing to add.** The replication design spends
the expensive resource (+35 node-steps) to buy the free one (−20 waves).

### 3.3 The payoff, re-derived

**Cost of replication:** +35 node-steps × 4.143–4.843 µs
= **+145 to +170 µs/layer** ⇒ **+7.0 to +8.1 ms/step** at 48 layers.

**Handoff actually removed:** less than FR13 charged.

- *Export writes: 45.99 µs.* Clean A/B (`L0_deployed − L0_no_export`). Trustworthy.
- *Parent reads: NOT isolated.* There is no `L1_no_parent_read` probe. FR13
  charged the whole of `L1_len1` (82.27 µs) to "parent reads + 1 step". But
  `L1_len1` also contains the launch floor and 11 real node-steps: at fit **b**,
  `82.27 − 11×4.752 = 30.00 µs` is *everything that is not a node-step* — floor
  **and** reads together. Level 0's comparable floor is 24.9 µs, which would
  leave reads ≈ 5 µs; even charging the floor at zero caps reads at 30 µs.

⇒ handoff removed ∈ **[46, 76] µs/layer** = **22–33%** of `two_launch_total`,
not the 56.1% headline. ⇒ **−2.2 to −3.7 ms/step.**

**Net R-A: +69 to +124 µs/layer ⇒ +3.3 to +6.0 ms/step REGRESSION** versus the
deployed two-launch route, and **+134 to +170 µs/layer worse than
`single_launch`**, which is already built, already default-off-gated, and whose
K0 arming path this campaign just finished porting.

*(Bookkeeping note for anyone re-checking: `L0_deployed + L1_deployed` = 249.92
but `two_launch_total` = 228.74. The two launches overlap by 21.18 µs on the
same stream, so every per-part attribution above — including FR13's 56.1% — is
an upper bound.)*

---

## 4. The structural argument, independent of any number

- Minimum node-steps for the tree is **32** — each node computed once.
- Zero handoff requires every state edge to stay inside one program.
- A program is a serial chain, so zero handoff with **P > 1** programs forces
  replication of every shared prefix ⇒ strictly **more than 32** node-steps.
- Therefore **zero handoff at minimum work has exactly one solution: P = 1** —
  which is `single_launch`, already in the tree.

Replication is not an unexplored point in the design space; it is a strictly
worse point than the one already occupied, on a machine where waves are free.
The only escape would be a machine model where extra *width* is free — and §3.1
measures that it is not.

**Corollary on the DSMEM alternative** (FR13 §6.3's own footnote: hand the state
tile between the 12 path-programs over Blackwell distributed shared memory
instead of HBM, keeping 32 node-steps *and* 12 waves). It is a real mechanism,
but its entire benefit over `single_launch` is the 20 waves it saves — and §3.2
measures those at *less than zero* net value. **Not worth the cluster-launch
risk inside an existing CUDA graph.**

**At B4 the conclusion strengthens, not weakens.** Width-4 raises occupancy
pressure, so extra node-steps get *more* expensive relative to waves, not less.
(`single_launch` rejects B4 by contract, so B4 keeps the two-launch route
regardless; a batched replication variant would be the most expensive point of
all.)

---

## 5. Variants considered and rejected on the same arithmetic

| variant | node-steps | launches | handoff | verdict |
|---|---:|---:|---|---|
| deployed | 32 | 2 | 48.0 MiB | baseline 228.74 µs |
| `single_launch` (built) | 32 | 1 | **0** | **188.64 µs, byte-gated** |
| **R-A** full replication | 67 | 1 | 0 | +145..170 µs recompute ⇒ regression |
| **R-C** long branches only | 39 | **2** | 36.0 MiB | +33 µs cost vs ~11–17 µs saved; and still two launches, so it achieves *neither* goal |
| R-B leaf-folding into root | 13 in one program | 2 | partial | makespan 13 > 12; strictly worse than R-A on its own objective |
| DSMEM cluster handoff | 32 | 1 | 0 (DSMEM) | buys only the free resource (§4) |

---

## 6. What would change this verdict

State them now, so the probe can look for them rather than the conclusion:

1. **The saturation reading is wrong.** If a width-12 wave really costs ~one
   width-1 wave, R-A lands near FR13's −6.2..−7.6 ms. §3.1's three fits say
   otherwise across widths 1→11, but 12 is extrapolation by one step and
   `L1_len_full_padded` is the only wide-wave datapoint.
2. **The handoff is much larger in serving than in the probe.** FR13's own
   caveat: the probe's 15.7 MiB of exported tiles stay hot in GB10's 24 MiB L2,
   while a real step interleaves 48 layers of other traffic. If the true
   in-serving handoff is 2–3× the probe's, the sign could flip. **This is the
   most likely way the verdict is wrong, and §7's probe cannot test it** — it
   needs an in-serving measurement, which is a separate (and much more
   expensive) instrument.
3. **Replication reduces per-step cost.** A replayed prefix has no parent-state
   load, so its first step may be cheaper than a normal step. Bounded small: the
   parent load happens once per program, not once per step.

---

## 7. The measurement that settles it — and it needs no new kernel

**The R-A schedule's cost is measurable today with the deployed kernel.**
`_tree_gdn_path_kernel` already takes the path descriptors as *data*
(`path_nodes`, `path_parent_slots`, `path_lengths`) and already has the two
specializations the design needs:

- `STATE_SOURCE=1` → start from `h0`, **no parent read** (`:10799-10815`)
- `EXPORT_MODE=2` → **no state export** (`:10916`)

So a grid of `(48, 16, 12)` with the §1 chains as `path_nodes`, all
`path_parent_slots = −1`, `path_lengths = [5,12,6,8,3,3,4,4,5,5,6,6]`,
`STATE_SOURCE=1`, `EXPORT_MODE=2` executes **exactly the R-A work and exactly
the R-A memory traffic**. The `out` values are wrong (nodes get written more
than once, by identical-valued racing programs) — irrelevant, because this arm
answers the **cost** question only, and cost is what the verdict turns on.

**Stage 0 (cost, ~10 GPU-minutes, offline, no credential).** Extend
`scripts/fr13_gdn_scan_fusion_cost_probe.py` with:

| new arm | what it measures |
|---|---|
| `replicated_R_A` | the §1 schedule, 12 programs, zero handoff |
| `replicated_R_C` | the 39-node-step two-launch variant |
| `width_sweep` | one node-step at grid-z ∈ {1,2,3,6,11,12} — **the direct test of §6.1**, and the reason to trust or discard the whole model |

Decision rule, pre-registered before the data exists:

- `replicated_R_A ≥ single_launch` ⇒ **the design is refuted; do not build it.**
  Report the null, bank the ~7 ms back out of the under-210 plan, and close.
- `replicated_R_A < single_launch` by a margin exceeding the probe's run-to-run
  spread ⇒ §3 is wrong; **then** build the `REPLAY_STEPS` kernel and go to
  stage 1.

**Stage 1 (byte + cost), only if stage 0 says build.** The `REPLAY_STEPS` kernel
against the two-launch reference: byte-identical `out`, `ring_k/v/a/b`, `flags`
and counter across ≥4 seeds × 5 input regimes at b=1 **and** b=4, plus the
kernel-level ms/layer. Same gate shape the `single_launch` arm passed 20/20.

**Why this order.** The house rule is measured-never-derived, and the cheapest
honest way to obey it here is to measure the thing the verdict turns on *before*
building the kernel it would justify. FR13 closed the fused-scan rung by pricing
the mechanism instead of building it; this is the same move applied to FR13's
own successor recommendation. The probe is offline, one container, CPU-launched,
`analysis_only=true`, `acceptance_valid=false` — no serve, no credential, no
step-envelope claim.

**Blocked on:** GPU free (`docker ps` empty) and parent clearance. A serve is
live at time of writing.

---

## 8. Recommendation to Mark, stated before the data

The under-210 plan's kernel lane carries *"GDN replication ~−7 ms"* as a banked
size. On the FR13 probe's own numbers that size is **the wrong sign**, and the
design is dominated by `single_launch`, which is already built and gated. My
recommendation is to **re-price the line to ~0 pending §7 stage 0**, and to
spend the kernel-weeks it was holding on TreeAttn-v2 (~−13 ms), which the same
arithmetic does not touch.

I have not measured this. §7 stage 0 is ~10 GPU-minutes and either confirms the
null or refutes me with a number; either outcome is worth more than the note.

---

## 9. STAGE-0 RESULT — measured 2026-08-17, verdict REFUTED

`gdn_replication_stage0_b1.json`, GB10, B1, 150 reps, synthetic, eager, one
container, no credential. `analysis_only=true`, `acceptance_valid=false`.

### 9.1 The verdict

| arm | µs/layer (p50) | node-steps | handoff |
|---|---:|---:|---:|
| `two_launch_total` (deployed) | 228.74 | 32 | 48.0 MiB |
| `single_launch` (built, gated) | **188.19** | 32 | 0 |
| `replicated_R_A` | **353.76** | 67 | 0 |

- **vs `single_launch`: +165.57 µs/layer = +7.95 ms/step**
- vs the deployed two-launch route: +125.02 µs/layer = +6.00 ms/step
- run-to-run spread: **2.11 µs** — the gap is **78×** it.

**Pre-registered rule (§7): BUILD only if the replicated schedule beats
`single_launch` by more than the spread. It loses by 78 spreads. → REFUTED.**
Stage 1 (the `REPLAY_STEPS` store-suppression kernel) is **not built**, which
was the entire point of pricing before building.

### 9.2 The load-bearing assumption, tested directly

Marginal cost of one node-step, now at **four** wave widths:

| width | µs/node-step | source |
|---:|---:|---|
| 1 | 4.112 | `L0` length sweep |
| 2–3 | 4.798 | `L1` length sweep |
| 11 | 4.842 | `L1_len_full_padded` |
| **12** | **4.787** | **`replicated_R_A` − `replicated_R_A_len1`** |

Flat to within 17% across a 12× width change. **A node-step costs ~4.1–4.8 µs
wherever it sits**, so cost tracks node-steps and the design's 2.09× node-step
multiplier is paid in full.

The width control separates the two effects cleanly: 12 programs running **one**
step each cost 90.46 µs; the same 12 programs replaying their prefixes cost
353.76. **+55 node-steps ⇒ +263.30 µs.** The cost is the recompute, not the
program count.

*Nuance, stated because the raw sweep looks like the opposite at first glance:* a
single-node-step launch at grid-z 12 costs 85.54 µs vs 25.95 at z=1 — only 3.30×
for 12× the programs, which reads as "width is cheap". It is not: ~20.5 µs of
that 25.95 is a fixed launch/`h0` floor, and the marginal is
`(85.54 − 25.95)/11 = 5.42 µs per added program-step` — the same ~5 µs a node-step
costs anywhere. Sublinear *totals*, linear *marginals*. Pricing a design off the
totals is how the ladder's 3.45 ms was produced.

### 9.3 The two banked-number corrections, confirmed on this run

1. **The handoff is 33.4%, not 56.5%.** Export writes A/B cleanly at 46.94 µs.
   `L1_len1` = 82.27 µs contains the launch floor *and* 11 real node-steps; at
   the measured marginal, floor+reads together are **29.49 µs**. So handoff
   ≤ **76.44 µs = 33.4%** of `two_launch_total`. The 56.5% figure the probe still
   prints in its legacy `derived` block charges 11 node-steps and a launch floor
   to parent reads; it should be read as an upper bound only.
2. **Launch overlap is real.** `L0_deployed + L1_deployed` = 251.42 but
   `two_launch_total` = 228.74 — the two launches overlap by 22.68 µs, so every
   per-part attribution (FR13's, mine, and this note's) is an upper bound.

### 9.4 Probe integrity

- Reproduces the banked 20260811 run: `two_launch_total` **228.74** vs banked
  228.74; `single_launch` **188.19** vs banked 188.64.
- `single_launch` vs two-launch byte A/B: **20/20 pass**, `max_abs = 0.0`, on
  `out` and all four ring surfaces — the parked kernel is still byte-clean at
  this HEAD, which is worth knowing independently of this rung.
- Zero containers left after the run.

### 9.5 What this does NOT close

§6.2 stands unrefuted and is the only surviving way the verdict could be wrong:
the probe's exported state tiles stay hot in GB10's 24 MiB L2, while a real step
interleaves 48 layers of other traffic. If the in-serving handoff were ~3× the
probe's, the arithmetic would move — but it would have to move **165.57 µs/layer**
to reach parity with `single_launch`, i.e. the true handoff would need to be
~2.2× the *entire* deployed kernel cost. That is not a live hypothesis.

**Recommendation to Mark, now measured rather than modelled: close the GDN
replication rung as a verified null and keep the "−7 ms" line at ~0.** The lever
that actually banks the handoff is `single_launch`, which is already built,
already byte-clean at this HEAD (§9.4), and whose K0 arming path this campaign
just finished porting — its −2.2 ms is the real version of this rung's promise.
