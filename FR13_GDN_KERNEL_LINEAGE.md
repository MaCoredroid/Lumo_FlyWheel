# GDN kernel lineage — design map (FR-7 → FR-13 replay route)

**Change policy (user 2026-06-10): any FACTUAL change to this table must be reported to the user immediately** (e.g., a status flips, a "why it died" is overturned, the byte A/B verdict lands, WY is revived). Routine progress does not need to touch this doc.

## The Half-2 kernel in one sentence
The verify scan stays exactly as it is; instead of materializing all 9 nodes' recurrent states to HBM and letting the next step pick one, we store each node's tiny *inputs* (~16 KiB bf16: k pre-l2norm, v, raw_a, raw_b) and — once the committer knows the accepted path — **re-execute just that chain from h0 with the literally identical instruction sequence** (one shared `@triton.jit` `_gdn_node_step` body, same constexprs, num_warps=8, ±0.0 handoff emulation, root→column 0 on zero-accept), writing results straight into the bank's linear columns.

```
scan (unchanged):  computes all 9 node states in registers → verify logits
                   ✂ deleted: export 9×3.1MB states + publish + remap   (the 5.14× tax)
new:               store k,v,raw_a,raw_b per node (~190× smaller)
committer picks path → replay kernel: h0 ──step──step──step──▶ bank columns 0..L-1
```
Traffic: 36 → 6 row-touches/layer/req = 21.74 → 3.62 GB/forward = **0.86× actual native**. Replay compute is free under the 99 ms weight-bandwidth floor. No `h_cache` in the replay kernel (one `[BV,DIM_K]` register tile walks the chain) → **spill-free at any tree size** — removes the N_PAD≈16 register wall, the scaling unlock for future suffix-decoding trees.

## The lineage table

| kernel (era) | approach | why it died / status |
|---|---|---|
| **native `fused_sigmoid_gating`** (vLLM decode/verify) | sequential rank-1, one token at a time, single carried state | the gold reference — everything is judged against its op order |
| **native `fla_chunk_gated_delta_rule`** (vLLM prefill; GB10 FLA route) | chunked/WY block math | prefill-only; its chunk-vs-recurrent ~6e-5 gap is why "byte-exact-vs-MTP-baseline" was the wrong losslessness bar |
| **FR7 packed tree** | one tree, ONE shared recurrent state across paths | architectural contamination — sibling state bled into path0 (STree-class problem); no-ship |
| **FR9 multi-spine** | no tree kernel — real co-scheduled sequences + full state *copies* per spine | copies don't scale bandwidth-wise; CLOSED_NON_SHIP; deep-dive (wm7zqhnu5) re-confirmed: not our later bugs, no speed crossover at any S, subsumed as a degenerate tree topology |
| **FR13 WY kernel** (archived, `fr13-wy-archive`) | whole-tree Gram + UT-solve (Householder) | ℝ-equal but a *different summation tree* — provably never byte-exact to native; chunk batching risks batch-variance; **parked as LAST-RESORT fallback** (triggers only on hard replay-route failure) |
| **current live `_tree_gdn_kernel`** (main) | sequential rank-1 per ancestor path, h_cache registers, strict-mask isolation; batch-invariant by construction | **correct + byte-exact-to-native-on-spine + batch-invariant — but exports/publishes all N states = 5.14× native state traffic, and hits the register wall at N_PAD≈16**; 2026-06-10 update: the p0-pos-35 "spine-commit flip" once indicting its legacy ssm publish/remap/h0 side is a cross-boot NEAR-TIE coin flip (flag-OFF boot b1 emitted the NATIVE token), NOT a stable defect — though its live handoff log still shows 2/113 nonzero ssm next-read deltas (branch-commit class) |
| **`_linear_remap_rows_kernel`** (main) | post-commit state shuffle to linear columns | had an in-place overlapping-permutation RACE (fixed: gather-then-scatter, cc008587); ssm half is **deleted** by the replay route, conv half **kept** |
| **NEW `_tree_gdn_replay_kernel`** (`fr13-replay-route@9d4d22e3`, GPU-gated 2026-06-10) | companion to the scan: re-executes only the accepted chain for the durable handoff | the speed route — same math, different *logistics*; **byte A/B PASSED** (codegen identity PROVEN: durable-bank int-view byte-equal 126/126 incl. −0.0/dst==h0/zero-accept/two-event + 48-layer sweep + old-vs-new scan binary byte-equal; 0 spill; cubin hashes pinned — risk R4 retired) — **but LIVE FAIL, gate-4 class** (accept 2.02→1.58, within-boot same-seed non-determinism, native forks at pos 11–17 vs 21–35+, in BOTH captured and eager regimes) ⇒ kernel exonerated, live wiring seam unlocalized (R1 prev-lens/commit indices, R6g native `get_temporal_copy_spec` neutrality, R8 ring/REQKEY churn); see `FR13_REPLAY_GPU_GATES_BIND.md` |

## The design lesson the lineage encodes
Every attempt that **changed the math** (FR7's shared scan, WY's reassociation) lost losslessness; every attempt that **copied state** (multi-spine) lost bandwidth. The replay kernel is the first that changes *neither* — pure state-logistics: the verify math is untouched (all byte-exact evidence on the scan and the FA2 fork carries per `FR13_REPLAY_GATE_TRANSFER_MATRIX.md`), and the only new claim — "the replayed chain equals the scan's chain bit-for-bit" — holds by IEEE determinism *iff* the compiler emits identical code for the shared step body in both kernels. That conditional was **DISCHARGED on GPU 2026-06-10** (byte A/B all-pass, `FR13_REPLAY_GPU_GATES_BIND.md`) — and the route still **fails LIVE** (gate-4 class): the open problem moved from the kernel to the live state-logistics wiring (publish ordering vs next-event h0 read, native copy-path neutrality, ring keying at request churn). Offline-bit-identical ≠ live multi-step, now proven twice.

Subtlety: the replay route deliberately **keeps the conv-bank machinery** (deleting it re-creates the old conv prior-window bug) — which is why the conv branch-commit fix transfers to, and is required by, both routes.
