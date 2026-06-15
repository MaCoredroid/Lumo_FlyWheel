# FR13 — LEAF CO-RESIDENCY PATH: the +16 width carrier is a TRAJECTORY-FORK at the LCP committer, not a per-forward co-residency seam

Date 2026-06-15. CPU-only, READ-ONLY (a GPU K1 boot `waao62oj0` runs concurrently — no code/boot touched).
vLLM source read DIRECTLY from the pinned image `vllm/vllm-openai@sha256:3dbe092e` (= 0.19.2rc1.dev134) via
`scripts/vllm_src.sh`, NEVER a /tmp cache. FRESH captures artifact-checked (timestamps below). Kernel lines
cited from `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` and `scripts/fr10_phase4_patch_vllm_tree_gdn.py`
(repo working tree, the served patch).

**Artifact freshness (checked):** `output/fr13_reshape_boot/rescore/{cat3w,chain3}_recur_flips.json` =
2026-06-15 02:27 / 02:36; `output/fr13_reshape_boot/{cat3w,chain3}_capture.json` = 02:09 / 02:17. NOT the
stale 06-14 `output/fr13_shape_sweep/`. Same recurrent-oracle frame, seed=1313, FLASH_ATTN, within-boot
det=[T,T,T,T] both arms (RECURRENT_PATH_ENGAGED=True, FR13_RESHAPE_AB_RECURRENT_BIND).

---

## 0. The clean A/B (recurrent-oracle frame), and the structural fact it turns on

| arm | shape (tree_choices, patch L10744-10749) | n_nodes | **N_PAD** | raw clear flips | de-cascaded (independent forks, gap>2) | accept/event |
|---|---|---|---|---|---|---|
| chain3 | `[(0,),(0,0),(0,0,0)]` — pure depth-3 spine, LEAF-FREE | 3 | **4** | **1** | **1** | 2.266 |
| cat3w  | `[(0,),(1,),(0,0),(0,1),(0,0,0)]` — same spine + root-sib `(1,)` + d1 `(0,1)` | 5 | **8** | **27** | **16** | 2.282 |

The ONLY difference is the 2 leaves at fixed depth-3, same spine, same frame. **+15 independent forks (16−1) =
the "+16 width carrier."** `N_PAD = 1<<(n-1).bit_length()` (kernel L159-163): adding the 2 leaves moves N_PAD
4→8 — load-bearing for path (c) below.

---

## 1. CARRIER-POSITION LOCALIZATION — LEAF-TOUCHING / FORK-BURST, **not** global (MEASURED from fresh JSON)

Per-prompt clear-flip positions and de-cascade (every number below is read from the two `_recur_flips.json`):

```
cat3w  p0 served=116  nonzero@{35,36,40 | 74 | 82,84 | 113,115}     5 independent forks, 8 clear
       p1 served=128  nonzero@{60 | 75 | 99,101}                    3 forks, 4 clear
       p2 served=72   nonzero@{21 | 61,62 | 71}                     3 forks, 4 clear
       p3 served=128  nonzero@{67,68,69 | 72,74,75,76 | 88 | 122,125,126}  5 forks, 11 clear
       => 16 INDEPENDENT forks total; deviations up to 22.7, 27.3 nat
chain3 p0 1 sub-margin (dev 0.5);  p1 2 sub-margin;  p2 1 clear (pos26 dev2.5, pos27→0.12 = re-converges);  p3 2 sub-margin
       => 1 fork total
```

**Three measured signatures that pin the carrier as a TRAJECTORY FORK, not a diffuse per-token co-residency seam:**

1. **SPARSE + CLUSTERED, not global.** cat3w p3 has 128 served positions; only **12 have ANY nonzero
   deviation**, in tight bursts (67-69, 72-76, 122-126) separated by long byte-identical (dev=0.0) stretches
   (e.g. pos 0-56 identical). A global batch/co-residency effect would smear small deviations across most
   tokens; instead we see re-convergence to dev=0 between bursts. The same shape holds in p0/p1/p2.

2. **HUGE deviations, not realization-ULP.** Fork-burst deviations are 8.6 / 9.2 / 12.4 / 22.7 / **27.3 nat**
   (served-token oracle-logprob ≈ e^-27). A per-forward realization seam (shared-h_cache bf16, FA2-MMA
   grouping) produces sub-1-nat near-tie flips. 27 nat is the served stream being teacher-forced through an
   off-distribution suffix — i.e. the committer already committed a *different token*, and the oracle scores
   the divergent continuation as near-impossible until the stream re-converges to a common phrase.

3. **chain3 (same spine, leaf-free) is essentially lossless** (1 clear flip, dev 2.5, instantly re-converges;
   all else sub-margin 0.25-0.62) **≤ native-E5's 3.** Remove the leaves → the forks vanish. The leaves CAUSE
   the forks. (FR13_BUG_CLASS_PLAYBOOK #12: this is the "non-like-for-like trajectories after a change" trap —
   the excess is a forked trajectory, not N un-aligned per-forward ops.)

**Localization verdict: LEAF-TOUCHING fork-bursts, not a global batch effect.** Each leaf introduces a point
where the served trajectory can diverge from the spine-only (leaf-free) trajectory; once it diverges, 1-3
downstream tokens are off-distribution (the burst), then re-converge.

---

## 2. OP-BY-OP MECHANISM ATTRIBUTION (a/b/c/d), with evidence

The committer serves `drafts[node]` along the **max-LCP root-to-leaf path** (patch L6818-6843, L6874-6875);
LCP = longest prefix where `drafts[node] == parent_targets[node]` (L6828-6831). The leaves add ALTERNATE paths
the LCP-max can pick AND co-resident verify rows that perturb the `parent_targets` the LCP comparison consumes.

### (d) LCP-COMMITTER TRAJECTORY FORK — **DOMINANT carrier** (MEASURED + mechanism-cited)
- Mechanism (cited): cat3w leaves `(1,)` (root sibling) and `(0,1)` (d1 runner-up sibling) are scored as
  full root-to-leaf paths (L6809-6843). When a leaf path's LCP ≥ the spine path's LCP, `best_path`/`best_leaf`
  becomes the LEAF (L6839-6843), and the committed token is `drafts[leaf]` — a token the spine-only arm never
  serves. Even when the spine wins LCP, a leaf co-resident in the verify forward shifts `parent_targets` at a
  near-tie node → the LCP boundary moves by one → a different bonus/correction token is served (L6877-6896).
  Either way the served trajectory FORKS.
- Evidence: §1 signatures 1-3. The 16-vs-1 fork count, the dev=0 re-convergence between bursts, and the
  >10-nat burst magnitudes are the trajectory-fork fingerprint, not a per-forward seam. **This is the same
  mechanism the recompute-rose result already exhibited** (FR13_SCAN_NOT_E2E_CARRIER_BIND: recompute changed
  near-tie verify logits → "different LCP-max path → ~369 token-diff stream," flips 23→32). MEASURED-dominant.

### (b) TREE-ATTENTION FA2-fork — **AMPLIFIER within each burst**, not the originator (INFERRED, bounded)
- The leaves enter the tree-attention bias mask, so the deep full-attn layers (16 of them) realize the
  spine's attention output slightly differently when leaves co-reside. FR13_REALIZATION_AGREEMENT (§ layer
  ratios) shows the largest per-layer divergence ratios at deep FULL-ATTN L35(1.61)/L47(1.32)/L51(1.34)/
  L62(1.29), where the FA2-fork 2-ULP floor (0.0039, project_fr13_fa2_fork_nocopy_floor) amplifies. But that
  floor is sub-1-nat and "no depth growth" — it cannot manufacture a 27-nat deviation on its own. It is the
  amplifier that converts a near-tie at a fork node into a clean argmax flip (turns a borderline LCP boundary
  into a committed fork), and it stretches the off-distribution burst. AMPLIFIER, not originator. INFERRED
  from the layer-ratio bind + the impossibility of a 2-ULP floor producing 27-nat.

### (a) SHARED GDN tree-scan h_cache/state co-residency — **REAL kernel gap, REFUTED as e2e carrier** (MEASURED)
- Mechanism (cited): one register tile `h_cache = tl.zeros((N_PAD,BLOCK_V,DIM_K))` (kernel L581) holds ALL
  tree-node states; the loop `for i in tl.static_range(0,N_PAD)` (L582) writes each node into the shared tile
  (`h_cache = tl.where(offs_n==i, state_i, h_cache)`, L651) and each node reconstructs its parent by reducing
  over the shared tile (`tl.where((offs_n==j), h_cache, 0)`, L586-590). So the spine's carried state IS
  realized in a tile whose size and reduction-tree depend on whether leaves are present. **This is exactly the
  path K1 touches** (the per-node `state_i.to(bf16).to(fp32)` store-boundary at L503-504).
- REFUTATION (MEASURED, FR13_SCAN_NOT_E2E_CARRIER_BIND): the RECOMPUTE route makes each node's STATE
  bit-exact to native packed-decode (int-view 0.0; recompute kernel L708-722 replays each node from the spine
  in an ISOLATED register tile, dropping the shared h_cache entirely = removes co-residency at the state
  level) — yet e2e clear flips **ROSE 23→32** and produced a DIFFERENT stream (~369 tok diffs). Removing the
  strongest per-forward state diff did NOT drop flips. So path (a)'s shared-h_cache state realization is a
  REAL kernel gap (OFF max_abs 0.0289 vs native packed-decode) but **NON-CAUSAL for the flips.** Its only
  effect is to nudge near-tie verify logits, which feeds back into (d).

### (c) Residual M-dim / grid / N_PAD codegen geometry — **secondary; codegen change real, but routes back into (d)** (MEASURED-bounded)
- Mechanism (cited): the launch grid is `(num_vh, cdiv(dim_v, BV))` (L1881) — NOT keyed on N_PAD; BLOCK_V=16
  (L18) and num_warps=8 (`_DEPLOYED_NUM_WARPS`, L1817) are constant across chain3/cat3w. So the launch
  geometry is identical. What changes is the **N_PAD constexpr (4→8)** → a DIFFERENT compiled kernel: the
  `tl.static_range(0,N_PAD)` unroll depth doubles, the `h_cache` register tile doubles, and the `tl.where`
  reduction over `offs_n` (L586-589) ranges over twice as many lanes. Bug-class #10 (shared source ≠ shared
  SASS): even the SPINE nodes (i=0,1,2) get a different FMA schedule / register pressure under the larger
  N_PAD. The in_proj_ba M-keyed GEMM co-residency was already FIXED (LUMO_FB pad, FR13_RESIDUAL13_RESOLVED;
  fp8 in_proj/o_proj M-invariant BLOCK_SIZE_M=64), so (c) is NOT a remaining M-keyed batch-variance — it is
  the N_PAD-keyed scan codegen. But this is a sub-ULP realization nudge of the same flavor as (a)/(b): it
  perturbs near-tie verify logits and routes back into the (d) fork; recompute pins geometry to native
  BV32/w1/s3 and STILL rose, so (c)'s codegen delta is also non-dominant on its own.

### Attribution summary
| path | what it is | role in the +16 | evidence | class |
|---|---|---|---|---|
| **(d) LCP committer fork** | leaf path wins/shifts LCP-max → committed token forks the stream | **ORIGINATOR (dominant)** | 16-vs-1 forks, dev=0 re-conv, 27-nat bursts (MEASURED); recompute-rose = same mechanism | #12 |
| (b) tree-attn FA2-fork | leaves in deep full-attn bias mask | AMPLIFIER of each fork node | layer-ratio bind L35/47/51/62; 2-ULP floor can't make 27 nat (INFERRED) | FA2-floor |
| (a) shared h_cache state | spine state realized in leaf-shared tile (K1 path) | REAL gap, NON-CAUSAL | recompute bit-exact state yet flips ROSE 23→32 (MEASURED) | #12/#10 |
| (c) N_PAD codegen | N_PAD 4→8 changes scan compile (NOT grid, NOT M) | secondary nudge → routes into (d) | grid not N_PAD-keyed (L1881); recompute pins geom + rose (MEASURED) | #10 |

---

## 3. K1 ON-PATH VERDICT — K1 is ON path (a) (the per-forward state realization), which is OFF the dominant (d) carrier ⇒ predicted ~0 / non-collapsing

**K1 = the per-node `state_i.to(bf16).to(fp32)` store-boundary round-trip (kernel L503-504), applied to every
GDN layer keeping cat9 geometry** (the SCAN_ALIGN MODE=body seam; the only seam with depth growth,
FR13_REALIZATION_AGREEMENT §3).

- **Is K1 on the dominant co-residency path?** K1 sits squarely on path (a) — it changes how the spine's
  carried scan STATE is realized (per-node bf16 round-trip on `state_i` after `out_i` is taken). That IS the
  shared-h_cache state-realization path the leaves' co-residency flows through.
- **But path (a) is REFUTED as the e2e carrier** (§2a, MEASURED): recompute already made the carried state
  BIT-EXACT to native (a STRICTLY STRONGER condition than K1's per-token bf16-store — if the final state is
  bit-exact the store boundary is necessarily matched) and flips ROSE 23→32. K1 is a WEAKER alignment than
  recompute on the SAME state-realization axis.
- **Therefore the dominant carrier (d, the LCP-committer trajectory fork) is OFF K1's path.** K1 perturbs
  near-tie verify logits (it must change SOME rounding, hence SOME token) but does not change the LCP-fork
  STRUCTURE — it just re-rolls which near-ties cross. The powered prior (recompute moved the same axis the
  WRONG way, ±9 flips of resolution on this exact harness) predicts:

  **PREDICTED K1 RESULT: flips stay ~23 or rise (re-rolled trajectory), NOT a collapse toward native-3.**
  K1 is ON the per-forward state path but that path is OFF the dominant trajectory-fork carrier.

- **Interpretation rule (settles relax-vs-fix):**
  - K1 fails (stays ~23 / rises) AND it is off-the-dominant-path (this prediction) ⇒ the carrier is the
    leaf-induced LCP-committer trajectory fork (d), and the next lever is the on-(d)-path topology lever, not
    a scan-state kernel seam. **The kernel-align lever is exhausted** (K1 was the only depth-growth candidate;
    K2-K5 are provably ~0, FR13_REALIZATION_AGREEMENT §4). RELAX to the accept/event arbiter or reshape.
  - K1 unexpectedly DROPS toward native-3 ⇒ a SYSTEMATIC 48×-applied alignment of a correlated diffuse floor
    DID cancel it (the only theoretical way one per-forward seam could; native-E5=3 proves a clean realization
    exists) ⇒ the floor was a correlated per-layer realization sum after all, and we KEEP the leaves + bake K1.
    This is the residual-doubt outcome; the §4 cheap test below is the cheaper pre-check.

---

## 4. NEXT-LEVER RANKING (for the on-(d) carrier that KEEPS the leaves = the 3.198 accept edge)

The accept edge IS the leaves (cat9 3.198 > native 3.08); chain3 (leaf-free) is lossless but slow (2.27). So
the lever must keep the leaves and cut their *fork* divergence, not remove them.

| rank | lever | what it does to the (d) fork | tag | cheap non-vacuous test |
|---|---|---|---|---|
| **1** | **TOPOLOGY reshape (shallower/root-sibling tree, FR13_RESHAPE)** | fewer/structurally-closer leaves ⇒ fewer LCP-fork points; the live front already localized this | **OUR-kernel-authorized (drafter packing only; no copy/dense)** | already running A/B; the reshape sweep IS the test — read its `_recur_flips.json` for fork count vs accept/event |
| 2 | **LCP-committer near-tie damping at the FORK node** (e.g. require a leaf to beat the spine LCP by a real margin before forking, not a ULP) | directly cuts (d): borderline leaf-wins that only exist because of a sub-ULP verify nudge stop forking | OUR-kernel-authorized IFF it is a deterministic committer rule that does NOT force-pick the spine (force-spine-commit = BANNED reward-hack; this must be a margin gate that still serves genuine leaf wins) — **needs user sign-off (changes accept logic, class #12 "non-like-for-like")** | CPU replay: re-run the committer (`_lumo_tree_path_lcp_max_greedy_sample`) on the BANKED cat3w verify logits with a +ε LCP-margin and count how many of the 16 forks were sub-ε near-ties vs genuine leaf wins — **zero GPU, uses existing capture** |
| 3 | per-leaf isolated h_cache bank rows / recompute-from-spine (path a alignment) | aligns state realization | **REFUTED** — recompute already did the strongest form (state bit-exact) and flips ROSE; do NOT re-pursue | n/a (done) |
| 4 | tree-attn realization alignment (FA2-fork → 0) | cuts the (b) amplifier | OUR-kernel-authorized (numerics-align) but bounded: 2-ULP MMA-grouping floor, no theorem (project_fr13_fa2_fork_nocopy_floor); amplifier-only, won't remove the originating fork | A/B the FA2-fork floor (already banked 0.0039); won't move the 16 forks |
| 5 | per-leaf isolated FORWARD (true no-co-residency) | removes a/b/c at the cost of separate forwards | **block-pool surgery (FR9), expensive, no cheap vLLM-0.19 path** | n/a |
| — | the residual after 1-2 | if forks are genuine leaf-LCP wins (not ULP near-ties) | **FUNDAMENTAL → relax** to accept/event-parity (cat9 fast-but-lossy) or ship chain3 (lossless-slow) | the rank-2 CPU replay settles fundamental-vs-fixable |

**Top next-lever = rank-1 (topology reshape, already running) with rank-2 (committer near-tie margin) as the
CPU-cheap discriminator that decides fixable-vs-fundamental WITHOUT a GPU boot.**

---

## 5. CHEAP NON-VACUOUS TEST (single, decisive, ZERO GPU)

**CPU committer-replay margin probe.** Re-run the LCP committer
(`_lumo_tree_path_lcp_max_greedy_sample`, patch L6818-6896) on the BANKED cat3w verify logits/drafts already
in `output/fr13_reshape_boot/cat3w_capture.json` (+ the per-node parent_targets if banked; else re-derive from
the recurrent-oracle sinks `output/fr13_reshape_boot/rescore/cat3w_sinks/`). For each of the 16 independent
fork positions, classify the fork as:
- **(A) genuine leaf-LCP win**: the leaf path's LCP beats the spine by a real (≥1 token, clear-margin)
  amount ⇒ FUNDAMENTAL (the leaf is genuinely the better continuation; removing the fork = removing the
  accept edge = relax).
- **(B) sub-ULP near-tie win**: the leaf wins only because a co-residency/FA2 nudge moved a near-tie
  `parent_targets` value across the accept boundary ⇒ FIXABLE by the rank-2 LCP-margin damp (keeps the leaf,
  cuts the spurious fork).

Non-vacuity (playbook #9): assert the replay reproduces the SAME served tokens as the capture for the dev=0
positions (committer is byte-faithful) BEFORE trusting any fork classification; if it does not reproduce the
served stream it is measuring nothing. Powered: it resolves a per-fork A/B label on all 16 forks; the band
(0 fixable ⇒ fundamental/relax … 16 fixable ⇒ rank-2 is the lever) is wide and discrete.

This is strictly cheaper than the running K1 GPU boot and answers the actual relax-vs-fix question (K1 only
closes the last kernel-state-realization doubt, which §2a already brackets as refuted).

---

## VERDICT (skeptic, for the user's relax-vs-fix decision — NOT a close/pass-fail)

- **Carrier position = leaf-touching trajectory FORK bursts, not a global per-token co-residency seam**
  (MEASURED: 16-vs-1 forks, dev=0 re-convergence between bursts, 8-27 nat burst deviations).
- **Mechanism = (d) LCP-committer fork (dominant) + (b) deep-full-attn FA2 amplifier; (a) shared-h_cache
  state and (c) N_PAD codegen are REAL but REFUTED-non-causal** (recompute made state bit-exact and flips
  ROSE 23→32).
- **K1 is ON path (a) — the per-forward state realization — which is OFF the dominant (d) carrier ⇒ predicted
  to stay ~23 / rise, not collapse to native-3** (it is a weaker form of the already-refuted recompute
  state-alignment).
- **Next lever = topology reshape (rank-1, running) + a CPU committer-margin replay (rank-2) that decides
  fixable-vs-fundamental with zero GPU.** The kernel-state-align lever (K1/recompute) is the weak/exhausted
  one. If the rank-2 replay finds the 16 forks are genuine leaf-LCP wins, the lossless+fast tension is
  FUNDAMENTAL ⇒ relax to accept/event-parity or ship chain3.

## Playbook rows quoted (FR13_BUG_CLASS_PLAYBOOK)
- **#12 Measurement traps / co-residency-trajectory** — "non-like-for-like trajectories after a change": the
  +16 is a forked trajectory (16 independent forks), NOT N un-aligned per-forward ops; raw counters only (read
  the `_recur_flips.json` directly, no hand-rolled TPS÷accept).
- **#10 Shared-source ≠ shared-SASS (codegen identity)** — path (c): N_PAD 4→8 recompiles the scan with a
  different unroll/register tile even though grid + num_warps + BLOCK_V are constant; int-view only, never atol.
- **#9 Silent fallback / vacuous instrument** — the §5 test's non-vacuity is the byte-faithful committer-replay
  assert on dev=0 positions BEFORE any fork classification; guards "a run passes while measuring nothing."

Links: [[project_fr13_22flip_carrier_l0gdn]], [[reference_diffuse_gdn_accumulation_explained]],
[[reference_multispine_not_lossless_closed_nonship]], [[reference_scalar_metric_per_token_blindspot]],
[[project_fr13_fa2_fork_nocopy_floor]], [[reference_gdn_tree_branch_oracle_losslessness]].
Sources: FR13_RESHAPE_AB_RECURRENT_BIND.md, FR13_SCAN_NOT_E2E_CARRIER_BIND.md, FR13_REALIZATION_AGREEMENT.md;
fresh `output/fr13_reshape_boot/rescore/{cat3w,chain3}_recur_flips.json` (2026-06-15).
