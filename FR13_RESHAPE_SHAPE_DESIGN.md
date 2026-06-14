# FR13 — Tree-Reshape Shape Design (ranked candidates, depth model, floor, feasibility, GPU test)

Date 2026-06-14. **READ-ONLY design workflow** (no kernel/patcher touched; pathspec commit of this doc
only). A concurrent GPU workflow runs the L0-GDN sub-op A/B — this doc does NOT modify any shared code.

Settled context (3 binds agree — `FR13_CONV_NOT_CARRIER_SCAN_STATEFEED_BIND.md`,
`FR13_NODE5_LADDER_DIFFUSE_BIND.md`, `FR13_FA2_CARRIER_OVERTURNED_BIND.md`): the cat9 22-flip carrier is the
**GDN recurrent SCAN STATE-FEED** — the live rank-1 tree-scan over the accepted spine chain vs the clean
chunked-prefill scan = the chunk-vs-recurrent ~1-ULP/node gap, born L0, amplified ~32× by gate 1/rms + deep
full-attn, crystallizing L60/L61. It **SCALES WITH ACCEPT DEPTH**. conv front DONE (sub-ULP); FA2-tile
EXCLUDED (downstream of L0); scan KERNEL M-invariant (BV-16). EXCLUDED: copy-recurrent multi-spine
(NOT-lossless-CLOSED), dense, splice (oracle-only). WY (bit-exact scan kernel) PARKED (no revival w/o user).

Frontier (each arm vs its OWN no-spec decode oracle, thr 1.0 nat — COMPARABLE across shapes):
| arm | topology | committed spine depth | flips | accept/event (cross-boot, class-12 confounded — directional) |
|---|---|---|---|---|
| native E5 (FLASH, MTP-5) | 5-spine linear | 5 | **3** | 3.076 |
| chain5 (our kernel) | 5-spine, no branches | 5 | **5** | 2.664 |
| cat9 (LOCKED build) | 5-spine + 4 leaves (d2-5) | 5 | **22** | 3.198 |
| cat10 (archive) | cat9 + root sibling `(1,)` | 5 | **22** | 2.932 |

---

## Playbook rows in force (quote — `FR13_BUG_CLASS_PLAYBOOK.md`)
- **Class 9 (silent fallback / vacuous instrument):** any NEW tree shape must pass the engagement gate
  (tok/draft == len(TREE)) BEFORE any flip/accept number is trusted. The patcher RAISES
  "FR10 caterpillar drafter disengaged" (`fr10_phase4_patch_vllm_tree_gdn.py:10834`) for any tree_mtp shape
  that is not exactly the hardcoded cat9 / chain5 (or cat10 on archive). Do NOT enable the BANNED
  `FR10_ALLOW_LINEAR_FALLBACK=1` (vacuous linear path). An unrealizable TREE FAILS LOUD — bring it to the user.
- **Class 8 (offline ≠ live multi-step):** the carrier is a LIVE multi-step state-feed; first gate of every
  boot = same-boot within-boot determinism (rep1≡rep2 byte-identical) — already step 4 of `fr13_shape_gate.sh`.
- **Class 11 (BI-flag / batch-composition sensitivity) + Class 12 (measurement traps):** accept/event is
  cross-boot, class-12 trajectory-confounded — DIRECTIONAL ONLY. The COMPARABLE metric across shapes is the
  flip COUNT each-vs-own-oracle. cat9+BI=34 (WORSE) → do NOT use BI to "fix" reshape flips. Per-depth accept
  RATE and within-arm d0 delta are the fair accept reads (not whole-window accept/event).

---

## 1. DEPTH-ACCUMULATION MODEL (quantitative)

### Mechanism (from the 3 binds + `FR13_DIFFUSE_GDN_EXPLAINED.md`)
A committed (accepted) spine of depth D is realized live by a **rank-1 tree-scan over the accepted chain**
(e.g. at num_accepted=4 the chain is `[0,1,3,5]` seeded from b_h0). The clean no-spec oracle realizes the
SAME logical recurrent state via a **chunked-prefill scan**. The two finite-precision realizations differ by
~**1 bf16 ULP per recurrent step (per accepted node on the chain)**. These per-node diffs are **correlated**
(lean the same way), accumulate across the ~48 GDN layers, are amplified ~32× by the gate `1/rms` and the
deep full-attn block, and crystallize the final-token argmax at L60/L61. A "flip" happens when the
accumulated `\`\`\``-vs-`Let`-class margin (cat9 carrier event: live 15.94 vs clean 26.60 on the loser logit)
crosses zero.

Two distinct accumulation axes — both observed:
- **DEPTH axis (intrinsic, per spine node):** longer accepted rank-1 chain ⇒ more chunk-vs-recurrent
  accumulation. This is the chain5=5 signal (a 5-deep scan's gap) and it is the floor (see §3).
- **WIDTH / co-residency axis (cat9 excess):** branches add ~17 flips over chain5 (5 → 22). Commit 2fe2c567:
  11/11 ch2 flips are ON the spine, 0 on leaves = SPINE_PERTURBATION — the co-resident branch nodes perturb
  the deep-spine row's GDN compute within one batched tree forward. (Whether this excess is pure deeper-accept
  [cat9 accepts to 3.198 vs chain5 2.664 ⇒ longer chains] or a genuine M-dependent co-residency op is exactly
  what the concurrent GPU L0-GDN sub-op A/B settles. If scan_out M10≈M5 deep-row ⇒ depth-intrinsic, reshape is
  the only lever; if nonzero ⇒ an alignable co-residency op exists.)

### Flips-vs-spine-depth fit (from the 2 measured points + node5/recurrent structure)
Anchor points (each vs own oracle, thr 1.0): **chain5 (depth 5, no width) = 5 flips**;
**native (depth 5) = 3 flips**. Native is the existence proof that a depth-5 chain *can* sit at ~3; our
chain5's 5 is the our-kernel-vs-native per-node realization excess (~+0.4 flip/node above native at depth 5).

A simple, honest model consistent with both axes (NOT presented as measured — it is a design predictor to be
checked by the GPU sweep):

  flips(D, W) ≈ flips_native(D)  +  c_kernel · D   +  c_cores · (branch-co-residency term)

- `flips_native(D)`: native's own depth-accumulation, ~0.6·D (3 flips at D=5). Reshape cannot go below this
  for a given committed depth — it is the chunk-vs-recurrent floor REALIZED AT NATIVE PRECISION (§3).
- `c_kernel · D`: our-kernel per-node excess over native. chain5: (5 − 3)/5 ≈ **0.4 flip per committed spine
  node**. LINEAR in committed depth (each accepted node adds one more rank-1 scan step that diverges ~1 ULP).
- `c_cores`: the branch term, ~+17 at cat9 (D=5, 4 leaves). If the A/B shows this is **deeper-accept-driven**
  (cat9 chains run longer because branches rescue more accepts), it folds into a higher effective D, NOT a
  separable width cost — meaning reducing committed depth attacks BOTH terms.

**Predictions (committed-spine depth D, our kernel, no width — i.e. chainD shapes):**
| committed spine depth D | predicted flips (our-kernel chainD) | basis |
|---|---|---|
| **3** | **~3** (native-ish; 0.6·3 + 0.4·3 ≈ 1.8+1.2 = 3.0) | shallow chain, ~3 rank-1 scan steps |
| **4** | **~4** (0.6·4 + 0.4·4 = 2.4+1.6 = 4.0) | interpolant |
| **5** | **5** (MEASURED, chain5) | anchor |

So **a committed depth-3 chain is predicted to land at ~native-3 flips** — the directive's "shallower spine =
fewer depth-flips." The width (leaf) nodes do NOT add committed depth (leaves are never fed back into any
forward/recurrent state — patcher comment `:10955`), so **leaf width does not move the DEPTH term**; it only
risks the co-residency term (`c_cores`), which is the open A/B question.

---

## 2. RANKED SHAPE CANDIDATES (TREE = list-of-tuples)

Design principle (corroborated by online research — Sequoia, EAGLE-2, TALON/OPT-Tree/C2T below): **cut
committed depth to cut depth-accumulation flips; recover accept with SHALLOW WIDTH (root/near-root siblings)
which add NO committed depth; gate the width on confidence to avoid the cat10 dilution.** Sequoia explicitly
finds that *limiting tree depth* yields larger e2e speedup than max-token trees, and that there are
*diminishing returns of width at deep layers* — i.e. width belongs at shallow depths, exactly the root-sibling
move.

Ranked by expected (flips → native 3-5 AND accept/event ≥ native ~3.16) AND drafter-buildability:

### Rank 1 — `cat3w` : depth-3 spine + 2 shallow siblings (root + d1)  [NEW PACKING — small code]
- TREE: `[(0,), (1,), (0,0), (0,1), (0,0,0)]`  (sorted (len,path) order)
- num_nodes 5, committed-spine depth **3**, root-sibling node `(1,)` + d1-sibling `(0,1)` (both shallow width).
- Predicted flips: **~3** (depth-3 chain term) — the depth model's native-floor prediction; width adds no depth.
- Predicted accept: the d0/d1 rescue (root sibling rescues the 62%-of-rejects-at-step-0; cat10 measured d0
  rate 0.871→0.906) recovers the shallow accept that the shorter spine gives up, while the depth-3 spine keeps
  d0-d2 acceptance high (cat9 rates 0.871/0.828/0.638 at d0/d1/d2). Net accept ~ native band is plausible but
  UNPROVEN (cat10 showed adding-width-unconditionally diluted deeper accept −0.27; this shape removes the deep
  spine that was being diluted, so the dilution target is smaller). Honest: accept ≥ 3.16 is the RISK axis.
- **Feasibility: NOT on HEAD** — needs a new hand-rolled packing branch (the disengagement raise fires).
  Width nodes are realizable (rank-2 only, see §4). RANK 1 because it is the cleanest test of "shallow spine
  hits native flips AND shallow width recovers accept."

### Rank 2 — `chain3` : pure depth-3 spine, NO width  [NEW PACKING — trivial code]
- TREE: `[(0,), (0,0), (0,0,0)]`  · num_nodes 3 · committed depth **3** · no siblings.
- Predicted flips: **~3** (the cleanest depth-floor probe; isolates the DEPTH term with zero co-residency).
- Predicted accept: LOW (chain5 was 2.664 at depth 5; depth-3 is lower still — no width rescue). This is the
  **floor-probe control**, not a deploy candidate: it answers "does cutting committed depth to 3 hit native-3
  flips?" If chain3 ≈ 3 flips, the depth model is confirmed and the DEPTH lever works; then cat3w/cat4w supply
  accept. Cheap, decisive, no co-residency confound.
- Feasibility: NOT on HEAD (new branch) but trivial (`_fr10_spine_only` generalized to depth 3).

### Rank 3 — `cat4w` : depth-4 spine + root sibling + 1 deep-trimmed leaf  [NEW PACKING — small code]
- TREE: `[(0,), (1,), (0,0), (0,0,0), (0,0,1), (0,0,0,0)]`
- num_nodes 6, committed depth **4**, root sibling `(1,)` + one d2 leaf `(0,0,1)`.
- Predicted flips: **~4-5** (depth-4 term + minimal width co-residency). Tests whether keeping ONE more spine
  node (better accept) costs only ~1 extra flip — the depth/accept knee.
- Predicted accept: HIGHER than cat3w (depth-4 spine keeps d3 accepts, cat9 d3 rate 0.483) + root rescue.
  Best accept-per-flip if the knee is at depth 4.
- Feasibility: NOT on HEAD (new branch).

### Rank 4 — `cat10` : cat9 + root sibling `(1,)`, CONFIDENCE-GATED  [ARCHIVE — buildable, no retrain]
- TREE: `[(0,), (1,), (0,0), (0,1), (0,0,0), (0,0,1), (0,0,0,0), (0,0,0,1), (0,0,0,0,0), (0,0,0,0,1)]`
- num_nodes 10, committed depth **5**, root sibling `(1,)`. **FREE confidence-gated variant:** emit `(1,)`
  only when the root top-2 margin is a near-tie (the runner-up is ALREADY computed — `_fr10_root_leaf_token`
  / `_fr10_top2[:,1]` — so the gate is FREE; see §5). This keeps cat10's +0.035 d0 rescue WITHOUT the
  unconditional-width dilution that cost −0.27 (commit cd30f5ad).
- Predicted flips: **22** (UNCHANGED — cat10 measured 22, FLAT vs cat9; depth-5 spine unchanged ⇒ depth term
  unchanged; root sibling adds no committed depth). This is the LOSSLESS-NEGATIVE candidate: it does NOT cut
  flips. Ranked low for the FLIP goal; kept because the confidence-gated root branch is the directive's named
  d0-accept recovery and is FREE — it belongs LAYERED ON a depth-cut shape (cat3w/cat4w), not on the depth-5
  spine. Use as the d0-rescue building block, not a standalone flip fix.
- Feasibility: on `origin/fr13-cat10-archive` (`FR13_CAT10_ROOT_SIBLING=1`, num_spec 10). Root sibling
  CONFIRMED buildable against code (§4). The confidence gate needs ~5 lines (a top2-margin threshold around
  the existing `_fr10_root_leaf_token` capture).

### Control — `chain5` / `cat9` (already banked: 5 / 22 flips) — re-boot for same-boot oracle re-baseline only.

**Sweep order (GPU serialized):** chain3 (floor probe) → cat3w (rank 1) → cat4w (rank 3) → [cat10-gated if
accept still short]. chain5/cat9 only if a fresh same-boot oracle re-baseline is needed.

---

## 3. CHUNK-vs-RECURRENT FLOOR ASSESSMENT — can reshape reach native ~3?

**There IS a floor, and reshape's job is to reach it, not beat it.** chain5's 5 ≠ native's 3 because chain5 is
OUR-kernel realization (rank-1 tree-scan) vs native's MTP realization at depth 5 — both have a chunk-vs-
recurrent per-node gap, but ours is ~0.4 flip/node LARGER (un-aligned seams, per `FR13_DIFFUSE_GDN_EXPLAINED`).

- **Is reshape floored at the chunk-vs-recurrent intrinsic?** PARTLY. Every committed spine node carries SOME
  per-node gap (chain5=5 IS "a 5-deep rank-1 scan's gap"). The shallowest MEANINGFUL spine (depth ~3, the
  minimum that keeps useful accept) still carries ~3 nodes of gap. So the **minimum reshape-achievable flip
  count is ~3** (depth-3 chain), which COINCIDES with native's 3.
- **Does that count as native-level within the floor?** YES, if it lands at ~3 — that is native's measured
  flip count and within the within-floor bar (`project_fr13_active_worker_codex_fr15`: per-depth-argmax +
  within-floor, NOT abs-0.0; native same-model fp8 drifts ~7× less = the existence proof a depth-5 chain *can*
  sit at 3). Reshape to depth ~3 is predicted to hit native's flip level **because it cuts the number of
  accumulating rank-1 scan steps to native's regime**, NOT because it aligns the kernel.
- **The honest caveat:** if the GPU A/B shows the cat9 excess is a real M-dependent CO-RESIDENCY op (scan_out
  M10 ≠ M5 on the deep row), then even a shallow spine WITH width re-introduces some co-residency flips — and
  the floor for a WIDE-shallow shape is above a NO-width shape. In that case the clean route is chain3/cat3w
  with confidence-gated (sparse) width, minimizing co-resident nodes. The truly irreducible MMA floor (~1e-12,
  `FR13_DIFFUSE_GDN_EXPLAINED:37`) is far below an argmax flip — NOT the binding floor here.

**Verdict:** reshape can plausibly reach native-~3 flips at committed depth ~3 (the depth term collapses to
native's regime); it canNOT go below ~3 (the chunk-vs-recurrent floor at native precision). Reaching ~3 AT
ACCEPT ≥ native is the open question — that is the accept/flip trade the sweep resolves.

---

## 4. DRAFTER FEASIBILITY (against the actual code — NOT an agent's claim)

Verified against `scripts/fr10_phase4_patch_vllm_tree_gdn.py` (HEAD) and `git show
origin/fr13-cat10-archive:...` (cat10):

- **Topology dispatch is HARDCODED per exact shape.** The drafter pattern-matches `tree_choices ==
  _fr10_caterpillar_choices` (cat9, num_spec 9) OR `== _fr10_spine_only_choices` (chain5, num_spec 5)
  (`:9941-9950`); cat10 adds a third exact match `== _fr10_cat10_choices` (archive `:9521-9528`). ANY other
  tree_mtp shape ⇒ the disengagement RAISE (`:10834`). The leaf-packing tensor is hand-assembled per shape
  (`:10384-10397` cat9; archive `:9993-10021` cat10). **So num_spec is auto-derived from len(TREE) by the
  LAUNCHER, but the drafter PACKING is NOT auto — a novel shape needs a new packing branch.** (The directive's
  "no code change for shape variants" is TRUE only for the already-built cat9/chain5/cat10; FALSE for cat3w /
  chain3 / cat4w. Each new shape = ~15-30 lines: an exact-match guard + a `torch.stack` packing in (len,path)
  order. The `fr13_shape_gate.sh:16-25` header documents this realizability constraint.)
- **Root sibling `(1,)` — CONFIRMED BUILDABLE (cat10 disproves the prior "no root sibling" claim).** Archive
  `:9509-9519` defines `_fr10_cat10_choices` with `(1,)` at flat index 1; `:9654-9679` captures the root
  runner-up `_fr10_root_leaf_token = torch.topk(_fr10_logits, 2).indices[:,1]`; `:9993` packs it into slot 1.
  The MTP drafter produces the rank-1 root token natively and the rank-2 (runner-up) is a free topk. cat10
  ran live, engaged (draft-toks/event=10.0). The false "no root sibling" claim was corrected in commit
  31e227cf.
- **rank ≥ 2 (top-3+) — NOT BUILDABLE without code (checked, not assumed).** Every leaf token in the drafter
  is `torch.topk(_fr10_step_logits, 2, dim=-1).indices[:, 1]` (`:10368-10379`) — ONLY the runner-up (rank-2)
  is captured. There is no rank-3 capture anywhere; a `(0,2)` / `(2,)` node would require widening the topk
  and a new packing slot. So the top-3 candidate shapes (e.g. a 3-way root fan) are **infeasible without a
  drafter change** — do NOT propose them as no-code variants. (rank-2 width — root sibling, d1/d2 siblings — IS
  feasible; that is what cat3w/cat4w use.)
- **Downstream consumers auto-adapt off `tree_choices`.** Archive comment `:9505-9508`: parent/ancestry masks,
  committer path enumeration, eager-pack replay rows, conv-fusion prior windows are ALL driven off the
  SPEC_CONFIG `tree_choices` and auto-derive — ONLY the drafter packing is hand-rolled. So a new shape needs
  the packing branch and nothing else.

**Net:** the realizable shape space for the next boot = {chain5, cat9} on HEAD + {cat10} on archive, with
no-code; cat3w / chain3 / cat4w need a small drafter packing branch each (rank-2 width only). Bring the
not-yet-built shapes to the user / hand to the GPU worker to add the packing branch (behavior-preserving,
flag-gated, default cat9) before the boot — do NOT enable `FR10_ALLOW_LINEAR_FALLBACK`.

---

## 5. ACCEPT-RECOVERY MECHANISM (root-sibling / shallow width + the FREE confidence-gated root branch)

The tension: shallower committed spine = fewer depth-flips BUT lower accept (chain5 = 5 flips / 2.664 accept).
The directive's resolution = recover d0/d1 accept WITHOUT adding committed depth via **shallow width**:

- **Root-sibling / shallow-width rescue (no committed depth):** a `(1,)` root sibling and `(0,1)` d1 sibling
  give a SECOND candidate at depth 0/1. cat10 MEASURED the d0-rescue: d0 accept rate 0.871→0.906 (+0.035) —
  REAL (it rescues the 62%-of-rejects-at-step-0 / the 2-horse-race near-ties). Width nodes are NEVER fed back
  into the forward/recurrent state (patcher `:10955`), so they add ZERO to the depth-accumulation term — they
  recover accept without re-introducing depth flips. This is why width belongs at the ROOT (shallow), where
  acceptance is highest and the online research shows width pays off (vs deep-layer width = diminishing
  returns, Sequoia/EAGLE-2).
- **Why cat10 still net-LOST accept (−0.27) and how the gate fixes it:** cat10 added the root sibling
  UNCONDITIONALLY on a depth-5 spine; the extra verify slot diluted deeper acceptance (d1-d4 all fell) more
  than the d0 gain (commit cd30f5ad / FR13_CAT10_BIND). Part of the −0.27 is also a class-12 trajectory-
  confound + sibling-stop denominator artifact (commit f8a9c032/cd30f5ad — de-confounded recovers d1|d0 to
  ~0.84), but the structural dilution is real.
- **The FREE confidence-gated root branch (directive's named mechanism, the fix):** emit the `(1,)` root
  sibling ONLY when the root top-2 margin is a near-tie (root argmax not confident). The runner-up token and
  its logit are ALREADY computed in the drafter (`_fr10_top2`/`_fr10_root_leaf_token`), so the
  margin = `top1_logit − top2_logit` is **FREE** — a scalar compare, no extra lm-head read. This keeps the
  +0.035 d0 rescue on the near-tie events (where it pays) and DROPS the sibling on confident-root events (where
  it only dilutes) — recovering d0 accept without the unconditional dilution. Confirmed plan in commit cd30f5ad
  ("LEVER = CONFIDENCE-GATED root branch ... FREE top2-margin gate, keeps +0.035 d0 rescue without the artifact
  dilution, shape-true"). This is exactly the adaptive/confidence-aware tree of TALON / OPT-Tree / C2T
  (online research) realized as a runtime gate, no retrain.
- **Layering:** the accept-recovery design = a SHALLOW spine (cut depth-flips) + confidence-gated shallow
  width (recover d0/d1 accept where it pays). cat3w/cat4w are these layered; cat10-gated is the d0-rescue
  building block to bolt on once the depth cut is proven. NOTE: the confidence gate makes draft-toks/event
  VARIABLE (sometimes 4, sometimes 5 for cat3w-gated) — the engagement gate (class 9) must assert tok/draft ∈
  {realizable set}, NOT a single integer; build the gated variant as its OWN exact-match shape with a measured
  variable-count engagement check, or boot it ungated first (cat3w) to get the clean flip number, then gated.

---

## 6. DECISIVE GPU TEST (single boot plan to gate the top candidates)

**Reuse `scripts/fr13_shape_gate.sh <name> "<TREE>"` verbatim** (one shape per invocation, GPU serialized,
recover_host_memory + MemAvailable≥95GiB + docker-empty hygiene before each boot, locked pipeline flags). It
already implements every gate below. Order: **chain3 → cat3w → cat4w** (→ cat10-gated only if accept short).

Per-shape gates (all in `fr13_shape_gate.sh`):
1. **Hygiene + engagement (class 9):** boot forked server with TREE override (num_spec auto-derived); warm
   request; assert `tok/draft == len(TREE)` (drafter ENGAGED; FAIL LOUD otherwise — a new shape with no
   packing branch raises "caterpillar drafter disengaged" here, recording nothing). For the confidence-gated
   variant, assert tok/draft ∈ the realizable variable set, not a fixed int.
2. **Within-boot determinism (class 8):** `fr13_gold_margin_probe.py capture --arm tree`; assert
   `within_boot_det_rep1_eq_rep2 == [T,T,T,T]` (same-boot byte-identical; the cross-boot byte gate is BANNED —
   `feedback_no_cross_boot_byte_gate`).
3. **FLIP COUNT (the comparable metric):** `fr13_oracle_stream_teacher_force.py run --arm tree` vs THIS boot's
   own no-spec decode oracle (thr 1.0 nat); read `total_clear_margin_flips` + per_prompt; assert
   `spec_metrics_delta_during_oracle == 0` (teacher-force must not advance spec counters) +
   `within_boot_det_all_prompts`. **This is the decisive number** — each shape vs its OWN oracle, comparable
   across shapes. Bar: chain3/cat3w → **~3-5** (native band) confirms the depth model; ≥ cat9's 22 refutes it.
4. **accept/event (DIRECTIONAL only, class 11/12):** from /metrics after capture (raw counters recorded);
   per-depth accept RATE (d0..d4) is the fair read; the d0 rescue delta (cat3w/cat10-gated vs chain3) pins the
   width recovery. Whole-window accept/event is trajectory-confounded — never a superset verdict alone.

**Decision logic after the sweep:**
- chain3 ≈ **3 flips** ⇒ depth model CONFIRMED, the chunk-vs-recurrent floor is reachable by depth-cut; then
  cat3w/cat4w supply accept. Pick the shape with flips ≤ ~5 AND best per-depth accept (target accept/event ≥
  native ~3.16 via the d0-rescue + shallow-spine high d0-d2 rates).
- chain3 ≫ 3 flips ⇒ the depth model is wrong / the carrier is NOT primarily committed-depth (would point back
  to a co-residency or scan-kernel seam) — bring to user; do NOT proceed to width shapes.
- Cross-check vs the concurrent L0-GDN sub-op A/B: if it found a real M-dependent co-residency op, prefer the
  NO-width / sparse-gated-width shapes (chain3, cat3w-gated) to minimize co-resident deep nodes.

**Final verdict instrument (separate, after a shape passes flips + accept):** e2e vs E5 (FLASH_ATTN native
MTP-5) at the within-floor bar — spine per-depth-argmax + bag-TV ≤ floor + accept/event ≥ same-shape native —
B=1 first per `project_fr13_speed_first_lossless_gate`; that pass/fail goes to the user. Per-layer 0.0 is a DEV
check only; the within-floor e2e is the gate (`FR13_NODE5_LADDER_DIFFUSE_BIND:63-78`).

---

## Sources (online research — tree topology vs accept/depth)
- Sequoia (hardware-aware, depth-limiting > max-token, dynamic-programming optimal tree): https://arxiv.org/abs/2402.12374
- SpecInfer (tree-based spec inference & verification): https://arxiv.org/pdf/2305.09781
- TALON (confidence-aware adaptive token trees — the confidence-gated branch family): https://arxiv.org/pdf/2601.07353
- OPT-Tree (adaptive draft tree structure; diminishing returns past ~130-140 nodes): https://arxiv.org/pdf/2406.17276
- C2T (classifier-based tree construction; gate which nodes to keep): https://arxiv.org/pdf/2502.13652

## Cross-refs
`FR13_CONV_NOT_CARRIER_SCAN_STATEFEED_BIND.md`, `FR13_NODE5_LADDER_DIFFUSE_BIND.md`,
`FR13_FA2_CARRIER_OVERTURNED_BIND.md`, `FR13_CAT10_BIND.md`, `FR13_DIFFUSE_GDN_EXPLAINED.md`,
`FR13_DIRECTION_AND_NUMBERS.md`, `scripts/fr13_shape_gate.sh`, `scripts/fr10_phase4_patch_vllm_tree_gdn.py`,
[[project_fr13_tree_reshape_unifying_lever]], [[reference_diffuse_gdn_accumulation_explained]],
[[feedback_check_artifact_before_concluding]], [[feedback_grind_all_fronts_dont_re_escalate]].
