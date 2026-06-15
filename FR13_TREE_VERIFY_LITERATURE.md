# FR13 — Tree-Verify Losslessness: Literature Survey + Mapping to Our LCP Committer

**Date:** 2026-06-15 · **Mode:** CPU-only, read-only research (a GPU committer-margin probe runs
concurrently). **Scope:** survey the state of the art on tree-verify commit rules + the
verify-vs-target realization gap, decide whether the user-proposed SPINE-BONUS / margin-aware
tie-break is lossless, deep-read Traversal Verification (2505.12398), and map back to our top-down
LCP committer (`scripts/fr10_phase4_patch_vllm_tree_gdn.py` L6818-6976).

**Tags:** `[LIT]` = cited from a paper/source · `[INF]` = our inference / mapping (not in any paper).

---

## 0. Our situation in one paragraph (the thing the literature must be mapped onto)

cat9 = 9-node caterpillar (depth-5 spine 0-1-3-5-7 + 4 leaves). Lossless bar = per-served-token argmax
== the **no-spec RECURRENT decode oracle**, at **native-E5 level** (native MTP-5 = ~3 clear-margin
flips = the within-floor bar; B=4 measured same-seed floor is **bag-TV ≤ 0.113**, `seed1313v1313`,
`FR13_DIRECTION_AND_NUMBERS.md`). cat9 = 23 raw / 18 de-cascaded flips. **The carrier**
(FR13_LEAF_CORESIDENCY_PATH, FR13_NOCOPY_LOSSLESS_LEAVES, both verify HOLDS) is an LCP-committer
**TRAJECTORY FORK** (bug-class **#12**, see §6): the verify forward (tree-batched, leaves co-resident,
GDN tree-scan) is a slightly different *numerical realization* than the single-sequence decode forward,
so `parent_targets` (verify argmax) ≠ decode argmax at a few **near-tie** nodes → the exact-match LCP
picks a leaf the spine never serves. **This is NOT an algorithm bug** — LCP is lossless *w.r.t. the
verify forward*; the gap is verify-vs-decode realization. Kernel fixes are closed (scan recompute made
state bit-exact yet flips rose 23→32; reshape-away-leaves = lossless-but-slow, REJECTED, leaves
mandatory). WY is **PARKED by the user** — not revived here.

---

## 1. Per-method commit / tie-break table

| Method | Commit rule | Tie-break | Verify==target assumption / realization-gap notion | Greedy(t=0)? | Cite |
|---|---|---|---|---|---|
| **SpecInfer** (2305.09781) | Token-tree verification. Stochastic: **multi-step speculative sampling** (recursive residual `norm(relu(p−q))` per tree level, naive/`SpecInfer` variants), Thm 4.1/4.2 lossless. Greedy: commit the **longest root-to-leaf path whose every token == target argmax** at that node, top-down. | "first/highest-prob matching child" (tree built in descending draft prob); ties not separately specified. | **Assumes verify dist == target dist exactly.** Thm proves equality to *the LLM's* `P(·|prefix)`; the parallel tree forward is taken to be that `P`. **No realization-gap notion.** | Yes (degenerate longest-match) | [LIT] §3-4 |
| **Sequoia** (2402.12374) | Sampling-**without-replacement** verify loop (`x~Q wo-repl; accept vs P; P←norm(relu(P−Q)); Q[x]←0`); optimal-transport (k=1) + cover property → **provably exact target output dist** across temps. | within-tree by draft prob; sibling order = descending Q. | **Assumes verify==target.** Losslessness is w.r.t. the given `P`,`Q`; **does not address tree-batched vs sequential forward drift.** | reduces to longest-prefix [LIT-implicit] | [LIT] |
| **SpecTr** (2310.…) | Multi-draft **optimal-transport** acceptance (draft selection ≈ k-sequential SpS); lossless to target by OT coupling. | OT plan resolves selection; no explicit numeric tie-break. | **Assumes verify==target.** No realization-gap notion. | reduces to longest-match | [LIT] (named in Sequoia/MARS as prior; OT lossless) |
| **EAGLE / -2 / -3** | Dynamic draft tree; target verifies in one forward; **accept the longest prefix matching the target's own predictions** (argmax for greedy; SpS-equivalent for sampling), continue from there. "without any impact on accuracy." | top-down longest-accepted prefix; among siblings the higher-confidence branch (tree reranked by global accept prob) is preferred. | **Assumes verify==target** (the target forward IS the ground truth). **No realization-gap notion** — single dense model, verify==decode by construction. | Yes | [LIT] EAGLE-2 paper / LMSYS / NVIDIA blog |
| **Medusa** (2401.10774) | Tree attention over head candidates; **typical-acceptance**: pick **longest plausible prefix** whose tokens clear a probability/entropy **threshold** (`p > min(ε, δ·exp(−H))`), NOT exact argmax. | longest qualifying prefix; ties by candidate order. | **LOSSY by design** — "diverges by **not insisting on exact correspondence** between the output and the LM distribution." Threshold-gated, not target-exact. | thresholded, not argmax | [LIT] §typical-acceptance |
| **DeFT** (2404.…) | An **IO/HBM-efficient flash tree-attention kernel** (prefix-aware, load-balanced KV partitions). It is an attention *primitive*, not a commit rule — commit logic is whatever method sits on top. | n/a (kernel) | n/a — but **it is the "no extra HBM traffic for shared-prefix KV" reference**; relevant to our 273 GB/s constraint. | n/a | [LIT] |
| **STree** (2505.14969) | Speculative **tree** decoding for **hybrid SSMs** (our exact regime). Packed-tree execution reusing accumulated **state-transition** matrices; verification is the standard tree accept on top of the SSM forward. | inherits the chosen accept (SpS/longest-match). | **Assumes the SSM tree forward == the SSM sequential forward.** The paper's whole contribution is making the *state* shareable; it **does not claim the tree-batched realization is bit-identical to the per-path recurrent realization** — exactly the gap we measure (STree root cause already in MEMORY: shared recurrent state degrades path0). [INF] this is the literature closest to our carrier and it has **no realization-gap remedy**. | reduces to longest-match | [LIT] + [INF] |
| **Traversal Verification** (2505.12398) | **Bottom-up leaf→root.** Per chain, cumulative accept prob root→leaf; sample η~U(0,1); if `η<p_α(leaf-path)` accept the **whole leaf-to-root chain**; else delete leaf, redistribute residual, recompute siblings; iterate. **Sequence-level** accept, not per-node. | post-order DFS over leaves; **no explicit tie-break / near-tie handling** (Appendix). | **Assumes verify==target** (`M_b`,`M_s` given as inputs; Lemma A.2 uses them directly). **No realization-gap / batching-artifact notion.** | **EXPLICITLY EXCLUDED** — App. G: "In greedy decoding … all verification approaches [are] functionally equivalent … eliminating any potential performance gains." Needs temp>0 for the `η<p_α` test. | **NO (temp>0 only)** | [LIT] §3.1, Thm 3.3, App. G |
| **vLLM rejection_sampler** (v1) | temp>0: per-token rejection sampling (`accept if u≤p_target/q_draft`, else recover from `norm(relu(p−q))`). **temp=0: pure argmax**, "greedy sampling with spec decode == greedy sampling without it" (the greedy-equality lossless test). | greedy: argmax (ties → lowest index, torch.argmax). For trees, longest matching path. | **Assumes verify==target** — the target forward's logits ARE the reference. The greedy-equality TEST is *same-engine* (verify==decode trivially in a dense single-seq model). **No realization-gap notion** — and that test does NOT cover our GDN tree-batched-vs-recurrent gap. | Yes | [LIT] docs.vllm.ai v1 rejection_sampler |
| **SGLang (EAGLE)** | Tree expanded `speculative_eagle_topk`, reranked to top-N; target verifies; longest accepted path. **Exposes `--speculative-accept-threshold-single/-acc`** = optional **lossy** typical-acceptance knobs (off ⇒ lossless argmax/SpS). | longest accepted prefix; reranked tree order. | Assumes verify==target. No realization-gap notion. | Yes (threshold off) | [LIT] SGLang docs |
| **MARS** (2601.15498) | **Margin-aware verification**: relaxes rejection **only when the target shows weak preference among top candidates** → accepts a **runner-up** draft token. | margin = "decision stability from target logits"; threshold τ (value not given in abstract). | **LOSSY relative to strict token-level verification** — "rejecting plausible runner-up tokens yields negligible information gain"; claims quality "preserved across benchmarks" but is **not distribution-exact**. | applies at low margin | [LIT] abstract (full PDF binary-blocked; abstract decisive on lossy-ness) |

**The one fact that matters across the whole table** [INF, but unanimous across [LIT]]: **every losslessness
theorem assumes the verify forward distribution IS the target distribution.** None of SpecInfer/Sequoia/
SpecTr/EAGLE/STree/Traversal/vLLM/SGLang has a *verify-vs-decode realization-gap* notion — because in a
dense single-sequence Transformer verify==decode by construction (the verify forward of node *i* is the
same kernel on the same prefix as the decode forward). **Our GDN tree-batched, leaf-co-resident
recurrent forward breaks that identity** — that is precisely why our problem is *not* in the literature,
and why "LCP is lossless" (true, w.r.t. the verify forward) and "we flip vs the decode oracle" (true) are
**both** correct. The lossy relaxations (Medusa typical-acceptance, MARS runner-up, SGLang accept-
thresholds) are **BANNED for us** (runner-up/top-k/threshold acceptance = LOSSY, reward-hack class).

---

## 2. Is SPINE-BONUS a known lossless pattern? Verdict + exact margin condition

**Verdict:** A deterministic *prefer-the-canonical-path* (spine-bonus) tie-break is **PROVABLY LOSSLESS for
greedy (temp-0) tree spec decoding ONLY when the deciding node is a sub-floor near-tie**, and **LOSSY
otherwise** (when it suppresses a confident genuine leaf win). It is **NOT a named method** in the
surveyed literature for the realization-gap case — it is **a sound novel tie-break** [INF], justified by
combining two cited facts:

1. **[LIT]** Greedy verify is lossless iff the committed token == the target argmax (EAGLE/vLLM greedy-
   equality). Among tokens that the target is *indifferent* between, committing either is lossless.
2. **[LIT]** Our floor is *measured*, not zero: native itself is non-deterministic at the deployed regime
   (bag-TV 0.113 floor, FR13_DIRECTION_AND_NUMBERS). So two tokens whose verify logits differ by **less
   than the realization floor are indistinguishable to the target** — the target's "true" argmax is not
   resolvable below that floor.

**Exact margin rule** [INF, the binding statement]:

> Let the deciding node be where a **leaf path's LCP strictly exceeds the spine's LCP**, i.e. the leaf was
> accepted because `drafts[node] == parent_targets[node]` (verify argmax) at a node the spine diverges on.
> Let `m = verify_top2_margin` at that node = `argmax_logit − runnerup_logit` of the **verify** forward
> row (already computed by our CAG instrument as `verify_top2_margin`, L6113-6116), where the runner-up
> is the spine's token.
>
> - **If `m ≤ FLOOR`** (the verify forward was nearly indifferent; the spine token is the runner-up within
>   the realization floor): leaf-vs-spine is **within realization noise** → committing the **spine** is
>   **LOSSLESS** (the target cannot resolve the order below the floor; bug-class #11 sub-ULP flips).
> - **If `m > FLOOR`** (the verify forward *confidently* prefers the leaf token over the spine token):
>   suppressing the leaf to commit the spine is **rejecting a real accept** → **LOSSY**.

`FLOOR` here is the per-node logit-margin counterpart of the B=4 bag-TV 0.113 floor — it must be
**measured** as the largest top-2 logit margin observed on a *native-vs-native* same-seed control arm at
the same shape (the GPU committer-margin probe is measuring exactly the per-fork deciding margins to bin
each fork near-tie vs confident). **Do not hard-code a logit number** — derive it from the control arm
(bug-class #12: single-draw floors are wrong; multi-sample p95).

**Contrast with the lossy relaxations** [LIT]: MARS (runner-up acceptance) and Medusa typical-acceptance
relax the accept rule for SPEED even when the target has a *clear* preference — they trade quality for
throughput and are **not distribution-exact**. Spine-bonus is the **opposite direction**: it only ever
*declines* a leaf in favor of the canonical spine, and *only* within the floor, so it **never accepts a
token the target disprefers** — it just resolves an unresolvable tie deterministically toward the
canonical path. That asymmetry is why it can be lossless where MARS/top-k cannot. **Novelty:** the
margin-gated tie-break itself is standard determinism; what's novel is **gating it by the
verify-vs-decode realization floor** (no surveyed paper has this object).

---

## 3. Traversal Verification (2505.12398) — deep-read assessment

**Mechanism** [LIT §3.1, Alg. 3]: bottom-up, leaf→root. Init cumulative accept probs root→leaf per chain.
Loop: pick chain to first unverified leaf; sample η~U(0,1); **accept the entire leaf-to-root chain** if
`η < p_α(leaf-path)`; else delete the leaf, redistribute residual/draft mass, recompute sibling chains;
iterate until accept or exhaustion. Theorem 3.3 proves the output equals the target distribution (Eq. 4).
It is **committer-only / plug-and-play** ([LIT] "plug-and-play replacement … minimal modification") — it
**reorders which path is committed from the SAME single verify forward**, no re-inference, **no KV/state
copy, no extra HBM traffic** — fits our 273 GB/s constraint on that axis.

**Does it avoid our leaf-fork?** [INF] **No — not the way we need.** Three disqualifiers:

1. **It is temp>0 ONLY. Greedy is EXPLICITLY EXCLUDED** ([LIT] App. G: "all verification approaches
   functionally equivalent" at t=0). Our binding bar is **greedy argmax vs the decode oracle**. At t=0
   Traversal reduces to the same longest-match every other method does — it has **nothing to add to
   greedy** and the user ruling is that the GREEDY branch rescue is the real bar (MEMORY
   feedback_fr13_speed_first). So it cannot be our greedy committer.
2. **It assumes verify==target** (Lemma A.2 uses `M_b`,`M_s` as exact inputs; **no realization-gap
   notion**). Its losslessness is w.r.t. *the verify forward's* distribution — **exactly like our LCP**.
   So on our GDN forward it would commit leaf chains the decode oracle never produces, the **same**
   trajectory fork, just bottom-up. It does **not** solve the verify-vs-decode gap — no method does.
3. **The fork it "avoids" is a different fork.** Traversal's advantage (over top-down) is that a *rejected
   parent does not discard a valid child subsequence* — it salvages deeper accepts that top-down throws
   away. That is the *opposite* of our problem: **we are over-committing leaves, not under-committing**.
   Traversal would commit *more* off-spine chains, **increasing** our trajectory forks, not reducing them.
   [INF]

**Is it strictly better than spine-bonus-margin-damp? When would we prefer it?** [INF]
- For **greedy** (our bar): **No.** It is inert at t=0 and, where it acts, it pushes toward *more* leaf
  commits — the wrong direction for our carrier. Spine-bonus is strictly better for the greedy lossless
  bar.
- For **temp>0 superset/speed** (deferred, after speed settles per MEMORY): Traversal is a **genuinely
  attractive committer-only, no-copy, no-HBM upgrade to acceptance LENGTH** — it salvages valid deep
  chains and is provably lossless w.r.t. the verify forward, with no memory tax. **Prefer it later as the
  temp>0 path-acceptance layer**, IF and only if we have first closed the realization gap (otherwise it
  inherits the same fork). It is **complementary**, not a substitute, to spine-bonus: spine-bonus fixes
  *which near-tie token* greedy commits; Traversal fixes *how many tokens* a sampling run accepts.

---

## 4. Mapping to OUR LCP committer

**What our committer does** (`fr10_phase4_patch_vllm_tree_gdn.py` L6818-6976): for each root-to-leaf path,
`lcp = ` longest prefix where `drafts[node] == parent_targets[node]` (parent_targets = argmax of the
VERIFY forward at the node, computed upstream). `best_path = argmax lcp`; **tie-break = earliest
flattened leaf**, and because vLLM emits **sorted** trees the spine is the earliest leaf, so **on EQUAL
lcp the spine already wins** (L6917-6923, comment "earliest flattened leaf wins … preserves the native
top-1/path0 chain"). Commits `drafts[node]` for the lcp prefix + one bonus token (self-target /
parent-target). **The fork is strictly `leaf_lcp > spine_lcp`** — a leaf wins because the verify argmax at
one node matched the leaf draft but not the spine.

**Does the literature support a margin-aware extension as lossless-within-floor?** [INF, grounded in [LIT]]
- **Spine-on-EQUAL-lcp** (what we already do) is exactly the EAGLE/vLLM **deterministic canonical-path**
  tie-break [LIT] — uncontroversially lossless (the two paths agree token-for-token through the tie).
- **Extending it to `leaf_lcp > spine_lcp` BUT within the floor** = the §2 margin rule. This is **lossless
  within the floor**, **not** a literature-named method, but a **sound novel tie-break** [INF]: it only
  declines a leaf at a node where the **verify top-2 margin `m ≤ FLOOR`**, i.e. where the verify forward
  itself could not resolve leaf-vs-spine above the realization floor. At such a node the leaf "win" is a
  *verify-realization artifact* (bug-class #11/#12), not a target preference — declining it commits the
  canonical token the decode oracle would have produced. **Lossless w.r.t. the decode oracle, which is the
  actual bar** (the literature's verify==target identity is exactly what's broken at these nodes, so
  "lossless w.r.t. the verify forward" — which raw LCP already is — is the WRONG target here).

**Is the user-proposed spine-bonus a known method, a sound novel tie-break, or does it need the probe?**
**[INF — the answer]:**
1. It is **NOT a known method** for our case — no surveyed paper models a verify-vs-decode realization gap.
   It is **a sound novel tie-break** that *composes* two cited facts (greedy lossless == match target
   argmax; the floor is measured, not zero).
2. It is **lossless ONLY confined to sub-floor forks**, and is **LOSSY if applied to confident forks**
   (`m > FLOOR` — suppressing a real accept = rejecting a token the target genuinely prefers). Therefore:
3. **It DOES need the probe first.** The principled fix is *conditional*: damp the spine→leaf boundary
   **only where `m ≤ FLOOR`**. To deploy it lossless we must **confirm, via the GPU committer-margin
   probe, that the forks are sub-floor** (and bin any `m > FLOOR` forks as *fundamental* — those are NOT
   spine-bonus-fixable; they are genuine verify-vs-decode divergence at a confident node and must be closed
   upstream, never damped). A blanket spine-bonus that fires regardless of margin would be **LOSSY** (it
   would reject confident leaf accepts) and is **banned as a reward-hack** in the same class as
   FR13_FORCE_SPINE_COMMIT (which the code already documents as DIAGNOSTIC-ONLY, "NEVER bind =1 into a
   committed serving config", L6855-6867).

**Our instrument is already in place** [code]: the CAG block (L7007-7125) computes, per served token,
`verify_top2_margin` (= argmax−runnerup of the verify row the committer used) and `ch1_margin`, and tags
`node_type` spine/leaf and `winner_path_idx` vs `spine_path_idx`. The margin-aware rule needs exactly
`verify_top2_margin` **at the deciding node** (`best_path[best_lcp]` for the boundary, or the first node
where leaf and spine diverge) — already captured. The probe's job is to histogram those margins across all
23 forks and decide the floor split.

---

## 5. Bottom line

- **Commit-rule landscape** [LIT]: top-down longest-accepted-prefix (SpecInfer greedy, EAGLE, vLLM/SGLang,
  STree on SSM) vs sampling-without-replacement (Sequoia) / OT (SpecTr) / SpS (vLLM temp>0) vs **bottom-up
  chain accept (Traversal)**. **All assume verify==target; none model a verify-vs-decode realization gap.**
  Lossy outliers: Medusa typical-acceptance, MARS runner-up, SGLang accept-thresholds — **BANNED for us.**
- **Spine-bonus** [INF]: provably **lossless within the realization floor** (sub-floor forks only),
  **lossy** for confident forks; a **sound novel margin-gated tie-break**, not a named method; **requires
  the probe** to confirm forks are sub-floor before binding (else it's a banned forced-spine reward-hack).
- **Traversal Verification** [LIT+INF]: committer-only, no-copy, no-HBM (fits constraints) but **temp>0
  only (greedy excluded)**, **assumes verify==target**, and pushes toward *more* leaf commits — **does NOT
  solve our greedy carrier**; bank it as a future **temp>0 path-acceptance-length** upgrade, complementary
  to spine-bonus, only after the realization gap is otherwise addressed.
- **Constraints honored:** spine-bonus = committer-only, **zero added HBM traffic** (it reads the
  `verify_top2_margin` already computed; commits a different already-verified row) — clean on the 273 GB/s
  GB10 B=1 bandwidth bound. No copy / no dense / no multi-spine / no runner-up/top-k. **WY stays PARKED.**

---

## 6. Bug-class playbook quote — #12 (trajectory / measurement traps)

> **#12 Measurement traps** (multiple retractions): TPS÷accept hand-rolls (retracted 2×); prompt-pairing
> mismatch (lcp=0 artifact, burned 3-4 boots); per-pos counters indexing accepted-path-length ("branches
> added 0" artifact); single-draw floors (0.0593 vs measured 0.113); non-like-for-like trajectories after
> fixes. **Fix:** raw counters only; capture-once pinned prompts; source-index traces; multi-sample p95
> floors; label every estimate. (FR13_BUG_CLASS_PLAYBOOK.md row 12)

Applied here: the spine-bonus FLOOR must be a **multi-sample p95** of a **native-vs-native** control arm
(NOT the single-draw 0.0593, NOT a hard-coded logit), the forks must be classified on **raw per-fork
`verify_top2_margin`** at the **source-indexed deciding node**, and any "all forks sub-floor → spine-bonus
lossless" claim must be measured on **like-for-like trajectories**, not inferred. Also relevant:

> **#11 Batch-composition / BI-flag sensitivity**: native itself only 0.714 draft-identical across BI
> flag; **near-ties flip on sub-ULP shifts**. (row 11)

— this *is* our carrier: the leaf-fork near-ties are sub-ULP verify-vs-decode shifts; the spine-bonus only
claims losslessness in exactly the regime (#11 sub-ULP) where the flip carries no target information.

---

## Sources

- SpecInfer — arxiv.org/abs/2305.09781 ; cs.cmu.edu/~zhihaoj2/papers/specinfer.pdf
- Sequoia — arxiv.org/abs/2402.12374 ; emergentmind.com/papers/2402.12374
- EAGLE-2/-3 — researchgate EAGLE-2 ; lmsys.org/blog/2025-12-01-eagle3-vertex ; developer.nvidia.com (intro to spec decoding)
- Medusa — arxiv.org/abs/2401.10774
- STree — arxiv.org/abs/2505.14969
- DeFT — openreview.net/forum?id=2c7pfOqu9k
- Traversal Verification — arxiv.org/abs/2505.12398 (html: arxiv.org/html/2505.12398)
- MARS — arxiv.org/abs/2601.15498
- vLLM rejection_sampler — docs.vllm.ai v1 sample/rejection_sampler ; spec_decode greedy-equality test
- SGLang — sgl-project.github.io/advanced_features/speculative_decoding.html
