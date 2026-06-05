# FR10 — Are the closeout papers a CATASTROPHIC no-go for no-copy GDN tree verify?

**Date:** 2026-06-05 · **Method:** deep-research harness (5 angles → 15 primary sources → 69 claims → 25 adversarially verified, 3-vote/2-of-3-refutes-to-kill → 21 confirmed, 4 killed). All claims cite primary paper text.

---

## VERDICT: NOT-CATASTROPHIC — "open path, unproven for GDN"

**No cited paper establishes a theoretical no-go.** There is no impossibility theorem, no "superset-only" statement, and no path0-degradation claim anywhere in the literature surveyed. The "no way out for no-copy" framing is **not supported by any paper text**. But neither is there a citable *save*: no paper supplies a validated no-copy recipe for the **gated delta rule** specifically. The live `1.77 vs 3.076` loss is an **engineering gap, not a proven mathematical wall**.

---

## STree (arXiv 2505.14969, Wu/Qin/Wong/Soatto) — the central paper

STree is the **opposite of a no-go** — it is "the first scalable algorithm to perform tree-based speculative decoding in SSMs and hybrid architectures," reports positive speedups (MT-Bench 1.74×, HumanEval 1.95×, GSM-8K 1.98× at temp=0), and frames tree-on-SSM as a viable speedup path.

**1. STree is itself genuinely no-copy / single-shared-state** (confirmed 3-0). It packs the entire token tree into ONE sequence with one shared initial state `x_0` + a tree mask `L`, "without requiring extra SSM states." Post-accept carry-forward is by **activation replay (recompute)**, not per-branch state copy. Its stated motivation is to *avoid* "one state per input sequence... an explosion of the memory footprint." → STree does **not** mandate copying state per path.

**2. STree explicitly preserves path0 losslessly** (confirmed 3-0). Verbatim: *"When L is a lower triangular causal attention mask, our method is the same as Mamba2 with a non-zero initial state."* The tree-ancestry mask is a subset of strict-lower-triangular, so the trunk is unmutated. → **This refutes the closeout's attribution that "shared recurrent state degrades path0."** Our own kernel independently confirms it: leaf contamination = 0.0, state delta 3.8e-6.

**3. ⚠ THE LOAD-BEARING CAVEAT — STree's exactness requires a DIAGONAL state transition** (confirmed 3-0). Verbatim: *"we enforce a diagonal structure on A_i to reduce the products... to a sum of logarithms."* STree is validated **only on diagonal Mamba2/S6** (Mamba2-2.7B/130M, MambaInLlama-8B, Mamba2-Llama3). **The string "delta" never appears in the paper.** The gated delta rule's rank-1 `(I − βkkᵀ)` **non-diagonal** update is exactly the regime STree's diagonal log-sum construction does **not** cover. STree's "arbitrary mask" generality is about tree *topology*, not the diagonal-A constraint.

**4. STree proves no formal losslessness theorem** (confirmed 3-0) — it's algebraic reformulation + RTX-3090/H100 empirics, no per-layer numerical-drift analysis. Precision/hardware regime is mismatched to GB10 fp8.

**5. STree covers only the SSM scan — NOT causal_conv1d, NOT the drafter.** The FR10 conv-coverage and drafter-topology issues are outside any paper's scope.

> **Why this is actually reassuring for our kernel:** our FR10 tree scan does *not* use STree's diagonal shortcut — it solves the full lower-triangular rank-1 delta-rule system directly (`fr10_gdn_tree_kernel.py`, the two `static_range` solver loops), and was proven contamination-0 / byte-native (1.5e-5). So **the non-diagonal case is already solved at the scan level.** STree's diagonal limitation tells us the *scan algebra* is the hard part in the literature — and we've banked that. That pushes the live drift **upstream of the scan** (conv / l2norm / drafter / fp8) — i.e. exactly the engineering seams workflow #1 is probing.

## Component-Aware Self-Spec (arXiv 2605.01106) — the scary 0.038 number

**ORTHOGONAL, not catastrophic** (confirmed 3-0). The 0.038 sequential-hybrid acceptance is an artifact of **component-level SELF-speculation** — replacing attention layers with identity functions to use the linear-attn subgraph as a zero-cost internal drafter, which pushes GDN layers off-distribution (81.96× perplexity blow-up). It is **not** a ceiling on sequential-hybrid speculation: generic LayerSkip on the *same* model gets **12× higher** (0.233 vs 0.019). Our Qwen3.6 is a sequential hybrid, but FR10 uses an **external MTP drafter + tree-verify**, which this figure does not bound. The paper **explicitly endorses our direction as future work**: *"Tree-based methods such as STree could increase the effective acceptance rate... Combining component-aware drafting with tree verification is a promising direction."*

## Gated DeltaNet-2 (arXiv 2605.22791, Hatamizadeh/Choi/Kautz, NVIDIA)

**ORTHOGONAL** (confirmed 3-0). Pure architecture evolution — decouples the scalar gate into channel-wise erase `b_t` (keys) + write `w_t` (values); reduces to GDN-1 (our target) when gates collapse. State stays `R^{d_k×d_v}`. **Zero** mention of speculative/tree/draft/verify. Neither establishes a no-go nor supplies a tree path.

## Impossibility search

**No impossibility theorem or lower bound exists** for lossless tree spec-verify under a fixed-size recurrent state (medium confidence — absence-of-evidence across the searched set). The BRW lower bound (2512.11718) is architecture-agnostic (depends only on verifier capacity P + output entropy), zero hits for recurrent/state-space/linear-attention. The "no-copy tree verify on linear attention is superset-only" claim is an **engineering observation, not a proven theorem**.

## What was KILLED (over-reaching claims, refuted ≥2-of-3)

- "STree documents an overhead tradeoff that mirrors our 1.77-vs-3.076 loss" — **refuted 0-3**, the cited cost-inequality is not in the paper. The live loss **cannot** be cited as a paper-documented expected tradeoff.
- "STree's losslessness is an exact *proven* path-independence theorem" — refuted 1-2; it's algebra + empirics, not a theorem.
- "0.038 is definitely not a fundamental ceiling" (over-asserted) — softened; the LayerSkip-12× counter stands but the framing was trimmed.

## Bottom line for the decision

| Question | Answer |
|---|---|
| Do the papers prove no-copy GDN tree verify is impossible? | **No.** No theorem, no no-go. |
| Does STree doom us (shared-state degrades path0)? | **No — STree refutes it**; our contamination-0 measurement agrees. |
| Is the scary 0.038 our regime? | **No** — it's component self-spec; we use external MTP + tree-verify (paper endorses this). |
| Does any paper hand us a validated no-copy recipe for the **gated delta rule**? | **No** — STree is diagonal-only (Mamba2/S6); GDN's rank-1 update is uncovered. |
| So what is the live 1.77-vs-3.076 loss? | **Engineering** — non-diagonal handled by our scan already; gap is upstream (conv / drafter-topology / fp8), per our own diffuse-drift data. |

**No-copy is not closed by the literature.** The honest status is **"open but unproven for GDN"**: nobody forbids it, nobody hands us the recipe, and our hardest piece (the non-diagonal tree scan) is already banked lossless. The remaining question — *is the upstream drift a fixable bug?* — is what workflow #1 (`w23supcd0`) is deciding empirically.

## Open questions carried into workflow #1

1. Does STree's `A_tree = L·A_log` extend exactly to GDN's non-diagonal rank-1 update? (We bypass it with a direct solver — but is our solver's *input* path identical to native's?)
2. Is the live loss from conv coverage, drafter topology (caterpillar-vs-parallel-chain, flat-vs-depth-RoPE), or fp8 on GB10 — and which single fix closes the most gap?
3. Given the scan is solved, is the copy-recurrent multi-spine pivot **premature**?

### Sources
- STree: https://arxiv.org/html/2505.14969v2 · https://openreview.net/pdf?id=a95Vd41o1u
- Component-Aware: https://arxiv.org/html/2605.01106
- Gated DeltaNet-2: https://arxiv.org/html/2605.22791v1
- BRW lower bound: https://arxiv.org/abs/2512.11718
