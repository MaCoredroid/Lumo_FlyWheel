# Final serving-claim and proposed-experiment review

Reviewed 24 September 2026. Read-only review of the revised manuscript and novelty synthesis; no inference, source modification, external-system execution, or additional workload measurement. This supplements `serving-prior-art-review-2026-09-24.md`; that dated report is unchanged.

## Verdict

**PASS for the revised manuscript's serving contribution boundary.** No remaining material serving-novelty inflation was found in the checked source. This is a source/literature review, not evidence that the system has a novel algorithm or an independently replicated workload advantage. Two small plan-language corrections are identified below; neither changes the manuscript's observations or requires another experiment to preserve its descriptive systems case study.

## Checked identities

Paths are relative to `papers/gdn-tree-scan-mlsys/v2/`.

| File | SHA-256 |
|---|---|
| `main.tex` | `3493bf4f1147ab75cde013321aea01eba57b90904a5719678f13f5660cc9754d` |
| `abstract.tex` | `0b3bdd53645100dadd5d097417b90411e03c0956853acd1330f6c546522f069c` |
| `ref.bib` | `2ad620c7496f093482d9beb43b123c5996a50784c72e48464cbcd8cb297ece8e` |
| `notes/novelty/novelty-review-2026-09-24.md` | `e290b4e745173020d52c1748693661c2a88065611a0a00600d54f24685b9bf63` |

## Serving claims and attribution

- **Attention partitioning:** `main.tex:57` now attributes KV splitting, partial-result combination, query/head reuse, and tree-aware partitioning to [Flash-Decoding](https://pytorch.org/blog/flash-decoding/), [FlashInfer](https://arxiv.org/html/2501.01005v1), and [DeFT](https://arxiv.org/html/2404.00242v2). This matches the primary sources inspected in the preceding independent review: Flash-Decoding's algorithm, FlashInfer Appendix A and split-KV treatment, and DeFT §§3.2–3.3. Lines 219–221 describe the specific two-query-head/four-split implementation and its qualification without claiming invention or a measured isolated gain.
- **Physical layout:** lines 94, 129 and 217 correctly distinguish a tree-specific spine-contiguous KV/mask-key permutation from the general numerical effect of reduction order already discussed in [batch-invariant attention work](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/). Logical query rows and positions remain unchanged. No firstness, universal determinism, or demonstrated superiority over compact verifiers is asserted. The remaining mechanism-specific benefit is explicitly an attribution question, not established by the component bounds alone.
- **Device acceptance and graphs:** lines 94 and 208–210 treat these as established execution strategies. The selected fixed32 path is distinguished from a generic host walk; the text does not say all engine control is device-resident. Lines 178–180 distinguish one graph replay of forty-eight native recurrent calls from a fused all-layer kernel and separate convolution/KV publication. This agrees with the current-production source/engagement review.
- **Fused selection:** lines 203–205 describe a specialized full-vocabulary fusion and pinned reference tie behavior, not a new selection algorithm or a universal top-k stability guarantee. The exact finite gates at lines 262–266 retain their component scope. This is consistent with the public [argmax](https://docs.pytorch.org/docs/2.9/generated/torch.argmax.html) and [top-k](https://docs.pytorch.org/docs/2.9/generated/torch.topk.html) contracts and the pinned implementation evidence.
- **Hybrid drafting:** line 120 now attributes suffix proposals and retrieval-plus-neural drafting to [SuffixDecoding](https://arxiv.org/abs/2411.04975v3), [Arctic](https://arcticinference.readthedocs.io/en/latest/suffix-decoding.html), and [SAM](https://arxiv.org/abs/2411.10666v3). The remaining claim concerns Hydra27's concrete head/tail/rescue packing. There is no general hybrid-drafting priority claim.
- **Overall claim:** lines 46, 50, 94 and 283, plus the abstract, describe a serving implementation and bounded workload observations. The added replay/chain attribution and explicit no-priority statement are consistent with the novelty synthesis. Repository timestamps are not promoted to verified public disclosure. The current scientific evidence does not establish a new general verifier algorithm, and the manuscript now says so in substance.

The new serving bibliography entries resolve to the primary sources used in the earlier review. Revision-year differences for papers with older initial arXiv submissions do not create priority claims in the manuscript. No old Cat10 performance, old numerical witness, external published speed ratio, or isolated-kernel speed result was reintroduced as a current workload result.

## N1–N3: keep the conditions aligned with the requested scope

The note's lines 80–88 explicitly mark these as proposed, require the current deployed route, and separate numerical diagnosis from workload speed. `main.tex:298–300` already makes additions conditional. That respects the user's latest-design/workload-performance restriction if applied as follows:

1. **N1 is eligibility and mechanism diagnosis.** Use the actual current FA2 route and logical-equivalent mappings. It can establish whether this specific layout intervention addresses a reproducible problem. It must not become a substitute kernel-rate headline. Reuse exact-current-build evidence first; no superseded route should be resurrected.
2. **N2 is conditional on a schedule/comparative-verifier claim.** The proposed SpecLA-style port plus a compact verifier is not a prerequisite for retaining the present systems case study, nor for N1 followed by a workload ablation of layout. A faithful implementation comparison would become necessary if a new claim asserts the current schedule's advantage over those alternatives. The novelty note should state this conditionality explicitly instead of presenting all of N1–N3 as one mandatory minimum sequence.
3. **N3 follows the selected distinguishing mechanism.** For an application-benefit claim, compare eligible current systems on declared agent tasks, preserve all attempts and outcomes, and report full agent duration alongside the already-defined pooled decode rate. A layout claim does not require completing an unrelated alternative-verifier port first. Whole-system competition and a within-system mechanism ablation answer different questions; only the latter attributes gain to that mechanism.
4. **Literal condition reversal at note line 90:** “MTP/suffix composition and individual fusion ablations are optional only if those components become claimed scientific advances” says the opposite of the intended rule. Replace with **“needed only if”** or **“optional unless.”** No new experiment is implied by correcting this sentence.

These are plan-language corrections, not numerical-result defects. No GPU experiment is requested by this review, and no extra experiment is required solely to retain the current descriptive claims.

## Closure after mechanism-led reframing (24 September 2026)

**PASS; no remaining actionable regression found in this bounded revision check.** The source now frames the paper around continuation consistency, temporary state versus recomputation, and serving overhead. Internal configuration names have been replaced with explanatory mechanisms in the main narrative. Evaluation Setup (`main.tex:252`) preserves the concrete 27-candidate/32-row geometry, maximum depth eleven, four post-root MTP forwards, two query heads per group, four context partitions, and 48 recurrent/sixteen full-attention layers. This is a presentation change, not a claim that those settings are universal or optimal.

The initial abstract wording accidentally described replay into both recurrent and convolution state. The parent corrected it during review: accepted-path replay reconstructs recurrent state; convolution history and attention entries are gathered from the same path. The checked abstract below contains that correction. The mechanism body still separates recurrent replay from convolution/KV publication (`main.tex:178–180,213,224–226`). No new equivalence, all-layer fusion, or individual optimization speedup claim was introduced.

The numerical claims remain unchanged from the preceding review: shared-task rates 29.09, 28.28 and 27.27 versus the eligible SGLang 26.89, relative difference 8.18%; separate ten-task rate 25.63; full four-task Sr12 rate 26.21; the same task outcomes, request/token counts, latency sums and retrospective patch-producing rule. These values still match the saved `patch-producing-rate-audit.json` (read-only check of its existing totals, not a new reduction or measurement). Component quantities also remain 1,368 cases × five settings = 6,840 configurations, 24 graph replays, sixteen determinism cases across two processes, 93.31% within two ULP, maximum absolute difference 0.00390625, maximum LSE difference four ULP, and nine declared checks. The narrative continues to separate component checks from complete-model correctness and workload speed.

Both plan-language findings above are closed: the novelty synthesis now explicitly makes N2 conditional on a schedule-advantage claim; N3 follows the selected mechanism; the final clause says individual component ablations are “needed only if” those components become claimed scientific advances. `review-experiments.md:5–11` agrees and explicitly makes neither N1 nor N2 a prerequisite for honestly describing the existing case study. No additional experiment is required by this wording review.

### Superseding checked identities

| File | SHA-256 |
|---|---|
| `main.tex` | `fefd9fb1a07199dd7d085c415930db4ad1266b22c3de27786188c01c37ac196e` |
| `abstract.tex` | `5bd67fda58efeb062a2f3fb13abb266b7d960c321f14312bc214de57f971f101` |
| `results/agent-workload/case-study.tex` | `091f567a2dc1367c67a31947c11608c53457920983751a603a62df07773dab35` |
| `ref.bib` (unchanged) | `2ad620c7496f093482d9beb43b123c5996a50784c72e48464cbcd8cb297ece8e` |
| `notes/novelty/novelty-review-2026-09-24.md` | `eaf464309d8ff89efffb2550a375178500dcdc030213aba143b91a141bcc68a8` |
| `review-experiments.md` | `14c4eee5d97fc88917c599cb13497e3412a58528b29518d4ffd380318af5850b` |

The final source identities include the reference-column balance change from trigger 22 to 27 and the case-study opening's plain-language dataset name. Those last changes alter no numerical or mechanism claim. The checked saved rate audit has SHA-256 `279475d182ecd259790e2c57663d65c37205f2ee67c80aa46b0bd2e9b4e460c4`; the unchanged current-component audit has SHA-256 `2d0904d5a6c77e0fa0f56a9850dbc0cce3cad4898227a3f78770758edc93c4c2`.

No literature rescan, GPU execution, raw-evidence mutation, or manuscript edit was performed. PDF rendering is handled separately by the parent.
