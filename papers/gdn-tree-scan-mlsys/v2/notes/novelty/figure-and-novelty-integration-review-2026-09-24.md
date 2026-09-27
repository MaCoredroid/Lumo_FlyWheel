# Figure 1 semantics and novelty-review integration

24 September 2026. Bounded read-only semantic review of Figure 1, its caption, and the manuscript locations implementing the completed novelty review. No manuscript edit, literature rescan, inference, or new measurement. Visual/layout QA is handled separately by the parent.

**PASS. No actionable semantic error or unsupported novelty claim found in the checked figure and caption. The novelty review materially feeds the manuscript rather than remaining only an external note.**

## Figure 1

`figures/pipeline.tikz` and `main.tex:39–44` now describe an illustrative logical tree and an accepted-path state boundary:

- The initial `m` is explicitly a materialized prefix. The selected `a,d` path exists in the proposed tree (`m→a→d`); it is selected after target verification rather than colored as accepted in advance. The example need not match the evaluated tree's complete geometry.
- Proposal probabilities `q` and target probabilities `p` feed selection. The figure makes no independent claim that numerical kernels supply exactly sequential target probabilities or that the illustration proves the sampling law.
- The published persistent state is `S(mad)`. The newly sampled `b*` is separately dashed and marked “not yet in state,” consistent with the pending-token boundary at `main.tex:99–105`. It is not incorrectly included in `S(mad)`.
- Recurrent replay, convolution-history gathering, and attention-KV remapping are separate operations. This matches `main.tex:178–180,213,224–226` and avoids the earlier replay-versus-gather ambiguity.
- Temporary cut states and replay operands remain within-step resources; the dashed operand connection supports recurrent replay. “Unselected branch state is not published” does not assert that all scratch must be zeroed, consistent with `main.tex:117`.
- The common descriptor is a coordination invariant beneath the stages, not an extra sequential model operation. The target-forward box groups state surfaces rather than depicting three independent target models.

The figure adds explanatory structure, not a new algorithm, firstness claim, equivalence proof, or isolated performance attribution. The caption and existing architecture text together preserve the materialized-state/pending-token distinction.

Final refresh: the target-forward labels now read “Convolution / path-local filtering” and “Attention / tree mask + KV map.” The first is more precise than the earlier causal-gather label: target convolution performs path-local filtering, whereas the commitment stage gathers the selected history. The shorter attention label preserves the reviewed mask/addressing meaning. The bibliography balancing trigger changed from 27 to 24, and the user-requested working-revision footnote was removed. These refinements preserve the semantic PASS and change no numerical claim. The hashes below identify this final checked version.

## Where the novelty review enters the manuscript

| Review decision | Current manuscript integration |
|---|---|
| Frame a scoped systems/design contribution rather than invention of replay or tree verification | Introduction `main.tex:46,50`; explicit contribution boundary `:93–94`; conclusion `:283` |
| Attribute chain scheduling and selected-path replay to close prior work | Weaver and SpecLA `:63–65`, comparison table `:83–88`; method attribution `:151`; Snakes and Ladders/STree `:60` |
| Treat split-K, head reuse, and tree partitioning as established families | Flash-Decoding, FlashInfer, DeFT `:57`; mechanism description `:217–219` contains no invention claim |
| Distinguish the specific spine-contiguous KV intervention from general floating-point layout sensitivity | Scope `:94`; KV/mask-key mapping `:129`; determinism attribution `:217` |
| Attribute retrieval-plus-neural proposals | SuffixDecoding, Arctic, SAM `:120`; the evaluated geometry is separately specified in `:252` |
| Do not claim general numerical or external-system superiority | Bole/TreeWY scope `:67–73`; alternatives `:198,272`; component/workload limits `:276,283,292–294` |
| Make added experiments conditional on the desired claim, and use workloads for application benefit | Matched workload protocol `:298`; conditional verifier comparisons and current logical-equivalent layout intervention `:300`; detailed N1–N3 conditions in `review-experiments.md:5–11` and novelty synthesis `:80–92` |

Not every search-log entry belongs in the main text. The manuscript carries the material overlap, narrowed claims, and evidence conditions; the detailed chronology, search coverage, and unverified-public-priority boundary appropriately remain in the supporting novelty report. The added figure does not weaken those limitations. No obsolete route or non-workload rate was reintroduced by this figure change.

## Checked identities

Paths are relative to `papers/gdn-tree-scan-mlsys/v2/`.

| File | SHA-256 |
|---|---|
| `main.tex` | `7fd3e3966a92e462122fc3700e39100c5abb766a663da1fcfd61253e0066eaf5` |
| `figures/pipeline.tikz` | `4eaed23cd58ff06d51a72b8a169d8dd1be13f0570b157c2f9d69d104d140fe5c` |
| `abstract.tex` | `5bd67fda58efeb062a2f3fb13abb266b7d960c321f14312bc214de57f971f101` |
| `ref.bib` | `2ad620c7496f093482d9beb43b123c5996a50784c72e48464cbcd8cb297ece8e` |
| `notes/novelty/novelty-review-2026-09-24.md` | `eaf464309d8ff89efffb2550a375178500dcdc030213aba143b91a141bcc68a8` |
| `review-experiments.md` | `14c4eee5d97fc88917c599cb13497e3412a58528b29518d4ffd380318af5850b` |
