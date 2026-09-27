# Serving and optimization prior-art review — 2026-09-24

**Verdict: a defensible implementation/integration contribution, but the serving-optimization list is not yet a strong independent novelty argument.** Split-K attention, head-group reuse, fixed speculative trees, GPU acceptance, CUDA graphs, and neural-plus-suffix speculation have direct prior art. The most promising narrow distinction is coordinating a **spine-contiguous physical KV layout with unchanged logical query order, tree-mask columns and accepted-state publication in a recurrent hybrid**. I did not find an exact prior description of that remedy in the primary sources checked; that is a bounded search result, not proof of priority. Its current-route benefit still needs a controlled witness if it is to carry the paper's novelty claim.

This review used live primary papers, author documentation and code plus the deployed-route inventory. No manuscript or experimental source was edited and no inference ran. GDN algebra/reconstruction prior art is being reviewed separately; this report does not re-open superseded Cat10 results or import old performance estimates.

## Direct overlap and the remaining distinction

### 1. Spine-first KV/mask layout: potentially specific, not a new general numerical principle

Current `main.tex:120,208` and `scripts/fr13_patch_fa2_tree_bias.py:9394–9447,9637` describe the relevant distinction: permute physical KV writes and mask key columns; leave query rows/positions in logical order; apply the same mapping during accepted-path publication. This is stronger and more specific than merely saying “tree attention” or “layout affects rounding.”

[DeFT, §3.2–3.3/Figure 3 and Appendix A.8](https://arxiv.org/html/2404.00242v2) already organizes tree KV by depth-first flattening, splits it into balanced blocks and uses KV-guided masks/grouping to reuse cache loads across queries. It explicitly evaluates speculative trees. Its stated motivation is memory traffic/load balance. I did not find a claim about making a preferred spine numerically match a sequential attention layout. **Distinction:** Lumo's selected-key order is a numerical/layout constraint, whereas DeFT's flattening is an IO grouping strategy; both must be acknowledged before claiming a new tree-layout mechanism.

[Thinking Machines, “Batch-invariant attention,” 10 September 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) already explains that cache/current-token partition boundaries and split sizes change reduction order, and proposes consistent KV layout and reduction schedules. [SGLang's corresponding implementation discussion, 22 September 2025](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/) makes this serving concern explicit. Thus floating-point sensitivity to packing, stable reduction order, and a coherent cache layout are established. Lumo must not present the underlying numerical observation as newly discovered.

[LLM-42, §4.2–4.4](https://arxiv.org/html/2601.17768v2) separately uses fixed-shape verification and rollback to enforce deterministic outputs, replacing KV entries with verifier-produced state. It overlaps the “numerical reference plus state consistency” concern, but not the specific spine permutation or GDN native replay. Lumo's four-split credential is bounded-error evidence, not the batch invariance or deterministic-output guarantee pursued by these works.

**Defensible claim:** an implementation of coordinated physical KV/mask/publication maps for this verifier, motivated by reduction-layout sensitivity. **Not established:** first such layout, universal numerical stability, or end-to-end determinism. Retaining the physical principal spine contiguously does not by itself prove that off-spine paths have sequential-reference arithmetic.

### 2. GQA-pair split-K FA2: established strategy with a specialized integration

[Flash-Decoding, “A faster attention for decoding,” 13 October 2023](https://pytorch.org/blog/flash-decoding/) already splits KV into chunks, computes partial attention and log-sum-exp in parallel, and rescales/combines the outputs. That is the algorithm in `main.tex:210`, not merely a remotely related technique. [FlashInfer's February 2024 author description](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html) explicitly applies split-KV to decode and append attention.

[FlashInfer, Appendix A](https://arxiv.org/html/2501.01005v1) explicitly fuses query heads with query rows so a shared-memory KV load serves all query heads of a group; it also identifies TensorRT-LLM XQA as related work. Sharing tiles between two query heads is a specialization of this established reuse principle. Choosing four partitions is a deployment parameter, not a new algorithm.

The narrower implementation work is the fixed32 tree-bias/slot layout, GQA-pair kernel at the actual model geometry, deterministic bounded-error admission, and binary-bound deployment receipts. These support “we implement and qualify,” not “we introduce grouped split-K attention.” If the paper claims a novel or superior attention kernel, compare the specialized kernel with an eligible existing split-K/head-group implementation and account for tree-mask support. If it makes no kernel-superiority claim, citing/positioning the prior art is sufficient; porting every external library is unnecessary.

### 3. Static tree/device acceptance/graphs: useful composition, not first principles

[Sequoia, §2–4](https://arxiv.org/html/2402.12374v1) predates this work with hardware-aware tree shapes, tree sampling/verification and an implementation using CUDA Graphs. It is already cited, but currently positioned mainly as tree construction. [FlashInfer, §3.4/Listing1](https://arxiv.org/html/2501.01005v1) describes reusable workspaces, planning separate from graph-captured execution and runtime graph selection. Static buffers and captured layer sequences are established serving practice.

GPU acceptance also has direct implementation prior art: [SGLang's June 2025 kernel](https://github.com/sgl-project/sglang/blob/cfceb83d057624dd8a8da5fd29e394ccb3dfe068/sgl-kernel/csrc/speculative/speculative_sampling.cuh#L30-L96) has a CUDA tree walk consuming child/sibling tables and emitting accepted token indices/counts. That kernel uses its own target-only rule; this is overlap in device execution and products, **not a claim that its sampling law equals Lumo's**. [FlashInfer's chain sampler](https://docs.flashinfer.ai/generated/flashinfer.sampling.chain_speculative_sampling.html) is another clear precedent for fused GPU speculative verification.

Lumo's narrower integration is the five fixed device products and their common use by convolution publication, native recurrent replay, attention remapping and next-drafter selection (`fr13_device_multidraft_kernel.py:6489ff`; patcher `:19980ff`). Capturing 48 native layer updates reduces orchestration but is neither a new CUDA-graph technique nor a single all-layer kernel. The scientific contribution would be showing why this particular shared continuation interface resolves a nontrivial hybrid-state failure and what cost it saves on the deployed route.

### 4. Single logits and fused argmax/top-three: a concrete compatibility kernel

Single-logits reuse removes a duplicate projection in this implementation; common-subexpression reuse is not a new drafting algorithm. Fused selection also belongs to an established optimization family: [FlashInfer's March 2025 sampling implementation](https://flashinfer.ai/2025/03/10/sampling.html) avoids sorting/multiple passes for GPU sampling, though it does not promise Lumo's exact pair of output tensors.

The concrete distinguishing detail is in `csrc/fr14_dfwd_full_topk.cu:19–55`: select one top-three set while separately reproducing the pinned argmax and top-k tie order, emitting both spine and ordered candidate outputs. [PyTorch documents first-maximum argmax](https://docs.pytorch.org/docs/2.9/generated/torch.argmax.html), while [top-k tied indices are not a stable API guarantee](https://docs.pytorch.org/docs/2.9/generated/torch.topk.html). Therefore this is a **pinned implementation-compatibility result**, not a general PyTorch top-k theorem. The 6,840-configuration gate is appropriate evidence for that bounded claim.

I did not locate a prior kernel with exactly this dual-output, pinned tie policy. That can be reported as an implementation contribution; an argmax/top-three specialization and index-augmented ordering are too narrow to establish a new general selection algorithm. A claimed workload benefit still needs the same-route replacement ablation; component parity alone does not establish it.

### 5. MTP plus Arctic suffix candidates: hybrid proposals already exist

[SuffixDecoding, §3](https://arxiv.org/html/2411.04975v1) already builds tree proposals from current-request and historical suffix indices. Lumo directly imports Arctic's `SuffixDecodingCache` (`scripts/fr13_merged_drafter.py:94–108`) and adapts its proposals (`:965ff`); the original method deserves explicit attribution.

[Arctic's own hybrid documentation](https://arcticinference.readthedocs.io/en/latest/suffix-decoding.html#combining-with-arctic-speculator) combines suffix and model-based speculation by selecting between sources per iteration. The same documentation is present at [a 2025 source revision](https://github.com/snowflakedb/ArcticInference/blob/8b1d693136c1b987a3b7dcd25056f5a669394910/docs/suffix-decoding.rst#L40-L49), and its [May 2025 author post](https://www.snowflake.com/en/blog/engineering/fast-speculative-decoding-vllm-arctic/) discusses agent workloads. Independently, [SAM Decoding](https://aclanthology.org/2025.acl-long.595/) integrates suffix-automaton retrieval with EAGLE-2. General neural-plus-retrieval speculation is not new.

The narrower Lumo distinction is **composition within one fixed tree**: an MTP head plus suffix tail/rescue paths under one Hydra27 descriptor, rather than merely choosing a neural or suffix proposer for the whole iteration. The exact 27-node shape is an engineering choice, not by itself an algorithmic contribution. If proposal composition becomes a contribution, its incremental benefit needs a matched-budget neural-only or source-selection comparison and proposal/accepted-node provenance. The current record establishes engagement, not that this splice is better than prior hybrid policies.

## Chronology: history is not public priority

Read-only `git log --all --reverse` locates the following local development records. These are author/committer timestamps and content identities; no claim is made that the commit was publicly accessible on that day.

| Mechanism | Local first relevant record inspected |
|---|---|
| FA2 tree-bias patcher | `29d3c8bd1`, 7 June 2026 |
| Single-logits reuse | `d407e5450`, 12 June 2026 |
| Spine-first KV/mask slot permutation | `6c11a359d`, 13 July 2026; preceding dense-suffix experiment `380749096` |
| MTP/suffix orchestration | `957e97df4`, 14 July 2026 |
| Fixed32 runtime | `39f78869f`, 30 July 2026 |
| GQA-pair split-K | `a9aa2a859`, 18 August 2026 |
| Fused full-vocabulary top-three | `ff9478266`, 18 August 2026 |

On 24 September 2026, GitHub's repository API reports `MaCoredroid/Lumo_FlyWheel` public, created 15 April 2026, with a 24 September push. Creation date, present visibility, and embedded commit dates do **not** establish when the repository or a specific mechanism became public. First/public-priority wording would need an independently dated public release, archived page, DOI, preprint or public discussion that actually discloses the mechanism. The 2023–2025 sources above are prior art regardless of the uncertain Lumo disclosure date. This review does not compare private local authorship timestamps against a later paper's publication date to infer priority.

## Minimal actions and evidence required by claim

1. **Position the implemented optimizations explicitly.** Add Flash-Decoding, FlashInfer, DeFT and SuffixDecoding/Arctic to the relevant method paragraphs. Numerical-layout discussion should cite the batch-invariance work and distinguish the narrower tree-layout goal. Sequoia's graph/static-tree overlap should be visible. These are attribution/positioning repairs, not reasons to launch a broad benchmark campaign.
2. **If spine layout is the novelty anchor:** show a same-input witness on the current fixed32 FA2 route. Compare logical-equivalent key layouts with consistent masks/remaps and fixed kernel settings, reference both the preferred path and off-spine paths, and separate arithmetic error from address errors. Preserve counterexamples where the permutation fails to restore reference behavior. A kernel witness can establish the mechanism; it cannot be promoted into a coding-agent speedup. Current-route evidence is needed rather than obsolete-route diagnostics.
3. **If the paper claims optimization benefit:** select the mechanism carrying that claim and run a qualified, matched same-route workload ablation, with complete attempted-task outcomes and the common pooled metric. It is unnecessary to ablate every engineering detail merely to describe it. A native-MTP workload comparison would support the value of the composed tree system; it would not identify the individual optimization's effect.
4. **If claiming new proposal composition or attention superiority:** use the narrow comparisons described above. These are conditional on those stronger claims. A 6,840-case selection gate or a bounded attention credential establishes eligibility, not superiority or end-to-end equivalence.

**Recommended contribution framing:** a source-bound implementation of a fixed-shape speculative verifier for a recurrent hybrid, with coordinated logical/physical continuation maps and qualified optimizations, plus workload-specific system observations. Treat the layout remedy and dual-output selection kernel as explicit, limited technical details. The current evidence is a useful systems case study; it does not yet isolate a broadly novel serving optimization or establish general superiority over the cited systems.

## Reviewed identities and scope

- `main.tex`: `4793cefe3803d70e817031cb02b743d66c5c5d0e177ccd2c6356708f5d154894`.
- `ref.bib`: `88ceccadbd0376d694f6037181ebf0dacd71f0936558d6c88f856666ccc05fc8`.
- `notes/latest-optimization-supersession-2026-09-24.md`: `3d9aa321a3894d72c9990060086955d4e1511e57ee96fb1ce2870e0887c89ed9`.
- `scripts/fr13_merged_drafter.py`: `6b0e8581dda53cc9baa0eebae07f95d9058800beee3d75c80a3a2eaa2a3e69be`.
- `csrc/fr14_dfwd_full_topk.cu`: `dfdafaef6f7deca 2458c8e09aa682cbb002f688896766145cea81ee2b9582e0c`.

All external URLs above were read on 24 September 2026. The search was bounded to the named serving mechanisms and primary sources; an unsuccessful exact-match search is not an exhaustive novelty or patent search. No external speed ratio was used to rank the deployed workload result.
