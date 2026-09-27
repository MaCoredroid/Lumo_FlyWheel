# Second literature and mechanism-comparison pass

Reviewed 26 September 2026. Requested scope: reassess branch-local computation and on-chip state reuse, compare the implemented mechanism with nearby literature, and recommend a paper name. This is a research and naming memo; the manuscript, PDF, experiments, and title were not changed. No inference or external author benchmarks were run.

## Recommendation

Use **LumoTree** as the system name, with the descriptive title:

> **LumoTree: Path-Parallel Speculative Verification for Hybrid Language Models**

The paper should center on path execution, state lifetime, and continuation through the hybrid model. Attention layout is one supporting optimization. The system name identifies the artifact; the subtitle explains its execution strategy without implying that the scheduling family was invented here.

The evidence supports an implemented system with a particular verification/commit policy and measured deployment observations. It does not yet isolate a new general path-scheduling algorithm or establish that our memory/numerical tradeoff dominates the alternatives. This is a claim boundary, not a reason to replace the design paper with an audit paper.

## What changed in this pass

1. **OneLA is an additional adjacent comparison.** Its September paper was absent from the previous source register. It matters for shared recurrent state and branch ancestry, although beam decoding and speculative acceptance are different workloads.
2. **FastTree strengthens the attention-side comparison.** Its grouping of queries and contexts explicitly targets on-chip reuse. It does not supply a GDN verifier.
3. **The TreeWY author implementation is now directly inspectable.** Its GDN component fuses prior accepted-state reconstruction with the next verification and supports tree masks. The paper's whole-model tree graph limitation must not be described as an inherent inability to capture a WY GDN kernel.
4. **Our allocated scratch and state traffic must be distinguished.** The active path kernel exports selected boundary states, but its preseeded export tensor reserves node-indexed capacity. Reduced writes are not automatically reduced allocation.
5. **The exact path scheduler has a more specific development date.** The tracked path kernel and heavy-path decomposition were added July 25, not at the June date of the initial GDN module. Broad module chronology cannot date every later mechanism.

## Source and implementation boundary

The current GDN kernel, topology file, and runtime patcher were hashed again and match the inputs in [the production audit](../../results/current-production/audit.json). The audit binds the reported deployment to source commit `e7af6b595a8b3a09b7e88d25a3653d381432b99c`; the working checkout HEAD is `ae3dfbe99ff1ec0590b5435c738f1df92e5e5053`. An enabled candidate in later source is not evidence that it served the reported workloads.

| Local evidence | What it establishes |
|---|---|
| [Path kernel](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:10923) | A program operates on one path, head, and value tile. It carries `state_i` across sequential updates rather than explicitly storing each intermediate state. |
| [Path source and export decisions](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:17270) | The deployed two-level route starts from the native base state, exports cut states, and launches terminal paths from those exports. |
| [Scratch allocation](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:6327) | The export tensor has shape `[n_actual, vh, dv, dk]` in FP32. The export mask limits writes; it does not compact the allocation to only cut nodes. This is reusable scratch, not a durable state for every speculative leaf. |
| [Native committer body](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:14551) | Accepted recorded operands drive native recurrent updates into running rows. The measured route captures per-layer calls; it is not one all-layer fused kernel. |
| [Deployment interpretation](../latest-production-design-evidence-2026-09-24.md) | Patched FA2, native publication, cache support, and the actual disabled single-launch candidate are distinguished from historical alternatives. |

The best precise description is **path-parallel verification with a state tile carried within each path and explicit inter-path handoffs**. Source-level register carriage is not a measured no-spill guarantee. On GB10, call exported storage GPU global memory rather than implying that its backing memory is HBM. Persistent request state, reusable global scratch, and on-chip temporaries have different lifetimes.

## Direct comparison by execution and state policy

| Work and inspected primary source | Verification and working state | Continuation policy | Implication for this paper |
|---|---|---|---|
| **Our active implementation** | Sequential GDN updates within paths; parallel ready paths/head/value tiles; explicit boundary exports | Immediate native replay from recorded operands; coordinated convolution and KV publication | Concrete combination and deployment. Main costs include boundary traffic, scratch allocation, replay, and surrounding hybrid-model work. |
| **SpecLA**, [§§4.1–4.3, 5, 8](https://arxiv.org/html/2607.16673v1) | Value-tiled resident recurrence; node-disjoint chains; ready-chain parallelism and boundary exports | Accepted factors are buffered and applied in a later fused update | Closest scheduling overlap. Its reported pure-GDN/H100 setting differs from our hybrid deployment; this is a scope difference, not evidence of our superiority. |
| **Trees from Marginals / Weaver**, [§3.4](https://arxiv.org/pdf/2607.06763v2) | Ancestor-masked triangular verification; committed state stays read-only during verification | Short recurrence over the accepted path; cached operands support commit | Separates the verifier arithmetic from the replay policy. Replay after a compact verifier is already a direct comparator. |
| **Bole**, [§§IV–VI](https://arxiv.org/html/2608.01651v1) | Shared tree factors and value-tiled finite-Neumann propagation; intermediate solve work stays on chip | Compact factors reconstruct the accepted state | Different arithmetic, but also on-chip reuse. Includes GB10 and coding-agent trace replay; neither hardware nor the agent-serving label is unique to us. |
| **TreeWY**, [§§3–4](https://arxiv.org/html/2608.20961v1) and pinned author source below | Ancestor-masked correction solve with compact pseudo-values; resident tiles in the fused kernel | Prior accepted state is reconstructed from its stash at the next forward in inspected code | A compact verifier need not pay a separate full state round trip for commit. Its published whole-model tree capture limitation concerns the integrated route. |
| **ReplaySSM**, [§§3–5 and Appendix A](https://dao-lab.ai/blog/2026/replayssm/) | Checkpoint plus buffered corrected inputs; GDN speculative windows use triangular correction algebra | Buffer acceptance/rollback and periodic state materialization | Another point in the materialization tradeoff; not a naive sequential-replay baseline. |
| **OneLA**, [§§3.2–3.4](https://arxiv.org/html/2609.12399v1) | Shared prompt state plus compact branch transition records; replays query/key projections without reconstructing every branch matrix; fused cross-beam state reuse | Dynamic beam ancestry references append-only records | Relevant adjacent work on branching state and local reuse. It is not a drop-in speculative acceptance or hybrid continuation implementation. |

Two comparison errors to avoid: our cached-operand replay does not rerun target projections/FFNs as SpecLA's expensive token-replay baseline does; and “replay” in an external system's name does not specify whether its verification uses recurrence, a solve, or projection reconstruction. Compare the executed operations and when they occur.

### TreeWY author-code corroboration

The author's [vLLM RFC](https://github.com/vllm-project/vllm/issues/54080) links the `sneha5gsm/vllm` branch `treewy-gdn-spec-decode`. The GitHub API resolved it to `b073ed6cacfa1cf5ae23111854729e662a72a6d1` (recorded commit date August 8, 2026). Read-only source inspection found:

- [`tree_wy_triton.py:751–880`](https://github.com/sneha5gsm/vllm/blob/b073ed6cacfa1cf5ae23111854729e662a72a6d1/vllm/third_party/flash_linear_attention/ops/tree_wy_triton.py#L751): tree-mask launcher and the fused computation it invokes. The per-program value-tile loop retains a bounded working tile, reuses value-independent work, and stores compact records.
- [`qwen_gdn_linear_attn.py:1396`](https://github.com/sneha5gsm/vllm/blob/b073ed6cacfa1cf5ae23111854729e662a72a6d1/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L1396): padded request slots, accepted-leaf descriptors, and the chain/tree dispatch for reconstruction plus verification.
- [`mamba_utils.py`](https://github.com/sneha5gsm/vllm/blob/b073ed6cacfa1cf5ae23111854729e662a72a6d1/vllm/v1/worker/mamba_utils.py): lazy-commit migration preserves the pending acceptance descriptor across state-block movement.

This is source corroboration, not a run of its tests or validation of its performance comments. A stale “launch 1/2” comment in the launcher must not override the actual call sequence. No SpecLA or Bole author kernel was verified in this pass; their papers establish mechanism overlap, not executed parity.

## Earlier foundations and adjacent systems

| Primary source revisited or added | Relevant boundary |
|---|---|
| [The Mamba in the Llama, §4.2](https://proceedings.neurips.cc/paper_files/paper/2024/file/723933067ad315269b620bc0d2c05cba-Paper-Conference.pdf) | Multi-step recurrent speculation already avoids materializing most intermediate states and advances a cached state lazily. It predates GDN-specific tree methods. |
| [STree, §3](https://arxiv.org/html/2505.14969v1) | Tree scan, fast-memory intermediates, and activation replay for hybrid SSMs. Its transition structure differs from GDN's delta update. |
| [DeFT, §3 and Appendix A.8](https://arxiv.org/html/2404.00242v2) | Tree-aware KV grouping, flattening, and partitioning address attention IO; they do not implement recurrent GDN handoffs. |
| [FastTree, §§4–5](https://proceedings.mlsys.org/paper_files/paper/2025/file/96894468eb44631a32d7ebd56f9892c7-Paper-Conference.pdf) | Query/context grouping, shared-memory KV reuse, and topology-adaptive work division. Important for claims about generic branch-local GPU reuse on the attention side. |
| [Thinking Machines, batch-invariant attention](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | Reduction boundaries, consistent KV ordering, and fixed split sizes affect numerical invariance. Our spine mapping is a specific intervention, not discovery of floating-point sensitivity. |

[HLX](https://doi.org/10.1145/3725843.3756115) and [Pimba](https://arxiv.org/abs/2507.10178) were screened at the primary abstract/project-description level as adjacent hybrid-model hardware work. They were not promoted to direct software-verifier baselines. Earlier drafting and sampling references remain in the September 24 review; this pass does not pretend to reread every background citation.

## What could distinguish the system, and what is established

| Possible framing | Assessment now | Evidence needed for a stronger claim |
|---|---|---|
| “We invent branch-local state caching in SRAM” | Too broad; the direct comparison already contains that family. | A materially different execution/storage mechanism, not a renamed path or cache. |
| **Path-parallel verification with native continuation in a hybrid serving engine** | Best current center. Source and deployment records support its realization; describing a system does not require claiming every component as new. | To claim an advantage, isolate the chosen verification/commit policy under the actual route. |
| Numerical and cost tradeoffs of sequential versus compact verification | Strong potential study, but current evidence does not establish a general winner. | Same captured current-route operands, comparable precision choices, candidate outputs and accepted states, and full verification-plus-commit costs. |
| Spine-contiguous physical layout | Supporting mechanism, potentially a focused result. | Current patched-FA2 intervention that changes physical placement while preserving logical ancestry, plus downstream checks. Old Triton witnesses do not qualify this route. |
| Better long-running agent serving | Relevant deployment evidence; not an automatic causal claim about the GDN kernel. | Task outcomes and full agent time alongside pooled decode rates; repeated declared cohorts for generalization. |

An analytical distinction worth measuring is **resident recurrence versus compact-factor execution**. Our path program carries roughly one value-by-key state tile, while its sequential work grows with path length. Factor methods pay setup and different intermediate/reduction costs to expose node parallelism. Hardware, tree geometry, state dimensions, precision, and commit timing decide which tradeoff wins. “Uses SRAM” is not enough to predict the outcome.

Likewise, algebraic equivalence does not imply floating-point identity, and floating-point discrepancy does not establish an intrinsic mathematical defect. Previously recorded local experiments can be reused only if they bind the current operands, implementation, and measurement definitions relevant to the new claim.

## Smallest useful next comparison, if we strengthen the mechanism claim

No experiments were launched. First reuse any current-source receipts already containing the required evidence.

1. **Account for the current kernel's actual residency and traffic.** Record compiled register/shared-memory usage and spills, touched boundary bytes, allocated scratch, and verification plus native-commit time. This turns a source-level residency description into a measured hardware statement.
2. **Change one execution policy at a time.** Compare current path recurrence/native replay with a qualified compact author implementation. The TreeWY source above supplies a concrete candidate; Weaver is another. Include accepted-state work and metadata. A SpecLA-family arm must isolate a real difference such as delayed commit; rerunning an equivalent chain scheduler cannot establish a new scheduling idea.
3. **Only after a useful difference appears, confirm it in the existing agent workload.** Complete systems may tune their own MTP depth and serving settings fairly. For causal kernel ablations, hold the relevant inputs fixed. These are complementary comparisons.

Stop extending a novelty claim if the proposed distinction reduces to an existing mechanism. Stop a performance comparison if an arm fails its numerical/continuation contract. A kernel win that vanishes after commit or full-model costs does not justify an application-speed claim.

## Chronology check

`git log -S 'def _tree_gdn_path_kernel'` and `git log -S 'def _subtree_decompose'` identify `bfd129089` on **July 25, 2026** as the tracked introduction of the register-carried path kernel and heavy-path decomposition. Commit `4a04999a2` on **July 31** batches the fixed32 schedule into two launches. The earlier June module/replay history remains valid development history, but does not date this exact path scheduler. SpecLA's arXiv v1 is [July 18](https://arxiv.org/abs/2607.16673v1).

These are the records inspected, not a universal priority ruling: renamed functions, untracked work, and public deployment history are separate evidence questions. Nothing here infers copying.

## Naming decision

**Prefer a memorable system name plus a factual mechanism subtitle.** No forced acronym expansion is needed.

| Candidate | Recommendation |
|---|---|
| **LumoTree: Path-Parallel Speculative Verification for Hybrid Language Models** | Preferred. Keeps the project identity, puts execution at the center, and covers the full verifier/continuation system. |
| **LumoTree: State-Resident Tree Verification for Hybrid Language Models** | Good memory-focused alternative. Define residency as within-path tile lifetime; avoid suggesting all branch state persists in SRAM. |
| **BranchScan: Tree Verification for Hybrid Language Models** | Mechanism-only alternative. Less connection to the existing project, and “scan” may suggest an associative parallel scan rather than sequential path updates. |

Avoid **Lumo** alone: it is already an AI-assistant name used by [Proton](https://proton.me/lumo), which weakens searchability. Avoid **PathSpec**: it is already used by a [speculative decoding paper](https://arxiv.org/abs/2606.10492). LumoTree searches did not locate a same-field research system, but did find unrelated commercial uses; this is a bounded searchability check, not a claim that the word is unused. LumoScan also has existing unrelated uses. “Layout-aware” would make one supporting detail carry too much of the paper's identity.

The paper can introduce the system as:

> We present LumoTree, a speculative verifier for hybrid language models that executes recurrent paths in parallel, carries state tiles across updates within each path, and publishes the accepted continuation through the serving engine's native state interface.

This is proposed wording for discussion. The current manuscript title has not been renamed.

## Reproducibility of this review

See [the review ledger](mechanism-comparison-2026-09-26.json) for search strings, source-read scope, code pins, hashes, and preserved manuscript hashes. This pass expands the September 24 review; it is not a guarantee of exhaustive coverage of all unpublished or unindexed work.
