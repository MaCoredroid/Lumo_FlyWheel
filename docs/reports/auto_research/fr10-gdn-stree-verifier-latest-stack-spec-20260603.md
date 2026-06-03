# FR10 GDN/STree Verifier Spec: Lossless Tree Verification on Latest Stack

Date: 2026-06-03

Status: proposed implementation track

Primary objective: build a lossless and efficient single-request tree verifier for
Qwen3.6-27B's hybrid attention + Gated DeltaNet path. This replaces the failed
independent-row spine-2 route with a real recurrent tree kernel.

## 1. Executive Decision

We should start a new FR10 track around a custom GDN tree-verification kernel on
a current NVIDIA/vLLM stack.

The previous spine-2 and token-tree attempts did not fail because the value of
branching is impossible. They failed because the runtime represented branches as
either co-resident sibling rows or a packed token tree without a tree-aware GDN
state transition. For Qwen3.6 Gated DeltaNet, that makes branch verification a
recurrent-state problem, not just a tree-attention-mask problem.

The new track should therefore be:

1. Upgrade the container/runtime first.
2. Prove the Gated DeltaNet tree algebra against a serial per-path reference.
3. Build a one-layer Triton/CUDA kernel microbench outside vLLM.
4. Integrate the kernel as a single-request token-tree verifier in vLLM.
5. Only then tune MTP/tree/suffix policy for speed.

Do not revive `unsafe_best_of_spines`, independent hidden-row winner selection,
or selector-off public claims. Those were useful diagnostics, but they are not
the implementation path.

## 2. What Main Now Says

After pulling `main@59dc8b6b`, the closeouts establish the following:

- `fr9-spine2-lossless-winner-closeout-20260602.md`: independent public
  `spines>1` is fail-closed. Lossy best-of-spines was deleted, and selector-off
  spine-2 was not proven cleanly lossless against spine-1.
- `fr9-spine2-copy-based-selectoroff-closeout-20260603.md`: copy-based
  selector-off spine-2 is closed as borderline/not cleanly lossless. The report
  also adds the key revision: a no-copy tree-delta kernel for Gated DeltaNet is
  not published as a ready-made solution, but it appears derivable.
- `fr9-isolated-forward-p0-20260603.md`: a pure vLLM-0.19 isolated-forward probe
  reached the right diagnostic hooks but failed as a production primitive because
  scratch KV/state was not scheduler-owned.
- `fr9-p0-gdn-state-isolation-feasibility-20260603.md`: vLLM has internal
  scheduler-owned Mamba/GDN state-copy paths, but not arbitrary snapshot/restore
  plus batch-1 hidden forward for branch verification.
- `fr7-tokentree-gdn-superset-ceiling-20260531.md`: packed token-tree
  verification without a recurrent tree kernel contaminated path0 and destroyed
  acceptance. A tree attention mask alone is insufficient for hybrid GDN.

Net: latest `main` leaves production at lossless `spines=1`, while pointing the
next serious spine/tree attempt at a custom single-request GDN tree verifier.

## 3. Research Grounding

STree, "Speculative Tree Decoding for Hybrid State-Space Models"
([arXiv:2505.14969](https://arxiv.org/abs/2505.14969)), is the right external
reference. It states that existing SSM speculative decoding lacked an efficient
way to compute token trees, and proposes a scalable tree-based speculative
decoding algorithm for SSMs and hybrid SSM/Transformer models. The paper's
mechanism exploits accumulated state transition structure and uses a
hardware-aware tree implementation.

The important distinction for us:

- STree validates the architectural direction: recurrent models need tree-aware
  state computation, not only tree attention.
- STree does not directly ship our Qwen3.6 kernel. Qwen3.6's Gated DeltaNet path
  has a delta-rule update with key/value factors and gates; it is not just a
  diagonal SSM transition.
- The closeout addendum in `3b217415` argues that the Gated DeltaNet tree update
  is still derivable because the within-chunk delta operator is causal. Appending
  a leaf should not mutate trunk rows; the verifier can compute trunk factors
  once and extend each branch from the correct parent state.

vLLM batch-invariance documentation
([stable docs](https://docs.vllm.ai/en/stable/features/batch_invariance/)) also
supports our caution. vLLM documents batch-invariant operation for tested dense
attention models, but our local audits found the GDN/Mamba backend path does not
give the needed invariant hidden recurrent branch behavior. So the plan should
not rely on "upgrade vLLM and the issue disappears." The upgrade is for newer
CUDA/PyTorch/vLLM infrastructure, not for free branch-safe GDN semantics.

NVIDIA's current container release notes show the stack is moving beyond the
repo's v0.19/CUDA-13.0 anchor. The CUDA DL 26.05 notes say release 26.05 is based
on CUDA 13.2.1, and the vLLM container release notes describe monthly NVIDIA
vLLM containers with tested upstream contributions. We should build on the
newest available NVIDIA vLLM or PyTorch/CUDA image, while retaining a source
checkout for patching custom GDN kernels.

## 4. Core Correctness Model

For attention layers, tree verification can share prefix KV and use a tree mask
or tree-position metadata so each speculative node attends only to its ancestors.

For Gated DeltaNet/GDN layers, every tree node must also receive the recurrent
state produced by its own parent path:

```text
root state
  -> a1 state
      -> a2 state
          -> a3 state
      -> b2 state
  -> b1 state
      -> c2 state
```

The verifier is lossless only if the packed tree produces the same logits and
post-token recurrent state as serially evaluating each path from the committed
prefix. A hidden sibling must not perturb the public path by sharing a mutable
row, batch reduction shape, or incorrectly reused recurrent state.

FR10 invariant:

```text
packed_tree_gdn(prefix, tree).node_state[node]
  == serial_gdn(prefix + ancestors(node)).state_after(node)
```

within a specified numerical tolerance, for every GDN layer and every node in
the tree.

For greedy runs, the public path0 output must be byte-exact against native
spine-1/non-tree decoding. For sampled runs, the acceptance/rejection sampler
must preserve the target distribution; no longest-accepted hidden winner may be
published.

## 5. Proposed Gated DeltaNet Tree Kernel

The kernel should compute a tree of recurrent states in one verifier pass while
sharing trunk work.

Working hypothesis from the closeout addendum:

- The within-chunk DeltaNet solve is causal/lower-triangular.
- Trunk rows and their transformed factors can be computed once.
- A branch leaf can extend from its parent/trunk state without replaying all
  earlier sibling rows.
- Scalar gate products along the tree can be represented as ancestor-path
  products, analogous to STree's accumulated transition matrix.

Implementation shape:

1. Input a fixed-shape tree descriptor:
   `node_id`, `parent_id`, `depth`, `sibling_index`, `token_id`,
   `ancestor_mask_or_offsets`, and per-node position.
2. Gather each node's parent recurrent state from committed prefix state or a
   previous tree node.
3. Compute GDN per-node projections and delta factors.
4. Reuse trunk transformed factors for branch leaves where the parent path is
   shared.
5. Write one private recurrent state per tree node.
6. Return logits for all nodes to the verifier.
7. Commit only the accepted path's final state back to the real request state.

The first kernel does not need to support arbitrary dynamic trees. It should
support a small fixed family:

- spine depth 1-6,
- shallow branch width 2-3,
- total nodes 2, 3, 6, 8, 14,
- static padded descriptors for CUDA graph capture.

## 6. Latest Stack Plan

Start from the existing `docker/Dockerfile.nvidia-vllm`, but update the stack
selection before implementing the kernel:

- Prefer the newest available NVIDIA vLLM container if it boots Qwen3.6 FP8/MTP.
- Otherwise use the newest NVIDIA PyTorch/CUDA DL devel image and install vLLM
  from source at a pinned release or commit.
- Keep a vLLM source checkout in the image for patching model runner metadata,
  tree descriptors, and GDN backend hooks.
- Record exact CUDA, driver, PyTorch, Triton, vLLM, FlashAttention, and model
  revision in every run artifact.

Acceptance for the stack bootstrap:

- Qwen3.6-27B FP8 boots.
- Current spine-1/MTP baseline reproduces expected lossless behavior.
- CUDA graph mode and any downgrades are logged explicitly.
- GDN backend capture status is audited on the new vLLM source.
- The old vLLM-0.19 clone-collapse/sibling-row route is not carried forward.

## 7. Implementation Phases

### Phase 0: Audit And Freeze Baseline

Run current lossless `spines=1` and E3/E5 baselines on the pulled main branch.
Preserve the raw traces. These are not speed targets yet; they are the reference
streams for correctness.

Required artifacts:

- exact prompts/tasks used,
- greedy token streams,
- sampled task outcomes,
- per-event accepted-token counters,
- engine-step latency,
- CUDA graph mode/capture status,
- kernel-level Nsight Systems/Nsight Compute traces for one representative E5
  and one FR9-style tree/spine run.

### Phase 1: Algebra Reference

Write a pure PyTorch reference for Gated DeltaNet tree evaluation:

- serial per-path evaluator,
- packed-tree evaluator,
- random small-tree generator,
- trunk-sharing evaluator,
- tolerance tests across dtype modes.

Pass condition:

- Every tree node state/logit matches the serial per-path reference.
- Appending a sibling leaf does not change any trunk node.
- The accepted path final state equals serial native decode for the same token
  path.

This is the most important correctness gate. No CUDA work should begin until it
passes on synthetic and captured real GDN tensors.

### Phase 2: One-Layer Kernel Microbench

Build the first Triton/CUDA kernel outside vLLM:

- one GDN layer,
- fixed hidden size/config from Qwen3.6,
- static tree descriptor,
- trunk + shallow branch layouts,
- comparison against per-node serial/copy baseline.

Initial performance target:

- beat the old per-node tree path that closed at roughly 243 ms/event and 6.4
  TPS versus E3's 16.7 TPS,
- demonstrate that one extra branch row is closer to "incremental leaf cost"
  than "replay the whole trunk",
- show cost by depth: branch at root, branch after token 1, branch after token 2.

The output of this phase should answer the user's recurring cost question:
"what is the marginal cost to verify one more row at a given branch depth?"

The measurement table should include:

```text
tree_shape | branch_depth | nodes | shared_trunk_tokens | new_leaf_tokens
kernel_us | memory_bytes | state_reads | state_writes | equivalent_serial_us
```

### Phase 3: CUDA Graph Capture Probe

Before full vLLM integration, prove the kernel can run in a capture-friendly
fixed-shape loop:

- no allocation inside active capture,
- all descriptor tensors preallocated,
- no cache-miss path during capture,
- static output buffers,
- fail-closed if a requested tree shape was not warmed/primed,
- graph replay produces identical outputs to eager for the same buffers.

Target: full capture for the custom GDN tree kernel and fixed scaffolding.

If vLLM still downgrades the whole model to piecewise because another backend is
unsafe, that must be logged separately. The GDN tree kernel itself should not be
the reason capture fails.

### Phase 4: vLLM Integration

Integrate as a single parent request with a token-tree verifier:

- use tree attention for attention layers,
- use the custom GDN tree kernel for Gated DeltaNet layers,
- keep all branch states private until accept/commit,
- commit only the accepted path final recurrent state and KV suffix,
- do not create sibling scheduler requests,
- do not depend on hidden row co-scheduling,
- keep tree descriptors fixed/padded for graph capture.

The verifier API should expose:

```text
public_path0_tokens
candidate_tree_tokens
per_node_logits
per_node_accept_decisions
accepted_path
committed_state_source_node
suppressed_superset_events
capture_mode
```

### Phase 5: Lossless Gates

Losslessness gates must precede speed tuning:

- Gate A: unsafe modes deleted/fail-closed. `unsafe_best_of_spines` remains
  unavailable in production configs.
- Gate B: greedy equality. Native E5/spine-1 and tree path0 produce identical
  tokens over fixed prompts and B=1/B=4 scheduling.
- Gate C: distribution gate. Sampled multi-draft/tree selection matches target
  distribution tests; no longest-accepted order-statistic selector.
- Gate D: serial parity. Every packed-tree node matches serial per-path GDN
  state/logit within tolerance.
- Gate E: agentic harness gate. Real tasks show no quality regression versus the
  accepted lossless baseline before any speed claim.

`tests/test_lossless_selector_gate_c_stub_design.py` is the right negative-control
shape for Gate C: a max/longest accepted hidden winner should fail.

### Phase 6: Speed Tuning

Only after Gates A-E pass:

- tune tree shapes,
- combine MTP-1/2 anchors with suffix decoding,
- use tiny branch budgets where the kernel proves cheap,
- mine harness traces for suffix cold-start patterns,
- compare against current E3/E5 and spine-1 baselines.

The expected policy direction remains narrow:

- MTP depth 1-2 anchors,
- suffix extension for cheap extra proposal length,
- branch budget 2-6 nodes first,
- 14-node trees only if marginal leaf cost is proven small.

## 8. Why This Could Beat E3

The old static 14-node verifier was too expensive because it paid for a large
expanded tree while branch acceptance value was low. The proposed tree kernel
changes the cost model:

- shared trunk GDN work is computed once,
- a later branch can reuse/copy the parent recurrent state,
- attention shares prefix KV and only appends branch suffix KV,
- fixed tree descriptors can be captured,
- suffix decoding can provide cheap longer candidates while MTP anchors the
  high-confidence first tokens.

The strategy is not "make 14 nodes cheap enough and hope." It is:

1. make one extra branch row cheap at a measured depth,
2. use suffix proposals to create more candidate length without expensive MTP
   depth,
3. let MTP-1/2 trim or rank the suffix tree,
4. scale branch width only where the kernel-level marginal-cost table justifies
   it.

## 9. Risks

- The Gated DeltaNet algebra may not match the closeout hypothesis for all
  model-layer details.
- Numerical drift may be acceptable for per-node state parity but still flip
  greedy logits at low margins.
- Latest vLLM may change internal model-runner hooks enough to slow integration.
- CUDA graph capture can still fail in surrounding vLLM scaffolding even if the
  custom kernel is capture-safe.
- A correct kernel may still be too slow if memory traffic dominates the leaf
  update.
- Suffix decoding may improve candidate length but not task quality if the
  harness patterns are too diverse.

Each risk has a stop gate. Do not proceed from algebra to CUDA, CUDA to vLLM, or
lossless to speed if the prior gate is ambiguous.

## 10. Deliverables

1. `docs/reports/auto_research/fr10-gdn-tree-algebra-proof-YYYYMMDD.md`
2. PyTorch serial-vs-packed tree reference tests.
3. One-layer Triton/CUDA GDN tree microbench.
4. Latest-stack Docker update or new Dockerfile variant.
5. CUDA graph capture probe for fixed tree descriptors.
6. vLLM integration design doc with exact hook points from the selected version.
7. Lossless gate suite updates.
8. Kernel-level cost report comparing:
   - E3,
   - E5/spine-1,
   - old per-node tree verifier,
   - new GDN tree kernel with 2/3/6/8/14-node shapes.

## 11. Stop/Go Criteria

Go to implementation only if:

- latest stack boots Qwen3.6 FP8/MTP,
- algebra reference proves packed-tree parity,
- one-layer kernel shows real marginal branch savings,
- capture probe is hit-only/fixed-buffer/fail-closed,
- lossless gates are defined before policy tuning.

Stop or narrow scope if:

- path0 differs from native decode,
- appending a sibling changes trunk state,
- kernel speed is near serial replay,
- capture requires dynamic allocation/cache misses,
- sampled selector behavior cannot pass the distribution gate.

## 12. Bottom Line

Yes, we should write the new GDN kernel if we want efficient and lossless tree
verification. The old approaches tried to encode branches in scheduler rows or
attention masks; Qwen3.6 needs the recurrent backend itself to understand the
tree. STree gives the right architecture, the latest closeout says the Gated
DeltaNet extension is derivable, and the newer NVIDIA/vLLM stack should be the
base for building it. The implementation order must be lossless first, kernel
inspection second, speedups third.
