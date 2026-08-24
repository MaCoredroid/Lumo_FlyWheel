# MAIN-DISH DESIGN PACKAGE — Hybrid Tree-Verify Spec Decode for vLLM
**Target:** vLLM `main` @ `23ab0cfdb` · **Status:** engineering document, supersedes the RFC's "Open Questions" framing on substrate · **Date:** 2026-08-24

---

## 0. Provenance and corrections to the RFC draft

**Re-verified in this pass** (via `git show main:<path>` from `/home/mark/shared/vllm-head`; worktree is on `feat/speculators-plural-proposal-methods` `728fc7793`, which differs from `main` only in `vllm/transformers_utils/configs/speculators/base.py` + one test):

| Claim | Result |
|---|---|
| `MambaSpecDecodeGPUContext` is V2-only | **FALSE.** Defined `vllm/v1/worker/mamba_utils.py:649`; imported by V1 at `gpu_model_runner.py:216` (wired `:1120`, `:1644`, `:1667`, `:2201`, `:4444`, `:4471`) and by V2 at `gpu/model_states/mamba_hybrid.py:25-26,102,138,146`. Shared. |
| `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` contains no hybrid arch | **TRUE.** 10 entries, all attention/MoE; `KimiLinearForCausalLM` and `MiniMaxM3Sparse*` appear only in `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`. Hybrid veto confirmed in `_is_default_v2_model_runner_model`. |
| AV budgeter docstring is the tree condition | **TRUE, verbatim:** *"Survival only decreases along a request, so a global top-k always admits continuously along steps with a request."* Impl: `confidence_probs[idx_mapping].cumprod(dim=1)` → `masked_fill(out_of_range, -inf)` → `topk(draft_budget)` → `sum(admitted, dim=1, out=capacities)`. |
| AV writes `query_start_loc` on device | **TRUE.** `adaptive_verification.py:418-434`, two `torch.cumsum(..., out=...)` into `_cu_num_logits` and `query_start_loc`. |
| `_rejection_kernel` has `tl.constexpr` mode axis | **TRUE.** `HAS_DRAFT_LOGITS`, `SYNTHETIC_MODE`, `USE_BLOCK_VERIFICATION` at `rejection_sampler_utils.py:536-539`; scalar out `tl.store(rejected_steps_ptr + req_idx, accepted_length)`. |
| `stride_indices_tok` dead-parameter bug | **TRUE.** Declared `fused_recurrent.py:52`, computed `:211-215`, passed `:243`, **never read in the kernel body** — `:110` and `:155` index bare `+ i_t`. Contrast `:296` (non-spec path) which drops the token term entirely. |
| FA2 rejects `mask_mod` | **TRUE.** `flash_attn_interface.py:313-314` `raise NotImplementedError("FA2 does not support mask_mod")`. |
| FA4 family gate excludes sm_121 | **TRUE.** `_is_fa4_supported()` `:72-86` requires family ∈ {9,10,11}; `121 // 10 == 12`. |
| `BaseSpeculator.propose() -> torch.Tensor`, `req_states.draft_tokens [max_num_reqs, K]` | **TRUE.** `gpu/spec_decode/speculator.py:42-69`; `gpu/states.py:72-77`. |
| `abstract.py` carry budget is already feature-dependent | **TRUE.** `MambaBase.get_kv_cache_spec` sets `num_speculative_blocks = 0 if use_kda_recoverssm else num_speculative_tokens`, with the comment *"RecoverSSM verifies the whole window off one checkpoint…"*. |

**New finding, load-bearing for §C:** `vllm/vllm_flash_attn/cute/` is **not tracked in the vLLM repo**. `git ls-tree main vllm/vllm_flash_attn/` returns exactly three blobs: `.gitkeep`, `__init__.py`, `flash_attn_interface.py`. The `cute/` directory (`mask.py`, `flash_fwd_sm100.py`, `compute_block_sparsity.py`, `softmax.py`, `interface.py`…) is populated by the `vllm-flash-attn` build. Consequence: the SM90 `Mask.apply_mask` mask_mod drop, the 5-point `fast_sampling` classifier, and the vectorized `__vec_size__ > 1` contract are all in a **different upstream repo**. The E-a PR can only *consume* the ABI; it cannot fix it. This is not a stylistic constraint — it is the reason the tree mask must be `causal=False` and fully self-contained, and it means the PR body must cite CuTe internals by **symbol name, not line number**.

**Three corrections the RFC must absorb before filing:**

1. Strike *"we target V2 because that is where recurrent spec-state lives."* It is false and the owning reviewer (author of #40172 / #42406 / #49291) will know it in ten seconds. The correct sentence is: *"the recurrent spec-state commit is substrate-shared (`mamba_utils.py:649`), so Phase 0 lands in shared code and benefits both runners regardless of substrate."* This makes the RFC's Phase 0 **stronger**, not weaker — it converts three interfaces from "V2 tax" into "shared infrastructure."
2. The RFC path itself is unresolved on this machine. `docs/vllm-upstream/RFC_hybrid_tree_lossless_spec_decode.md` exists only on branch **`vllm-upstream/rfc-draft`** (`3203bc09d`); the execution plan is `docs/vllm-upstream/EXECUTION_PLAN.md` (`d11e5fa71`). Three of four deep reads could not find it and wrote against the prompt text. Fix the canonical path in the campaign notes before the next pass.
3. Tree attention was **deleted** upstream (campaign commit `692c9e046`; `git grep -l "tree_attn\|TreeAttention\|spec_token_tree" main -- vllm/ tests/` → nothing). The RFC must open by saying *why a mask primitive is the right re-entry point*, not assume goodwill.

---

# A. SUBSTRATE DECISION MEMO — V1 vs Model Runner V2, phases 0–3

## A.1 Decision

**Target Model Runner V2 for phases 0–3.**

With one framing correction that is not a hedge: **phases 0 and 1 are substrate-neutral by construction.** Their insertion points are files both runners import. "Targeting V2" costs literally nothing there, and the RFC should stop presenting it as a choice. The choice becomes real only at **phases 2–3**, where the proposal buffer, the verification kernel, and the commit live in runner-specific files — and there the evidence is one-sided.

Concretely, of the 12 production files the design touches across phases 0–1:

| Shared by both runners | V1-only | V2-only |
|---|---|---|
| `vllm/v1/worker/mamba_utils.py` · `model_executor/layers/mamba/abstract.py` · `layers/mamba/ops/mamba_ssm.py` · `layers/mamba/ops/causal_conv1d.py` · `third_party/flash_linear_attention/ops/fused_recurrent.py` · `csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu` · `v1/kv_cache_interface.py` · `v1/attention/backends/gdn_attn.py` · `v1/attention/backend.py` · `v1/attention/backends/flash_attn.py` · `v1/attention/backends/flex_attention.py` · `v1/attention/backends/utils.py` · `v1/attention/backends/recoverssm_metadata.py` · `v1/core/sched/{scheduler,output}.py` | `v1/spec_decode/metadata.py` · `v1/worker/gpu_model_runner.py` · `v1/sample/rejection_sampler.py` | `v1/worker/gpu/{model_runner,input_batch,states}.py` · `gpu/spec_decode/*` · `gpu/model_states/{mamba_hybrid,recoverssm}.py` |

Phase 0's three interfaces: 8 shared files, 1 V2 file (`gpu/model_states/recoverssm.py`), 1 V1 file (the `gpu_model_runner.py:1639` dispatch site, which needs a hook V2 already has). Phase 1: 4 shared production files, 0 runner files (the PR adds no producer — §C.5).

## A.2 The three strongest reasons

### Reason 1 — The verification kernel: V2 has the extension seam; V1 has nine hardcoded lines

V2's `_rejection_kernel` already carries a mode axis: three `tl.constexpr` switches (`HAS_DRAFT_LOGITS`, `SYNTHETIC_MODE`, `USE_BLOCK_VERIFICATION`, `rejection_sampler_utils.py:536-539`) selecting between synthetic verification, Sun et al. block verification (`:595-627`, with `cumulative_log_p` and `_compute_global_residual_mass`), and the default Leviathan path (`:629-665`). `TREE_VERIFICATION` slots in as a fourth peer. More importantly, **block verification already does cross-position residual-mass bookkeeping** — which is the single hardest correctness component of a lossless tree walk (renormalizing the residual across rejected siblings). We would be extending a kernel whose author already thought about global residual mass, not writing that reasoning from scratch.

V1's `rejection_greedy_sample_kernel` (`vllm/v1/sample/rejection_sampler.py:713-770`) has **no mode axis at all**. Nine lines with four independent chain assumptions: `for pos in range(...)` in physical order (parent ≡ `pos-1`); `target_argmax_ptr + start_idx + pos` (physically adjacent comparison row); one scalar `rejected` latch; output `[batch, MAX_SPEC_LEN+1]` with the bonus at fixed offset and `bonus_token_ids_ptr + req_idx` pre-selected per request — **no accepted-leaf identity is ever emitted.** Plus `MAX_SPEC_LEN = 128` (`:35`, enforced `:123`) and an entry-point assert block (`:413-425`) pinning `draft_token_ids.ndim == 1`.

The two are the same *amount* of new logic, but V2's lands as a `constexpr` branch in a file that already has three; V1's lands as a rewrite of a kernel with none.

### Reason 2 — Non-uniform, device-decided verification width is a shipped contract on V2 only

This is the objection every tree-verify proposal has died on: *"you cannot vary per-request query length inside a captured CUDA graph without a CPU round-trip."* Upstream answered it three PRs ago, for unrelated reasons:

- **Device-authoritative offsets.** `AdaptiveVerificationManager.reallocate_drafts()` computes `capacities` on device and writes `query_start_loc` via `torch.cumsum(capacities + num_non_draft_tokens_gpu, out=self.query_start_loc[1:num_reqs+1])` (verified `adaptive_verification.py:418-434`). `InputBatch.num_scheduled_tokens` is documented as an *upper bound* under AV (`gpu/input_batch.py:56-58`).
- **Backends were taught to read it.** `AttentionBackend.supports_device_cpu_query_lens_mismatch()` (`vllm/v1/attention/backend.py:208`), enforced by `get_query_lens_mismatch_unsupported_backend()` (`gpu/attn_utils.py:168-181`). Already `True` on **two backend families**: FlashInfer trtllm-gen on SM100 (#52157 `6a9c69fa8` — `q_cu_seq_lens = qo_indptr[:num_decodes+1]`, uniform-query assert relaxed at `backends/flashinfer.py:1642`) and the DSv4 MLA indexer on SM90 (#52795 `5df31ea52`, `backends/mla/indexer.py:158`).
- **Varlen decode CUDA graphs.** `ModelCudaGraphManager(..., varlen_decode=self.adaptive_verification is not None)` (`gpu/model_runner.py:599-601`) → one graph per token count accepting *any* 1..`decode_query_len` mix per request (`cudagraph_utils.py:230-247`). **A tree of ≤ `decode_query_len` nodes replays inside already-captured graphs, unchanged.**

On V1: none of these exist. Not the flag, not the device cumsum, not the varlen graphs. `AdaptiveVerificationManager` is V2-only (`gpu/spec_decode/adaptive_verification.py:114`, referenced only at `gpu/model_runner.py:136,296,551`).

### Reason 3 — The AV budgeter is already a connected-subtree admission rule

Nobody in the RFC has costed the *runtime tree-shape chooser*. On V2 it is a ~10-line diff to a shipped, `torch.compile(dynamic=True)`, profiled component. Verified verbatim from `adaptive_verification.py:34-65`:

```python
survival = confidence_probs[idx_mapping].cumprod(dim=1)
out_of_range = steps[None, :] >= capacities[:, None]
survival = survival.masked_fill(out_of_range, -float("inf"))
winners = flat.topk(draft_budget).indices
torch.sum(admitted.view_as(survival), dim=1, dtype=capacities.dtype, out=capacities)
```

with the docstring invariant *"Survival only decreases along a request, so a global top-k always admits continuously along steps with a request."*

**That invariant is the tree condition.** Path survival along a root→node path is monotonically non-increasing, so a global top-k over `[request, node]` scored by path survival admits a **connected subtree** — every admitted node's parent scores at least as high and is therefore also admitted. Replace `cumprod(dim=1)` with a parent-scan over the node array; the `capacities`-in/`capacities`-out contract, the `-inf` out-of-range masking, and the `flat.topk` all carry over verbatim. It inherits the profiled cost model `get_num_tokens()` (`:267-335`) maximizing `estimated_accepted_tokens / cost` over `draft_cost_ms[num_reqs] + verify_cost_ms[…]` curves from `set_initial_cost_curves` (`:195-214`), and the double-buffered confidence D2H (`record_confidences`, `:239-265`) fed by `speculator.draft_token_confidence_probs [max_num_reqs, K] fp32` (`dspark/speculator.py:82-84`).

On V1 this is ~600 lines from scratch, including a profiling harness.

## A.3 What it costs — enumerated honestly

| # | Cost | Size | Mitigation |
|---|---|---|---|
| C1 | **Our 27B hybrid GDN target defaults to V1.** `_is_default_v2_model_runner_model` vetoes `is_hybrid` unless allow-listed, and no hybrid arch is on the list (both verified). We run phases 0–3 under `VLLM_USE_V2_MODEL_RUNNER=1` (`envs.py:2037`, short-circuit `config/vllm.py:650-652`), which *bypasses* the gate rather than satisfying it. Our numbers come from an opt-in path; reviewers will say so. | 1–3 eng-wk to fix | The fix is half-done upstream: **#51410 `44351f81d`** (2026-08-08) deleted the hybrid skip in `tests/v1/e2e/spec_decode/mtp/test_mtp.py`, added `test_mtp_correctness[qwen3_5-hybrid]` to `.buildkite/test_areas/model_runner_v2.yaml:123-125`, and removed `not Jamba` from MRV2 PP tests on CUDA and AMD. Hybrid + MRV2 + MTP spec decode is **CI-gated today**. The residual is an allowlist entry — file it as a separate PR with our benchmark as evidence. |
| C2 | **Cache drafters (suffix, ngram_gpu) do not exist on V2.** 7 methods gated off at `config/vllm.py:2474-2487`. | 4–8.5 eng-wk | **Phases 0–3 do not use them.** They are Phase 4 (proposer composition). File as independent PRs on their own clock, starting with the 0.2-wk freebie: promote `draft_logits` and `supports_mm_inputs` onto `BaseSpeculator` with `None`/`False` defaults — they are read unguarded at `gpu/model_runner.py:1360` and `:731` but only ever set in `DraftModelSpeculator.__init__:126-142`. A genuine latent ABC defect; easy to land; establishes standing with the V2 reviewers before RFC-A. Drop `ngram` (numba/CPU, 293 L) entirely — `ngram_gpu` supersedes it. |
| C3 | **AV's inherited constraints**, if the tree budgeter rides AV: no LoRA, no `enforce_eager`/`cudagraph_mode=none`, no PP>1 (`config/vllm.py:2506-2531`); every builder must report `AttentionCGSupport.ALWAYS` (`adaptive_verification.py:462-470`); the trimmed batch must fit **one** rejection chunk (`:128-131`, `get_max_chunk_logits(vocab_size)`). Plus AV is method-gated to `dspark` (`config/speculative.py:528-530`). | gate widening + doc | Same constraints a tree would have needed to invent. Widening the method gate is 2 lines; see Open Question 2. |
| C4 | **Forgone: V1's ragged drafter path.** `propose() -> list[list[int]] | torch.Tensor` (`gpu_model_runner.py:5115`), `update_scheduler_for_invalid_drafts` (`ngram_proposer_gpu.py:475-515`), async count D2H on a dedicated stream (`:5182-5188`). | — | **Not a loss for a tree.** A tree proposal is rectangular-with-padding by nature (fixed max node count per step). And V2's *verify* side is already ragged: `cu_num_logits` gives a per-request `num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS` inside the kernel (`gpu/input_batch.py:410`), `InputBatch.num_draft_tokens_per_req: np.ndarray | None` (`:66`). Only the drafter *output* contract is rectangular. |
| C5 | Reviewer churn on V2: 193 commits to `vllm/v1/worker/gpu/` since 2026-06-01 vs 90 to `gpu_model_runner.py`. Higher rebase load. | ongoing | This is also the reason to be there: new spec-decode features land V2-first, and `config/vllm.py:673-678` *forces* V2 for DFlash/DFlash2/DSpark. |

**Net vs. Report 3's Path A/B tables:** Report 3 priced V2 at +6…+13 eng-wk (B0a–B0e). **Every one of those items is C2 or C1 — cache drafters and default-enablement. Neither is in phases 0–3.** For the scoped work, the premium is C1 only (1–3 wk, separable), against which V1 carries an unavoidable deferred A8 (8–14 wk to re-port the tree work *and* our 36 runner patch functions) because upstream tree-verify will land on V2 whether or not we do it there.

## A.4 What changes in the RFC's phase wording

1. **Delete** the "recurrent spec-state lives in V2" sentence. Replace with the shared-context framing (§0, correction 1).
2. **Retitle the open question.** Not *"V1 or Model Runner V2?"* but *"which runner owns the tree verification path?"* — and **answer it in the body**. An open substrate question invites the reviewer to answer it for us, and there is no evidentiary reason to leave it open.
3. **Relabel Phase 0** as *"substrate-neutral, lands first, independently useful."* All three interfaces are additive with `None`/default values (§B). Prepend the `stride_indices_tok` correctness fix as a standalone PR that lands before any of them.
4. **Phase 1 (E-a) drops any runner claim.** It touches `v1/attention/backend.py`, `backends/flash_attn.py`, `backends/flex_attention.py`, `backends/utils.py` — all shared — and sets no producer in either runner.
5. **Phase 3 must name RNG re-keying as a deliverable.** `gumbel.py:111` `gumbel_seed = tl.randint(seed, pos)` and `rejection_sampler_utils.py:569` `u = tl_rand32(seed, pos, includes_zero=False)` both key on **absolute position**, and the drafter deliberately matches (`speculator.py:346-355` passes `positions + 1`; `dspark/speculator.py:139-149`: *"the target verifies it with the predecessor's Gumbel key (Q-1)"*). **In a tree, all siblings at one depth share `pos`, hence share `u`.** Sibling draws become perfectly correlated and residual mass is never renormalized across rejected siblings. This is the design's one *silent* losslessness hazard — it degrades acceptance rate rather than crashing — and the RFC does not currently mention it.
6. **Add DFlash2 as Phase 2's precedent and fallback.** `dflash2/speculator.py:15-106` already ships propose-a-lattice → linearize → verify-with-the-stock-kernel: `_selector_walk_kernel` walks a `[step, prev_k, k]` transition-score lattice via `gumbel_noised_argmax` carrying `previous = index`, writes one linearized path to `draft_tokens`, and `_cache_draft_logits_kernel` writes only the K candidate columns into a `-inf`-filled `draft_logits` so the unmodified rejection sampler consumes a truncated-but-correct distribution. Phase 2 is a *generalization of shipped code*, not a new idea.
7. **Move the GB10/FA4 exclusion to the front of Phase 1.** sm_121 → family 12 → `_is_fa4_supported()` False → `fa_version = 2` (`fa_utils.py:98-100`) → `NotImplementedError("FA2 does not support mask_mod")`. The FLEX_ATTENTION twin is the portability answer and must be in the PR, not promised (§C.4).

---

# B. PHASE-0 INTERFACE PROPOSAL (RFC technical annex)

Three additive interfaces. All default to today's behavior. **None of them requires a tree to exist** — each is independently reviewable and independently mergeable.

**Landing order and rationale.** `B0` (correctness fix) → `B1` → `B2` → `B3`. `B2` and `B3` land in code created by **#51855 `70afdedc1`** (2026-08-17, K3 RecoverSSM), which is one week old and actively moving; `B1`'s Triton sites are the quietest surface in the stack (`fused_recurrent.py` last touched `2d24355eb` #52030, 2026-08-13) and `B0` is independently justifiable.

---

## B0 (prerequisite) — Fix the `stride_indices_tok` dead parameter

**Not part of the tree design. A pre-existing correctness gap that a tree layout is the first thing to expose.**

`vllm/third_party/flash_linear_attention/ops/fused_recurrent.py`: `stride_indices_tok: tl.constexpr` is declared at `:52`, computed at `:211-215` (three branches: `1,1` / `stride(0),1` / `stride()`), passed at `:243` — and **never referenced in the kernel body.** Lines `:110` and `:155` do:

```python
state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t)
```

hard-assuming the token axis of `ssm_state_indices` is unit-stride. The Mamba2 twin gets this right (`mamba_ssm.py:351,436` use `stride_*_indices_T`). Any layout that slices or permutes the slot axis reads garbage silently.

**Patch:** `+ i_t * stride_indices_tok` at both sites. **Test:** build `ssm_state_indices` as a non-unit-stride view (e.g. `full[:, ::2]` or a transposed `[S, N].T`), run `fused_recurrent_gated_delta_rule` with `num_accepted_tokens > 1`, assert `torch.equal` against the contiguous-`.contiguous()` reference. Fails at HEAD.

---

## B1 — Per-node parent indexing for branch-local scans

### Proposed signature

```python
# vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:523
# mirrored: vllm/model_executor/layers/mamba/ops/ssu_dispatch.py:36,323
#           vllm/_custom_ops.py:2780
def fused_recurrent_gated_delta_rule(
    ...,
    ssm_state_indices: torch.Tensor | None = None,   # [N, S] slot table (existing)
    num_accepted_tokens: torch.Tensor | None = None, # [N]    chain depth (existing)
    node_parent_slot: torch.Tensor | None = None,    # NEW: int32 [N, S]
) -> tuple[torch.Tensor, torch.Tensor]:
```

`node_parent_slot[i_n, t]` = index **into `ssm_state_indices[i_n, :]`** of token `t`'s parent state, or `-1` meaning *"select the request's initial state the legacy way."* Note it indexes the slot table, **not** the block table — the `NULL_BLOCK_ID == 0` sentinel semantics of `ssm_state_indices` are untouched, and `-1` is a distinct, unused value in the slot-index space.

Kernel body (site A; B and C mirror):

```python
# fused_recurrent.py:105-120, replacing the IS_SPEC_DECODING branch
if HAS_NODE_PARENTS:
    i_t = tl.load(node_parent_slot + i_n * stride_parent_seq)      # branch root
    if i_t < 0:
        i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1  # chain fallback
elif IS_SPEC_DECODING:
    i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
else:
    i_t = 0
state_idx = tl.load(ssm_state_indices
                    + i_n * stride_indices_seq
                    + i_t * stride_indices_tok)   # ← B0 fix, same patch
```

with `HAS_NODE_PARENTS = lambda args: args["node_parent_slot"] is not None` added to the existing `@triton.heuristics` block.

### Exact insertion points at HEAD (all three must move together)

| | file:line | symbol | what changes |
|---|---|---|---|
| A | `vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:106-116` | `fused_recurrent_gated_delta_rule_fwd_kernel` | initial-state select |
| B | `vllm/model_executor/layers/mamba/ops/mamba_ssm.py:334-354` | `_selective_scan_update_kernel` | `init_token_idx = max(num_accepted-1, 0)` → parent slot |
| C | `csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu:173-177` | `gdn_decode_post_conv_mtp_kernel` | `source_slot = state_indices[req*w + accepted-1]` → parent slot |

Metadata carrier: `GDNAttentionMetadata` (`vllm/v1/attention/backends/gdn_attn.py:42-79`) gains `node_parent_slot: torch.Tensor | None = None` beside `num_accepted_tokens` (`:66`), populated in `GDNAttentionMetadataBuilder.build` (`:210`) next to `spec_state_indices_tensor` (`:307-309`, `:328-330`). **CUDA-graph buffer must be `[decode_cudagraph_max_bs, num_spec+1]` with tail filled `-1`** — note the existing `num_accepted_tokens` CG buffer is 1-D `(decode_cudagraph_max_bs,)` (`:162-166`) and its staging path fills the tail with `1` (`:470-474`); the parent buffer's neutral value is `-1`, not `1`.

### Why it is a no-op for chains

Three independent mechanisms, in order of strength:

1. **Compile-time elision.** All existing call sites pass `node_parent_slot=None`. The `triton.heuristics` flag resolves to `HAS_NODE_PARENTS: tl.constexpr = False`, the branch is dead-code-eliminated before codegen, and the emitted PTX is byte-identical to HEAD's. Same for the CUDA path via a template parameter or `if (parent == nullptr)` hoisted out of the loop.
2. **Runtime sentinel.** Even with the tensor supplied, `-1` routes to the exact legacy expression.
3. **Semantic identity.** For a chain, `node_parent_slot[i_n, 0] = num_accepted_tokens[i_n] - 1` reproduces the legacy select by definition.

**What is deliberately *not* claimed:** branch-local scans (reloading the accumulator from a different ancestor mid-loop) are **not** in this interface. `b_h` (`fused_recurrent.py:102-146`), `state` (`mamba_ssm.py:385-433`), `h[row][i]` (`.cu:270-307`) are still carried monotonically across the whole loop. B1 only decouples *which parent the scan starts from*. Multi-branch-per-CTA is Phase 2 work; keeping it out is what makes B1 reviewable in isolation.

### Test that proves the no-op

```
tests/kernels/mamba/test_node_parent_indexing.py

T1 (codegen identity, no GPU needed beyond compile):
    compile fused_recurrent_gated_delta_rule_fwd_kernel with the new arg absent
    and with it present-but-None; assert kernel.asm["ptx"] identical, and
    assert the cache key tuple identical.

T2 (sentinel identity, bit-exact):
    random GDN decode, num_accepted_tokens ∈ [1, γ] per request.
    run A: node_parent_slot=None
    run B: node_parent_slot = full((N, S), -1, int32)
    assert torch.equal on out AND on the written state pages.

T3 (chain-encoded parents == legacy, bit-exact)  ← the theorem
    node_parent_slot[:, 0] = num_accepted_tokens - 1
    assert torch.equal(state_pages_A, state_pages_B) and torch.equal(out_A, out_B)

T4 (B0 regression, fails at HEAD):
    ssm_state_indices as a strided view; assert equality vs .contiguous().

T5 (site C parity):
    same three assertions through ops.fused_gdn_decode_post_conv_mtp, gated on
    _can_use_fused_gdn_mtp_decode (qwen_gdn_linear_attn.py:1801-1816).
```

`torch.equal` is correct here (integer slot indices, and the state writes are a pure function of the same reads) — unlike the float-output case in §C.2 where `-0.0` forces max-abs-diff.

### Collisions

Site C is owned by **#51674 `1be362836`** (2026-08-14, +557 new `.cu`), with follow-ups `cdb8545a9` (#52539, head-ratio widening) and `5af7c8dad` (#51812, gate alignment). `gdn_attn.py` moved as recently as `6df7adc17` (#53077, 2026-08-20, empty-draft-schedule reset at `:237-244`). Sites A and B are quiet. **`kMaxMtpTokens = 8`** (`.cu:178`, mirrored `MAX_FUSED_GDN_MTP_TOKENS = 8` at `qwen_gdn_linear_attn.py:89`) is a hard node-count cap in the native path — above it the kernel zero-fills and returns (`.cu:178-187`); the tree gate must tighten `_can_use_fused_gdn_mtp_decode`, not rely on the fallback.

---

## B2 — Declared carry-slot budget

### Proposed signature

```python
# vllm/v1/kv_cache_interface.py, adjacent to MambaSpec:811
@dataclass(frozen=True)
class SpecCarryBudget:
    """Carry slots a speculative step may write, per carry geometry.

    Temporal (SSM) state carries one slot per *block-table column*; conv state
    carries extra *token columns inside one block*. A tree scales these by
    different factors, so one integer cannot express both.
    """
    temporal_slots: int      # block-table columns past the running block
    conv_tokens: int         # extra token columns inside the running block
    max_branch_depth: int    # longest root→leaf path (== chain length for chains)

    @classmethod
    def chain(cls, gamma: int) -> "SpecCarryBudget":
        return cls(temporal_slots=gamma, conv_tokens=gamma, max_branch_depth=gamma)


# MambaSpec, after num_speculative_blocks (:817):
    spec_carry_budget: SpecCarryBudget | None = None   # None ⇒ chain(num_speculative_blocks)

    @property
    def carry_budget(self) -> SpecCarryBudget:
        return self.spec_carry_budget or SpecCarryBudget.chain(self.num_speculative_blocks)
```

**`num_speculative_blocks` remains the authoritative allocator number**, with the invariant `num_speculative_blocks == carry_budget.temporal_slots`. This is the key design choice: it means `MambaManager` (`single_type_kv_cache_manager.py:1510-1513,1573-1574,1600-1642`), `mamba_get_block_table_tensor` (`v1/attention/backends/utils.py:1156-1166`), `max_memory_usage_bytes` (`kv_cache_interface.py:840-851`), `MambaAttentionMetadataBuilder.__init__` (`mamba_attn.py:119-159`) and NIXL (`kv_connector/v1/nixl/base_scheduler.py:146-154`) are **all untouched in phase 0**.

### Exact insertion point

`vllm/model_executor/layers/mamba/abstract.py:76-82` — verified verbatim at HEAD:

```python
            # RecoverSSM verifies the whole window off one checkpoint, so it
            # never writes the baseline's per-draft-token state slots.
            num_speculative_blocks=(
                0
                if vllm_config.cache_config.use_kda_recoverssm
                else vllm_config.num_speculative_tokens
            ),
```

becomes three-way; the tree branch returns `SpecCarryBudget(temporal_slots=tree.num_nodes, conv_tokens=tree.depth, max_branch_depth=tree.depth)` **and** `num_speculative_blocks=tree.num_nodes`.

### Why it is a no-op for chains

- `spec_carry_budget=None` is the dataclass default; every existing construction site is unchanged, and `carry_budget` synthesizes `chain(num_speculative_blocks)`.
- **The one real hazard is dataclass equality.** `MambaSpec` is `@dataclass(frozen=True)` and specs are equality-compared across layers (`vllm/v1/worker/mamba_utils.py:611` `assert all(mamba_specs[0] == spec for ...)`; `gpu/model_states/mamba_hybrid.py:128`) and hashed through `is_uniform_with_collection` (`kv_cache_interface.py:865-873`). A `None` default makes the new field compare equal for **every** existing config; `is_uniform_with_collection` must add `spec_carry_budget` to its comparison so a future mixed-budget config fails loudly instead of silently.

### Precedent to cite in review

Two, both already in the file:
1. **#51855 already converted line 76** from an unconditional `num_speculative_tokens` into a feature-dependent *declared* budget (`0` under RecoverSSM). The seam is open; we are widening it, not opening it.
2. **`MambaSpec` already carries two independent block budgets:** `num_prefill_checkpoint_blocks` (`:818`, consumed at `max_memory_usage_bytes:846-849`, built by `KimiK3KDAMetadataBuilder` at `kda_metadata.py:570-624`). A budget *struct* is not a novel shape here.

### Test that proves the no-op

```
tests/v1/core/test_mamba_spec_carry_budget.py

T1 (golden memory table): parametrize mamba_cache_mode ∈ {none, align, all}
    × num_speculative_tokens ∈ {0,1,3,7} × use_kda_recoverssm ∈ {F,T};
    assert max_memory_usage_bytes and max_num_blocks_per_req match a golden
    table generated at HEAD^ (pin the golden file, don't recompute).

T2 (equality/hash identity): for every config above,
    assert MambaSpec(**kw) == MambaSpec(**kw) and
           hash(...) == hash(...) and
           is_uniform_with_collection unchanged vs HEAD^.

T3 (property identity): assert spec.carry_budget == SpecCarryBudget.chain(
        spec.num_speculative_blocks) whenever spec_carry_budget is None.

T4 (allocator invariance): MambaManager.get_num_blocks_to_allocate over a
    synthetic request trace, assert identical block counts vs HEAD^.
```

---

## B3 — Accepted-path replay hook

### Proposed signature

```python
# vllm/v1/attention/backends/recoverssm_metadata.py:21
@dataclass(frozen=True)
class AcceptedPath:
    """Which emitted nodes survived verification, per request."""
    node_ids:  torch.Tensor   # int32 [N, D]  slot ids along root→accepted leaf, -1 pad
    path_lens: torch.Tensor   # int32 [N]     valid prefix length of node_ids
    is_linear: bool           # True ⇒ node_ids[i, :path_lens[i]] == arange(path_lens[i])

    @classmethod
    def linear(cls, num_accepted: torch.Tensor, max_depth: int) -> "AcceptedPath": ...


class RecoverSSMMetadata(abc.ABC):
    @abc.abstractmethod
    def commit_recoverssm_state(
        self,
        num_accepted_tokens: torch.Tensor,
        accepted_path: AcceptedPath | None = None,   # NEW; None ⇒ chain
    ) -> RecoverSSMPostprocessMetadata | None: ...
```

and correspondingly `RecoverSSMState.commit_step(..., accepted_path: AcceptedPath | None = None)`.

### Exact insertion points

| | file:line | what |
|---|---|---|
| ABC | `vllm/v1/attention/backends/recoverssm_metadata.py:21-26` | widen `commit_recoverssm_state` |
| V2 caller | `vllm/v1/worker/gpu/model_states/recoverssm.py:38-67` (`commit_step`), forwarded at `:52`; driven from `gpu/model_states/mamba_hybrid.py:331-337` | thread the new arg |
| impl | `vllm/models/kimi_k3/nvidia/ops/recoverssm.py:909-918` (`KDARecoverSSMCommitContext.commit`), plan kernel `:233` (`_prepare_commit_plan_kernel`, launched `:967` with `SPEC_QUERY_LEN`) | early-out when `accepted_path is None or .is_linear` |
| align commit | `vllm/v1/worker/mamba_utils.py:414` call site into `_copy_mamba_state_block` (`:123`) | see below |
| V1 dispatch | `vllm/v1/worker/gpu_model_runner.py:1639-1675` inside `_update_states_after_model_execute` (def `:1619`) | symmetric free function beside `postprocess_mamba_align_gpu` (`mamba_utils.py:1388`) |

**The align-commit generalization is one line, and the code is already shaped for it.** `_copy_mamba_state_block` (`mamba_utils.py:123-127`) already takes `src_col` and `token_bias` as *separate* arguments; the temporal branch (`:284`) does `block_table[bt_row, src_col + token_bias]` while the conv branches (`:188-279`) use `token_bias` as an intra-block slide. So:

```python
# mamba_utils.py:414 call site
accepted_slot = (tl.load(accepted_slot_ptr + req_idx)
                 if HAS_ACCEPTED_SLOT else src_block_idx + accept_token_bias)
```

Temporal takes `accepted_slot`; **conv keeps `token_bias` unchanged** — conv carry is depth-indexed even in a tree, because the conv window slides along the accepted *path*, whose length is still ≤ `max_branch_depth`. This is precisely why B2 needs two numbers.

### Why it is a no-op for chains

- Default `None` on both the ABC and the V2 state class; the V1 free function returns immediately on `None`.
- `is_linear=True` is an explicit second escape: `AcceptedPath.linear()` produces `node_ids = arange`, for which the tree gather is provably the identity, so the fast path is taken without re-deriving it.
- `HAS_ACCEPTED_SLOT` is a Triton `constexpr` heuristic → the align kernel's generated code is unchanged when absent.

### Why this is the right hook (argument for the reviewer)

**The align commit is already an accepted-path replay** — it *selects* a carry slot by accepted count (`mamba_utils.py:284`), *materializes* it at the canonical block-aligned destination `dest_block_idx = aligned_new_computed // block_size - 1` (`:401`), and *re-normalizes the anchor* by resetting `num_accepted_tokens := 1` (`:406-407` GPU, `:1326` CPU, `preprocess_mamba_align_fused_kernel:479-480` V2). **That reset is literally re-linearization.** The only structural gap is that the selector is `token_bias = aligned_new_computed - num_tokens_running_state` (`:400`) — a pure function of token *counts*, which cannot name *which node at a depth* was accepted.

Likewise `RecoverSSM` is already replay-from-checkpoint, and it already proves that a lossless method can declare a **zero** carry budget (`abstract.py:76`; `kda_metadata.py:318` `spec_state_slots = 1`; `:323-327` shrinks `spec_state_indices_tensor` to `[max_num_reqs, 1]`). What it lacks is node identity: `commit_recoverssm_state` takes only `num_accepted_tokens`, `_prepare_commit_plan_kernel` derives `commit_lens` from it plus `query_start_loc`, and `_postprocess_recoverssm_align_kernel` (`worker/gpu/model_states/recoverssm.py:74-101`) writes `state_idx[req] = min((num_computed+num_sampled)//BS, width-1)` and `num_accepted[req] = 1`. Every one of those is correct for a contiguous `[bos, bos+k)` window and wrong for a scattered path.

**Note the page-layout constraint** that B3 does *not* fix: `get_state_shape` under RecoverSSM adds `(H, spec_query_len, head_dim)` and `(H, spec_query_len, 2*head_dim)` cache tensors (`layers/mamba/mamba_utils.py:307-325`) sized by `spec_query_len = 1 + num_speculative_tokens` (`kda_metadata.py:347`) — a chain-length assumption baked into the page layout. For a tree that becomes `1 + num_nodes`, which is a **B2** consumer, not a B3 change. Say so explicitly in review; a maintainer will look for it.

### Test that proves the no-op

```
tests/v1/worker/test_accepted_path_replay.py

T1 (None ≡ HEAD): real Qwen3.5-GDN hybrid decode step with spec tokens,
    mamba_cache_mode ∈ {none, align}; run with accepted_path=None and compare
    every mamba state page against a HEAD^ reference dump.
    assert (a - b).abs().max().item() == 0.0   (states are float; -0.0 risk)

T2 (linear ≡ None): accepted_path = AcceptedPath.linear(num_accepted, γ)
    assert byte-identical pages AND identical num_accepted_tokens_gpu after
    postprocess_state.

T3 (no new syncs): wrap the commit in torch.cuda.graph capture; assert capture
    succeeds. A CPU read of path_lens would fail here — this is the test that
    keeps the interface device-resident.

T4 (kernel-count invariance): with a Triton launch hook, assert the same set of
    kernels launches for accepted_path=None as at HEAD^.

T5 (align selector parity): unit-test _copy_mamba_state_block directly with
    HAS_ACCEPTED_SLOT=False vs =True with accepted_slot = src_col + token_bias;
    assert torch.equal on the destination block.
```

### Collisions

**Tightest in the design.** #51855 (`70afdedc1`, 2026-08-17) created `RecoverSSMMetadata`, `worker/gpu/model_states/recoverssm.py`, and the 1067-line `kimi_k3/nvidia/ops/recoverssm.py`, and rewrote `abstract.py:76-82`. `mamba_hybrid.py` was touched `da329cc30` (#50272, 2026-08-22, short_conv/LFM2 spec decode). `mamba_utils.py` has five touches in the last three weeks (`f936a267f` #48109 XPU pointer overflow; `a02cfccbc` #50729 overlapping state-copy race; `fac808b36` #49436 `_TEMPORAL_TILES` 3-D grid; `c2881ce60` #50432 cross-block `num_accepted` race in MRv2 align). **Land B3 last, rebase-check weekly, and open a courtesy issue on #51855's author before filing.**

Config gates to widen if the hook generalizes beyond K3: `config/vllm.py:2646-2651` (*"RecoverSSM is only supported for Kimi-K3 KDA"*), `:2658-2661` (`none`/`align` only), `:2662-2668` (align ⇒ V2 only), `:2669-2672` (PP=1).

---

# C. E-a IMPLEMENTATION SPEC — tree visibility via FA4 `mask_mod`

**Thesis for the PR:** *this PR adds the ability to express a tree visibility mask. It adds no producer of one.* Every new field defaults to `None`; no scheduler, proposer, rejection-sampler, or re-linearization code is touched; the tests construct parent tables directly. Serving behavior is provably unchanged because the mask_mod is **unreachable** without a caller setting `spec_parent_indices`. This is also what makes it substrate-free: it touches neither runner's spec-decode path.

## C.1 The encoding — one row per query token, O(1) per element

Copy the shape of `_make_mm_prefix_mask_mod` (`vllm/v1/attention/backends/flash_attn.py:1426`): **all structure lives in one per-query-token row, loaded once and hoisted out of the unrolled element loop**, with `cu_seqlens_q` as `aux_tensors[1]` mapping local `q_idx` → packed row index. No key-side lookup.

Per scheduled query token `t`, two int32:

| row kind | `pfx[t]` | `anc[t]` |
|---|---|---|
| prefill / plain-causal decode | `q_abs + 1` | `0` |
| tree node at draft slot `s` | `ctx_off` (= `seqlen_k − seqlen_q`) | bitmask of `{s} ∪ ancestors(s)` over slots `0..K−1` |

**Predicate, uniform across all row kinds:**

```
keep(q, kv) = (kv < pfx[q]) ∨ (0 ≤ kv − ctx_off < K ∧ bit(anc[q], kv − ctx_off))
```

**The theorem that makes the whole PR reviewable:** a prefill/decode row reduces to exactly `kv ≤ q_abs` — the causal mask. And a **fanout-1 tree** yields `anc = (1 << (s+1)) − 1`, for which the predicate is *bit-identically* `kv ≤ q_abs`. Not approximately; by construction. That is what §C.2 Tier 1 asserts on CPU and Tiers 2–3 assert on the kernel.

No `tree_base` aux tensor is needed: for a spec-decode step the draft window is exactly `[seqlen_k − seqlen_q, seqlen_k)`, i.e. `ctx_off` is already in `seqlen_info`. `K ≤ 32` fits one int32; `K ≤ 64` needs a second word (`anc_hi`, row shape `(n, 3)`, word selected by `slot >> 5`). Ship `K ≤ 32` for v1 and make `K` a `functools.cache` key so the compile key differentiates.

## C.2 Code sketch

```python
# vllm/v1/attention/backends/flash_attn.py — sibling of _make_mm_prefix_mask_mod (:1426)

@functools.cache                                   # REQUIRED — see C.3
def _make_tree_mask_mod(max_tree_slots: int = 32):
    """CuTe-DSL mask_mod: each query token sees the full prefix plus exactly its
    own ancestor set inside the draft window.

    aux_tensors[0]: (num_actual_tokens, 2) int32 — per scheduled query token,
        [0] = leading KV positions visible unconditionally
              (q_abs + 1 for a non-tree row; ctx_off for a tree node),
        [1] = bitmask over draft slots of {self} ∪ ancestors(self); 0 if non-tree.
    aux_tensors[1]: (num_reqs + 1,) int32 — cu_seqlens_q, mapping the kernel's
        LOCAL q_idx to a packed row index (identical role to mm_prefix).

    A fanout-1 tree yields anc = (1 << (slot + 1)) - 1, for which the predicate is
    bit-identically `kv_idx <= q_abs`; see tests/v1/attention/test_tree_mask.py.
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Uint32
    from vllm.vllm_flash_attn.cute.utils import scalar_to_ssa, ssa_to_scalar

    @cute.jit
    def _load_row(q_idx, seqlen_info, aux_tensors, batch_idx):
        # Depends only on q_idx -> hoists out of the unrolled element loop.
        # ssa_to_scalar reads lane 0, so __vec_size__ must be pinned to 1.
        rows, cu_seqlens_q = aux_tensors[0], aux_tensors[1]
        b = batch_idx[0]
        q_local = cutlass.min(ssa_to_scalar(q_idx), seqlen_info.seqlen_q - Int32(1))
        token_idx = cutlass.max(cu_seqlens_q[b] + q_local, Int32(0))
        return rows[token_idx, 0], rows[token_idx, 1]

    @cute.jit
    def tree_mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        pfx, anc = _load_row(q_idx, seqlen_info, aux_tensors, batch_idx)
        ctx_off = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        kv   = ssa_to_scalar(kv_idx)                     # exact: __vec_size__ == 1
        slot = kv - ctx_off
        # Clamp the shift: a count outside [0,31] is UB. Neutralised by in_win.
        sh   = cutlass.min(cutlass.max(slot, Int32(0)), Int32(max_tree_slots - 1))
        bit  = Int32((Uint32(anc) >> Uint32(sh)) & Uint32(1))

        pfx_s  = scalar_to_ssa(pfx, Int32)
        base_s = scalar_to_ssa(ctx_off, Int32)
        bit_s  = scalar_to_ssa(bit, Int32)
        zero   = scalar_to_ssa(Int32(0), Int32)
        one    = scalar_to_ssa(Int32(1), Int32)
        cap    = scalar_to_ssa(Int32(max_tree_slots), Int32)

        slot_s = kv_idx - base_s
        in_win = (slot_s >= zero) & (slot_s < cap)
        return (kv_idx < pfx_s) | (in_win & (bit_s == one))

    tree_mask_mod.use_fast_sampling = False   # tree tiles are NOT 5-point-samplable
    tree_mask_mod.__vec_size__ = 1            # _load_row / kv scalar take lane 0
    return tree_mask_mod
```

Every DSL construct above has a HEAD exemplar: 2-D aux indexing with a runtime `Int32` (`flash_attn.py:1486` `q_ranges[token_idx, 0]`); `cutlass.min/max` on DSL scalars (`:1483-1484`); `>>`/`&` on `Uint32` (CuTe `mask.py`, `curr_mask_val >> j & 1`); `&`/`|`/comparisons on `TensorSSA` (`:1504-1509`, `:1579-1582`). The final combine stays in SSA space deliberately — returning a scalar-built `Boolean` SSA has **no exemplar at HEAD**, whereas an SSA-comparison result is exactly what both shipped exemplars return.

## C.3 Three non-obvious constraints to pre-empt in the PR body

1. **`@functools.cache` is mandatory, not an optimization.** FA4's `hash_callable` mixes `repr()` of every closure cell into the compile key; the nested `_load_row` is a closure cell whose repr contains its address. Without the cache, **every forward pass JIT-recompiles.** Cite the mm_prefix docstring at `flash_attn.py:1434-1437`, which says exactly this. (Note `_make_rswa_mask_mod` at `:1535` is *not* cached and is rebuilt per forward at `:1065` — it survives only because it has no closure cells, and it still pays `inspect.getsource` + sha256 per step. Do not copy it.)
2. **`use_fast_sampling = False` is a correctness requirement.** The block-sparsity classifier drops to 5 threads and samples 4 corners + center; a block classified "full" **skips mask_mod entirely** (`flash_fwd_sm100.py`, comment *"Full blocks dont need mask_mod"*). A tree mask is non-convex within a tile — two siblings at opposite corners can both be self-visible while their cross terms are masked — so 5-point sampling will misclassify partial blocks as full and silently drop the mask. Today this is inert on the vLLM path (`use_block_sparsity = block_sparse_tensors is not None`, and **no vLLM caller ever builds `block_sparse_tensors`**), but the attribute must be set now, because it becomes live the moment anyone wires the precompute kernel.
3. **`__vec_size__ = 1` is load-bearing.** The vectorized path hands you `vec_size` *different* columns per call and expects a bit-packed `Uint32` back. That contract is a natural fit for an ancestor bitmask (one packed word per 32 columns, zero shifts) and is the obvious perf follow-up — but its lanes come from the TMEM fragment layout, whose contiguity is not guaranteed and which **no shipped exemplar exercises**. Ship v1 scalar; name the follow-up explicitly rather than being silent about it.

## C.4 Equivalence test — four tiers

House idiom for bit-exact kernel assertions is **max-abs-diff == 0**, not `torch.equal` (`tests/kernels/attention/test_mla_cross_layer_kernel_equivalence.py:70,147,217,303`, module docstring *"Bit-exact kernel equivalence…"*). Structural template is `tests/v1/attention/test_mm_prefix.py` (640 L), the only FA4-mask_mod test at HEAD.

**Tier 1 — predicate equality (CPU, no GPU, runs on every CI host).** This is where the theorem lives; it catches every table-construction bug and is arch-independent.

```python
def test_fanout1_tree_rows_are_exactly_causal():
    rows = build_tree_rows(parents=[-1, 0, 1, 2, 3], ctx_off=C, seq_len=S)   # chain
    for s in range(5):
        pfx, anc = rows[s]
        got = np.array([(kv < pfx) or (0 <= kv - C < 32 and (anc >> (kv - C)) & 1)
                        for kv in range(S)])
        np.testing.assert_array_equal(got, np.arange(S) <= C + s)
```

**Tier 2 — bit-exact, apples-to-apples (FA4; the primary kernel assertion).** Run **the same mask_mod twice** with different aux *contents*: run A = chain parent table (`pfx=ctx_off`, `anc=(1<<(s+1))−1`), run B = causal-encoded table (`pfx=q_abs+1`, `anc=0`). The compile key is identical — same `mask_mod_hash`, same `aux_tensor_metadata` (dtype/align/leading-dim tuple), same tiles, `causal=False` both — so it is *literally the same compiled kernel with the same `n_block` range and the same accumulation order*. Any difference is a table bug.

```python
assert (out_a - out_b).abs().max().item() == 0.0
assert (lse_a - lse_b).abs().max().item() == 0.0     # return_softmax_lse=True
```
LSE is the sharper probe: a direct function of the mask (row max + row sum), insensitive to the PV product.

**Tier 3 — bit-exact vs *native* causal (FA4, SM100/SM110 only, tiles pinned).** Compare `causal=True, mask_mod=None` against `causal=False, mask_mod=chain_tree`, calling `vllm.vllm_flash_attn.cute.interface._flash_attn_fwd` **directly** with `tile_mn=(128,128)` for both (`flash_attn_varlen_func` does not expose `tile_mn`).

Why it should be bit-exact: the only structural difference is `n_block_max` — causal clips it, non-causal uses the full `ceil_div(seqlen_k, tile_n)` — so the mask_mod run processes extra *leading, fully-masked* blocks. Those contribute exactly nothing: a fully-masked block gives `row_max = −inf → row_max_safe = 0.0, acc_scale = 0.0`; the first live block then computes `acc_scale_ = (−inf − real)·scale = −inf`, `exp2(−inf) = 0.0`, so `acc_O *= 0.0` and `row_sum = 0.0·0.0 + new` — reproducing the `is_first=True` initialization exactly. The rescale threshold (8.0 for 16-bit) does not fire, since `−inf >= −8.0` is False.

**State the residual risk in the docstring:** the causal run zero-inits `acc_O` via `zero_init=True`; the mask_mod run accumulates `0.0 + x`. If any `acc_O` element were `−0.0`, `0.0 + (−0.0) = +0.0` — a one-bit sign flip. Hence `abs().max() == 0.0`, not `torch.equal`.

**Gate hard on device family 10/11.** On SM90 this comparison is *not* bit-exact by construction: the SM90 tile-size selector returns 192×144 non-causal vs 192×128 causal for `head_dim ≤ 96` → different KV partitioning → different online-softmax split points. And on SM90, `Mask.apply_mask` **drops mask_mod entirely under causal**. Also assert `cu_seqlens_q is not None` in the test, to keep the `use_2cta_instrs` `not causal` term neutralized by the varlen path.

**Tier 4 — behavioural, fanout > 1 (FA4).** Dense fp32 masked-softmax reference in the mm_prefix style over real 4-ary and mixed trees, `atol=rtol=2e-2`. **Include the mm_prefix-style negative control** (`test_mm_prefix.py:392-396`): assert the tree reference *differs* from the causal reference, so a degenerate table cannot pass silently.

## C.5 Runtime metadata path

**It does not exist at HEAD, in either runner.** `vllm/v1/spec_decode/metadata.py:9-24` (`SpecDecodeMetadata`) is seven flat linear fields with no parent/child/depth channel; `bonus_logits_indices` is `[batch_size]`, presuming exactly one terminal position per request. `CommonAttentionMetadata` (`vllm/v1/attention/backend.py:369-450`) has no tree field. V2's `req_states.draft_tokens` is `[max_num_reqs, K]` (verified `gpu/states.py:72-77`) and `BaseSpeculator.propose` returns a bare `torch.Tensor` (verified `speculator.py:42-69`). The only tree in the whole stack is `suffix_decoding.py`'s CPU-side suffix *trie*, which emits a linear draft.

**The E-a PR adds the channel and leaves it `None` in both runners.** Six pieces, mirroring `rswa_prefix_lens` / `mm_req_doc_ranges` exactly:

1. `CommonAttentionMetadata` (`v1/attention/backend.py`, near `:435-441`) += `spec_parent_indices: torch.Tensor | None = None`, `spec_num_draft_tokens: torch.Tensor | None = None`, doc-commented in the `rswa_prefix_lens` house style. **Shared by both runners** — this is the substrate-free part.
2. `fill_tree_ancestor_rows(out_np, parent_indices, num_draft, query_start_loc_cpu, seq_lens_cpu) -> int` in `v1/attention/backends/utils.py`, sibling of `fill_mm_prefix_query_ranges` (`:84`). Same contract: writes into a caller-owned `(max_num_batched_tokens, 2)` int32 staging array, returns rows written, returns `0` meaning *"skip the mask_mod entirely."* Ancestor bits are one upward walk per node in preorder: `anc[s] = (1 << s) | anc[parent[s]]`, O(K).
3. `FlashAttentionMetadata` += `tree_rows_tensor: torch.Tensor | None`.
4. `FlashAttentionMetadataBuilder.__init__` allocates persistent pinned-CPU + GPU staging of `max_num_batched_tokens × 2` (`flash_attn.py:486-499` pattern); `build()` fills + `copy_(..., non_blocking=True)` (`:704-729` pattern). **Required for CUDA-graph capture** — `forward()` must never allocate (`:731-733`).
5. `FlashAttentionImpl.forward` gate beside the mm/rswa gates (`:1016-1071`):

```python
tree_mask_mod = tree_aux = None
if (attn_metadata.tree_rows_tensor is not None
        and self.vllm_flash_attn_version == 4
        and not is_dynamic_causal):
    tree_mask_mod = _make_tree_mask_mod(max_tree_slots=32)
    tree_aux = [attn_metadata.tree_rows_tensor, attn_metadata.query_start_loc]
    causal = False              # mask_mod is the entire mask
    sliding_window_size = None
```
then `mask_mod=tree_mask_mod or rswa_mask_mod_fn or mm_mask_mod` at `:1109-1110`. The existing `or`-chain is mutually exclusive by construction — **a tree + mm_prefix batch must raise, not silently let one win.** Add the explicit `NotImplementedError`.
6. `SpecDecodeMetadata.parent_indices: torch.Tensor | None = None` (+ `make_dummy` support, `:29-66`) so the V1 producer has somewhere to land later; V2's producer lands on `req_states`/`InputBatch` in Phase 2. **Neither is populated in this PR.**

Exclusions to encode: `dynamic_causal` is SM90-only and both shipped exemplars refuse it (`flash_attn.py:1021,1063`); DCP takes a different branch (`:989-1002`); cascade attention bypasses the FA4 call entirely (`:1114-1129`).

## C.6 Capability gating — and the GB10 answer

**FA4 family gate (verified).** `_is_fa4_supported()` (`flash_attn_interface.py:72-86`) requires `is_device_capability_family` ∈ {90, 100, 110}, i.e. `capability // 10 ∈ {9,10,11}`. **sm_121 (GB10 / DGX Spark) → `121 // 10 == 12` → excluded.** The whole 12.x family is out, not just 121. Fall-through: `get_flash_attn_version` → `fa_version = 2` (`fa_utils.py:98-100`) → `flash_attn_varlen_func` raises `NotImplementedError("FA2 does not support mask_mod")` (verified `:313-314`).

Additional gates to encode in test skips: FA4 is only the *default* at `device_capability.major == 10` (`fa_utils.py:95-97`); SM90/SM110 need explicit `attention_config.flash_attn_version = 4`. `VLLM_BATCH_INVARIANT` forces FA2 (`:165-170`); `head_size == 256` and `head_size > 128` (non-DeepSeek) force FA2 on Blackwell (`:176-198`).

**Is the mask expressible on the FA2 path? No, and the reason is structural.** The FA2 varlen call (`flash_attn_interface.py:317-342`) exposes exactly four mask/bias arguments: `alibi_slopes`, `causal`, `window_size_left/right`, `softcap`. No `attn_bias`, no mask tensor, no callback. `alibi_slopes` is a fixed `−slope·|i + seqlen_k − seqlen_q − j|` monotone distance penalty and is *finite*, so it cannot mask at all, let alone enforce a lossless contract. The MInference sparse path (`sparse_attn_varlen_func`, `:576-665`) fails on granularity — `block_count`/`block_offset`/`column_count`/`column_index` are `(batch, nheads, cdiv(seqlen, BLOCK_M))`, i.e. **one KV column set per BLOCK_M query rows**, so with `BLOCK_M = 64` every draft node in a tile shares one set — and it is unreachable anyway (`git grep sparse_attn_varlen_func -- vllm/` hits only its own definition).

**And per §0, we could not fix this even if we wanted to in this PR:** `vllm/vllm_flash_attn/cute/` is not tracked in the vLLM repo. Only `flash_attn_interface.py` is. Any FA2 mask hook, any SM90 `Mask.apply_mask` fix, any `fast_sampling` change is a PR against the **vllm-flash-attn** repo, on a separate release cadence.

**So GB10 gets a variant, and it is FLEX_ATTENTION.** `vllm/v1/attention/backends/flex_attention.py` is arch-agnostic (torch `flex_attention` + `create_block_mask`) and already carries the matched-pair precedent: every mask feature shipped for FA4 has a torch twin here fed from the same `CommonAttentionMetadata` field — `mm_prefix_range` (`:406`), `rswa_prefix_lens`/`rswa_window` (`:411-412`), composed via `and_masks`/`or_masks` in `get_mask_mod` (`:622-642`). The closest template is `get_rswa_mask_mod` (`:576-620`): a small `(q_req, logical_q_idx, logical_kv_idx) -> Tensor` predicate wrapped by a `final_mask_mod` applying `torch.where(is_valid, ..., False)`, with physical→logical conversion already handled by `get_paged_mask_mod`/`_convert_physical_to_logical` (`:456-481`).

**A `get_tree_mask_mod()` there is ~45 lines and reuses the identical `(pfx, anc)` row table via `torch.gather`.** One tensor, two backends. That is the sm_121 story, and shipping it in the same PR makes the portability claim concrete rather than aspirational — which is exactly the kind of thing that decides whether a mask primitive gets merged after a tree feature was deleted.

## C.7 PR shape and LOC

| file | change | ~LOC |
|---|---|---|
| `vllm/v1/attention/backend.py` | `CommonAttentionMetadata` += `spec_parent_indices`, `spec_num_draft_tokens` (default `None`) | 8 |
| `vllm/v1/attention/backends/utils.py` | `fill_tree_ancestor_rows(...)` | 70 |
| `vllm/v1/attention/backends/flash_attn.py` | metadata field · builder persistent buffers · `build()` fill · `forward()` gate · `_make_tree_mask_mod` | 150 |
| `vllm/v1/attention/backends/flex_attention.py` | `tree_rows` field · `get_tree_mask_mod()` · `and_masks` wire-in | 60 |
| `vllm/v1/spec_decode/metadata.py` | `parent_indices: torch.Tensor \| None = None` + `make_dummy` | 8 |
| `vllm/v1/worker/gpu_model_runner.py` | thread through to `cm_base` (`:2498`), **guarded, always `None` today** | 10 |
| `tests/v1/attention/test_tree_mask.py` | new, modelled on `test_mm_prefix.py` | ~300 |

**Production ≈ 306 L; the FA4-mask-only slice ≈ 230 L** — consistent with the RFC's 150–300 L estimate, and over it only if you count the FlexAttention twin (which you should ship anyway, per C.6).

**Review-surface notes the PR body must pre-empt:** why `use_fast_sampling = False`; why `causal=False` rather than composing with `causal=True` (SM90 silently drops mask_mod under causal, and we cannot fix that from this repo); why `@functools.cache`; why `__vec_size__ = 1` with the packed-`Uint32` follow-up named explicitly; **and, per AGENTS.md §1**, the duplicate-work check (`gh pr list --search "tree attention mask_mod"`), the exact test commands with results, and an explicit AI-assistance statement. Given that tree attention was *deleted* upstream, the PR body must lead with why a **mask primitive with no producer** is the right re-entry point.

---

# OPEN QUESTIONS FOR THE RFC THREAD (5)

1. **RNG node identity.** `gumbel.py:111` and `rejection_sampler_utils.py:569` both key on absolute `pos`, and DSpark *deliberately* relies on the predecessor's key (`dspark/speculator.py:139-149`). In a tree, siblings at one depth share `pos` and therefore share `u` — correlated draws, no residual renormalization, silent acceptance-rate loss. Do maintainers want the stream re-keyed on `(request, node_index)`, on a path hash, or on `(pos, sibling_ordinal)`? Whatever is chosen must match bit-for-bit between drafter and verifier, and must preserve today's chain behavior exactly.

2. **Should tree width ride `AdaptiveVerificationManager`, or be a peer mechanism?** AV is method-gated to `dspark` (`config/speculative.py:528-530`) and carries hard constraints (no LoRA, no PP>1, no eager, one-rejection-chunk cap). Riding it gets a shipped, profiled, connected-subtree budgeter for ~10 lines; forking it duplicates the cost model. Which do you prefer — and if riding it, are you willing to widen the method gate for a `tree` method?

3. **Carry budget: struct or scalar?** Is `SpecCarryBudget(temporal_slots, conv_tokens, max_branch_depth)` the shape you want on `MambaSpec`, or would you rather keep one integer and have tree methods report `num_speculative_blocks = num_tree_nodes`, accepting conv over-allocation of `(nodes − depth)` token columns per request per layer? The scalar is a one-line change; the struct is the honest model of two different carry geometries.

4. **Non-FA4 portability.** Is a FLEX_ATTENTION twin an acceptable non-FA4 path (sm_121/GB10, sm_8x), or do you want the tree mask expressed in the FA2 kernel? The latter is a change in the **vllm-flash-attn** repo (FA2 varlen exposes no mask hook, and `vllm/vllm_flash_attn/cute/` is not tracked in vllm), so it is a separate PR on a separate cadence — worth agreeing before we build.

5. **Re-entry framing.** Tree attention was deleted upstream (`git grep -l "tree_attn|TreeAttention|spec_token_tree" main -- vllm/ tests/` → nothing). Is a **mask primitive with no producer** the re-entry point you want, or would you rather see the whole path behind one flag with an end-to-end acceptance-rate benchmark on a hybrid model before *any* of it lands? Our preference is the former — three independently useful PRs (the `stride_indices_tok` fix, the `BaseSpeculator` ABC completion, the mask primitive) that each stand alone — but we will follow the maintainers' sequencing.

---

# THE THREE FACTS THAT MOST STRENGTHEN THE RFC

**1. The runtime tree-shape chooser already exists upstream, and its own docstring states the tree condition.**
`_assign_draft_token_budget` (`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:34-65`, `torch.compile(dynamic=True)`) scores every `(request, step)` slot by `confidence_probs[idx_mapping].cumprod(dim=1)` and admits a global `topk(draft_budget)`, documented as *"Survival only decreases along a request, so a global top-k always admits continuously along steps with a request."* Path survival is monotone non-increasing along a root→node path, so the identical rule over `[request, node]` admits a **connected subtree**. Swap `cumprod(dim=1)` for a parent-scan and the `capacities`-in/`capacities`-out contract, the `-inf` masking, and the profiled cost model `get_num_tokens()` (`:267-335`) all carry over verbatim. This converts *"how do you pick the tree shape adaptively at runtime without a CPU sync?"* — the question that usually sinks tree proposals — into a ten-line diff against shipped, benchmarked code.

**2. The maintainers have already merged the propose-a-tree/verify-losslessly pattern; it just linearizes at the end.**
`DFlash2Speculator` (`vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:15-106`) walks a `[step, prev_k, k]` transition-score lattice with `_selector_walk_kernel` + `gumbel_noised_argmax`, carrying `previous = index` across steps, writes one linearized path into `draft_tokens`, and `_cache_draft_logits_kernel` writes only the K candidate columns into a `-inf`-filled `draft_logits` so the **unmodified** rejection sampler consumes a truncated-but-correct distribution. RFC Phase 2's re-linearization is therefore not a novel proposal — it is a generalization of code that is already in `main`, already lossless, and already forces V2 (`config/vllm.py:673-678`). It is also the cheapest credible first landing: a tree proposal that re-linearizes touches zero attention masks, zero rejection kernels, and zero scheduler code.

**3. Non-uniform, device-decided verification width is already the shipped contract on two backend families — answered upstream, by other people, for other reasons.**
`AttentionBackend.supports_device_cpu_query_lens_mismatch()` (`vllm/v1/attention/backend.py:208`) with the CPU token count documented as an *upper bound* (`gpu/input_batch.py:56-58`) and the truth in a device-side `torch.cumsum` into `query_start_loc` (verified `adaptive_verification.py:418-434`); enabled on FlashInfer trtllm-gen SM100 (**#52157** `6a9c69fa8`) and the DSv4 MLA indexer SM90 (**#52795** `5df31ea52`); with varlen decode CUDA graphs accepting any 1..`decode_query_len` per-request mix (`cudagraph_utils.py:230-247`). Add **#51410** `44351f81d` (2026-08-08), which un-skipped hybrid MTP spec decode on MRV2 and added `test_mtp_correctness[qwen3_5-hybrid]` to `.buildkite/test_areas/model_runner_v2.yaml:123-125`. The two objections a tree-verify RFC normally has to argue its way past — *"you can't vary per-request query length under CUDA graphs"* and *"hybrid + MRV2 + spec decode isn't real yet"* — are both already answered in `main`, with PR numbers we can cite instead of arguments we have to win.