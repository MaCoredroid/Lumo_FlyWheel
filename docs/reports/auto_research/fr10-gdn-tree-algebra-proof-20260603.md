# FR10 GDN Tree-Algebra Proof

Date: 2026-06-03

Status: Phase 1 CPU oracle gate passed.

This report proves the Gated DeltaNet tree algebra needed before any CUDA work.
It is not the final implementation kernel. The CPU recurrent implementation is
used only as the correctness oracle; the production FR10 kernel must be a
GPU-only, CUDA-graph-capturable Triton/CUDA implementation.

## Grounding

The proof uses the vLLM 0.22 CPU GDN reference:

`/tmp/vllm-0.22-src/vllm-0.22.0/vllm/model_executor/layers/mamba/ops/cpu/recurrent_gated_delta_rule.py`

The reused oracle functions are:

- `recurrent_gated_delta_rule()`
- `gdn_gating()`
- `l2norm()`
- `chunk_gated_delta_rule()` as an independent single-path chunk oracle

Qwen3.6 dimensions come from `/models/qwen3.6-27b-fp8/config.json` and
`QwenGatedDeltaNetAttention` in vLLM 0.22:

| Field | Value |
|---|---:|
| hidden layers | 64 |
| GDN / `linear_attention` layers | 48 |
| `linear_num_key_heads` | 16 |
| `linear_num_value_heads` | 48 |
| `linear_key_head_dim` | 128 |
| `linear_value_head_dim` | 128 |
| `linear_conv_kernel_dim` | 4 |
| recurrence state dtype | float32 |
| scale | `linear_key_head_dim ** -0.5` |
| decode L2 norm | enabled |

The P1 proof operates on real post-conv GDN tensors `q/k/v/g/beta`. Convolution
and projection are upstream of the recurrent algebra gate.

## Recurrence

For one head, vLLM's CPU recurrent rule updates state `S` as:

```text
S'       = exp(g_t) * S
kv_mem   = S' @ k_t
delta    = (v_t - kv_mem) * beta_t
S_t      = S' + delta outer k_t
out_t    = S_t @ q_t
```

Therefore a node state is a pure function of the committed prefix state and the
tokens on that node's root path. A sibling cannot change a trunk node unless the
implementation uses a linear mask that lets siblings interact or mutates a shared
parent state.

## Tree Chunk Form

vLLM's chunked rule builds a lower-triangular linear-time solve. FR10 replaces
the linear causal mask with a tree-ancestry mask.

For topologically ordered nodes:

```text
A[i,j] = 1 iff node j is a strict ancestor of node i
cum_g[i] = sum(g[j] for j in root_path(i))
decay[i,j] = exp(cum_g[i] - cum_g[j]) for ancestor pairs
system = I + A * interaction * decay
```

Because every ancestor precedes its descendant, `A` is a subset of the strictly
lower-triangular mask. `system` remains lower triangular with unit diagonal, so
`torch.linalg.solve_triangular(..., upper=False)` is valid. Each row depends
only on ancestors, not on preceding siblings.

Per-node state is reconstructed with the tree-visible rows:

```text
S_i = S0 * exp(cum_g[i])
      + sum_{j in ancestors(i) union {i}}
          transformed_value[j] outer k[j] * exp(cum_g[i] - cum_g[j])
```

Outputs use the same tree-visible mask for `q @ k.T`: ancestors plus self are
visible; siblings are not.

## Implemented Artifacts

- `scripts/fr10_gdn_tree_algebra_reference.py`
- `tests/test_fr10_gdn_tree_algebra.py`
- `scripts/fr10_p0_audit_baseline_artifacts.py`
- `output/fr10_p0_baseline_audit_20260603.json`

Evaluators:

1. Serial per-path oracle replaying `recurrent_gated_delta_rule()`.
2. Packed private-state evaluator, retained as a correctness baseline.
3. Tree-ancestry masked chunk evaluator, the algebraic trunk-sharing form for the
   future kernel.
4. Single-path `chunk_gated_delta_rule()` oracle.

## Gates Passed

Command:

```bash
.venv/bin/pytest -q tests/test_fr10_gdn_tree_algebra.py
```

Result:

```text
15 passed in 3.60s
```

Coverage:

- Node families `{2,3,6,8,14}`.
- Random small trees with spine depth 1-6 and branch width capped at 3.
- fp32 parity with `atol=2e-5`, `rtol=2e-5`.
- bf16/fp16 dtype sweep with documented synthetic tolerance `8e-2`.
- Packed and tree-masked chunk state/output/logits match serial per-path.
- Single-path chunk oracle matches serial per-path.
- Appending a sibling leaf does not change trunk node state/logits.
- Accepted-path final state equals serial native decode for that path.

Standalone smoke:

```bash
.venv/bin/python scripts/fr10_gdn_tree_algebra_reference.py
```

Result: 8-node synthetic tree, max state delta `0.0`, max logit delta `0.0`
against the serial oracle.

## Negative Controls

All required red-team controls are represented:

- Linear-mask leak: using plain `tril` instead of tree ancestry fails parity.
- Shared mutable parent state: mutating a reused parent state fails parity.
- Greedy drift: synthetic logits use a deterministic 256-wide projection. The
  test asserts per-node argmax identity between serial and tree-masked logits
  and keeps `logit_delta <= 2e-5`; top-2 margin is diagnostic only. Real Gate B
  is byte-exact-by-identical-kernel: public path0 uses the same verifier kernel,
  not a tolerance-based selector.
- Longest-accepted hidden winner: the Gate C order-statistic selector shape fails
  the target distribution check.

## P0 Audit Result

The accepted FR9 baseline tag
`fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z` has:

- sampled task outcomes,
- per-event accept counters,
- engine-step latency,
- all 16 per-task `vllm_request_metrics.jsonl` files nonzero,
- x86 metadata for all tasks.

Missing for full FR10 P0 without targeted follow-up:

- greedy token streams,
- CUDA graph/capture status,
- kernel-level Nsight traces,
- exact stack version record for CUDA, driver, PyTorch, Triton, vLLM,
  FlashAttention, and model revision.

The audit recommends not rerunning the full B=4 temp=0.6 spines=1 campaign for
already-present fields. Run only targeted collection for missing streams unless
an existing artifact location is supplied.

## GPU Kernel Implication

The Phase 2 kernel should fork the vLLM 0.22 FLA ops under:

`/tmp/vllm-0.22-src/vllm-0.22.0/vllm/model_executor/layers/fla/ops/`

Surgical points:

- `solve_tril.py`: linear triangular mask to tree-ancestry mask.
- `chunk_scaled_dot_kkt.py`: KKT interaction to ancestry-masked interaction.
- `cumsum.py`: linear cumsum to ancestor-path gate accumulation.
- `chunk_delta_h.py` and `chunk.py`: chunk state recurrence to per-node tree
  state.

The kernel must run inside `lumo-vllm-audit:v0.22.0-cu129-min`, use static
padded descriptors for `{2,3,6,8,14}`, allocate nothing during capture, and fail
closed on unwarmed shapes.
