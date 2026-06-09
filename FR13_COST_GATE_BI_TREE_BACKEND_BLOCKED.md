# FR13 Cost-Gate Backend Read-First Blocker

Date: 2026-06-09

Commit: this bind

## Question

Can the forked-FA2 `-inf` tree-bias path boot under `VLLM_BATCH_INVARIANT=1` as a
batch-invariant-accepted `FLASH_ATTN` backend?

## Read-First Evidence

Batch-invariant guard:
`/tmp/vllm_live_019/vllm/model_executor/layers/batch_invariant.py:993-1017`

The accepted backend enum names are:

- `FLASH_ATTN`
- `TRITON_ATTN`
- `FLASH_ATTN_MLA`
- `TRITON_MLA`

If the selected backend is not one of those enum values, the guard raises:
`VLLM batch_invariant mode requires an attention backend in ...`.

FR13 forked-FA2 tree launcher:
`scripts/fr13_launch_forked_fa2_tree_server.sh:11,74-75,88-89`

- Default selected attention backend: `ATTENTION_BACKEND=${ATTENTION_BACKEND:-TREE_ATTN}`.
- Batch-invariant envs are passed through as `VLLM_BATCH_INVARIANT` and
  `LUMO_BATCH_INVARIANT_VLLM`.
- `FR13_FA2_PREFILL_NATIVE=1` is live by default in this launcher.

Live tree backend registration:
`/tmp/vllm_live_019/vllm/v1/attention/backends/registry.py:44-79`

- `AttentionBackendEnum.FLASH_ATTN` maps to native
  `vllm.v1.attention.backends.flash_attn.FlashAttentionBackend`.
- `AttentionBackendEnum.TREE_ATTN` maps to
  `vllm.v1.attention.backends.tree_attn.TreeAttentionBackend`.

Live tree backend name:
`/tmp/vllm_live_019/vllm/v1/attention/backends/tree_attn.py:50-52`

- `TreeAttentionBackend.get_name()` returns exactly `TREE_ATTN`.

FR13 patch:
`scripts/fr13_patch_fa2_tree_bias.py`

- Adds the `varlen_fwd_tree_bias` FA2 op.
- Routes tree decode to `flash_attn_varlen_func(..., tree_bias=tree_bias)` inside the
  `TREE_ATTN` backend when `FR13_FA2_TREE_BIAS=1`.
- Does not register the tree-bias backend as `AttentionBackendEnum.FLASH_ATTN`.
- Does not make `TreeAttentionBackend.get_name()` return `FLASH_ATTN`.

## Verdict

No. The forked-FA2 tree-bias path is still selected as backend name `TREE_ATTN`,
not `FLASH_ATTN`. Therefore `VLLM_BATCH_INVARIANT=1` cannot boot this path through
the accepted backend allowlist.

The decisive GPU Step2 is blocked under the strict gate. Running it anyway would
only reproduce the known `TREE_ATTN` batch-invariant boot failure from `e9f9267c`,
or require a fallback/renamed backend path that has not been validated and would
not satisfy this cost-gate.

## Bound Action

No GPU server was booted. No fallback run was bound. GDN scan remains exonerated
per the existing bindings; it was not re-investigated.
