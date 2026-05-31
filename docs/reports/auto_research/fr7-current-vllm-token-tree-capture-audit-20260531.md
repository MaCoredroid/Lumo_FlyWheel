# FR7 Current-vLLM Token-Tree Capture Audit - 2026-05-31

## Scope

Non-destructive Step 1 audit for FR7. The running 0.19 baseline container
`lumo-vllm-track-b-suffix` was not restarted or modified.

## Artifacts

- Current-main source checkout: `/tmp/lumo_vllm_main_audit/vllm`
  - `main` at `6bdabbad5bce747865fd3a249658518a4269cc22`
- Release source checkout: `/tmp/lumo_vllm_main_audit/vllm-v0.22.0`
  - `v0.22.0` at `0b3ba88f165976e77ca5e6a7a3f5bba4562b80af`
- Separate audit image: `lumo-vllm-audit:v0.22.0-cu129-min`
  - Built from the released v0.22.0 `+cu129` aarch64 wheel.

## Findings

Current vLLM `main` and v0.22.0 do not contain the 0.19 token-tree API:

- `speculative_token_tree`: 0 hits
- `propose_tree`: 0 hits
- `TreeAttention`: 0 hits
- `TREE_ATTN`: 0 hits
- `tree_attn`: 0 hits

The same result was confirmed inside the separate v0.22 audit image against both
the installed wheel and cloned source.

CUDA graph support in current vLLM has improved for Qwen3.5/GDN but does not
answer the token-tree question because TreeAttention is absent:

- `GDNAttentionMetadataBuilder._cudagraph_support = AttentionCGSupport.UNIFORM_BATCH`
- `LinearAttentionMetadataBuilder._cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`
- base attention metadata default remains `AttentionCGSupport.NEVER`

PR `#35520` is merged into current `main`: commit `7a08b34fb` is an ancestor of
`6bdabbad`. That confirms MRV2 Qwen3.5/Mamba hybrid support is present upstream.

## Conclusion

There is no current-main or v0.22 native single-request token-tree capture mode
to benchmark directly. The 0.19 TreeAttention path remains the correctness
implementation surface for FR7. For the later speed phase, current upstream gives
useful MRV2/GDN capture infrastructure, but token-tree support would require a
minimal forward-port/fork before a true GDN + TreeAttention FULL-capture audit can
be run.
