# FR10: the tree kernel must match the ORIGINAL vLLM GDN kernel's dtype flow exactly

Author: Claude (researcher/red-team). Date: 2026-06-03.
User directive: "bf16 in, fp32 state per mamba_ssm_dtype=float32 need to be same
as original vllm kernel." For the tree kernel to be a **lossless drop-in**, its
dtype flow must replicate the production vLLM 0.22 GDN kernel exactly — not a
hand-cast approximation. Source: `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`.

## Which kernel is "original"? — determine the ACTIVE backend, then match it
`_resolve_gdn_prefill_backend(vllm_config)` picks the active GDN prefill/verify
backend (the multi-token forward used for prefill AND for our tree verify):
- `additional_config.gdn_prefill_backend` default = `"auto"`.
- `auto` + flashinfer libs intact (CUDA 13 → `_is_libs_cu13_install_intact()`) ⇒
  **flashinfer** (`forward_cuda` → `fi_chunk_gated_delta_rule`).
- `cutedsl` is opt-in ⇒ `forward_cutedsl`.
- else ⇒ **native Triton/FLA** (`forward_native` → `fla_chunk_gated_delta_rule`).

ACTION: in the serving container, instantiate `ChunkGatedDeltaRule()` and read
`.gdn_prefill_backend` (and `_log_gdn_backend_decision` output) under the SAME
config used for E3/E5. Match THAT backend. codex's current `native_gpu_oracle`
uses the FLA native fallback with a hand `.to(bfloat16)` — that may not be the
active backend and its casts differ from the real wrapper.

## Exact dtype contract — flashinfer CUDA path (`fi_chunk_gated_delta_rule`)
```
q = l2norm_fwd(q)                 # reuse vLLM l2norm_fwd (fla/ops/l2norm.py); do NOT reimplement
k = l2norm_fwd(k)                 # q,k stay in ACTIVATION dtype (bf16)
v : activation dtype (bf16), unchanged
g : fi_g = g.to(float32);  passed as torch.exp(fi_g)   # gate exp'd in fp32 OUTSIDE kernel
beta : fi_beta = beta.to(float32)                        # fp32
initial_state : .to(float32)                             # fp32   (mamba_ssm_dtype=float32)
scale = 1/sqrt(key_head_dim)      # =1/sqrt(128)
internal chunk solve : fp32 accumulation
output o : activation dtype (bf16)
final_state : fp32
A_log : fp32 parameter; gdn_gating = -exp(A_log.float()) * softplus(a.float()+dt_bias.float())
```
So: **q/k/v bf16 (q/k l2-normed), g=exp(fp32), beta fp32, state fp32, accumulate
fp32, output bf16.** (My earlier "bf16 in, fp32 state" was directionally right but
incomplete — the gate is pre-exp'd in fp32 and the output is cast back to bf16.)

For cutedsl/native backends, read `forward_cutedsl` / `forward_native` +
`fla_chunk_gated_delta_rule` and replicate THEIR casts instead, if that is the
active backend.

## Gate D test (corrected)
Single spine path (no branch), matched precision, against the EXACT active
backend wrapper (e.g. call `ChunkGatedDeltaRule().forward_cuda(...)` /
`fi_chunk_gated_delta_rule(...)`), not a hand-cast bf16 FLA call. On a single path
the tree-ancestry mask == the linear causal mask ⇒ algebraically identical ⇒
expect agreement to floating-point round-off. If the tree kernel is built fp32-in
while production is bf16-in, that gap (~7e-4) is a dtype mismatch, not a kernel
defect — but the DELIVERABLE must run the production dtype to be lossless, so
build it bf16-in/fp32-state/bf16-out to match.

## Note on the E3/E5 baseline stack
E3/E5 were measured on vLLM 0.19; the tree kernel targets 0.22. "Lossless" must be
defined vs the 0.22 native non-speculative decode on the SAME serving config —
which is exactly why P0 must (re)freeze the spines=1 baseline on the pulled/new
stack before any speed claim. See [[project_fr10_gdn_tree_kernel_track]].
