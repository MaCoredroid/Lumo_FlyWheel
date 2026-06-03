# FR10 Phase 4 Real-Tensor De-Risk

Date: 2026-06-03
Branch: `fr10-gdn-tree-kernel`
Stack: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
Local image ID: `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`

## Purpose

Prove the FR10 GDN tree verifier on real Qwen3.6 GDN tensors before serving-loop surgery.
The validation uses a captured layer-0 speculative GDN payload from cu130-nightly with an
11-node two-spine tree, real production dimensions, and the real native decode/update primitive.

## Captured Tensor

Payload path:
`output/fr10_phase4_tree_capture_probe_20260603/tensors/language_model_model_layers_0_linear_attn_spec_gdn.pt`

The payload is intentionally not committed because it is 6.6 GiB. The committed compact evidence is:

- `output/fr10_phase4_tree_capture_probe_20260603/tensor_summary.json`
- `output/fr10_phase4_tree_capture_probe_20260603/real_tensor_validation.json`

Tree parent vector:
`[-1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]`

Tensor contract:

- q/k heads: `16`
- value/state heads: `48`
- head dim: `128`
- `initial_state_before_spec`: fp32, shape `[1126,48,128,128]`
- q/k/v spec tensors: bf16

## Command

```bash
docker run --rm --gpus all --ipc=host \
  -v /home/mark/shared/lumoFlyWheel:/workspace \
  -w /workspace \
  -e PYTHONPATH=/workspace/src \
  --entrypoint bash \
  vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776 \
  -lc "python3 scripts/fr10_phase4_real_tensor_validation.py \
    --payload output/fr10_phase4_tree_capture_probe_20260603/tensors/language_model_model_layers_0_linear_attn_spec_gdn.pt \
    --out output/fr10_phase4_tree_capture_probe_20260603/real_tensor_validation.json"
```

## Results

GQA grouping:

- confirmed mapping: consecutive / `repeat_interleave`
- strided mapping diagnostic is loudly wrong: output `2.422473669052124`, state `28.254070281982422`

Native-tree contamination:

- cu130 native linear GDN vs tree reference on non-linear branch nodes: `0.5618356466293335`
- This proves vLLM's native `speculative_token_tree` is not GDN-lossless on Qwen3.6: tree attention is wired, but GDN recurrent state is still linear.

FR10 verifier vs production decode/update:

- tree kernel vs native `fused_sigmoid_gating_delta_rule_update` serial root-to-node paths:
  - output max abs: `7.450580596923828e-09`
  - state max abs: `7.62939453125e-06`
- tree kernel vs GQA tree reference state: `5.7220458984375e-06`
- `gate_d_real_tensor_decode_update_pass=true`

The older `chunk_gated_delta_rule` serial wrapper comparison is retained only as a prefill-wrapper diagnostic on a decode-captured payload:

- output max abs: `0.015625`
- state max abs: `0.10932159423828125`

It is not the Step 2/Gate D decode oracle.

## Red-Team Reproduction

Claude red-team independently reproduced the exact Step 2 numbers in a fresh cu130 container:

- tree kernel vs native decode/update output `7.45e-09`
- tree kernel vs native decode/update state `7.63e-06`
- native linear GDN contamination on non-linear branch nodes `0.5618`

The red-team also verified the oracle is non-vacuous: `native_update_serial_per_path`
runs the native production primitive over each node's full root-to-node path, including branch nodes.

## Conclusion

Step 2 de-risk is complete. The FR10 kernel is proven correct on real model tensors against the native production decode/update primitive, and native vLLM tree spec is empirically contaminated for GDN branches. Proceed to Phase 4 integration: extend `GDNAttentionMetadata` with static tree descriptors, route GDN spec verification to the FR10 tree kernel when tree metadata is present, and commit accepted-path recurrent state through the native decode/update primitive.
