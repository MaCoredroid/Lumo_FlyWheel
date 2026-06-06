# FR-12 Layer Parity Results

## 2026-06-06 lossless-first pivot

Scope: parity diagnosis only. The serving tree GDN scan is restored to the
original pre-WY forward-substitution path for stability while diagnosing
tree-spine-vs-native divergence. This is **not** a proposed final lossless fix:
the revised plan still forbids per-spine copy, separate native-shape relaunches,
weight re-streaming, and dense slow paths as the route to production.

Existing same-event captures used:

- Tree: `output/fr10_match0_layer_tree_20260605T100044Z/logs/layer_tree.call0.pt`
- Tree logits: `output/fr10_match0_layer_tree_20260605T100044Z/logs/spine_tree.call0.pt`
- Native MTP-5: `output/fr10_match0_layer_native_20260605T101521Z/logs/layer_native.call0.pt`
- Native logits: `output/fr10_match0_layer_native_20260605T101521Z/logs/spine_native.call0.pt`
- Compare JSON: `output/fr10_match0_layer_compare_20260605T_now/match0_layer_compare.json`
- Aligned-row compare JSON: `output/fr10_match0_layer_compare_20260605T_now/aligned12_layer_compare.json`

Gate facts:

- Spine draft tokens match on the selected request:
  `[271, 248069, 271, 8179, 5073]`.
- Layer input hidden rows are equal: max abs `0.0` at all five depths.
- First composite layer-output divergence is decoder layer 0,
  `linear_attention`, max abs `0.015625`.

Per-layer head of table:

| Layer | Type | Max abs | Depth deltas |
|---:|---|---:|---|
| 0 | linear_attention | 0.015625 | [0.015625, 0.00390625, 0.015625, 0.0078125, 0.01171875] |
| 1 | linear_attention | 0.00732421875 | [0.00732421875, 0.001953125, 0.0029296875, 0.0013427734375, 0.003173828125] |
| 2 | linear_attention | 0.046875 | [0.046875, 0.0078125, 0.015625, 0.0234375, 0.015625] |
| 3 | full_attention | 0.0078125 | [0.0078125, 0.00537109375, 0.00360107421875, 0.003173828125, 0.00732421875] |
| 4 | linear_attention | 0.006591796875 | [0.0048828125, 0.006591796875, 0.004150390625, 0.003994598984718323, 0.0037841796875] |
| 5 | linear_attention | 0.009765625 | [0.00390625, 0.009765625, 0.005859375, 0.00390625, 0.0068359375] |
| 6 | linear_attention | 0.015625 | [0.01171875, 0.008056640625, 0.015625, 0.00390625, 0.00927734375] |
| 7 | full_attention | 0.006927490234375 | [0.00531005859375, 0.005889892578125, 0.005126953125, 0.0052490234375, 0.006927490234375] |

Tail confirms compounding:

| Layer | Type | Max abs |
|---:|---|---:|
| 59 | full_attention | 3.65234375 |
| 60 | linear_attention | 4.25 |
| 61 | linear_attention | 1.87109375 |
| 62 | linear_attention | 2.5 |
| 63 | full_attention | 53.0 |

Source-read decomposition for first layer:

`Qwen3NextDecoderLayer.forward` runs input RMSNorm, then
`QwenGatedDeltaNetAttention`, then post-attention RMSNorm + MLP. For a
`linear_attention` layer, `QwenGatedDeltaNetAttention.forward_cuda` consists of:

1. `in_proj_qkvz(hidden_states)` and `in_proj_ba(hidden_states)` fp8 projections.
2. Spec conv path via `causal_conv1d_update`.
3. Spec recurrent GDN path via `fused_sigmoid_gating_delta_rule_update` or the
   FR10 tree replacement.
4. `RMSNormGated` plus `out_proj`.
5. Decoder layer post-attention RMSNorm/residual and MLP.

Current verdict: the first confirmed divergence boundary is the layer-0
`linear_attention` composite output. The existing capture does not yet isolate
which internal sub-kernel within layer 0 first diverges; the next required
diagnostic is layer-0 sub-taps around `in_proj_qkvz`, `in_proj_ba`, conv output,
GDN core output, gated norm output, `out_proj`, post-attention norm, and MLP.
