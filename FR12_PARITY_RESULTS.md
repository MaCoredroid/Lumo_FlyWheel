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

Per-layer max-abs table:

| Layer | Type | Max abs |
|---:|---|---:|
| 0 | linear_attention | 0.015625 |
| 1 | linear_attention | 0.00732421875 |
| 2 | linear_attention | 0.046875 |
| 3 | full_attention | 0.0078125 |
| 4 | linear_attention | 0.006591796875 |
| 5 | linear_attention | 0.009765625 |
| 6 | linear_attention | 0.015625 |
| 7 | full_attention | 0.006927490234375 |
| 8 | linear_attention | 0.01171875 |
| 9 | linear_attention | 0.01171875 |
| 10 | linear_attention | 0.016845703125 |
| 11 | full_attention | 0.05419921875 |
| 12 | linear_attention | 0.0244140625 |
| 13 | linear_attention | 0.02294921875 |
| 14 | linear_attention | 0.026611328125 |
| 15 | full_attention | 0.0296630859375 |
| 16 | linear_attention | 0.0279541015625 |
| 17 | linear_attention | 0.046875 |
| 18 | linear_attention | 0.140625 |
| 19 | full_attention | 0.125 |
| 20 | linear_attention | 0.0625 |
| 21 | linear_attention | 0.0634765625 |
| 22 | linear_attention | 0.1796875 |
| 23 | full_attention | 0.0831298828125 |
| 24 | linear_attention | 0.1875 |
| 25 | linear_attention | 0.1328125 |
| 26 | linear_attention | 0.1875 |
| 27 | full_attention | 0.296875 |
| 28 | linear_attention | 0.1634521484375 |
| 29 | linear_attention | 0.203857421875 |
| 30 | linear_attention | 0.18359375 |
| 31 | full_attention | 0.19677734375 |
| 32 | linear_attention | 0.40673828125 |
| 33 | linear_attention | 0.5009765625 |
| 34 | linear_attention | 0.265625 |
| 35 | full_attention | 0.40625 |
| 36 | linear_attention | 0.146484375 |
| 37 | linear_attention | 0.19091796875 |
| 38 | linear_attention | 0.33203125 |
| 39 | full_attention | 0.2244873046875 |
| 40 | linear_attention | 0.2734375 |
| 41 | linear_attention | 0.2734375 |
| 42 | linear_attention | 0.671875 |
| 43 | full_attention | 0.34375 |
| 44 | linear_attention | 0.3330078125 |
| 45 | linear_attention | 0.314453125 |
| 46 | linear_attention | 0.78125 |
| 47 | full_attention | 0.8125 |
| 48 | linear_attention | 0.3758544921875 |
| 49 | linear_attention | 0.3359375 |
| 50 | linear_attention | 1.25 |
| 51 | full_attention | 1.40625 |
| 52 | linear_attention | 0.9375 |
| 53 | linear_attention | 0.62890625 |
| 54 | linear_attention | 1.7421875 |
| 55 | full_attention | 2.25 |
| 56 | linear_attention | 0.84375 |
| 57 | linear_attention | 0.81640625 |
| 58 | linear_attention | 4.0 |
| 59 | full_attention | 3.65234375 |
| 60 | linear_attention | 4.25 |
| 61 | linear_attention | 1.87109375 |
| 62 | linear_attention | 2.5 |
| 63 | full_attention | 53.0 |

Depth deltas at the first confirmed divergent layer:
`[0.015625, 0.00390625, 0.015625, 0.0078125, 0.01171875]`.

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
