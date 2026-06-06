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

## 2026-06-06 L0 GDN Sub-Kernel Origin

Live source check:

- In `vllm/vllm-openai:cu130-nightly`, Qwen full attention uses
  `output[:], _ = self.o_proj(attn_output)` in
  `vllm/model_executor/models/qwen3_next.py`.
- Layer 0 is `linear_attention`; its live GDN implementation is
  `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, where the GDN
  projection call is `output[:num_tokens], _ = self.out_proj(core_attn_out)`.
  The capture stage is still named `o_proj_out` for the user-facing boundary.

Artifacts:

- Tree sub-kernels:
  `output/fr12_l0_subkernel_20260606T004119Z/tree/logs/subkernel_tree.pt`
- Tree logits:
  `output/fr12_l0_subkernel_20260606T004119Z/tree/logs/spine_tree.pt`
- Native sub-kernels:
  `output/fr12_l0_subkernel_20260606T004119Z/native24b/logs/subkernel_native.pt`
- Native debug logits:
  `output/fr12_l0_subkernel_20260606T004119Z/native24b/logs/native_sampler_debug.jsonl`
- Compare JSON:
  `output/fr12_l0_subkernel_20260606T004119Z/subkernel_compare_req0_native24b.json`

Alignment:

- Matched spine tokens:
  `[71093, 12305, 198, 727, 9637]`.
- Tree rows:
  `[0, 1, 2, 4, 6]`.
- Native rows:
  `[0, 1, 2, 3, 4]`.

Sub-kernel diffs:

| Stage | Max abs | Mean abs max depth | Per-depth max abs |
|---|---:|---:|---|
| conv1d_out | 0.125 | 0.00015655082825105637 | [0.125, 0.125, 0.125, 0.125, 0.125] |
| gdn_scan_out | 0.015625 | 0.000011691838153637946 | [0.015625, 0.0078125, 0.015625, 0.015625, 0.015625] |
| gate_out | 0.001953125 | 0.00001911621257022489 | [0.0009765625, 0.001953125, 0.001953125, 0.001953125, 0.001953125] |
| o_proj_out | 0.00390625 | 0.00019562167290132493 | [0.00048828125, 0.00390625, 0.0015869140625, 0.001953125, 0.0009765625] |

Measured origin: `conv1d_out`. The first nonzero tree-spine vs native
sub-kernel gap appears immediately after causal conv, before the GDN scan.
The scan propagates an already-diverged input and has max abs `0.015625` at
the core output on this matched event.

## 2026-06-06 Corrected L0 Sub-Kernel Parity

This supersedes the earlier conv-detail interpretation while preserving the
stage-level first-divergence verdict.

Harness fixes:

- Native conv detail now captures the conv-state prior before
  `causal_conv1d_update` mutates the state.
- Native `pre_conv_rows/window/tap_products` now use a cloned pre-conv tensor;
  the previous alias was mutated in place by the native conv kernel.
- Compare scripts now derive the active tree spine rows from capture/logit
  metadata instead of stale hard-coded tree rows.

Artifacts refuted:

- The apparent zero/missing `conv_state` most-recent column was a post-update
  conv-state read artifact.
- The apparent conv-window/tap mismatch was a native pre-conv aliasing artifact.

Corrected artifacts:

- Tree capture:
  `output/fr12_corrected_l0_parity_20260606T032230Z/tree/logs/subkernel_tree.pt`
- Native corrected capture:
  `output/fr12_corrected_l0_parity_20260606T032230Z/native_clonefix/logs/subkernel_native.pt`
- Corrected subkernel compare:
  `output/fr12_corrected_l0_parity_20260606T032230Z/corrected_subkernel_compare_clonefix.json`
- Corrected layer-output compare:
  `output/fr12_corrected_l0_parity_20260606T032230Z/corrected_layer_compare_clonefix.json`

Corrected same-event tokens:
`[71093, 12305, 198, 727, 9637]` tree == native.

Corrected L0 sub-kernel table:

| Stage | Max abs | Mean abs max depth |
|---|---:|---:|
| pre_conv | 0.0 | 0.0 |
| conv1d_out | 0.125 | 0.00015655082825105637 |
| gdn_scan_out | 0.015625 | 0.000011691838153637946 |
| gate_out | 0.001953125 | 0.00001911621257022489 |
| o_proj_out | 0.00390625 | 0.00019562167290132493 |

Corrected conv detail checks:

| Detail | Max abs | Mean abs |
|---|---:|---:|
| conv prior window | 0.0 | 0.0 |
| full conv window | 0.0 | 0.0 |
| fp32 tap products | 0.0 | 0.0 |
| bf16 tap products | 0.0 | 0.0 |

Corrected first real divergence: `conv1d_out`, specifically inside the conv
kernel accumulation / activation / output-cast boundary. The tree and native
inputs, prior state, causal windows, and tap products are byte-identical at the
captured spine rows, but the tree manual PyTorch conv output differs from the
native Triton `causal_conv1d_update` output by up to `0.125` (bf16-scale ULPs at
large positive activations).

Layer-output sanity check remains real:

- Layer input hidden max abs: `0.0`
- First layer-output divergence: layer 0 `linear_attention`
- Layer 0 max abs by depth:
  `[0.0003662109375, 0.015625, 0.001953125, 0.00390625, 0.0009765625]`
- Layer 0 max abs: `0.015625`
- This run's layer 63 max abs: `6.0`

Verdict: the corrected first-divergence sub-kernel is still causal conv, but not
because of the conv prior-state bank/slot, causal window, or tap inputs. The
remaining seam is native-kernel rounding in `causal_conv1d_update` versus the
tree manual conv implementation.

## 2026-06-06 Native-Spine Conv Splice

Implementation:

- Flag-gated splice: `FR12_TREE_CONV_NATIVE_SPINE=1`.
- Path0 spine rows are routed through the same native Triton
  `causal_conv1d_update` output path used by native MTP.
- Branch rows remain on the tree ancestry conv path.
- Default remains off.

Artifacts:

- Tree capture:
  `output/fr12_native_spine_conv_20260606T040439Z_eager/tree/logs/subkernel_tree.pt`
- Tree logits:
  `output/fr12_native_spine_conv_20260606T040439Z_eager/tree/logs/spine_tree.call2.pt`
- Sub-kernel compare:
  `output/fr12_native_spine_conv_20260606T040439Z_eager/corrected_subkernel_compare_native_spine.json`
- Layer-output compare:
  `output/fr12_native_spine_conv_20260606T040439Z_eager/corrected_layer_compare_native_spine.json`
- Native reference:
  `output/fr12_corrected_l0_parity_20260606T032230Z/native_clonefix/`

Alignment:

- Spine tokens match:
  `[71093, 12305, 198, 727, 9637]`.
- Tree rows:
  `[0, 1, 2, 4, 6]`.
- Native rows:
  `[0, 1, 2, 3, 4]`.

Post-splice L0 sub-kernel table:

| Stage | Max abs | Mean abs max depth |
|---|---:|---:|
| pre_conv | 0.0 | 0.0 |
| conv1d_out | 0.0 | 0.0 |
| gdn_scan_out | 0.00000095367431640625 | 0.00000000015522050311744806 |
| gate_out | 0.0000019073486328125 | 0.0000000003104597967595879 |
| o_proj_out | 0.0001220703125 | 0.000000026775524020195007 |

Conv detail checks remain exact:

| Detail | Max abs | Mean abs |
|---|---:|---:|
| conv prior window | 0.0 | 0.0 |
| full conv window | 0.0 | 0.0 |
| fp32 tap products | 0.0 | 0.0 |
| bf16 tap products | 0.0 | 0.0 |

Verdict: the native-spine conv splice eliminates the measured causal-conv
origin. `conv1d_out` drops from `0.125` to `0.0`, and the previous
`gdn_scan_out = 0.015625` gap drops to `9.5367431640625e-7`; the scan gap was
therefore propagated from conv rather than intrinsic on this event.

Production caveat: this diagnostic splice calls native `causal_conv1d_update`,
which mutates `conv_state`, and the tree path then writes its canonical tree
conv states. That double-write is acceptable for this diagnostic parity probe,
but the production form must avoid corrupting canonical state while still
bit-reproducing native spine-row conv output.

Post-splice layer-output parity:

- Input hidden max abs: `0.0`
- First layer-output divergence: layer 0 `linear_attention`
- Layer 0 max abs by depth:
  `[0.0, 0.0, 0.000244140625, 0.0, 0.0]`
- Layer 0 max abs: `0.000244140625`
- Final norm max abs: `0.75`
- Layer 63 max abs: `6.25`

Layer-output max abs by layer:

| Layer | Type | Max abs |
|---:|---|---:|
| 0 | linear_attention | 0.000244140625 |
| 1 | linear_attention | 0.0015869140625 |
| 2 | linear_attention | 0.0234375 |
| 3 | full_attention | 0.00439453125 |
| 4 | linear_attention | 0.00433349609375 |
| 5 | linear_attention | 0.003979682922363281 |
| 6 | linear_attention | 0.009765625 |
| 7 | full_attention | 0.0068359375 |
| 8 | linear_attention | 0.0064697265625 |
| 9 | linear_attention | 0.0089111328125 |
| 10 | linear_attention | 0.007343292236328125 |
| 11 | full_attention | 0.0146484375 |
| 12 | linear_attention | 0.0135498046875 |
| 13 | linear_attention | 0.017578125 |
| 14 | linear_attention | 0.013671875 |
| 15 | full_attention | 0.0302734375 |
| 16 | linear_attention | 0.0244140625 |
| 17 | linear_attention | 0.0546875 |
| 18 | linear_attention | 0.15625 |
| 19 | full_attention | 0.09375 |
| 20 | linear_attention | 0.15625 |
| 21 | linear_attention | 0.1328125 |
| 22 | linear_attention | 0.0650634765625 |
| 23 | full_attention | 0.1171875 |
| 24 | linear_attention | 0.2265625 |
| 25 | linear_attention | 0.140625 |
| 26 | linear_attention | 0.375 |
| 27 | full_attention | 0.2890625 |
| 28 | linear_attention | 0.099029541015625 |
| 29 | linear_attention | 0.109375 |
| 30 | linear_attention | 0.109375 |
| 31 | full_attention | 0.19921875 |
| 32 | linear_attention | 0.12890625 |
| 33 | linear_attention | 0.126953125 |
| 34 | linear_attention | 0.375 |
| 35 | full_attention | 0.65625 |
| 36 | linear_attention | 0.111328125 |
| 37 | linear_attention | 0.107421875 |
| 38 | linear_attention | 0.21875 |
| 39 | full_attention | 0.171875 |
| 40 | linear_attention | 0.09375 |
| 41 | linear_attention | 0.1953125 |
| 42 | linear_attention | 0.21875 |
| 43 | full_attention | 0.25 |
| 44 | linear_attention | 0.1875 |
| 45 | linear_attention | 0.1796875 |
| 46 | linear_attention | 0.11083984375 |
| 47 | full_attention | 0.21875 |
| 48 | linear_attention | 0.15234375 |
| 49 | linear_attention | 0.171875 |
| 50 | linear_attention | 0.671875 |
| 51 | full_attention | 1.0703125 |
| 52 | linear_attention | 0.4375 |
| 53 | linear_attention | 0.240234375 |
| 54 | linear_attention | 1.4375 |
| 55 | full_attention | 0.6875 |
| 56 | linear_attention | 0.2425537109375 |
| 57 | linear_attention | 0.2421875 |
| 58 | linear_attention | 1.875 |
| 59 | full_attention | 1.0 |
| 60 | linear_attention | 0.28125 |
| 61 | linear_attention | 0.6640625 |
| 62 | linear_attention | 1.12255859375 |
| 63 | full_attention | 6.25 |

Verdict: the conv splice fixes the L0 GDN core and reduces the layer-0
post-MLP output gap by roughly 64x (`0.015625` to `0.000244140625`), but this
event is not full 64-layer lossless. The remaining small L0 post-core/output
residual still compounds through later layers.

## 2026-06-06 Token-Level Argmax Gate With Conv Splice

Purpose: check the actual losslessness gate, not just max-abs parity. The tree
arm ran with `FR12_TREE_CONV_NATIVE_SPINE=1`; native MTP-5 and tree ran
sequentially with real `FR10_SPINE_LOGIT_CAPTURE` tensors.

Artifacts:

- Run directory:
  `output/fr12_layer2_scan_origin_20260606T043504Z/`
- Native logits:
  `l0later_native/logs/spine_native.call2.pt`,
  `l0later_native/logs/spine_native.call3.pt`
- Tree logits:
  `l0later_tree/logs/spine_tree.call2.pt`,
  `l0later_tree/logs/spine_tree.call3.pt`
- Argmax/layer compares:
  `argmax_splice_call2_layer_compare.json`,
  `argmax_splice_call3_layer_compare.json`
- L0 sub-kernel compares:
  `l0_argmax_splice_subkernel_call2.json`,
  `l0_argmax_splice_subkernel_call3.json`

L0 sub-kernel parity remained at the post-splice floor on both full events:

| Stage | Max abs |
|---|---:|
| pre_conv | 0.0 |
| conv1d_out | 0.0 |
| gdn_scan_out | 0.00000095367431640625 |
| gate_out | 0.0000019073486328125 |
| o_proj_out | 0.0001220703125 |

Call 2 token/probability parity:

| Depth | Draft token | Tree argmax | Native argmax | Tree prob(draft) | Native prob(draft) |
|---:|---:|---:|---:|---:|---:|
| 0 | 71093 | 248068 | 248068 | 0.0 | 0.0 |
| 1 | 12305 | 12305 | 12305 | 1.0 | 1.0 |
| 2 | 198 | 12305 | 198 | 0.0 | 1.0 |
| 3 | 727 | 198 | 1005 | 0.0 | 0.12000831216573715 |
| 4 | 9637 | 1005 | 9637 | 0.0 | 1.0 |

Call 3 token/probability parity:

| Depth | Draft token | Tree argmax | Native argmax | Tree prob(draft) | Native prob(draft) |
|---:|---:|---:|---:|---:|---:|
| 0 | 271 | 198 | 271 | 0.5 | 0.6513549089431763 |
| 1 | 248069 | 248069 | 248069 | 1.0 | 1.0 |
| 2 | 271 | 248069 | 271 | 0.0 | 1.0 |
| 3 | 71093 | 271 | 71093 | 0.0 | 1.0 |
| 4 | 12305 | 2 | 12305 | 0.0 | 1.0 |

Verdict: `FR12_TREE_CONV_NATIVE_SPINE=1` is real progress for L0 numeric
parity, but it is not sufficient for token-level losslessness. The spine tokens
match, yet the verified target distribution still flips argmax at multiple
depths. The next fix target must be chosen by the token-level gate: remove the
remaining post-core/logit residual that changes argmax, not merely reduce
sub-kernel max abs.

## 2026-06-06 Hard Redirect: Splice Is Oracle-Only

The native-spine conv and scan splices are no longer valid implementation fixes.
They are retained only as bit-exact oracles for diagnosis. In the live patch
path, `FR12_TREE_CONV_NATIVE_SPINE=1` and `FR12_TREE_SCAN_NATIVE_SPINE=1` are
ignored unless `FR12_NATIVE_SPINE_ORACLE=1` is also set. Lossless progress must
come from our tree kernels matching native numerics, not from routing path0 rows
through native kernels.

Boot-free scan check:

- Script:
  `scripts/fr12_spine_scan_rounding_probe.py`
- Run directory:
  `output/fr12_real_kernel_rounding_20260606T070013Z/`
- Payload:
  `output/fr10_scan_capture_replay_20260604T191801Z/logs/fr10_tree_gdn_scan_capture.pt`
- Reference:
  `vllm.fused_sigmoid_gating_delta_rule_update`
- Our serving kernel:
  `lumo_flywheel_serving.fr10_gdn_tree_kernel.launch_tree_gdn_prepared`

Single-spine result:

| Check | Max abs |
|---|---:|
| scan output | 0.000000476837158203125 |
| recurrent state | 0.00000476837158203125 |

Full captured-tree replay result:

| Check | Max abs |
|---|---:|
| output vs serial oracle | 0.000000476837158203125 |
| state vs serial oracle | 0.0000057220458984375 |

Verdict: on this captured event, the current serving tree scan is already at
small fp32/bf16 floor relative to the native recurrent reference and serial
oracle. The measured L0 `0.125` origin was therefore not in the scan for this
event.

## 2026-06-06 Real-Kernel Conv Rounding Cut

Source-grounded native arithmetic:

- Live `causal_conv1d_update` loads bf16 `matrix_x` and `matrix_w`, computes
  `acc += matrix_x * matrix_w` into fp32, applies SiLU in fp32, then stores to
  the output dtype.
- Qwen3-Next GDN conv has `bias=False`, so the seam is tap product rounding,
  not bias or state.

Boot-free replay:

- Script:
  `scripts/fr12_conv_rounding_replay.py`
- Output:
  `output/fr12_real_kernel_rounding_20260606T071500Z/conv_rounding_replay.json`
- Tree capture:
  `output/fr12_corrected_l0_parity_20260606T032230Z/tree/logs/subkernel_tree.pt`
- Native reference:
  `output/fr12_corrected_l0_parity_20260606T032230Z/native_clonefix/logs/subkernel_native.pt`

Alignment checks:

| Check | Max abs |
|---|---:|
| pre-conv window | 0.0 |
| fp32 tap products tree vs native | 0.0 |
| bf16 tap products tree vs native | 0.0 |

Replay result:

| Variant | Max abs vs native | Mean abs | Nonzero |
|---|---:|---:|---:|
| Captured tree default, fp32 products | 0.125 | 0.00014408888819161803 | 15305 |
| bf16 products, fp32 SiLU, bf16 store | 0.0 | 0.0 | 0 |

First old mismatch:

- Index `[0, 6100]`
- Captured tree: `18.25`
- Native: `18.375`
- Abs: `0.125`

Patch status:

- The real tree-conv path now defaults to native bf16 tap-product rounding via
  `FR12_TREE_CONV_NATIVE_BF16_TAPS=1`.
- The legacy `FR11_TREE_CONV_NATIVE_BF16_TAPS` knob remains as a compatibility
  override, but the FR12 server launcher now defaults it to `1`.
- This is our-kernel arithmetic alignment, not a native-spine splice.

Serving gate, splice OFF:

- Run directory:
  `output/fr12_real_kernel_conv_bf16_20260606T071207Z/`
- Tree capture:
  `tree/logs/subkernel_tree.pt`
- Manual five-row compare:
  `manual_subkernel_compare_vs_native_clonefix.json`
- Native reference:
  `output/fr12_corrected_l0_parity_20260606T032230Z/native_clonefix/logs/subkernel_native.pt`
- Tree rows:
  `[0, 1, 2, 4, 6]`
- Native rows:
  `[0, 1, 2, 3, 4]`
- Engagement:
  `21/21` GPU tree metadata rows ok, `61` tree accept rows.
- Diagnostic one-prompt eager `accepted_per_draft_event`:
  `1.1016949152542372` (not an acceptance verdict).

Serving L0 sub-kernel parity with our tree-conv kernel:

| Stage | Max abs | Mean abs | Nonzero |
|---|---:|---:|---:|
| pre_conv | 0.0 | 0.0 | 0 |
| conv1d_out | 0.0 | 0.0 | 0 |
| gdn_scan_out | 0.00000095367431640625 | 0.000000000031078205980916707 | 4 |
| gate_out | 0.0000019073486328125 | 0.00000000007082311126449525 | 4 |
| o_proj_out | 0.0001220703125 | 0.0000000053551048040390015 | 11 |

Verdict: the first real-kernel conv numerics cut is verified in serving. With
native-spine splice disabled, our tree conv now bit-matches native
`causal_conv1d_update` at `conv1d_out` for the aligned L0 spine rows. The
remaining L0 core residual is downstream floor (`gdn_scan_out`, `gate_out`,
`o_proj_out`) rather than the previous `0.125` conv-origin mismatch.

## 2026-06-06 Argmax Lag Red-Team Check

Input:

- Red-team note:
  `FR12_REDTEAM_ARGMAX_LAG.md`
- New checker:
  `scripts/fr12_compare_argmax_lag.py`

Important harness correction:

- The older `fr10_layer_hidden_spine_compare.py` indexed `target_logits` by
  target model row id. That is invalid for tree captures because
  `target_logits_indices` contains duplicate branch-sibling row ids.
- The corrected checker maps each spine depth through the captured
  `target_logits_indices` entry first, then reads that entry's logits.
- `fr10_layer_hidden_spine_compare.py` has been patched to use the same
  entry mapping for probability and argmax reporting.

Old splice-ON recheck with corrected logit-entry mapping:

| Capture | Argmax mismatch depths | One-depth lag depths |
|---|---|---|
| `output/fr12_layer2_scan_origin_20260606T043504Z/argmax_lag_call2_recheck.json` | `[]` | `[]` |
| `output/fr12_layer2_scan_origin_20260606T043504Z/argmax_lag_call3_recheck.json` | `[0, 3]` | `[]` |

Verdict: the reported splice-ON `tree_argmax[d] == native_argmax[d-1]` pattern
was a compare-basis artifact, not a proven structural one-depth lag.

Splice-OFF argmax gate:

- Run directory:
  `output/fr12_spliceoff_argmax_20260606T072901Z/`
- Tree config:
  `FR12_NATIVE_SPINE_ORACLE=0`,
  `FR12_TREE_CONV_NATIVE_SPINE=0`,
  `FR12_TREE_SCAN_NATIVE_SPINE=0`,
  `FR12_TREE_CONV_NATIVE_BF16_TAPS=1`
- Tree engagement:
  `10/10` GPU tree metadata rows ok, `31` tree accept rows.
- Native MTP-5 rerun:
  `native/logs/fr10_mtp_draft_trace.jsonl`
- Native rerun note:
  the native rerun confirmed the same draft events, but did not write
  `spine_native*.pt`; comparison used the previous deterministic native
  logit captures for those same draft events.

Current native draft-event alignment:

| Event | Tree path0 draft tokens | Current native MTP draft tokens |
|---|---|---|
| call2 / native idx0 | `[71093, 12305, 198, 727, 9637]` | `[71093, 12305, 198, 727, 9637]` |
| call3 / native idx1 | `[271, 248069, 271, 71093, 12305]` | `[271, 248069, 271, 71093, 12305]` |

Splice-OFF corrected argmax result:

| Capture | Argmax mismatch depths | One-depth lag depths | Lag from first branch gap |
|---|---|---|---|
| `argmax_lag_spliceoff_call2_vs_native_call2.json` | `[]` | `[]` | `false` |
| `argmax_lag_spliceoff_call3_vs_native_call3.json` | `[]` | `[]` | `false` |

Per-depth splice-OFF argmax:

| Event | Depth | Tree row | Native row | Draft | Tree argmax | Native argmax |
|---|---:|---:|---:|---:|---:|---:|
| call2 | 0 | 0 | 0 | 71093 | 248068 | 248068 |
| call2 | 1 | 1 | 1 | 12305 | 12305 | 12305 |
| call2 | 2 | 2 | 2 | 198 | 198 | 198 |
| call2 | 3 | 4 | 3 | 727 | 1005 | 1005 |
| call2 | 4 | 6 | 4 | 9637 | 9637 | 9637 |
| call3 | 0 | 0 | 0 | 271 | 271 | 271 |
| call3 | 1 | 1 | 1 | 248069 | 248069 | 248069 |
| call3 | 2 | 2 | 2 | 271 | 271 | 271 |
| call3 | 3 | 4 | 3 | 71093 | 71093 | 71093 |
| call3 | 4 | 6 | 4 | 12305 | 12305 | 12305 |

Verdict: with splice OFF and our real bf16-rounded tree conv enabled, the
per-depth argmax gate passes for these two matched events. The one-depth lag
does not persist. Next seam should be selected by the next failing token-level
gate, not by the old lag table.

## 2026-06-06 Many-Event Argmax + Distribution Gate

Purpose:

- Red-team coverage check for the one-event splice-OFF green result above.
- Gate basis: splice OFF, native-spine oracle OFF, our bf16-rounded tree conv
  ON, real SWE-Bench Verified prompt sample, `B=4`, `temperature=0.6`,
  `top_p=0.95`, `mtp5`.
- Comparator: `scripts/fr12_compare_argmax_distribution.py`.

Run directory:

- `output/fr12_spliceoff_many_argmax_20260606T075933Z/`

Serving configs:

| Arm | Key config |
|---|---|
| tree | `FR12_NATIVE_SPINE_ORACLE=0`, `FR12_TREE_CONV_NATIVE_SPINE=0`, `FR12_TREE_SCAN_NATIVE_SPINE=0`, `FR12_TREE_CONV_NATIVE_BF16_TAPS=1`, tree n9 |
| native | native MTP-5, eager rerun for capture, `LUMO_TREE_SAMPLER_DEBUG_LOG=/logs/tree_sampler_debug.jsonl` |

Capture note:

- Native logit capture did not fire until `LUMO_TREE_SAMPLER_DEBUG_LOG` was set,
  because the current capture hook is nested inside the sampler-debug branch.
- Tree produced `40` tree capture files plus the base file; two early files had
  no tree parent indices and were skipped by the comparator.
- Native produced `40` native capture files after the debug-log rerun.

Probe metrics for this bounded SWE prompt sample:

| Arm | accepted_per_draft_event | spec accepted | spec drafts |
|---|---:|---:|---:|
| tree_mtp | 0.6586021505376344 | 245 | 372 |
| native MTP-5 | 1.623728813559322 | 479 | 295 |

Coverage:

| Metric | Value |
|---|---:|
| Tree path0 events in captures | 96 |
| Native path events in captures | 136 |
| Exact draft-token matched events | 6 |
| Unmatched tree events | 90 |
| Unmatched native events | 130 |
| Rows compared | 30 |

Argmax / lag:

| Metric | Value |
|---|---:|
| Argmax mismatches | 5 / 30 |
| Argmax mismatch rate | 0.16666666666666666 |
| Events with any argmax mismatch | 3 / 6 |
| One-depth lag matches | 0 / 30 |
| Events with first-branch lag pattern | 0 |

Distributional drift on matched rows:

| Metric | Mean | P50 | P90 | Max |
|---|---:|---:|---:|---:|
| TV | 0.23948737420141697 | 0.049303941428661346 | 0.7638193368911743 | 1.0 |
| JS nats | 0.1182111736619845 | 0.0037247389554977417 | 0.4454859495162964 | 0.6931471228599548 |
| Draft-prob abs delta | 0.1316683748116096 | 0.028470218181610107 | 0.3891814351081848 | 0.867035761475563 |

Per-depth summary:

| Depth | Rows | Argmax mismatch rate | TV mean | TV max | Draft-prob abs-delta mean | Draft-prob abs-delta max |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 0.3333333333333333 | 0.344734862446785 | 1.0 | 0.09633595868945122 | 0.3997858762741089 |
| 1 | 6 | 0.16666666666666666 | 0.261039358874162 | 0.8670357465744019 | 0.24425327281157175 | 0.867035761475563 |
| 2 | 6 | 0.16666666666666666 | 0.2734779603779316 | 0.7638193368911743 | 0.0963319248209397 | 0.298944890499115 |
| 3 | 6 | 0.0 | 0.17994595877826214 | 0.42518046498298645 | 0.09386170158783595 | 0.3891814351081848 |
| 4 | 6 | 0.16666666666666666 | 0.1382387305299441 | 0.5393196940422058 | 0.1275590161482493 | 0.47524142265319824 |

Verdict: the many-event gate does **not** pass. The prior one-event argmax green
result was real but not broad enough: on this SWE prompt sample, comparable
matched events already show argmax failures and large distributional TV, while
the exact-match coverage itself is low because tree and native sampled
continuations diverge quickly. The explicit one-depth lag pattern remains
absent, so the next work should target the remaining distribution/argmax
propagator, not the old lag hypothesis.

User-reconciled aggregate:

- Overall TV mean over the full gate basis: `0.34`.
- Overall TV p90: `1.0`.
- Argmax mismatch rate: `16.7%`.
- Outcome: tree `accepted_per_draft_event = 0.659` vs native MTP-5
  `accepted_per_draft_event = 1.739`.

The table above is the matched-event comparator subset; the user-reconciled
aggregate is the cost-gate basis.

## 2026-06-06 Scan-BF16 Lever Rejected; FP8 Activation Quantizer Probe

Scan-bf16 status:

- The in-server `FR12_TREE_SCAN_FLA_BF16_BOUNDARIES=1` run was stopped before it
  burned more GPU.
- Boot-free scan probe:
  `output/fr12_scan_bf16_boundary_probe_20260606T083638Z/`
- Baseline serving scan vs native:
  `out_max_abs = 5.96e-08` (already near bit-exact).
- Forced FLA bf16 boundaries:
  `out_max_abs = 0.0078125`.

Verdict: the scan bf16 boundary lever is wrong by about four orders of
magnitude on the boot-free probe. The scan lever stays default-off; scan is not
the dominant residual.

Boot-free fp8 activation quantizer probe:

- Script:
  `scripts/fr12_fp8_gemm_batch_invariance_probe.py`
- Run directory:
  `output/fr12_fp8_gemm_batch_invariance_20260606T084909Z/`
- JSON:
  `output/fr12_fp8_gemm_batch_invariance_20260606T084909Z/fp8_gemm_batch_invariance_l0_o_proj.json`
- Tree capture:
  `output/fr12_real_kernel_conv_bf16_20260606T071207Z/tree/logs/subkernel_tree.pt`
- Native capture:
  `output/fr12_corrected_l0_parity_20260606T032230Z/native_clonefix/logs/subkernel_native.pt`
- Aligned rows: tree `[0, 1, 2, 4, 6]` vs native `[0, 1, 2, 3, 4]`.

L0 `o_proj` boundary:

| Check | Value |
|---|---:|
| `gate_out` input max abs tree vs native | `0.0000019073486328125` |
| `gate_out` input nonzero | `4` |
| `o_proj_out` output max abs tree vs native | `0.0001220703125` |
| `o_proj_out` output nonzero | `11` |

Live fp8 per-token-group activation quantizer:

| Check | FP8 byte mismatches | Scale max abs |
|---|---:|---:|
| Tree full batch vs row-only | `0` | `0.0` |
| Tree full batch vs reversed-row context | `0` | `0.0` |
| Native full batch vs row-only | `0` | `0.0` |
| Native full batch vs reversed-row context | `0` | `0.0` |
| Tree row-only vs native row-only | `2` | `0.0` |

Verdict: the live activation quantizer is row-independent on this capture. The
specific hypothesis that co-resident tree rows make the same row's fp8
activation bytes or scales differ is not reproduced boot-free. The `o_proj`
boundary residual is real, but this probe does not support activation-quant
batch-invariance as its cause. Full fp8 GEMM replay remains unmeasured because
the existing captures contain `o_proj` input/output tensors but not
RowParallelLinear fp8 weights or block scales; `in_proj` is also unmeasured
because the captures start after the input projections at `pre_conv`.

## 2026-06-06 Full FP8 GEMM Batch-Invariance Probe

Purpose:

- Exhaust the two fp8 seams the activation-quant probe could not reach:
  full Cutlass block-fp8 GEMM weight/block-scale behavior and layer-0
  `in_proj`.
- Boot-free only: no server, no full model load.
- Splice OFF: uses captured real-kernel tree/native tensors and local
  `/models/qwen3.6-27b-fp8/layers-0.safetensors` weights.

Source-grounding:

- Live cu130 container path:
  `vllm.model_executor.kernels.linear.scaled_mm.cutlass.CutlassFp8BlockScaledMMKernel`
- Low-level op used by the probe:
  `torch.ops.vllm.padded_cutlass`
- Activation quant used by that path:
  `per_token_group_quant_fp8(..., group_size=128, column_major_scales=True)`
- Weight layout: checkpoint fp8 weights are `[out, in]`; block scales are
  `weight_scale_inv` with shape `[ceil(out/128), ceil(in/128)]`.
- The low-level op was fed block scales as `float32`; this exactly reproduced
  captured `o_proj_out`.

Script:

- `scripts/fr12_fp8_full_gemm_batch_invariance_probe.py`

Run directory:

- `output/fr12_fp8_full_gemm_batch_invariance_20260606T181323Z/`

JSON:

- `output/fr12_fp8_full_gemm_batch_invariance_20260606T181323Z/fp8_full_gemm_batch_invariance_l0.json`

Aligned rows:

- Tree `[0, 1, 2, 4, 6]`
- Native `[0, 1, 2, 3, 4]`

Context-invariance results:

| Module | Context | Full vs row-only max abs | Full vs reversed-context max abs | Nonzero |
|---|---|---:|---:|---:|
| `out_proj` | tree | `0.0` | `0.0` | `0` |
| `out_proj` | native | `0.0` | `0.0` | `0` |
| `in_proj_qkv` | tree | `0.0` | `0.0` | `0` |
| `in_proj_qkv` | native | `0.0` | `0.0` | `0` |
| `in_proj_z` | tree | `0.0` | `0.0` | `0` |
| `in_proj_z` | native | `0.0` | `0.0` | `0` |

Replay validation:

| Check | Max abs |
|---|---:|
| Tree `out_proj` replay vs captured `o_proj_out` | `0.0` |
| Native `out_proj` replay vs captured `o_proj_out` | `0.0` |

Tree-vs-native row-only replay:

| Module | Input max abs | Output max abs | Nonzero output |
|---|---:|---:|---:|
| `out_proj` | `0.0000019073486328125` | `0.0001220703125` | `11` |
| `in_proj_qkv` | `0.0` | `0.0` | `0` |
| `in_proj_z` | `0.0` | `0.0` | `0` |

Serving boundary:

| Boundary | Max abs |
|---|---:|
| `pre_conv` tree vs native | `0.0` |

Verdict: the last fp8 batch-dependence lever is exhausted on this boot-free
cost gate. Full block-fp8 Cutlass GEMM does not change spine-row outputs with
co-resident tree rows for `out_proj`, `in_proj_qkv`, or `in_proj_z`.
`in_proj` starts the layer bit-matched; `out_proj` exactly replays the captured
`1.22e-4` tree-vs-native residual, but that residual comes from its already tiny
input delta, not from M-dependent fp8 GEMM behavior. No server/fix run is
justified from this probe; the remaining lossless deficit is the diffuse
multi-layer numeric wall rather than a fixable fp8 batch-invariance seam.

## 2026-06-06 GDN Scan N-Independent Direct-Load Fix

Hard-gate target: L0 GDN scan output only, boot-free, splice OFF / oracle OFF.

Mechanism confirmed:

- WIP scan batch-invariance probe first showed full-tree spine and
  reversed-context spine were already output-identical, but spine-only differed
  because the old kernel selected rows by reducing over the padded node axis.
- That row-select reduction made even a single row's fp32 op order depend on
  `N_PAD` and co-resident rows.
- The serving tree kernel now replays each node's visible ancestor path in the
  same recurrent statement order as live
  `fused_sigmoid_gating_delta_rule_update`, using direct row loads for
  `q/k/v/a/b` and computing `g`/`beta` from raw `a/b/A_log/dt_bias` inside the
  kernel. This is our kernel arithmetic, not a native splice.

Script:

- `scripts/fr12_scan_batch_invariance_probe.py`

Artifact:

- `output/fr12_scan_direct_raw_clean_20260606T183545Z/scan_batch_invariance_l0.json`

Scan output results on the captured tree payload:

| Comparison | `out.max_abs` |
|---|---:|
| original full-tree spine vs native FLA | `0.0` |
| spine-only vs native FLA | `0.0` |
| spine-first full-tree spine vs native FLA | `0.0` |
| reversed sibling DFS full-tree spine vs native FLA | `0.0` |
| spine-only vs original full-tree spine | `0.0` |
| reversed sibling DFS full-tree spine vs original full-tree spine | `0.0` |

Per-depth output max abs is `0.0` at all five aligned spine depths
`[0, 1, 2, 4, 6]`. The recurrent state still has a small fp32 internal delta
vs native (`state.max_abs = 4.76837158203125e-7`), but the user-facing scan
output boundary is bit-exact and N-independent.

Next hard-gate step: verify the downstream L0 cascade with fresh splice-OFF
serving captures: `gdn_scan_out == 0.0`, then `gate_out == 0.0`, then
`o_proj_out == 0.0`.

## 2026-06-06 L0 GDN Hard Gate Passed: Scan → Gate → O-Projection

Hard-gate target: L0 GDN sub-kernels `(3) gdn_scan`, `(4) RMSNormGated gate`,
and `(5) o_proj`, splice OFF / oracle OFF, our kernel computing.

Serving captures:

- Tree, splice OFF:
  `output/fr12_scanfix_l0_gate_20260606T183732Z/tree/logs/subkernel_tree.pt`
- Native linear MTP-5:
  `output/fr12_scanfix_l0_native_linear_b1_20260606T185459Z/native/logs/subkernel_native.pt`
- Manual aligned-row compare:
  `output/fr12_scanfix_l0_native_linear_b1_20260606T185459Z/manual_tree_vs_native_linear_b1_subkernel_compare.json`

Alignment:

- Tree rows: `[0, 1, 2, 4, 6]`
- Native rows: `[0, 1, 2, 3, 4]`
- Tree capture metadata: `tree_scan_active=true`, `tree_parent=[-1,0,1,1,2,2,4,4,6,6]`
- Native capture metadata: `tree_scan_active=false`, `tree_parent=[-1,0,1,2,3,4]`

Stage results:

| Stage | Max abs | Mean abs | Nonzero |
|---|---:|---:|---:|
| `pre_conv` | `0.0` | `0.0` | `0` |
| `conv1d_out` | `0.0` | `0.0` | `0` |
| `gdn_scan_out` | `0.0` | `0.0` | `0` |
| `gate_out` | `0.0` | `0.0` | `0` |
| `o_proj_out` | `0.0` | `0.0` | `0` |

Verdict: the FR12 hard gate is satisfied at L0 for scan, gate, and o-projection.
This was verified against a true native linear MTP-5 capture; an intermediate
compare against a tree-shaped native config was also green but is not the gate
basis.

Next step: propagate this L0 result across all 64 layers and verify full
layer-output/logit behavior splice OFF before any other kernel work.
