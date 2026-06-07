# FR13 GDN Conv Offline Replay

Status: diagnostic/offline workflow only. Do not bind Gate A from single-layer replay.

## Native op order readout

Source: `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`, `_causal_conv1d_update_kernel`.

For width 4 decode/update, native order is:

1. Load prior state columns `col0`, `col1`, `col2` and current `x`.
2. Load weights `w_col0..w_col3`.
3. Accumulate in source order: `acc_preload + col0*w0 + col1*w1 + col2*w2 + x*w3`.
4. If enabled, apply SiLU as `acc / (1 + tl.exp(-acc))`; lowered PTX uses `ex2.approx.f32`.
5. Store to bf16 output.

Observed failed fixes:

- fp32 tap products are the wrong direction: live matched-token data overshot broadly.
- A positive-midpoint SiLU/bf16 tie patch fixed old L45 offline but regressed live layer 0, so it is not a valid fix.

## Required Offline Gate Before Next Live Run

One future capture must include at least:

- Layers: `language_model.model.layers.0.linear_attn,language_model.model.layers.45.linear_attn`.
- Rows: default replay selection is spine rows plus first branch row; override with `FR13_CONV_REPLAY_NODES=all` or a comma-list.
- Payload: `pre_conv`, `conv1d_out`, `h0_state_in`, selected windows/taps, raw `conv_weights`, raw `conv_bias`.

Capture flags:

```bash
FR12_TREE_CONV_STATE_FULL_CAPTURE=1
FR12_SUBKERNEL_CAPTURE=/logs/gdn_conv_replay.pt
FR12_SUBKERNEL_CAPTURE_LAYER_PREFIX=language_model.model.layers.0.linear_attn,language_model.model.layers.45.linear_attn
FR12_SUBKERNEL_CAPTURE_NUM_TOKENS=10
FR12_SUBKERNEL_CAPTURE_LIMIT=2
FR12_SUBKERNEL_CAPTURE_INPUT=1
FR12_SUBKERNEL_CAPTURE_Z=1
```

Offline reducer:

```bash
python3 scripts/fr13_conv_replay_multilayer.py \
  --tree <tree L0 capture> <tree L45 capture> \
  --native <native L0 capture> <native L45 capture> \
  --out <run>/conv_replay_multilayer.json
```

A valid candidate must report aggregate `max_abs=0.0`, `nonzero=0` across all targetable rows/layers before any live full-ladder run. Branch rows without a native-on-path target are reported as `missing_native_on_path_oracle`; do not treat them as passed.

If torch cannot reproduce native `ex2.approx`/bf16 store simultaneously across L0, L45, and branch-target rows, stop and present the native `causal_conv1d_update` diagnostic-oracle/reroute option for user approval. Do not reroute compute unilaterally.
