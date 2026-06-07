# FR13 Top-Down Ladder Pass 1

Date: 2026-06-07

Run directory:

`output/fr13_preprocess_input_compare_20260607T050411Z`

## Scope

This pass follows the top-down rule: compare verifier input first, then layer
outputs in order. It uses one tree capture and one native capture, then diffs
offline.

Diagnostic wiring added:

- `FR13_PREPROCESS_INPUT_CAPTURE`
- `FR13_PREPROCESS_INPUT_CAPTURE_NUM_TOKENS`
- `FR13_PREPROCESS_INPUT_CAPTURE_SKIP`
- `FR13_PREPROCESS_INPUT_CAPTURE_LIMIT`

The capture records the GPU model runner token ids and `inputs_embeds` handed to
the target model before Qwen receives them.

## Spine

Matched tree/native event:

- Tree: `tree/logs/tree_layer_hidden.call0.pt`
- Native: `native/logs/native_layer_hidden.call0.pt`
- Compare: `spine_ladder_compare_call0.json`

Spine tokens match:

`[579, 264, 7047, 1817, 25]`

Input hidden max-abs by depth:

| depth | max_abs |
| --- | ---: |
| 0 | 0.0 |
| 1 | 0.0 |
| 2 | 0.0 |
| 3 | 0.0 |
| 4 | 0.0 |

First nonzero stage, walking in order:

| stage | layer type | max_abs | max_abs_by_depth |
| --- | --- | ---: | --- |
| layer 3 | full_attention | 0.001953125 | `[0.001953125, 0.001953125, 0.00041961669921875, 0.001953125, 0.001953125]` |

Layers 0, 1, and 2 are exactly zero on the matched spine event.

## Branch Input Stage

Tree branch capture:

- `tree_branches/logs/tree_preprocess_inputs.call0.pt`
- `tree_branches/logs/tree_preprocess_inputs.call1.pt`
- `tree_branches/logs/tree_layer_hidden.call0.pt`
- `tree_branches/logs/tree_layer_hidden.call1.pt`
- Compare: `tree_branches_input_stage_compare.json`

Tree paths:

`[(0,), (0, 0), (0, 1), (0, 0, 0), (0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 1)]`

Branch node indices:

`[2, 4, 6, 8]`

For both captured calls:

- runner `model_inputs_embeds` vs target `input_hidden`: `max_abs_all_rows = 0.0`
- branch row max-abs for nodes 2, 4, 6, 8: all `0.0`

Result: the input stage is clean for the captured spine and all captured branch
rows. The next fix target is the first nonzero top-down stage: layer 3
`full_attention`.
