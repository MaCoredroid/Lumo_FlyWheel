# FR13 L4 Clean Native Re-Localization

Date: 2026-06-09

## Setup

- Native clean pre-update reference:
  `output/fr13_l4_clean_native_ref_20260609T034637Z/native/logs/native_l4_gdn_subop.pt`
- Matching tree arm:
  `output/fr13_l4_clean_pair_20260609T035452Z/tree/logs/tree_l4_gdn_subop.pt`
- Pinned prompt: `Explain hash tables.`
- Native request mode: `naive_mtp`
- Tree request mode: `tree_mtp`
- Both arms: eager, `max_model_len=65536`, `max_num_seqs=1`
- Capture filters:
  - native L4 GDN subop: `num_tokens=6`
  - tree L4 GDN subop: `num_tokens=10`

## Clean Reference Check

The native L4 capture used the clean pre-update conv state:

- `native_conv_detail.prior_window_source = pre_update`
- `prior_bank_row = 1`
- `prior_cols = [0, 1, 2]`
- `spec_state_indices = [[1, 2, 3, 4, 5, 6]]`
- `metadata_num_accepted_tokens = [1]`

Debug logs selected the intended verify captures, not the 2048/profile pass:

- native: one `capture_created` at `num_tokens=6`
- tree: one `capture_created` at `num_tokens=10`

## Gate A Re-Localization

Clean paired ladder:
`output/fr13_l4_clean_pair_20260609T035452Z/gateA_spine_ladder_clean_pair.json`

Result:

- `input_hidden = 0.0`
- first nonzero: `layer_hidden`, layer `3`, `max_abs=0.08203125`
- layer 3 type: `full_attention`
- layer 4 hidden max: `0.03515625`
- final norm max: `0.6875`
- logits max: `1.6875`

Per-row max abs for path0 rows `[0, 1, 2, 4, 6, 8]` vs native rows `[0, 1, 2, 3, 4, 5]`:

| Layer | Type | Per-row max |
| --- | --- | --- |
| 0 | linear_attention | `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` |
| 1 | linear_attention | `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` |
| 2 | linear_attention | `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` |
| 3 | full_attention | `[0.0, 0.0, 0.0, 0.08203125, 0.02734375, 0.03515625]` |
| 4 | linear_attention | `[0.0, 0.0, 0.0, 0.03515625, 0.0089111328125, 0.008331298828125]` |

## L4 Subop Read

Row-mapped L4 subop report:
`output/fr13_l4_clean_pair_20260609T035452Z/l4_subop_spine_clean_pair_rowmapped.json`

At L4, the GDN subop is already downstream of the layer-3 full-attention divergence:

- `input_hidden.max_abs = 0.09375`
- `pre_conv.max_abs = 0.28125`
- `conv1d_out.max_abs = 0.078125`
- `h0_state_in.max_abs = 0.0`
- `gdn_scan_out.max_abs = 0.00048828125`

Rows 0-2 remain zero through L4 conv; the nonzero L4 conv deltas are inherited on deeper rows from layer 3 full attention. The prior conv localization against the post-update native window is not a valid root cause.

## Conclusion

With a clean native pre-update reference, the first real divergence is not L4 conv. The next front is layer 3 full attention on deeper spine rows.
