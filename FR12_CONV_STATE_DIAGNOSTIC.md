# FR12 Conv State Diagnostic

Date: 2026-06-06
Branch: `fr12-wy-tree-kernel`

## Scope

This records the corrected conv-state diagnostic for the FR12 layer-0 conv
origin investigation. The goal was to determine whether the no-copy tree conv
path was reading a conv-state bank/slot that lacked the native prefill prior
state.

## Runs

- Tree full-state capture:
  `output/fr12_conv_fullstate_20260606T024801Z/tree/logs/subkernel_tree.pt`
- Native post-update capture, superseded for prior-state comparison:
  `output/fr12_conv_fullstate_20260606T024801Z/native/logs/subkernel_native.pt`
- Native pre-update capture, authoritative for prior-state comparison:
  `output/fr12_conv_fullstate_20260606T024801Z/native_pre/logs/subkernel_native_pre.pt`
- Summary:
  `output/fr12_conv_fullstate_20260606T024801Z/conv_fullstate_preupdate_summary.json`

## Result

The earlier apparent native-vs-tree mismatch came from reading native
`conv_state` after `causal_conv1d_update` had shifted and written the state.
After adding a native pre-update capture, the tree prior rows match the true
native pre-update prior state exactly:

- Tree `conv_state` shape: `[1152, 10240, 12]`
- Native `conv_state` shape: `[1177, 10240, 8]`
- Tree prior rows: `[1, 32, 63, 94]`
- Tree prior cols: `[0, 1, 2]`
- Native prior row: `1`
- Native pre-update prior absmean: `[1.6899105310, 1.8764997721, 2.3334021568]`
- Tree prior absmean: `[1.6899105310, 1.8764997721, 2.3334021568]`
- Tree vs native pre-update prior max_abs: `0.0`
- Tree vs native pre-update prior mean_abs: `0.0`

## Verdict

`CONV_PRIOR_STATE_SLOT_BUG_REFUTED_BY_PREUPDATE_CAPTURE`.

No conv-state bank/slot fix was applied. The default-off tail-read experiment
remains refuted, and the corrected pre-update measurement shows that changing
the tree prior-state read would move it away from the native pre-conv read.

Next investigation should return to the layer-0 conv parity harness and compare
the actual same-event spine rows with a pre-update native prior reference, not a
post-update reconstructed conv-state window.
