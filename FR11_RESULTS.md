# FR11 Results

## Probe beta: event-0 state-handoff byte compare

Command:

```bash
python3 scripts/fr11_probe_beta_event0_handoff.py \
  --payload output/fr10_conv_handoff_confirm_20260605T023734Z/logs/fr10_src_native_handoff.pt \
  --out output/fr10_nocopy_resolve/fr11_probe_beta_event0_state_handoff.json
```

Valid capture:
- `output/fr10_conv_handoff_confirm_20260605T023734Z/logs/fr10_src_native_handoff.pt`
- `layer_prefix`: `language_model.model.layers.0.linear_attn`
- `tree_parent`: `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]`
- `accepted_len`: `4`
- `accepted_gdn_node_id`: `7`
- `accepted_gdn_node_path`: `[1, 2, 4, 7]`
- `accepted_bank_row`: `8`
- `next_read_bank_row`: `4`
- `address_coincide`: `false`

Byte-compare result:
- h0 tree accepted row vs native next-read h0: bit-exact, `max_abs=0.0`, mismatched elements `0`, shape `[48, 128, 128]`, dtype `float32`.
- conv decisive prior window, native `conv_state_token_offset=accepted_len-1=3`, columns `[3, 6)`: bit-exact, `max_abs=0.0`, mismatched elements `0`, shape `[10240, 3]`, dtype `bfloat16`.
- conv full row: bit-exact, `max_abs=0.0`, mismatched elements `0`, shape `[10240, 12]`, dtype `bfloat16`.

Verdict:

`MATCH_WRONG_INITIAL_STATE_EXCLUDED`

Probe beta does not find a #40738-class wrong-initial-state bug on the valid event-0 handoff capture. Bank addresses differ, but the loaded recurrent h0 and the conv prior-state window bytes match exactly, so address inequality alone is not evidence of a state-handoff bug.

Rejected older captures:

Several older `fr10_src_native_handoff.pt` payloads under `output/fr10_*confirm*/` were intentionally not used because their `accepted_node_id` and `accepted_node_path` metadata disagree. The Probe beta script fails these loudly instead of silently selecting a row.
