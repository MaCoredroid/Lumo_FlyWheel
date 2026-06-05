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

## Probe alpha attempt 1: conv seam through post-attention only — incomplete

Command:

```bash
docker run --rm --gpus all --entrypoint python3 \
  -v /home/mark/shared/lumoFlyWheel:/workspace \
  -v /models:/models \
  -w /workspace \
  vllm/vllm-openai:cu130-nightly \
  output/fr10_nocopy_resolve/gpu_conv_seam_replay.py \
  --out output/fr10_nocopy_resolve/gpu_conv_seam_replay_result.json
```

Inputs:
- Layer hidden capture: `output/fr10_match0_layer_native_20260605T101521Z/logs/layer_native.call0.pt`
- Handoff/conv prior capture: `output/fr10_conv_handoff_confirm_20260605T023734Z/logs/fr10_src_native_handoff.pt`
- Real layer-0 safetensors: `/models/qwen3.6-27b-fp8/layers-0.safetensors`
- Tokens replayed: `11`
- Conv read column: `3`, columns `[3, 6)`
- Tree engagement asserted from `tree_parent`.

Replay path:
- Reconstructed layer-0 input Gemma RMSNorm from captured `input_hidden`.
- Reconstructed real `in_proj_qkv`, `in_proj_z`, `in_proj_a`, and `in_proj_b` from layer-0 weights.
- Compared native conv semantics (`bf16*bf16` tap product, fp32 accumulate) against FR10 tree semantics (`fp32*fp32` tap product, fp32 accumulate).
- Propagated both branches through the same GDN recurrent scan, `RMSNormGated(norm_before_gate=True)`, real dequantized FP8 `out_proj`, and residual add.

Numbers:
- Preconv absmean: `2.2228586673736572`
- Conv output absmean: `0.1273835301399231`
- GDN core absmean: `0.0073541151359677315`
- Attention output absmean: `0.019381709396839142`
- Residual-hidden absmean: `0.0216512493789196`

Deltas:
- Conv native vs tree: `max_abs=0.25`, `rms=0.00271211308427155`, mismatched elements `31887 / 112640`.
- GDN core native vs tree: `max_abs=0.015625`, `rms=0.00020703690825030208`, mismatched elements `26394 / 67584`.
- RMSNormGated native vs tree: `max_abs=0.0078125`, `rms=9.100384340854362e-05`, mismatched elements `27985 / 67584`.
- Real out_proj attention output native vs tree: `max_abs=0.001953125`, `rms=0.00010767146159196272`, mismatched elements `36221 / 56320`.
- Residual-hidden native vs tree: `max_abs=0.001953125`, `rms=0.00011625278420979157`, mismatched elements `30765 / 56320`.

Status:

`INCOMPLETE_RETRACTED`

This attempt stopped at the post-attention residual (`attn_out + residual`) and did not propagate through `post_attention_layernorm -> MLP gate/up/down`, while the live FR10 layer compare measures layer output after the MLP (`layers[0].hidden`, peak around `1.9`). The `0.001953125` post-attention residual number is therefore not a valid verdict against the live `0.0156` post-MLP drift. Probe alpha must be extended through the real MLP path before any causal-sufficiency verdict.

## Probe alpha extended: conv seam through post-attention norm and MLP

Command:

```bash
docker run --rm --gpus all --entrypoint python3 \
  -v /home/mark/shared/lumoFlyWheel:/workspace \
  -v /models:/models \
  -w /workspace \
  vllm/vllm-openai:cu130-nightly \
  output/fr10_nocopy_resolve/gpu_conv_seam_replay.py \
  --out output/fr10_nocopy_resolve/gpu_conv_seam_replay_result_post_mlp.json
```

Live source checked:
- `/tmp/vllm-0.22-src/vllm-0.22.0/vllm/model_executor/models/qwen3_next.py`: `Qwen3NextDecoderLayer.forward` does `input_layernorm -> linear_attn -> post_attention_layernorm(hidden_states, residual) -> mlp -> return hidden_states, residual`.
- `scripts/fr10_phase4_patch_vllm_tree_gdn.py`: `layers[0].hidden` capture records the returned post-MLP `hidden_states`; `layers[0].residual` records the separate residual state.

Replay path extension:
- Reused the post-attention native/tree branches from attempt 1.
- Applied the real layer-0 `post_attention_layernorm` with residual semantics.
- Applied real layer-0 MLP `gate_proj`, `up_proj`, `silu(gate) * up`, and `down_proj` using the FP8 safetensor weights.
- Reported both returned MLP output (`layers[0].hidden` equivalent) and the diagnostic hidden-plus-residual stream.

Numbers:
- Post-attention residual native vs tree: `max_abs=0.001953125`, `rms=0.00011625278420979157`.
- Post-attention norm native vs tree: `max_abs=0.00830078125`, `rms=0.0005177347920835018`.
- MLP output native vs tree: `max_abs=0.015625`, `rms=0.00012635976599995047`, mismatched elements `35508 / 56320`.
- Diagnostic MLP-plus-residual stream native vs tree: `max_abs=0.00390625`, `rms=0.000167013073223643`.
- Captured live layer-0 hidden absmax: `1.8515625`; replay MLP hidden absmax: `1.8359375`.
- At captured layer-0 hidden peak `(row=5, col=3994, captured=1.8515625)`: replay native `1.8359375`, replay tree `1.8359375`, delta `0.0`.
- Across captured layer-0 hidden elements with `abs >= 1.75`: `2` elements, MLP output native/tree `max_abs=0.0078125`, `rms=0.005524271633476019`.
- Across replay MLP output elements with `abs >= 1.75`: `1` element, MLP output native/tree `max_abs=0.0`.
- Whole-MLP max-diff point: `(row=1, col=3994)`, native `1.2578125`, tree `1.2734375`, delta `0.015625`.

Replay-to-capture check:
- Native replay MLP output vs captured `layers[0].hidden`: `max_abs=0.59765625`, `rms=0.010052364319562912`.
- Native replay residual vs captured `layers[0].residual`: `max_abs=2.25`, `rms=0.016501644626259804`.

Verdict:

`PRECISION_FLOOR_CANDIDATE_POST_MLP_WITH_HIGH_PEAK_CAVEAT`

Extending through the real post-attention norm and MLP shows the conv tap-dtype seam can produce a post-MLP `0.015625` delta, so the previous post-attention-only `NOT_CAUSALLY_SUFFICIENT` verdict is invalid. The strongest high-output-region check is smaller than the live red-team reference: at the captured `~1.85` peak the delta is `0.0`, and across captured `abs>=1.75` elements the max is `0.0078125`; the whole-tensor `0.015625` occurs at native value `1.2578125`. Because the boot-free replay is not byte-close to the captured live layer output, alpha supports a precision-floor mechanism but does not by itself prove that the exact live `0.0156` peak is solely the conv seam. Acceptance A/B is required to isolate user-visible impact.

## Acceptance A/B: conv tap dtype flag

Configuration:
- Image: `vllm/vllm-openai:cu130-nightly` via `scripts/fr10_launch_speed_server.sh`.
- Endpoint: `http://127.0.0.1:9950`.
- Decode mode: `tree_mtp`.
- Batch: `4`.
- Sampling: `temperature=0.6`, `top_p=0.95`.
- Max tokens: `64`.
- Tree: 9-node MTP-5-style caterpillar, `expected_draft_count=9`.
- Metrics mode: `FR10_METRICS=0`.
- Engagement assertion: required; explicit metrics-off logs enabled with `LUMO_TREE_SAMPLER_DEBUG_LOG=/logs/tree_sampler_debug.jsonl` and `LUMO_TREE_PATH_LCP_LOG=/logs/tree_path_lcp_max.jsonl`.
- Native reference: MTP-5 `accepted_per_draft_event=3.076`.

Launcher/probe fix:
- A first acceptance attempt with `FR10_METRICS=0` failed loudly before reporting metrics because tree engagement logs were still gated on `FR10_METRICS=1`.
- Commit `aa233aed` changed the instrumentation gate so explicit tree-log env vars write engagement traces without enabling metrics, and passed those env vars through the launcher.
- Focused tests after the patch: `pytest -q tests/test_fr10_phase4_sampled_committer_wiring.py tests/test_fr10_tree_conv.py` -> `16 passed, 1 skipped`.

Baseline, `FR11_TREE_CONV_NATIVE_BF16_TAPS=0`:
- Run dir: `output/fr11_conv_fp32tap_baseline_accept_20260605T194754Z`.
- Tree engagement logs: `tree_sampler_debug.jsonl` `1679` rows; `tree_path_lcp_max.jsonl` `1064` rows.
- `accepted_per_draft_event=0.87535953978907`.
- `accepted_per_draft_token=0.09726217108767445`.
- `spec_accepted_tokens=913.0`, `spec_draft_tokens=9387.0`, `spec_drafts=1043.0`.
- Returned tokens: `1944`; request count: `8`; records: `32`.
- Warm decode TPS: `3.4234543038004497`.
- Delta vs native MTP-5 event acceptance: `-2.20064046021093`; ratio vs native: `0.2845772235985273`.

Conv fix, `FR11_TREE_CONV_NATIVE_BF16_TAPS=1`:
- Run dir: `output/fr11_conv_bf16tap_fix_accept_20260605T200003Z`.
- Tree engagement logs: `tree_sampler_debug.jsonl` `1729` rows; `tree_path_lcp_max.jsonl` `1083` rows.
- `accepted_per_draft_event=0.7597340930674265`.
- `accepted_per_draft_token=0.08441489922971404`.
- `spec_accepted_tokens=800.0`, `spec_draft_tokens=9477.0`, `spec_drafts=1053.0`.
- Returned tokens: `1857`; request count: `8`; records: `32`.
- Warm decode TPS: `3.2202026669903643`.
- Delta vs native MTP-5 event acceptance: `-2.3162659069325735`; ratio vs native: `0.24698767654987855`.

Flag delta:
- Fixed minus baseline event acceptance: `-0.11562544672164354`.
- Fixed minus baseline token acceptance: `-0.012847271857960404`.

Acceptance verdict:

`CONV_TAP_DTYPE_FIX_DOES_NOT_RECOVER_ACCEPTANCE`

The native-bf16 tap-product flag did not improve no-copy GDN tree acceptance. It reduced accepted/event from `0.87536` to `0.75973`, while native MTP-5 remains `3.076`. This isolates the conv tap-dtype seam as not the acceptance-limiting bug, even though alpha shows it can create post-MLP bf16 precision-floor deltas.
