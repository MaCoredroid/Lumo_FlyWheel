# DRAFT_COMMENT.md — NOT POSTED. Nothing was posted anywhere. 178 words.

@jschmied — a configuration where it is reachable: **hidden-state cache layers**, not a drafter.

I booted your two arms plus two more on 0.28.0 (GB10, TP=1, Qwen3.8-27B-FP8, `--enable-prefix-caching
--mamba-cache-mode align`). A separate DFlash2 drafter is **not** sufficient: its attention group lands
at the same block size (832/832, `divergent=False`), because `unify_kv_cache_spec_page_size` only ever
*raises* a group's block size. `--block-size 816` gets used as the alignment grain → 1632/1632.

What does it: `--speculative-config '{"method":"extract_hidden_states","num_speculative_tokens":1,
"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[32]}}}'` + the
ExampleHiddenStatesConnector. Log: `interface.py` "Setting attention block size to 800 tokens";
`kv_cache_utils.py` "Using block size 200 for hidden-state cache layer cache_only_layers.64"; then
`min(800,800,800,800,200)` → `cache_config.block_size = 200` vs `MambaSpec.block_size = 800`.

On main @382970ee6c the `min` is filtered to `prefix_cacheable` groups — `HiddenStateCacheSpec`
inherits `prefix_cacheable = True`, so it still applies. I read that; I did not run main.

Full log excerpts: {{ARTIFACT_LINK}}. AI assistance was used.
