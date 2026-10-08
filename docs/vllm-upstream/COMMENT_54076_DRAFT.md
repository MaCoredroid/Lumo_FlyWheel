# #54076 reply to jschmied (funded item E) — v1 (Codex GO on this exact text after the re-pin check, 2026-09-21; AWAITING MARK GO)
> **POSTED 2026-09-21T22:39:19Z (Mark GO 2026-09-21):** https://github.com/vllm-project/vllm/pull/54076#issuecomment-5768495023
> Codex E_review.md: NO-GO on the agent's draft; GO on this text as ONE reply on #54076; no merge
> recommendation; no #58021 cross-post. Bundle corrected (CORRECTIONS.md) at 632817854588fd36fe0e00ed71d178cad323e52a; evidence files
> byte-identical to the Codex-verified 92fea914a tree.

---

@jschmied — candidate for your current-main rerun: Qwen/Qwen3.8-27B-FP8, GB10/TP=1, `--enable-prefix-caching --mamba-cache-mode align`, `extract_hidden_states` (`num_speculative_tokens=1`, `eagle_aux_hidden_state_layer_ids=[32]`) with `ExampleHiddenStatesConnector` (`kv_producer`). This is the documented extraction workflow for drafter-training data. Executed vLLM 0.28.0 only; stock INFO excerpts:
> Setting attention block size to 800 tokens
> Using block size 200 for hidden-state cache layer cache_only_layers.64; page alignment wastes 1228800 bytes (37.50%) per block

Logging-only wrappers recorded `cache_config.block_size=200` versus `MambaSpec.block_size=800`, with the same cache-config object at initialization exit and scheduler construction. The server booted and answered a request.

Static main @382970ee6c retains this hidden-state reduction path and includes these groups in its minimum (`prefix_cacheable=True`); I have not run main.

This establishes startup geometry only: no `_mamba_block_aligned_split` runtime trace or correctness conclusion.

Our Qwen3.8-FP8 + incoai/DFlash2 k=7 arm yielded 832/832 on 0.28.0, so it did not reproduce the PR body's 816/1648 geometry. [Pinned logs, hook, exact argv and revisions](https://github.com/MaCoredroid/Lumo_FlyWheel/tree/632817854588fd36fe0e00ed71d178cad323e52a/results/upstream/54076).

AI assistance was used.
