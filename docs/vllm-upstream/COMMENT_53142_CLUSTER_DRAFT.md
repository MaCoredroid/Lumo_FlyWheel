# Align-seed duplicate-cluster cross-reference — for issue #53142 (funded item D) — v1 (Codex replacement verbatim)

> Codex GO (2026-09-17): post ONCE on #53142; exempt from the weekly ask cap (code/test evidence); no pings; do not duplicate across PRs. NEEDS MARK GO. Body below the separator.

---

Cross-reference for this issue and #55600; these related changes differ in initialization and scope:

- #53798 uses `MambaSpec.block_size`, resolved eagerly through `ModelState.set_kv_cache_config`, and includes unit tests.
- #55507 uses `self._mamba_spec.block_size` once resolved, falling back to the global `cache_config.block_size` while unbound.
- #55601 reads `cache_config.mamba_block_size` directly.
- #55688 uses the Mamba divisor under `_use_flashinfer_replayssm`; its other align path retains the global divisor.

In the inspected main construction path, `MambaSpec.block_size` comes from `cache_config.mamba_block_size`; that does not establish equivalence with #55507's unbound fallback. Our [model-free restore-fidelity test](https://github.com/vllm-project/vllm/compare/af5357c2b90b37bd2033578bbc97d0ddfa6cc69f...MaCoredroid:vllm:p8-restore-fidelity) byte-compares restored state under unequal geometry. It passes against #53798 and fails with the old divisor. Its fixture uses #53798's initialization API; other variants need adaptation that preserves their production binding, and those ports have not been run. This oracle distinguishes incorrect restoration, not which correct implementation should land. AI assistance was used.
