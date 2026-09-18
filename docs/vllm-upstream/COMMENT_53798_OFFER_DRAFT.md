# Offer to ptorsten on #53798 — the align-mode restore-fidelity regression (P8, route 1) — v2 (Codex replacement verbatim: two-commit series stated; rerun scope accurate; AI disclosure)
> **POSTED 2026-09-13T18:29:59Z (Mark GO):** https://github.com/vllm-project/vllm/pull/53798#issuecomment-5655221523

> Mark: "B go" (2026-09-12). Branch p8-restore-fidelity pushed to MaCoredroid/vllm
> as staging (af5357c2b → ca1d410ae → bb9d7569d, test-only, DCO-signed).
> Codex GO on the artifact; this is the OFFER text → Codex check → Mark GO →
> post as a PR comment on #53798 (the thread already carries our review).
> Body below the separator.

---

Following up on the review: here is a two-commit, test-only addition atop `af5357c2b`, through `bb9d7569d`: [diff](https://github.com/vllm-project/vllm/compare/af5357c2b90b37bd2033578bbc97d0ddfa6cc69f...MaCoredroid:vllm:p8-restore-fidelity). Both commits are DCO-signed and touch only `tests/v1/worker/test_mamba_hybrid_model_state.py`.

Model-free conv/SSM pools with synthetic padding exercise real `set_kv_cache_config → add_request → preprocess_state` and both Triton kernels. With M=1648/global=816, admission at 3M, one scheduled token, a nonidentity table and slot 1, the byte oracle requires column 2 restored into column 3, preserving other blocks and padding. The old divisor seeds column 6 and fails on contents; the equal-geometry control passes. Suppressed/misdirected-copy controls check the oracle. GB10 runs passed all 10 tests before and after trimming; the original commit also received an independent rerun. AI assistance was used.

This covers worker restore fidelity, not scheduler publication or model outputs. If useful, cherry-pick `ca1d410ae` then `bb9d7569d`; otherwise no action needed.
