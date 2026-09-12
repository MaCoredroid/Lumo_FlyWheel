# Offer to ptorsten on #53798 — the align-mode restore-fidelity regression (P8, route 1) — v1

> Mark: "B go" (2026-09-12). Branch p8-restore-fidelity pushed to MaCoredroid/vllm
> as staging (af5357c2b → ca1d410ae → bb9d7569d, test-only, DCO-signed).
> Codex GO on the artifact; this is the OFFER text → Codex check → Mark GO →
> post as a PR comment on #53798 (the thread already carries our review).
> Body below the separator.

---

Following up on the review above with something you can take or leave: a test-only commit on top of `af5357c2b` that exercises the unequal-geometry case end to end rather than as an integer check — https://github.com/vllm-project/vllm/compare/af5357c2b90b37bd2033578bbc97d0ddfa6cc69f...MaCoredroid:vllm:p8-restore-fidelity (one file, `tests/v1/worker/test_mamba_hybrid_model_state.py`, DCO-signed).

It builds a model-free padded state pool, runs the real `set_kv_cache_config → add_request → preprocess_state` path with both Triton kernels (M=1648, global block 816, admission at 3M, one scheduled token, non-identity block table, slot 1), and compares byte views of the logical state: column 2 must be restored into column 3 with every other block and the padding untouched. With the pre-fix seed it fails on the content check (the seed lands on column 6); on this head it passes; an equal-geometry control and two negative controls (suppressed / misdirected copy) are included. Verified twice independently on a GB10.

Scope is worker restore fidelity only, not scheduler publication. If it's useful, cherry-pick it or tell me a shape you'd prefer; if not, no action needed.
