# #55291 negative-result comment (funded item C) — v1 (Codex GO on this exact text after the bundle correction, 2026-09-18)
> **POSTED 2026-09-18T02:46:06Z (Mark GO "go" 2026-09-18):** https://github.com/vllm-project/vllm/issues/55291#issuecomment-5724389616
> Codex C_review.md: NO-GO on the agent's kernel-path conclusion; GO on the
> corrected negative-result comment. Placement: ONE evidence comment on issue
> #55291 (answers sizzlecar's "reproduce on 0.28.0" request; exempt from the
> weekly ask cap). No cause claim, no "likely why", no L20-D prediction.
> Bundle corrected (F8) at 1e03bf4acafac1ce50340542febb072746618167. AWAITING MARK GO. Body below the separator.

---

Did not reproduce in this bounded 0.28.0 run: 298 requests, 249,544 generated tokens, ~3h16m, zero detected collapses/errors. Model `Qwen/Qwen3.6-27B-FP8@e89b16eb`, stock wheel, GB10/TP1, prefix caching/chunked prefill, GPU utilization 0.50. Peak context was 36,496 tokens; the 20-worker phase lasted ~50 minutes. All 26 canaries avoided collapse but exhausted 100 tokens in reasoning. Choice-level token IDs covered reasoning and content; longest `!` run was one character.

Kernel path (the bundle's F2/F7 are corrected in F8): startup `cuda` is configuration. The 0.28.0 fused CUDA MTP path requires speculative metadata and 8:1 heads; this run had neither (3:1). Default decode-only uses packed Triton; mixed decode/prefill can call #54146's readout. No runtime kernel trace was captured. Auto SSM dtype resolves to FP32; activations were BF16, so the specific FP16 overflow threshold does not apply. Prefill conditionally zeroes initial SSM state; that does not prove all recycling paths safe.

Untested: 0.21.0, TP2/L20-D, filled 128k contexts. P3 used repeated assistant snippets. [Artifacts and deviations](https://github.com/MaCoredroid/Lumo_FlyWheel/tree/2f91527dfca0908ace46fa259c70021557beaae2/results/upstream/55291). This negative result does not identify the cause or prove absence.

AI assistance was used.
