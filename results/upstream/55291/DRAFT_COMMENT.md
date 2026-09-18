> **CORRECTION 2026-09-18 (independent Codex check, see F8 in STATIC_FINDINGS.md):** the kernel-path conclusion in F2/F7 (and the 'likely why' in RESULT.md / DRAFT_COMMENT.md) is WRONG. The startup line 'GDN decode kernel: cuda' is configuration only; the fused CUDA MTP path additionally requires speculative metadata and 8:1 V/K heads, and this run had neither (speculative_config=None; 48:16 = 3:1). Default decode-only batches use the packed Triton path; mixed decode/prefill batches can call #54146's readout. No per-batch kernel trace was captured. The L20-D bypass prediction is withdrawn. The negative result itself stands. The evidence tarball is the pristine agent output and still contains the uncorrected text.

<!-- DRAFT ONLY — NOT POSTED BY THE AGENT. {{...}} filled from the final run. -->

Tried the 0.28.0 reproduction @sizzlecar asked for.

**Did not reproduce.** 298 requests / 249,544 generated tokens over 3.3h,
`Qwen/Qwen3.6-27B-FP8` @ `e89b16eb`, stock 0.28.0 wheel, 1×GB10, TP=1, prefix caching +
chunked prefill on, `--max-num-seqs 20`, `--max-model-len 128000`, serial and 20-way
concurrent, peak context 36k tokens. No run of ≥16 `!` characters in any
completion's token ids; your canary stayed coherent throughout.

Likely why: in 0.28.0 `VLLM_GDN_DECODE_KERNEL` defaults to `cuda`, and this model meets
every condition, so GDN decode uses the fused CUDA kernel — not the Triton readout
#54146 patches. Startup logs `qwen_gdn_linear_attn.py:505 GDN decode kernel: cuda`.
The only GPU condition is capability ≥8.0, so your L20-D should take that path too.

**Worth checking on your side:** if your log instead says `Falling back to the Triton
GDN decode path: <reason>`, the unfixed readout is still live and the reason matters.

For 0.28.0: `auto` still resolves to float32 (Q2); the model is bf16, so #54146's fp16
overflow can't fire (Q3); `KVBlockZeroer` still skips Mamba, but GDN prefill zeroes
state for sequences without an initial state (Q4/Q5).

Untested: 0.21.0, TP=2, L20-D. Artifacts: {{ARTIFACT_LINK}}
