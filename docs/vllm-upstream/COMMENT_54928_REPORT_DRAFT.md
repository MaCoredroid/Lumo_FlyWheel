# #54928 reproduction report — v1 (Phase 2 item 1; funded run complete 2026-09-11 00:29 UTC)

> Results-only, cause-neutral, scoped as a RELATED reproduction (FP8,
> single GB10, stock v0.28.0) per EXPERIMENT_CARD_54928.md v2. Evidence
> archive pinned on the review branch (commit 1108c76c1). Codex independent
> review → Mark GO → post as one issue comment on #54928. No tree mention,
> no oracle-floor argument. Body below the separator (~430 words).

---

**Related reproduction on a DGX Spark GB10 (sm_121), stock v0.28.0, FP8 target: reproduces; deterministic; E == V ≠ A at 0–1-ulp margins.**

Setup (all pinned; details and raw data in the archive linked below): official `vllm-0.28.0` aarch64 wheel (SHA256 `817b8181…`), torch 2.13.0+cu130, CUDA 13.0, driver 590.48, one GB10 (117 GiB unified). Target `Qwen/Qwen3.8-27B-FP8` @ `017b9c7a`, draft `incoai/Qwen3.8-27B-DFlash2` @ `dedf8df6` (bf16), `num_speculative_tokens: 7`, TP=1, prefix caching off, `max_num_seqs 4`, `max_model_len 4096`, default compilation, `--logprobs-mode raw_logprobs`, `gpu_memory_utilization 0.50` on both arms. Requests: temperature 0, top_p 1, top_k −1, fixed seed, 256 new tokens, `logprobs` top-5, `return_token_ids` + `return_tokens_as_token_ids`. Cases: **G** = @Windless84's git-squash prompt, `enable_thinking: false`, seed 1234; **T** = this issue's Spanish tetris prompt, `reasoning_effort: medium`, `preserve_thinking: true`, seed 20260902. Two launches per arm (A/B then B/A), five repeats per case per launch — 40 responses, all with token IDs aligned 1:1 to logprob entries, zero errors, all `finish_reason: length`. To run at all on this box: `ninja` on the server's PATH (FlashInfer JIT-builds its sampling kernels on sm_121) and a page-cache drop before each launch (the startup free-memory check counts it on unified memory). No source patches.

| case | first divergence | target-only (A): emitted / top-2 gap | DFlash2: emitted E / verifier argmax V unique / gap | within-launch | across launches |
|---|---|---|---|---|---|
| G | 69 | 5339, exact tie with 8932 (0.000) | 8932 / yes / 0.125 | 5/5 both arms | identical, both arms |
| T | 53 | 579 over 728 by 0.125 | 728 / yes / 0.125 | 5/5 both arms | identical, both arms |

Pattern in both cases and both launches: **E == V ≠ A** — the emitted token is the verifier's own argmax, which differs from the single-token forward's argmax at a position where the target-only top-two gap is 0 or 0.125 nats. In G the target-only forward is at an exact tie; in T it prefers 579 by 0.125 and the verify forward inverts that by the same amount.

On the shared prefix the two forwards are never numerically identical: the emitted token's logprob differs at every position (G: mean 0.008, max 0.066 nats over 69 positions; T: mean 0.041, max 0.147 over 53), while the top-1 matches at 69/69 and 52/53 (the exception is an exact tie in target-only). Target-only has 6 (G) and 10 (T) positions per 256 with a top-two gap ≤ 0.125 nats, of which 2 and 4 are exact ties; not every such position flips. The returned top-two gaps are quantized to multiples of 0.125 at these magnitudes (bf16 logits: one ulp at |logit| in [16, 32)), so both divergence positions are 0- or 1-ulp gaps.

Spec path exercised (`/metrics`, cumulative within a launch after each case's five requests): G 290 drafts / 2030 draft tokens / 985 accepted; T 735 / 5145 / 1820.

Scope: a related reproduction — FP8 single-device on sm_121, not the original BF16 TP=4 setup nor sm_120; stock wheel only; no `--enforce-eager` arm; no current-main comparison. It localizes a ranking discrepancy between the block-shaped verify forward and the single-token forward at 0–1-ulp gaps; it does not identify a kernel cause or rule out state effects.

Raw responses, requests, `/metrics`, server logs, resolved argv, lock file and SHAs: https://github.com/MaCoredroid/Lumo_FlyWheel/blob/1108c76c1a04ab0378372cc18b3f6ba54089aea9/results/upstream/54928/evidence_54928_20260911T003009Z.tar.gz (SHA256 `e8ee636c…`). If one discriminator would help most (eager pair on both arms, K=1, or current main with a paired baseline), say which and I'll run that one.
