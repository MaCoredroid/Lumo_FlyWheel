# #54928 reproduction report — v2 (Codex review applied verbatim) (Phase 2 item 1; funded run complete 2026-09-11 00:29 UTC)

> **POSTED 2026-09-11 23:18 UTC (Mark GO, Codex GO):** https://github.com/vllm-project/vllm/issues/54928#issuecomment-5641733441
> v2 = Codex's 324-word replacement (v1 was 573 words; cut: bf16-ulp
> attribution [gap ~12.0625 is a counterexample], loose E==V!=A for G [A
> argmax is a tie], unfunded follow-up offer). Results-only, cause-neutral, scoped as a RELATED reproduction (FP8,
> single GB10, stock v0.28.0) per EXPERIMENT_CARD_54928.md v2. Evidence
> archive pinned on the review branch (commit 1108c76c1). Codex independent
> review → Mark GO → post as one issue comment on #54928. No tree mention,
> no oracle-floor argument. Body below the separator (~430 words).

---

**Related GB10 reproduction: repeatable output divergence with different reported rankings.**

Stock official v0.28.0 aarch64 wheel (`817b8181…`), torch 2.13.0+cu130, CUDA 13.0, driver 590.48.01, one GB10 (sm_121). Target: `Qwen/Qwen3.8-27B-FP8` @ `017b9c7a`; draft: `incoai/Qwen3.8-27B-DFlash2` @ `dedf8df6`, K=7. A is target-only; B adds DFlash2. Both use TP=1, prefix caching off, max_num_seqs=4, max_model_len=4096, default compilation, raw_logprobs, and gpu_memory_utilization=0.50. We put Ninja on PATH and cleared page cache before launches; matrix.log records earlier aborted setup attempts. No source patches. Full settings and hashes are archived.

G uses @Windless84's git-squash prompt, enable_thinking=false, seed=1234. T uses the issue's Spanish tetris prompt, reasoning_effort=medium, preserve_thinking=true, seed=20260902. Requests use temperature=0, top_p=1, top_k=-1, max_tokens=256 and top-5 logprobs. Launch order A/B then B/A; five repeats per case per launch. All 40 completed responses contain 256 generated IDs aligned with every logprob entry, no response errors, finish_reason=length. Rendered prompt IDs match across arms. Each arm/case's token IDs and returned logprobs match across all ten requests.

First differing generated positions are zero-based: G=69, T=53, in both launches. G: A emits 5339, tied with 8932 at the reported maximum; B emits 8932, its unique reported maximum, ahead by 0.125 nats. T: A emits 579 ahead of 728 by 0.125; B emits 728 ahead of 579 by 0.125. Defining A as the target-only emitted token, E as the speculative emitted token, and V as B's unique reported maximizer, both cases have E=V≠A. Before divergence, the common emitted token's logprob differs at every position: absolute mean/max differences are G 0.007620/0.066414 nats over 69 positions and T 0.040926/0.147165 over 53. These returned values do not establish an underlying BF16 ulp scale or a kernel cause.

Speculation counters match across B launches: after G's five requests, cumulative drafts/draft tokens/accepted tokens are 290/2030/985; after G+T, 735/5145/1820. This is a related FP8 single-GB10 reproduction, not the original BF16 TP=4 configuration or sm_120. No eager arm or current-main comparison was run; state effects remain possible. [Requests, responses, metrics, server logs, argv and dependency lock](https://github.com/MaCoredroid/Lumo_FlyWheel/blob/1108c76c1a04ab0378372cc18b3f6ba54089aea9/results/upstream/54928/evidence_54928_20260911T003009Z.tar.gz), archive SHA256 `e8ee636ca08864aa7029212e2e307625505f8a37a596071e37e8842b16c7f612`.
