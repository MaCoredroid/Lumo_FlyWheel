# Making K≥2 strictly beat K=1 in speculative decoding — a SOTA survey

*Research report · 2026-05-28 · broad state-of-the-art survey*

**The question.** Under what conditions does a wider candidate set (K=2 / a verified token *tree*) strictly beat a single-branch (K=1 / chain) draft on both acceptance and throughput — and how do you cheaply verify candidates regardless of whether they come from an MTP head, from suffix/retrieval decoding, or from a *merged* suffix+MTP candidate tree?

**The short answer.** A tree strictly beats a chain on **expected accepted tokens per target forward pass**, and that gain is mathematically guaranteed for any acceptance rate > 0. But that is *acceptance*, not *throughput*. Whether the extra branch also wins on throughput is governed entirely by a **cost model**: the second branch only pays off while the target's verify pass is still memory-bandwidth-bound, so its extra candidates ride along "for the price of one." Once the batch × candidate-count product pushes the verify pass into the compute-bound regime, the extra branch adds FLOPs that aren't repaid and throughput *drops*. The frontier methods (EAGLE-2/3, Sequoia, SuffixDecoding, RASD, ReSpec) are all, in effect, machinery for **maximizing accepted tokens per node while keeping the node count under the compute-bound cliff** — and for **merging cheap heterogeneous candidates** (n-gram/suffix + model) into one lossless-verified tree.

---

## 1. Why a verified tree beats a chain — the math

A single drafted chain of length γ with per-token acceptance rate α has a closed-form expected accepted length:

> **E[accepted] = (1 − α^(γ+1)) / (1 − α)**

This is the canonical Leviathan et al. (2023) result. The chain's fatal property: it accepts up to the *first* rejected position and discards everything after, so its expected length is capped by α even as γ grows. ([Leviathan et al. 2023](https://arxiv.org/abs/2211.17192); formula restated in [aman.ai primer](https://aman.ai/primers/ai/speculative-decoding/), [DistillSpec](https://arxiv.org/html/2310.08461v2))

A **tree** offers several candidate tokens at each position, so the probability that *at least one path* survives verification rises. Sequoia states the principle directly: "using a token tree — instead of sequence — can increase the number of tokens accepted by the LLM by providing several options for each token position." ([Sequoia](https://arxiv.org/html/2402.12374v2))

Medusa gives the exact expected-accepted-length of a candidate tree as a sum over all tree nodes of the product of per-token acceptance probabilities along each root-to-node path:

> **E[accepted] = Σ_{paths [i₁…i_k]} Π_{j=1}^{k} a_j^(i_j)**

where a_k^(i) is the acceptance probability of the i-th candidate at depth k. Every node you add appends a non-negative term, so **expected accepted length is monotonically non-decreasing in tree size.** This is the precise mechanism by which a wider tree raises accepted tokens. ([Medusa §2.3.3](https://arxiv.org/html/2401.10774v3))

EAGLE-2 makes the dependency structure explicit: a node is accepted only if *all* its prefixes are accepted, so a node's "global acceptance rate" is the product of acceptance rates along its path from the root. Tree growth therefore targets the highest global-probability nodes. ([EAGLE-2 §4.1](https://arxiv.org/html/2406.16858v1))

**Diminishing returns are equally well established.** Both Sequoia and Medusa analyses find the acceleration grows only *logarithmically* with tree size, while verification cost grows with node count (bigger matmuls + larger attention). ([Sequoia](https://arxiv.org/html/2402.12374v2); [Medusa analysis](https://medium.com/data-science/exploring-medusa-and-multi-token-prediction-de7f8312e4a7)) Crucially, **topology matters more than raw width**: Sequoia proves that the "k independent sequences" topology used by SpecInfer is *asymptotically bounded* in expected accepted tokens regardless of size, whereas its DP-optimal tree is *unbounded*, growing logarithmically. ([Sequoia](https://arxiv.org/html/2402.12374v2)) Empirically, a sparse 64-node Medusa tree beats a dense 256-node one. ([Medusa](https://arxiv.org/html/2401.10774v1))

**Takeaway for K=2:** a second branch is guaranteed to add expected accepted tokens — but only a *little*, and only if those nodes sit where acceptance is genuinely uncertain (context-dependent), not on arbitrary fixed positions.

---

## 2. Where candidates come from — drafters

### 2.1 MTP heads (model-based, sequential, causal)

DeepSeek-V3 trains *D* sequential MTP modules, each predicting one additional future token while **keeping a full causal chain** — each module conditions on the previously predicted token. The released V3 ships D=1 (one extra transformer layer, layer 61, ~14B params on top of 671B, sharing the embedding and output head). At inference the MTP module(s) can be discarded *or* repurposed for speculative decoding. ([DeepSeek-V3 §2.2](https://arxiv.org/html/2412.19437v2); [DeepWiki MTP](https://deepwiki.com/deepseek-ai/DeepSeek-V3/4.4-multi-token-prediction-(mtp)))

DeepSeek reports the **second-token (depth-1) acceptance rate is 85–90%**, yielding **~1.8× TPS**. ⚠️ This is for *one* extra token; deeper speculation reuses the single module autoregressively and acceptance falls off — the gap that EAGLE-3's training-time test and Qwen3-Next's multi-step MTP training explicitly target. ([DeepSeek-V3](https://arxiv.org/abs/2412.19437))

**Qwen3-Next** (80B-A3B, Oct 2025) ships a *native* MTP module purpose-built for speculative decoding and is "specifically optimized … [for] multi-step inference … through multi-step training that maintains consistency between training and inference." It quotes a qualitative "high acceptance rate" (no headline number). Note: the dense Qwen3 release did **not** ship an MTP head — MTP is a Qwen3-*Next* feature. ([Qwen3-Next blog](https://www.alibabacloud.com/blog/602580))

A key architectural distinction: **parallel heads at one position** (Medusa-style — weaker, because the predicted tokens are not causally conditioned on each other) vs. **sequential causal modules** (DeepSeek-MTP / EAGLE-style — stronger per added depth). DeepSeek explicitly contrasts the two. ([DeepSeek-V3 §2.2](https://arxiv.org/html/2412.19437v2))

### 2.2 EAGLE family (model-based, feature-level autoregressive)

EAGLE autoregresses at the target's **feature (penultimate hidden-state) level** rather than the token level, and generates multiple draft tokens per position to form a tree verified with tree attention. EAGLE-2 makes the tree **context-aware and dynamic**, expanding/pruning using the draft model's (well-calibrated) confidence as an acceptance-rate proxy — no retraining. EAGLE-3 drops the feature-prediction constraint, predicts tokens directly, fuses low/mid/high target features, and uses a "training-time test" simulating multi-step decoding — which unlocks a *scaling law* (more draft data → more speedup) absent in earlier versions. ([EAGLE-2](https://arxiv.org/abs/2406.16858); [EAGLE-3](https://arxiv.org/html/2503.01840v1))

Reported speedups: EAGLE-2 ≈ 3.05–4.26×; EAGLE-3 up to ~5.6× (greedy), with mean acceptance length τ up to ~6.6 tokens, and it out-throughputs vanilla vLLM up to batch size 56. ([EAGLE-2 Table 1](https://arxiv.org/html/2406.16858v1); [EAGLE-3 Table 1](https://arxiv.org/html/2503.01840v1))

### 2.3 Medusa (model-based, parallel heads)

Medusa adds K≈5 lightweight decoding heads on the frozen backbone's last hidden state; each emits its top-k tokens, and the **Cartesian product** of per-head predictions forms a static tree (total nodes Σ_{k} Π_{i≤k} s_i) verified in one pass with a tree-attention mask. The motivation for a tree is calibration: head top-1 accuracy on the "next-next" token is ~60% but top-5 is >80%, so multiple candidates per head recover most of the loss. Medusa-1 (frozen) ≈ 2.2× lossless; Medusa-2 (fine-tuned) ≈ 2.3–2.8× (the README/v1 cites up to 3.6×). ([Medusa](https://arxiv.org/abs/2401.10774); [Together.ai blog](https://www.together.ai/blog/medusa))

### 2.4 Suffix / retrieval / n-gram (model-free, near-zero cost)

These produce candidates at **near-zero compute** — no draft-model forward pass — which makes them ideal as one source feeding a verified tree. The tradeoff is *variable, workload-dependent* acceptance (high on repetitive / input-grounded / agentic text, low on novel text).

- **SuffixDecoding** (Snowflake/CMU, NeurIPS 2025 Spotlight) caches token sequences from the prompt and prior outputs in **suffix trees**, builds a candidate tree by greedily expanding the highest empirical-frequency continuations, and **adapts speculation length** to match length. It drafts at **~20 µs/token on CPU with zero GPU cost**. ⚠️ *Version caveat:* v1 (Nov 2024) reported up to ~2.9× vs the SpecInfer baseline; v3 (Oct 2025) reports up to **5.3×** vs vanilla decoding on agentic benchmarks (2.8× faster than EAGLE-2/3, 1.9× faster than Token Recycling) — different baselines, same line of work. Strongest on repetitive/self-referential agentic loops; on open-ended single-turn (Spec-Bench) it is *beaten* by EAGLE-2/3. ([SuffixDecoding](https://arxiv.org/abs/2411.04975); [CMU blog](https://www.cs.cmu.edu/~csd-phd-blog/2025/suffix-decoding/); [Snowflake blog](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/))
- **REST** (NAACL 2024) retrieves continuations from a non-parametric datastore by longest-suffix match, builds a Trie, prunes low-frequency branches, and verifies the subtree with tree attention. 1.62–2.36× on 7B/13B; datastores ~12 GB (chat) / ~27 GB (code) in CPU RAM; speedup grows with datastore size. ([REST](https://arxiv.org/abs/2311.08252); [repo](https://github.com/FasterDecoding/REST))
- **Prompt Lookup Decoding (PLD) / n-gram** copies the k tokens following an exact n-gram match in the context — no model, no datastore. ~2.4× on input-grounded tasks (summarization, QA, extraction), near-zero on novel text. Native in vLLM (`method: "ngram"`). ⚠️ Known failure: `prompt_lookup_min=2` can corrupt structured/tool-call output on Qwen3-class models ([vLLM #40875](https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/)) — illustrating unsafe acceptance when verification is greedy-mismatched. ([prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding))
- **Lookahead / Jacobi decoding** reframes decoding as Jacobi fixed-point iteration, generating disjoint n-grams in parallel into an n-gram pool, verified losslessly. ~1.8× MT-Bench, >2× CodeLLaMA/HumanEval, up to 4× with 8-GPU Lookahead Parallelism. ([Lookahead](https://arxiv.org/abs/2402.02057); [LMSYS blog](https://lmsys.org/blog/2023-11-21-lookahead-decoding/))
- **Token Recycling** stores recently-seen candidate tokens in an adjacency matrix M ∈ ℝ^(|V|×k) (<2 MB), runs a BFS-like walk to assemble an 80-node / 6-layer draft tree, verified with tree attention. Train-free, ~2× across sizes, beating prior train-free methods by ~30%. ([Token Recycling](https://arxiv.org/abs/2408.08696))

---

## 3. Verifying the tree cheaply — tree attention

SpecInfer is the canonical tree verifier: it organizes drafts (from one or more small models) as a token tree and **verifies all candidate paths in a single target forward pass**, fusing tree attention into one kernel via a **topology-aware causal mask** that encodes parent-child relationships and shares computation across common prefixes. Its headline single-variable result: switching chain → tree verification raises per-token verification success from **52–57% to 96–97%** under stochastic decoding, for end-to-end speedups of 1.5–2.8× (distributed) / 2.6–3.5× (offloading). ([SpecInfer](https://arxiv.org/abs/2305.09781))

The cost of a tree verify pass scales with **total node count** (matmul rows + attention positions), so the design goal is *accepted tokens per node*:

- **Static trees** (Medusa, original EAGLE) fix the candidate pattern by position.
- **Dynamic trees** (EAGLE-2) reshape per context, because acceptance is "not only position-dependent but also highly context-dependent." Its ablation shows each tree-shaping component adds accepted tokens: full EAGLE-2 = 3.62× / τ 4.98 → without expansion+reranking = 2.81× / τ 3.92 (Vicuna-7B, temp 0). ([EAGLE-2](https://arxiv.org/html/2406.16858v1))
- **Hardware-aware optimal trees** (Sequoia) use dynamic programming to maximize expected tokens for a given size/depth, then a hardware optimizer to pick the size/depth per GPU (e.g., 64–128 nodes on-device vs ~768 for offloading). Up to 4.04× (Llama2-7B, A100) and ~10× (70B offloading, L40). ([Sequoia](https://arxiv.org/abs/2402.12374); [Together.ai](https://www.together.ai/blog/sequoia))

---

## 4. Merging multiple candidate sources into one tree

This is the direct answer to "cheaply verify candidates *regardless of how generated*." The frontier is **one merged tree, one verify pass**, with the merge bounded so node count stays under the cost cliff.

- **RASD** (Retrieval-Augmented Speculative Decoding, ACL 2025 Findings) fuses a **draft-model tree** with a **retrieval tree** via *longest-prefix matching* into one unified tree, then prunes the retrieval branches using the draft model's probability distribution to keep node count (and verify FLOPs) bounded. It works with any tree-attention method (it uses EAGLE-2) and targets exactly the case where a draft model alone has a "low upper limit on acceptance length." SOTA on DocQA / Summary / Code / In-Domain QA. ([RASD](https://arxiv.org/abs/2503.03434))
- **Arctic Inference hybrid** (Snowflake) combines model-free SuffixDecoding with a trained LSTM/MLP draft, routing **per sequence** using SuffixDecoding's own frequency-based acceptance score as the selector: if the suffix score clears the draft's token budget, use suffix candidates and skip the draft pass; else fall back to the draft. The hybrid matches or beats either source on every workload — e.g., on a mixed Llama-3.1-70B workload, LSTM-only 154 tok/s, Suffix-only 155, **hybrid 209 tok/s**. ([Snowflake blog](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/))
- **ReSpec** ("When, What, and How," arXiv 2511.01282) replaces heuristic switching with three principled rules: **WHEN** (fire retrieval only when suffix predictive entropy is low, else use the model drafter); **WHAT** (keep an EMA acceptance score per match position, retrieve all, filter, keep top-K≈3 in the tree); **HOW** (*source-aware* verification — strict/lossless for model drafts, relaxed top-K for retrieved drafts). >33% over EAGLE-2; up to 5.21× on summarization. ([ReSpec](https://arxiv.org/abs/2511.01282))

**The recurring merged-tree principle:** never blindly union all candidates — cap the fused set (top-K, draft-probability pruning, or single-source routing), because verify cost grows with merged-tree node count. RASD prunes by draft probability; ReSpec keeps top-K≈3; Arctic picks one source per sequence.

---

## 5. Keeping it lossless — tree sampling correctness

The original speculative-sampling rule: accept draft token t̂ with probability **min(1, p(t̂)/q(t̂))**; on rejection, resample from the normalized residual **norm(max(0, p − q))** and discard the rest. Leviathan et al. (2023) and Chen et al. (2023) prove this **exactly preserves the target distribution** (within hardware numerics). ([Leviathan](https://arxiv.org/abs/2211.17192); [Chen et al.](https://arxiv.org/abs/2302.01318))

Extending this to a **tree** while staying lossless:

- **SpecInfer's multi-step speculative sampling (MSS)** verifies sibling candidates at a node one-by-one with **residual renormalization** (sampling-without-replacement style): a rejected sibling's probability mass is redistributed to the next sibling rather than lost. Theorem 4.2 proves the SpecInfer output distribution equals incremental decoding exactly. ([SpecInfer](https://arxiv.org/abs/2305.09781))
- **Sequoia's sampling-without-replacement** verifier is **temperature- and top-p-robust** — it prevents the draft from "making the same mistake twice" while preserving the target distribution, beating SpecInfer/top-k verification by 65%/27%. Sampling-with-replacement fails at low T; top-k verification fails at high T; without-replacement is robust at both. ([Sequoia](https://arxiv.org/abs/2402.12374))
- **EAGLE-2/3 stay strictly lossless**: "EAGLE-2 does not … relax acceptance conditions … the distribution of the generated text remains exactly the same … provably." EAGLE-3 "uses strict speculative sampling acceptance conditions." ([EAGLE-2](https://arxiv.org/html/2406.16858v1); [EAGLE-3](https://arxiv.org/html/2503.01840v1))
- **Lossy outlier — Medusa's "typical acceptance"**: uses temperature as a threshold to admit "plausible" tokens, trading exact-distribution preservation for higher acceptance. EAGLE-2 explicitly declines to benchmark against it head-to-head because it "does not guarantee lossless acceleration." Same category: BiLD, CLLMs, SPACE, Medusa-2. ([EAGLE-2 §5.1, §6](https://arxiv.org/html/2406.16858v1))

**Implication:** a multi-source K=2 tree can be fully lossless if siblings are verified with residual renormalization (MSS) or sampling-without-replacement (Sequoia). You only sacrifice exactness if you deliberately opt into typical/relaxed acceptance (Medusa, ReSpec's retrieved-branch path) for extra speed.

---

## 6. The cost model — when K=2 pays off vs. hurts (the crux)

This is where "strictly beats K=1 on throughput" is won or lost. Acceptance gains are necessary but **not sufficient**.

**The throughput equation.** SD-latency / AR-latency = (1 / Ω(γ,α)) · (γ·T_D/T_T + T_V(γ)/T_T), where Ω(γ,α) = (1 − α^(γ+1))/(1 − α) is expected accepted tokens per step, T_D the draft cost, T_V(γ) the verify cost, T_T one target pass. The decisive term is **T_V(γ)/T_T** — the verify-cost ratio. While it stays near 1, extra candidates are nearly free; when it grows, wider speculation stops paying. ([MagicDec Eq. 1](https://infini-ai-lab.github.io/MagicDec/))

**The regime that decides everything.** At **small batch**, decoding is memory-bandwidth-bound (arithmetic intensity ~1 FLOP/byte), so you can verify several extra candidate tokens *per weight load* "for roughly the price of one." At **large batch / high concurrency**, the target's linear layers become compute-bound, "which reduces the availability of compute resources that speculative decoding utilizes for parallel verification, essentially increasing the verification-to-decoding cost ratio." Then a second branch's extra FLOPs are *not repaid*. ([MagicDec](https://arxiv.org/pdf/2408.11049); [Doubleword roofline](https://www.doubleword.ai/resources/behind-the-stack-ep-13---faster-inference-speculative-decoding-for-batched-workloads))

**Concrete crossovers:**

- vLLM, Llama3-70B, 4×H100: SD gives up to **1.5× speedup at QPS=1** but a **1.4×–1.8× slowdown at high QPS** "due to the additional compute required to propose and verify tokens." ([vLLM spec-decode blog](https://blog.vllm.ai/2024/10/17/spec-decode.html))
- SmartSpec "goodput" (tokens both verified *and* generated): optimal speculation length **k drops as batch grows** — BS=1 peaks at k≈4–5, BS=64 at k≈1–2, extreme load drives k→0 (speculation off). Cuts average latency up to 3.2×. ([SmartSpec](https://arxiv.org/html/2406.14066v1))
- Production rule of thumb: SD helps wall-clock latency below ~4–8 concurrent requests; above that the gains are eaten by verification overhead (teams benchmarking at BS=1 → 2.5× then shipping at concurrency 16 see ~5% *regressions*). ([production traps](https://tianpan.co/blog/2026-04-17-speculative-decoding-production-hidden-traps))
- Acceptance gate: at α=0.6, ~2.4× with 5 spec tokens; α=0.8 → 3.7×; below α≈0.5 verification overhead makes you *slower than baseline*. ([production traps](https://tianpan.co/blog/2026-04-17-speculative-decoding-production-hidden-traps))
- Draft-length diminishing-then-negative returns: expected accepted tokens rise with k, but a larger k raises rejection probability and wastes target compute, so there is an optimal k beyond which throughput falls. ([DistillSpec](https://arxiv.org/html/2310.08461v2); adaptive: [SpecDec++](https://arxiv.org/pdf/2405.19715), [BanditSpec](https://arxiv.org/abs/2505.15141))

**The long-context exception (important if your sequences are long).** MagicDec shows the "SD only helps small batch" rule is *wrong* for long context: beyond a critical sequence length, KV-cache loading dominates and decoding is memory-bound *even at large batch*, so SD speedup **increases** with batch. Llama-2-7B-32K self-spec rises 1.18× (BS=32) → 1.63× (BS=128). The trick is a draft with a **fixed small KV budget** (StreamingLLM, ~256–512 tokens) so draft cost barely grows with batch while acceptance stays high (0.84 at 4K → 0.79 at 100K). ([MagicDec](https://infini-ai-lab.github.io/MagicDec/))

**Implementation overhead is part of the cost too.** Arctic cut verifier latency 3.5× (1.34→0.38 ms) and proposer 3.1× via FP8-quantized speculator, sharded Top-K before all-gather, full-loop CUDA-graph capture, and greedy verification instead of rejection sampling — reaching up to 91% of theoretical speedup. A second branch's "verify cost" is implementation-dependent, not just a FLOPs count. ([Snowflake blog](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/))

**Silent-failure modes where K=2 hurts regardless of FLOPs:** MoE expert-routing mismatch between draft and target "breaks the acceptance-rate math entirely" (SD on MoE "often performs worse than baseline"); constrained/structured generation can drop tokens during batch verification. ([production traps](https://tianpan.co/blog/2026-04-17-speculative-decoding-production-hidden-traps))

---

## 7. Benchmarks where wider trees beat chains

Holding the drafter fixed and only widening chain → tree raises both accepted tokens and speedup:

| Evidence | Chain | Tree | Source |
|---|---|---|---|
| SpecInfer per-token verify success (stochastic) | 52–57% | 96–97% | [SpecInfer](https://arxiv.org/abs/2305.09781) |
| Spec-Bench, Vicuna-7B (A100, greedy), mean accepted tokens | SpS 2.28 | EAGLE-2 4.34 | [Spec-Bench](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md) |
| EAGLE lineage, Vicuna-13B (temp 0), mean τ | SpS 2.24 | EAGLE 3.96 → EAGLE-2 4.83 → EAGLE-3 6.62 | [EAGLE-3 Table 1](https://arxiv.org/html/2503.01840v1) |
| EAGLE-2 static→dynamic tree ablation, Vicuna-7B | static 3.92 τ / 2.81× | dynamic 4.98 τ / 3.62× | [EAGLE-2 Table 3](https://arxiv.org/html/2406.16858v1) |

Spec-Bench (ACL 2024 Findings) is the standard harness, reporting speedup vs vanilla AR and mean accepted tokens per step across six task domains, all on identical hardware. ([Spec-Bench](https://github.com/hemingkx/Spec-Bench))

---

## 8. Synthesis — how to actually make K=2 strictly beat K=1

Pulling the threads together, a second branch wins on **both** acceptance and throughput only when **all** of these hold:

1. **Spend the second branch where acceptance is uncertain, not on fixed positions.** A static K=2 (extra candidate at every position) mostly adds nodes that don't get accepted. The proven win is a **context-dependent / dynamic** allocation: branch only when the drafter's confidence (or predictive entropy) at that position is low. This is exactly EAGLE-2's finding ("acceptance is highly context-dependent; static trees have inherent limitations") and ReSpec's entropy-gated trigger. An *adaptive root-diversity K=2* — top-2 only at position 0 and only when position-0 confidence is low — is the minimal form of this idea. ([EAGLE-2](https://arxiv.org/html/2406.16858v1); [ReSpec](https://arxiv.org/abs/2511.01282))

2. **Keep the merged node count under the compute-bound cliff.** The extra branch is only "free" while the verify pass is memory-bound. Cap total verified tokens (RASD's draft-prob pruning, ReSpec's top-K≈3, Sequoia's hardware-aware sizing) and measure the actual T_V/T_T at your batch size — if you're already compute-bound at your serving concurrency, *no* K=2 tree shape will win on throughput. ([MagicDec](https://infini-ai-lab.github.io/MagicDec/); [SmartSpec](https://arxiv.org/html/2406.14066v1))

3. **Merge cheap heterogeneous candidates rather than widening one drafter.** The most reliable real-world K≥2 wins come from *combining* a near-zero-cost source (suffix/n-gram) with a model source into one tree — RASD (draft+retrieval, longest-prefix fusion), Arctic (suffix+LSTM, per-sequence routing), ReSpec (source-aware verification). The suffix/n-gram branch costs ~0 GPU, so it widens the tree without moving T_D. This is the cheapest possible "second branch." ([RASD](https://arxiv.org/abs/2503.03434); [Snowflake](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/))

4. **Verify the merged siblings losslessly** with residual renormalization (SpecInfer MSS) or sampling-without-replacement (Sequoia, temperature-robust) — so K=2 is provably a strict superset of K=1's distribution, not an approximation. Opt into typical/relaxed acceptance only as a deliberate speed-for-fidelity trade. ([SpecInfer](https://arxiv.org/abs/2305.09781); [Sequoia](https://arxiv.org/abs/2402.12374))

5. **If your context is long, invert the intuition.** With long sequences, a fixed-small-KV draft keeps SD (and a wider tree) winning *even at large batch*, because KV loading keeps the target memory-bound. ([MagicDec](https://infini-ai-lab.github.io/MagicDec/))

6. **Mind the engine-level traps.** On MoE targets, ensure draft/target expert routing is consistent or acceptance math breaks; on structured/tool-call output, n-gram/suffix branches can silently corrupt — gate them. ([production traps](https://tianpan.co/blog/2026-04-17-speculative-decoding-production-hidden-traps))

The blunt conclusion: a wider K=2 tree *always* helps expected accepted tokens, but only a **context-adaptive, node-capped, cheaply-sourced, losslessly-verified** K=2 — sitting in a memory-bound verify regime — strictly beats K=1 on *throughput*. If the workload's accepted-suffix events are rare or the verify pass is already compute-bound at serving batch, the literature predicts parity or regression — which matches a "K=2 ≈ K=1" outcome.

---

## References

1. Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. https://arxiv.org/abs/2211.17192
2. Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023. https://arxiv.org/abs/2302.01318
3. Miao et al. *SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification.* ASPLOS 2024. https://arxiv.org/abs/2305.09781
4. Cai et al. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* ICML 2024. https://arxiv.org/abs/2401.10774 · https://www.together.ai/blog/medusa
5. Li et al. *EAGLE-2: Faster Inference of LLMs with Dynamic Draft Trees.* EMNLP 2024. https://arxiv.org/abs/2406.16858
6. Li et al. *EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test.* 2025. https://arxiv.org/abs/2503.01840
7. Chen et al. *Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding.* NeurIPS 2024. https://arxiv.org/abs/2402.12374 · https://www.together.ai/blog/sequoia
8. Luo et al. *Turning Trash into Treasure: Accelerating Inference of LLMs with Token Recycling.* ACL 2025. https://arxiv.org/abs/2408.08696
9. DeepSeek-AI. *DeepSeek-V3 Technical Report* (Multi-Token Prediction §2.2). 2024. https://arxiv.org/abs/2412.19437 · https://deepwiki.com/deepseek-ai/DeepSeek-V3/4.4-multi-token-prediction-(mtp)
10. Qwen Team. *Qwen3-Next* (native MTP). Oct 2025. https://www.alibabacloud.com/blog/602580
11. Oliaro et al. *SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications.* NeurIPS 2025. https://arxiv.org/abs/2411.04975 · https://www.cs.cmu.edu/~csd-phd-blog/2025/suffix-decoding/ · https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/
12. He et al. *REST: Retrieval-Based Speculative Decoding.* NAACL 2024. https://arxiv.org/abs/2311.08252 · https://github.com/FasterDecoding/REST
13. Saxena. *Prompt Lookup Decoding.* 2023. https://github.com/apoorvumang/prompt-lookup-decoding
14. Fu et al. *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding.* ICML 2024. https://arxiv.org/abs/2402.02057 · https://lmsys.org/blog/2023-11-21-lookahead-decoding/
15. *RASD: Retrieval-Augmented Speculative Decoding.* ACL 2025 Findings. https://arxiv.org/abs/2503.03434
16. *ReSpec: When, What, and How — Rethinking Retrieval-Enhanced Speculative Decoding.* 2025. https://arxiv.org/abs/2511.01282
17. Chen et al. *MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation.* 2024. https://arxiv.org/abs/2408.11049 · https://infini-ai-lab.github.io/MagicDec/
18. *SmartSpec: Optimizing Speculative Decoding for Serving LLMs Using Goodput.* 2024. https://arxiv.org/abs/2406.14066
19. vLLM. *How Speculative Decoding Boosts vLLM Performance by up to 2.8x.* 2024. https://blog.vllm.ai/2024/10/17/spec-decode.html · n-gram docs: https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/
20. Pan. *Speculative Decoding in Production: Hidden Traps.* 2026. https://tianpan.co/blog/2026-04-17-speculative-decoding-production-hidden-traps
21. Zhou et al. *DistillSpec: Improving Speculative Decoding via Knowledge Distillation.* 2023. https://arxiv.org/abs/2310.08461
22. Xu et al. *Spec-Bench: A Comprehensive Benchmark for Speculative Decoding.* ACL 2024 Findings. https://github.com/hemingkx/Spec-Bench
23. AMD ROCm + SGLang. *Multi-Token Prediction (NEXTN) serving tutorial.* https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/mtp.html

*Caveats: SuffixDecoding's headline number depends on version/baseline (v1 ~2.9× vs SpecInfer; v3 5.3× vs vanilla). Medusa-2 is cited as 2.3–2.8× (paper v3) or up to 3.6× (README/v1). A handful of SpecInfer figures and a few node-count specifics were corroborated via search-result excerpts of the primary papers rather than full-text fetches; treat exact digits as paper-version-dependent and re-check against the cited PDF before quoting in print.*

---

## Addendum (2026-05-29) — target-model correction: Qwen3.6-27B-FP8

An earlier draft referenced Qwen3-**Next** (80B-A3B MoE) specs. The actual target is **`Qwen/Qwen3.6-27B-FP8`**, the flagship **dense** (non-MoE) model of the Qwen3.6 family. It uses the *same* gated-delta-network hybrid attention as its MoE siblings, so the hybrid-attention conclusion is unchanged — only the layer counts differ. Verified against the [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-27B) and the HF model card (as captured in the repo's `round-Fa-qwen-hybrid-low-cost-tree-verifier-spec` note):

| Property | Value |
|---|---|
| Params / layers | 27B / 64 |
| Hidden | 5120 |
| Layer pattern | 16 × [ 3×(Gated DeltaNet → FFN) → 1×(Gated Attention → FFN) ] = **48 GDN + 16 softmax** (3:1) |
| Gated DeltaNet (linear, O(n); recurrent state **+ short-conv state**) | 48 V heads, 16 QK heads, head dim 128 |
| Gated Attention (softmax, GQA) | 24 Q heads, 4 KV heads, head dim 256, partial RoPE (dim 64) |
| Quantization | fine-grained FP8, block size 128; KV cache `fp8_e5m2` |
| Context | 262,144 native (≈1M with YaRN) |
| MTP | native, multi-step trained; vLLM method `mtp` / `qwen3_next_mtp`, `num_speculative_tokens` 1–2 (linear chain) |
| Modality | multimodal (vision + text) |

**Implications for K=2 (these supersede any Qwen3-Next-specific phrasing above):**

- **The hybrid-attention blocker is stronger, not weaker: 48 of 64 layers (75%) are Gated DeltaNet** linear-attention layers, each carrying a recurrent state *and* a short-conv state, with no per-pair attention matrix to mask. A topology-aware tree mask corrects only the 16 softmax (Gated Attention) layers; it cannot prevent the 48 GDN layers from absorbing sibling/cousin-branch contributions when a unique-node tree is flattened into a sequence.
- **F_a (unique-node packed tree) therefore requires a *state-tree* verifier, not tree attention** — per depth, gather parent GDN+conv state by `parent_id`, run one batched GDN step, scatter to `node_id`, with tree-aware rejection sampling over siblings. This is exactly the design in the repo's `round-Fa` note and matches the external SSM/hybrid evidence: [STree](https://arxiv.org/abs/2505.14969) and [SpecMamba](https://arxiv.org/abs/2509.19873) both state SSM/linear layers "lack a mechanism to specify the tree structure" and need a bespoke tree-scan kernel rather than an attention mask. (The repo note also cites two 2026 papers — a hybrid self-speculation study, arXiv 2605.01106, and SMART on super-linear big-tree cost, arXiv 2604.09731 — not independently re-verified here.)
- **F_b (row-1 from the unselected MTP top-p alternative) remains the correct cheap near-term K=2 path.** At K=2 the row layout's only real penalties are one extra recurrent-state branch and minor prefix recompute — far cheaper than building and maintaining a GDN state-tree kernel. The "no extra compute" property is exact for the *draft* (row 1 is read from logits already produced for row 0); the *verify* side still needs row 1's own GDN+conv state branch.
- **The MoE expert-routing failure mode in §6 does NOT apply** — Qwen3.6-27B is dense.

Bottom line for making K=2 strictly beat K=1 on this model: ship the gated F_b row-1 path (cheap, correct), and treat F_a's GDN state-tree verifier as a separate research-grade kernel project justified only if the row-1 candidate quality proves insufficient and node-count microbenchmarks show the state-tree is genuinely cheaper than path-rows at N ≥ 8.
