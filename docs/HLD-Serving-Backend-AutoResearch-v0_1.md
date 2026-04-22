# HLD · Serving Backend & Auto-Research Optimization Loop

> Codex-Bench · High-Level Design
> Scope: The serving-and-optimization stack that runs on a single DGX Spark — vLLM-based serving, kernel-level tuning, request shaping, an auto-research agent that optimizes serving config offline before each campaign, and the weight-update path that connects online RL / SFT to live serving.
> Status: DRAFT v0.1 — supersedes LLD-01 (vLLM Serving Layer), LLD-02 (Data Pool Manager), LLD-03 (Task Orchestrator), LLD-04 (Latency Telemetry Capture), LLD-13 (Codex-Long Scenario Framework). Those five LLDs are retained in `docs/` with DEPRECATED banners once this HLD is approved; implementation in `src/lumo_flywheel_serving/` continues to run until the re-platforming pass derived from this HLD lands.
> Target hardware: 1× NVIDIA DGX Spark (GB10 Superchip) — no second machine, no cluster assumption anywhere in this doc.
> Target model: [Qwen 3.5 27B](https://huggingface.co/Qwen/Qwen3.5-27B-FP8), served from FP8 weights (public HuggingFace artifact: `Qwen/Qwen3.5-27B-FP8`). Locked as the single model in scope for this HLD. The architecture is dense (every parameter activates on every token — no MoE routing), but the attention stack is **hybrid**: 64 transformer blocks arranged as 16 repeats of `(3 × Gated DeltaNet block → 1 × Gated Attention block)` per the public model card. Gated DeltaNet blocks use linear attention with an O(N) recurrent state (no per-token KV cache; 48 V heads / 16 QK heads / head_dim=128); Gated Attention blocks use standard multi-head attention (24 Q heads / 4 KV heads / head_dim=256, RoPE) with a per-token KV cache. The model is also vision-enabled (early-fusion multimodal) and ships with thinking-mode on by default. Model-structure alternatives (MoE variants in `model_registry.yaml`) are explicitly out of scope; their rows appear in §3.10 as envelope reference only. Revisit trigger in §11 open questions.
> Vision-tower disable is load-bearing for this HLD. Coding workloads never send images, so vLLM is launched with the vision branch disabled (`--limit-mm-per-prompt image=0` or the equivalent model-config flag); every verification item in §9 assumes the vision tower is inactive. Re-enabling the vision tower is a v0.2 problem.
> Thinking-mode contract. Qwen 3.5 thinks by default; vLLM exposes `enable_thinking` (chat_template_kwargs) and `thinking_token_budget` (extra_body) as first-class controls, and we use them as such. For agentic coding tasks the serving layer runs with `enable_thinking=true` and a generous `thinking_token_budget` (default: 8192 reasoning tokens per turn; family spec may override). The thinking contract is part of the serving-backend contract in §4, not a free-floating knob.
> Responses API: Qwen 3.5 supports the OpenAI-compatible Responses API natively via vLLM's `/v1/responses` endpoint; no separate compatibility gate is required. The Qwen3 reasoning parser (`--reasoning-parser qwen3`) is enabled so thinking content surfaces in `reasoning_content` rather than leaking into `content`. Any future model added to the roster must re-open both this decision and the thinking-contract decision above.
> vLLM version is pinned. The attention backend on Blackwell has churned from FA2 → FA3 → FA4 during 2025–2026, and vLLM's Blackwell auto-selection priority (FlashInfer first, FlashAttention second, with FA4 as the default FA version when FlashAttention is selected) has its own moving pieces; this HLD targets vLLM 0.19.x with the **attention backend for the Gated Attention family force-pinned in every tuned bundle** rather than left to upstream auto-select. The pinned version and the pinned `attention_backend` travel with every tuned-config bundle per §5.9 and are enforced by the bundle-validity rule in §6.4.
> Audience: coding-agent implementer + verifier pair. Every load-bearing requirement lands in §9's binary pass/fail checklist.

---

## Changelog

| Version | Change |
|---|---|
| v0.1 | Initial draft. Model locked to Qwen 3.5 27B (FP8) — dense, hybrid-attention (3 × Gated DeltaNet + 1 × Gated Attention per 4-layer group, 16 groups = 64 decoder blocks per the public Qwen3.5-27B-FP8 model card), vision-tower-disabled for coding, thinking-on with configurable budget, vLLM 0.19.x pinned with an explicit (force-selected) `attention_backend` per tuned bundle rather than upstream auto-select. Establishes hardware envelope for single DGX Spark, the serving-backend contract (three-dimensional latency SLO: TTFT, TPOT, per-turn), the auto-research agent design (layered action space: L0 kernel — split across the two attention families, with sub-levels for selection → parameter-tune → LLM-authored mutation gated on a numerical-parity harness; L1 vLLM config including thinking-mode knobs; L2 request shaping; L3 LoRA management; one-time per `(model, family)` pair, no automatic re-tuning; built on OSS inner optimizers — Optuna TPE, `@triton.autotune`, CUTLASS Python tuner, vLLM `benchmarks/benchmark_serving.py`), the weight-update path (PyTorch writes → KV invalidate → resume), the single bundle-validity rule (hard pins on model/family/vllm/kernel/triton/cutlass/optuna versions + explicit `attention_backend` + serving posture including thinking-mode defaults; weight_version_id is metadata only), the admission-layer policy that rejects request-level thinking overrides on campaign traffic, the telemetry contract in which `/metrics` exposes vLLM-native histograms (`vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:e2e_request_latency_seconds`) plus LLD-SB-02-added `tokens_reasoning` / `tokens_response` counters and p95s are derived via PromQL `histogram_quantile()` client-side, the machine-restart resumability contract, the pass/fail verification checklist for the implementer + verifier agent pair, and the deprecation migration plan for LLD-01 / 02 / 03 / 04 / 13. Explicitly out of scope: Responses-API compatibility gating (Qwen 3.5 supports it natively), model-structure alternatives (single-model lock per above), vision workloads (tower disabled), and automatic bundle re-tuning (v0.2 scope). |
| v0.1.1 | Correctness revision driven by sign-off review. Four blockers fixed: (P1.1) layer-count math re-anchored to the public Qwen3.5-27B-FP8 model card — 64 decoder blocks = 16 × (3 DeltaNet + 1 GatedAttn), 16 Gated Attention layers (not 12), 48 Gated DeltaNet layers (not 36); per-token KV recomputed to 2 × 16 × 4 × 256 × 1 = 32 KB/token (not 24), 131k-context full-fill footprint ~4.2 GB (not ~3); Appendix B subtitles and §3.10/§3.11/§3.12/§3.15 updated accordingly. (P1.2) Blackwell backend contract rewritten: vLLM auto-selection on Blackwell resolves to FlashInfer (with TRTLLM) first and FlashAttention (with FA4 as default FA version) second, *not* FA4-by-default; FlashInfer on Qwen3.5 on Blackwell has a known accuracy issue (GitHub vllm-project/vllm #35138), so the HLD pins an explicit force-selected `attention_backend` per tuned bundle rather than relying on auto-select; §9.3.13 rewritten to require an explicit selection (not to mandate FA4); Appendix A's "vllm-default resolves to FA3" corrected; Appendix B.2 reordered to reflect actual auto-select order with the caveat. (P1.3) Telemetry overclaim fixed: `/metrics` exposes vLLM-native histograms (`vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:e2e_request_latency_seconds`) and counters, not pre-aggregated `ttft_p95` / `tpot_p95` / `turn_latency_p95` series; p95s are derived client-side via PromQL `histogram_quantile(0.95, sum by (le, instance) (rate(vllm_*_bucket[5m])))`; `tokens_reasoning` / `tokens_response` are LLD-SB-02 additions to the telemetry plane, not native vLLM metrics. (P2.1) Thinking-override contract made explicit and enforceable: campaign traffic MUST NOT override `enable_thinking` or `thinking_token_budget` at request level — the admission layer rejects any request that sets `chat_template_kwargs.enable_thinking` or `extra_body.thinking_token_budget` with a structured 4xx; §4.1, §5.9 posture pin, §6.4 bundle-validity rule, and §9.3.9 all updated. Everything else unchanged. |

---

## 1. Purpose & Scope

This HLD defines what a single DGX Spark must deliver to run the Codex-Bench training flywheel end-to-end — fast enough that the bottleneck becomes the model, not the serving stack. It is the architectural parent of every LLD in the serving + orchestration path going forward.

### 1.1 What is in scope

- The vLLM serving layer that exposes Qwen 3.5 27B (FP8) via an OpenAI-compatible `/v1/responses` endpoint to Codex threads.
- The auto-research agent that tunes the serving layer offline, pre-campaign, per `(model, family, eval-workload)` tuple — producing a frozen tuned-config bundle consumed at campaign time.
- The auto-research action space, explored as a layer stack (L0 kernel → L1 vLLM config → L2 request shaping → L3 LoRA). L0 itself has three sub-levels — kernel **selection** from a pinned compatibility matrix, kernel **parameter-tune** via `@triton.autotune` / CUTLASS Python tuner, and LLM-proposed kernel **mutation** gated on a numerical-parity harness. L1/L2/L3 are knob-space.
- The weight-update path: how PyTorch training processes push new weights into the live server, how KV cache is invalidated, and how in-flight requests are handled.
- The machine-restart resumability contract — what `make resume` must do after a power cycle.
- The measurement harness that reuses LLD-04's `/metrics` delta-sampling plane plus a synthetic Codex-thread load generator that mirrors the eval workload.
- The pass/fail verification checklist (§9) that the implementer + verifier agent pair uses as the acceptance gate.
- The deprecation and migration plan for LLD-01 / 02 / 03 / 04 / 13 — all five are superseded by this HLD, but `src/lumo_flywheel_serving/` keeps running until the derived LLDs land.

### 1.2 What is out of scope

- Multi-machine / distributed training or serving of any kind (see §3.9 for the explicit non-feature list).
- The training algorithms themselves (QLoRA SFT, DAPO RL) — those live in HLD Training Flywheel LLD-06 → LLD-11. This HLD only defines how the output of those algorithms lands in the serving layer.
- Benchmark-family design, variant progression, and eval scoring — see `HLD-Family-Test-Requirements-v0_1.md`.
- Model-structure alternatives. Qwen 3.5 27B is locked for v0.1 of this HLD. The MoE entries in `model_registry.yaml` appear in §3.10 only as envelope reference, and revisiting them is tracked as §11.1.
- Responses-API compatibility gating. Qwen 3.5 supports Responses API natively; there is no per-model certification step in this HLD. A future model addition would re-open this.
- FP4 serving. vLLM's FP4-on-Blackwell kernel maturity is moving; tracked as §11.2 risk and a potential future auto-research lever, not a v0.1 deliverable.

### 1.3 Audience

This HLD is read primarily by **two coding agents**:

- **Implementer agent** — derives LLDs and code from this HLD. Owns §8's downstream LLD list and the migration of `src/lumo_flywheel_serving/`.
- **Verifier agent** — runs §9's pass/fail checklist against the implementer's work and reports binary results.

Human reviewers (the owner of `Lumo_FlyWheel`) are the secondary audience. Every decision in this HLD is written to be verifiable from artifacts the implementer produces, not from reviewer judgment. Where a decision is subjective ("the agent should produce useful tuned configs"), the verification item in §9 converts it to something measurable (objective function improvement > baseline on held-out workload).

### 1.4 Relationship to HLD Training Flywheel (LLD-06 → LLD-11)

The Training Flywheel HLD (`HLD-Training-Flywheel-LLD06-11-v0_1.md`, v0.8) is the parent on the training axis. This HLD is the parent on the serving + orchestration axis. The two meet at two well-defined seams:

| Seam | Training-side contract | Serving-side contract (this HLD) |
|---|---|---|
| Weight publication | Training process writes `/models/active/weights.pt` via atomic write-then-rename. | §6 weight-update path: serving watches for the rename, invalidates KV, resumes. |
| Telemetry | Training consumes trajectory rows (scorer outputs, integrity flags) from the event store. | §4.4 eval-and-train contract: serving tags every response with `weight_version_id`, `tuned_config_id`, and LLD-04 telemetry row ID. |

The two HLDs must remain consistent. If this HLD changes the weight-publication path, the Training Flywheel HLD §6 must update in lockstep — a concern §11.3 tracks.

---

## 2. Core Design Decisions

Six load-bearing calls. Every downstream section in this HLD derives from one or more of them. Changing any of these reopens the whole design.

1. **Single-machine assumption is load-bearing.** One DGX Spark (GB10 Blackwell). No TP / PP / DP sharding, no cluster, no NCCL, no cross-machine NVLink. This removes a large class of design problems (distributed weight sharding, cross-machine collectives, multi-node failure handling) and a large class of levers (tensor-parallel width, pipeline-stage count) from the auto-research agent's action space. See §3.9.
2. **Model is locked to Qwen 3.5 27B (FP8), with a specific architectural shape and serving posture.** Single model in scope for this HLD. The load-bearing details that ripple into §4 (backend contract), §5 (auto-research agent action space), §6 (weight-update path), and §9 (verification checklist) are:
    - **Weights.** ~27 GB at FP8 (block-128 fine-grained), all-active (every parameter participates in every token — no MoE routing).
    - **Hybrid attention stack** (per the public `Qwen/Qwen3.5-27B-FP8` model card). **64 decoder blocks arranged as 16 repeats of `(3 × Gated DeltaNet → 1 × Gated Attention)`.** Gated DeltaNet blocks are *linear attention* with an O(N) recurrent state — no per-token KV cache (48 V heads / 16 QK heads / head_dim=128). Gated Attention blocks are *standard attention* with a per-token KV cache (24 Q heads / 4 KV heads / head_dim=256, RoPE). This asymmetry is load-bearing for §3 KV math, §5.3.1 kernel tuning (two distinct kernel families), and §9 verification. Hidden size 5120, FFN size 17408, vocab 248320.
    - **Vision tower disabled.** Qwen 3.5 27B is a multimodal early-fusion model; coding workloads never send images, and we launch vLLM with the vision branch disabled (`--limit-mm-per-prompt image=0` or equivalent). This removes the image-encoder weight footprint from the hot path and eliminates a whole class of edge cases from §4 and §9.
    - **Thinking-on by default, with a configurable budget, and a closed request-level override surface for campaign traffic.** vLLM exposes `enable_thinking` (chat_template_kwargs) and `thinking_token_budget` (extra_body) as first-class controls, and — per current vLLM reasoning docs — request-level `chat_template_kwargs` overrides server defaults by default. This HLD closes that override surface for campaign traffic: the serving layer runs `enable_thinking=true` with a family-spec-overridable `thinking_token_budget` (default 8192 reasoning tokens per turn), and the admission layer rejects any incoming request that sets `chat_template_kwargs.enable_thinking` or `extra_body.thinking_token_budget` (see §4.1). Thinking tokens count toward TPOT and turn latency but not TTFT (see §4.1 three-dim latency contract).
    - **Native Responses API.** vLLM `/v1/responses` + `--reasoning-parser qwen3` surfaces thinking content in `reasoning_content`; no separate compatibility gate.
    - **vLLM 0.19.x pinned, `attention_backend` force-selected per tuned bundle.** vLLM's own Blackwell auto-selection resolves to FlashInfer (with TRTLLM) first and FlashAttention (with FA4 as the default FA version) second — *not* FA4-by-default; additionally, FlashInfer on Qwen3.5 on Blackwell carries a known accuracy issue (GitHub vllm-project/vllm #35138). To get a reproducible bundle we therefore **force-select** `attention_backend` in every tuned bundle rather than rely on auto-select (see §5.3.1.a, §9.3.13). FA2 / FA3 / FA4 / FlashInfer / Triton all remain candidates for L0-GatedAttn-a to pick among; the empirical winner — typically but not guaranteed to be FA4 on SM100+ after accuracy screening — is pinned in the bundle and enforced by the bundle-validity rule (§6.4).

    Re-evaluation trigger in §11.1.
3. **Auto-research is offline and pre-campaign.** Not continuous, not during-training. Runs once per `(model, family, eval-workload)` tuple, outputs a frozen tuned-config bundle consumed by the serving stack during the campaign. This is the Karpathy closed-loop research-agent framing: propose a config → measure on the fixed workload → iterate → produce a locked artifact. Cadence detail in §5.1.
4. **Weight-update path is intentionally trivial.** PyTorch writes model weights → KV cache invalidated → next request served against fresh weights. No hot-swap-while-preserving-KV, no partial re-shard, no gradient-aware KV rebuild. Simplicity is the feature; see §6.
5. **Kernel tuning is the bottom layer of the auto-research agent's stack, is tuned first, is the layer where "research" actually happens, and on Qwen 3.5 27B spans two kernel families.** The other three layers (L1 vLLM, L2 shaping, L3 LoRA) are knob-space — the agent picks values from a fixed config surface. Kernel (L0) is richer. Because the model's attention stack is hybrid, L0 operates over **two kernel families that are tuned independently**:
    - **Family A — Gated DeltaNet (linear-attention) kernels.** The chunked-delta / state-update kernels used by the **48 DeltaNet layers**. Typically Triton-based; the library of mature choices is smaller than for standard attention, which makes L0b autotune and L0c mutation more valuable at this family.
    - **Family B — Gated Attention (standard-attention) kernels.** The prefill / decode / chunked-prefill kernels used by the **16 Gated Attention layers**. Candidate pool: FlashAttention 4 / FA3 / FA2, FlashInfer (with TRTLLM on Blackwell — carries GitHub vllm-project/vllm #35138 accuracy caveat on Qwen3.5), Triton fallback, vllm-default. No candidate is "the default" for tuning purposes — L0-GatedAttn-a (§5.3.1.a) force-selects the empirical winner and that selection is hard-pinned in the bundle (§5.9, §6.4). On SM100+ the winner is typically FA4, but that must be *measured and selected*, not assumed.

    Within each family L0 has three sub-levels: **selection** (pick a kernel from Appendix B's per-family menu), **parameter-tune** (autotune the selected kernel's block sizes / tiling / pipeline depth via `@triton.autotune` or the CUTLASS Python tuner), and **mutation** (the LLM proposer edits or writes a Triton / CUTLASS kernel — fused ops, new tilings, novel schedule). Mutation is the Karpathy-style research part — and it is gated on a numerical-parity harness (§5.7 rail 9) so a faster-but-wrong kernel cannot slip through and silently corrupt eval. The parity harness runs per-family: every mutation is scored against the same-family reference kernel at its autotuned parameters. See §5.3.1 for the sub-level structure and Appendix B for the per-family menus. Kernels that demonstrably break Responses-API streaming or determinism are excluded from the action space at selection time, not tuned against. Vendor-library kernels (FA4, FA3, FlashInfer) are L0-select candidates only; L0b is a no-op for them and L0c is not applicable (we do not fork vendor binaries — the escape hatch is a Triton replacement).
6. **Measurement reuses LLD-04.** The auto-research agent does not rewrite the telemetry plane. It consumes vLLM `/metrics` delta-sampling (TTFT, throughput, cache-hit rate, per-turn latency) that LLD-04 already delivers. The new work is the driver (synthetic Codex-thread load generator that mirrors the eval workload) and the optimization loop on top — not the metrics.

---

## 3. Hardware Envelope — Single DGX Spark (GB10 Blackwell)

### 3.1 The machine

One DGX Spark. No cluster, no second box, no remote GPU. The entire serving + auto-research + online-training stack lives on this one device.

**GB10 Grace Blackwell Superchip** (figures committed to NVIDIA-published spec-sheet numbers):

| | |
|---|---|
| Grace CPU | 20 Arm cores (mix of Cortex-X925 performance + Cortex-A725 efficiency) |
| Blackwell GPU | 1 PFLOPS FP4 sparse peak |
| Unified memory | 128 GB LPDDR5x, shared CPU + GPU |
| NVLink-C2C (internal, CPU ↔ GPU) | ~900 GB/s |
| Main memory bandwidth | ~273 GB/s |
| Power envelope | ~170 W typical sustained |

Post-install microbenchmarks to measure real-world deviations from published peaks are a §9 verification item, not a drafting dependency.

### 3.2 Unified memory — weights, KV, activations share one 128 GB pool

Unlike H100 / H200 setups (separate HBM + host DRAM with a PCIe wall in between), GB10 has one coherent memory pool:

- Weight residency, KV cache, activations, and PyTorch workspace all allocate out of the same 128 GB.
- Cross-CPU-GPU transfers via NVLink-C2C (~900 GB/s internal) are effectively free compared to PCIe (~64 GB/s).
- There is no "offload to CPU" design lever worth pulling — it is already the same memory.

This removes several classical optimization patterns — host-side paging, CPU offload of rarely-used layers, pinned-memory staging — from the auto-research agent's action space. It also removes a mistake surface.

### 3.3 Memory-bandwidth wall — the real decode bottleneck

**273 GB/s is the single number that shapes every optimization decision downstream.** For autoregressive decode, each generated token requires reading the full set of *active* parameters from memory exactly once. The ceiling on per-stream decode throughput is:

```
tokens/sec ≤ bandwidth / (active_weight_bytes_per_token + per_stream_state_bytes_read_per_token)
```

At FP8 (1 byte per param) for Qwen 3.5 27B, *every* parameter is active on every token (dense, no MoE routing): `273 GB/s / 27 GB ≈ 10.1 tokens/sec per single stream` — memory-bound, single decode, no batching.

**Hybrid-attention nuance — not every layer contributes per-token KV.** Of the 64 decoder blocks, only the 16 Gated Attention blocks carry a per-token KV cache; the 48 Gated DeltaNet blocks hold an O(N) recurrent state that does not grow with sequence length. This matters for §3.11 KV math and §3.12 concurrency math: the "state read per decode step" term is much smaller than a pure-standard-attention 27B model would have. The linear-attention layers still read their recurrent state each step, but that state size is fixed per sequence regardless of context length, so it does not scale with max context and does not dominate at realistic N.

Batching and paged KV amortize the weight read: N concurrent streams that share weight reads approach `273 / (27 + N × per_stream_state_bytes_per_step)` — single-digit milliseconds amortized per stream at realistic N. In practice KV contention (on the 16 Gated Attention layers), prefill interleaving, recurrent-state updates (on the 48 DeltaNet layers), and decode-vs-prefill scheduling all eat into the ideal. §5.6 (measurement harness) is what distinguishes "the model is bandwidth-bound as expected" from "something else is slowing it down."

### 3.4 Compute envelope

NVIDIA-published peaks (precision and sparsity as noted):

| Precision | Peak | Notes |
|---|---|---|
| FP4 sparse | ~1 PFLOPS | NVIDIA headline number; assumes 2:4 structured sparsity |
| FP4 dense | ~500 TFLOPS | Derived |
| FP8 dense | ~250 TFLOPS | The number that matters for Qwen 27B FP8 serving today |
| FP16 dense | ~125 TFLOPS | Derived |

FP4 peak is the marketing number; dense FP8 at ~250 TFLOPS is the one that matters because `model_registry.yaml` commits to FP8 weights. FP4 serving is a potential future lever the auto-research agent may propose (§5.3), but vLLM's FP4-on-Blackwell kernel maturity is a 2026 moving target — §11.2 tracks this as a risk.

### 3.5 Grace CPU — background, not foreground

20 Arm cores. Relevant for tokenizer prefill, Python host orchestration, Docker management, task dispatching, file I/O. Never the bottleneck for any realistic Codex-Bench workload; treated as a free resource in this HLD. Does not appear in the auto-research action space.

### 3.6 Storage

Two NVMe devices, both empty at spec-write time. Proposed layout (both drives are part of the design; space permitting, flexibility is fine):

| Device | Capacity | Real throughput | Role |
|---|---|---|---|
| Internal NVMe | 1 TB | ~3 GB/s read | **Hot path** — model weights (active + 1 spare slot), vLLM runtime scratch, KV spill (if any), prefix-cache persistence, active campaign's event-store write-ahead region, last-known-good serving config (for §3.8 resume) |
| External USB 3.2 Gen 2 NVMe | 4 TB | ~1 GB/s real (USB 3.2 Gen 2 ceiling) | **Cold path** — raw event store archive, trajectory replay library, SFT / DPO training data, checkpoint archive, LLD-04 telemetry log archive, older LoRA adapters |

**Rationale.** Cold-start weight load for 27 GB FP8 off internal NVMe is ~9 seconds (acceptable). The same off USB at ~1 GB/s is ~27 seconds — fine for a one-time pull but not a steady-state hot-swap path. Bulk training artifacts live on external because that is where capacity sits; the 4:1 external-to-internal ratio matches the bulk-vs-hot ratio of the workload.

**One-line rule.** Bandwidth-sensitive goes on internal; capacity-sensitive goes on external. Staging between the two is a nightly cron, not part of the serving hot path.

### 3.7 Power, thermals, uptime

~170 W typical under sustained load. Machine assumed to be 24/7 live during a campaign — prefix cache accumulates, warm weights stay resident, LLD-04 telemetry logs continuously. This HLD does not prescribe a thermal-throttle mitigation because DGX Spark is designed for sustained operation, but thermal throttling under prolonged FP4 / FP8 load is a §11.4 risk to watch.

### 3.8 Machine-restart resumability

**Hard requirement.** After a machine restart (power cycle, OS reboot, OOM kernel panic), the serving stack must come back to a known-good state via a documented path — not necessarily auto-resume, but a single command or concise runbook the operator can follow.

Minimum guarantees:

- **Last-known-good serving config** (vLLM flags, kernel selection, LoRA adapter set, tuned-config bundle reference) persisted to internal NVMe and re-loadable by one `make resume` (or equivalent CLI) invocation.
- **Event store writes are append-only.** Any in-flight task interrupted by restart produces an `integrity_flag = 1` trajectory row (per the Training Flywheel HLD §3.1) rather than being silently dropped or silently completed.
- **Prefix cache is rebuilt from scratch** on restart — not preserved — because bandwidth cost to warm-reload is lower than the complexity cost of persisting and validating it.

Implementation belongs in the derived LLD; the HLD locks down the contract.

### 3.9 What GB10 removes from the design space

Explicit non-features. Calling them out keeps reviewers from asking "why didn't you consider X":

- No multi-GPU tensor parallel (one GPU).
- No pipeline parallel (one GPU).
- No data parallel across devices (one device).
- No NCCL, no cross-machine NVLink, no InfiniBand, no RDMA.
- No HBM — this is LPDDR5x. Bandwidth ceiling is the LPDDR number, not the HBM number.
- No PCIe-based CPU↔GPU transfer tuning — unified memory makes it moot.
- No second machine to parallelize the auto-research agent's sweeps — sweeps run serially or time-multiplexed on this box.

### 3.10 Weight-footprint table

Back-of-envelope weight residency at FP8, with KV + activations headroom under 128 GB. The Qwen 3.5 27B row is the load-bearing one; the MoE rows are envelope reference only — model alternatives are out of scope per §2 item 2.

| Model | Total params | FP8 weight (LM path, vision tower disabled) | Headroom (KV + activations) | Verdict |
|---|---|---|---|---|
| **Qwen 3.5 27B dense-hybrid (locked)** | **27 B** | **~27 GB** (vision tower adds ~1–2 GB when enabled; disabled here) | **~101 GB** | **Fits very comfortably** |
| Qwen 3.5 35B-A3B MoE | 35 B total, 3 B active | ~35 GB | ~93 GB | Fits (reference only; not in scope) |
| Qwen 3 Coder Next 80B-A3B MoE | 80 B total, 3 B active | ~80 GB | ~48 GB | Fits tight (reference only; not in scope) |
| Qwen 3.5 122B-A10B MoE | 122 B total, 10 B active | ~122 GB | ~6 GB | Does not fit at FP8 (reference only; not in scope) |

Numbers are weight-footprint only; activations + workspace + CUDA graphs + PyTorch overhead consume an additional ~4-8 GB. The vision-tower weights are counted out of the budget because §2 item 2 locks the serving posture to vision-off; if vision is ever re-enabled (v0.2), the tower adds ~1–2 GB to weight residency and a non-trivial activation workspace on image inputs.

### 3.11 KV-cache-budget math (concrete, hybrid-aware)

**Only 16 of 64 layers carry per-token KV cache.** The 48 Gated DeltaNet layers hold a fixed-size recurrent state per sequence (independent of context length); the 16 Gated Attention layers are the only contributors to the per-token KV footprint.

Per-token KV footprint (Gated Attention layers only) ≈ `2 × num_attention_layers × num_kv_heads × head_dim × bytes_per_element`:

```
2 × 16 × 4 × 256 × 1 = 32,768 bytes/token ≈ 32 KB/token   (FP8 KV cache)
```

That is still ~4× smaller per token than a hypothetical pure-standard-attention 27B would be (which would have ~64 layers' worth of per-token KV in the same per-head math), but it is higher than a naïve 12-layer GatedAttn count would suggest. At 131,072 max-context, one fully-filled sequence is ~**4.2 GB** of KV cache alone (131072 × 32 KB). The concurrency envelope and §5 tuning must reflect this 32 KB/token figure, not an older 24 KB/token number derived from the wrong layer count.

**Recurrent-state footprint (Gated DeltaNet layers).** Per sequence, the DeltaNet recurrent state is `48 layers × state_dim_per_layer × bytes_per_element` — a fixed ~hundreds of MB per stream regardless of context length (state is accumulated per DeltaNet head — 48 V heads × head_dim_v=256 per layer, fp16 by default per §11.11). Exact size is a §9.2.1 post-install measurement; the per-stream state is tracked in the KV-budget accounting alongside the attention KV.

**Implication.** With 27B weights resident (~27 GB) + OS / CUDA / activations (~8 GB), ~93 GB remains for KV + DeltaNet state. Because per-token KV is small (32 KB/token), a 131k-context stream costs ~4.2 GB of attention KV alone; concurrency is ceiling'd by the sum of (a) GatedAttn KV at long contexts, (b) DeltaNet recurrent-state aggregate, and (c) activation workspace. Effective concurrency at `max_model_len=32k` realistic campaign settings lands in the ~20–50 concurrent Codex-thread range (the envelope narrows slightly vs the 48-layer draft because per-token KV is 33% larger; still wider than a pure-attention model would allow). The exact number is what the §5 auto-research agent is for, and it is measured, not predicted.

### 3.12 Concurrency-ceiling math (hybrid-aware)

Decode latency per step on GB10 is dominated by memory bandwidth. For a dense-hybrid model serving N concurrent streams, each doing one decode step, the per-step bandwidth consumption is the union of:

```
decode_step_latency ≈ (active_param_bytes + N × attention_KV_read + N × deltanet_state_read) / bandwidth
                    ≈ (27 GB + N × ~16 KB + N × recurrent_state_read) / 273 GB/s
                    ≈ ~99 ms + N × (small, dominated by recurrent-state read)
```

where `attention_KV_read` is "read the active 16-layer KV up to current position" (bounded, typically a small multiple of KB per step under paged attention — far smaller than the 32 KB/token *total* footprint, because paged attention reads only the active block for this step), and `deltanet_state_read` is the per-step recurrent-state read across 48 DeltaNet layers (bounded per sequence, does not grow with context length).

Single-stream decode is ~99 ms per token (~10 tok / s). Adding concurrent streams is **almost free** until one of: (a) DeltaNet-state aggregate pressure starts costing wall-clock, (b) Gated-Attention KV pressure at long contexts (now 32 KB/token across 16 layers), (c) prefill-vs-decode scheduling collides, or (d) kernel launch overhead dominates at the smallest per-step read. That is the central fact the auto-research agent is exploiting: GB10 is very good at concurrent decode for this hybrid-attention 27B, and the optimization target is finding the `N` at which something other than bandwidth starts dominating — which is now *two* candidate bottlenecks (attention KV pressure, DeltaNet state pressure) to distinguish, not one.

### 3.13 Cold-start and weight-loading costs

| Action | Source → target | Approx. wall-clock |
|---|---|---|
| Load 27 GB FP8 weights (Qwen 27B) | Internal NVMe → unified memory | ~9 s |
| Load 27 GB FP8 weights | External NVMe → unified memory | ~27 s |
| Swap LoRA adapter (~200 MB) | Internal → unified memory | < 1 s |
| Full cold restart (CUDA init + vLLM warmup + first-request compile) | — | ~45-60 s for 27B dense |

**Policy.** Cold restart is acceptable as an occasional event (< 1 per hour steady-state). Frequent restarts would dominate campaign wall-clock — which is why §6's weight-update path is "PyTorch writes → KV invalidate → resume," not "cold restart per weight update."

### 3.14 Implications for training

- **QLoRA SFT on Qwen 27B FP8.** Fits cleanly — base weights (~27 GB) + LoRA adapters (< 1 GB) + optimizer state + gradients well within 128 GB. LoRA adapter placement on a hybrid-attention model must target both kernel families: Gated Attention Q/K/V/O and FFN projections, and the DeltaNet linear-attention projections. Adapter placement is a Training Flywheel concern, not a serving concern, but the serving layer must be able to load adapters against both families (§5.3.4).
- **Full SFT on Qwen 27B.** Does not fit — FP16 master weights (~54 GB) + Adam optimizer state (~108 GB) + gradients exceed 128 GB. Blocked without FSDP, which requires a cluster we do not have.
- **DAPO RL on 27B dense-hybrid.** Fits via QLoRA. Policy + reference co-residency would not fit at full FT — the Training Flywheel HLD prescribes reference-free DPO-style losses or offloaded reference. DAPO RL on a dense (non-MoE) model avoids the routing-divergence failure mode that MoE RL suffers — one of the reasons the model-structure lock in §2 item 2 holds dense. Training dynamics under hybrid attention (how the DeltaNet recurrent-state gradients flow under LoRA + DAPO) is a Training Flywheel HLD question; this HLD only needs the serving layer to faithfully reproduce the hybrid architecture at inference.

### 3.15 Summary — GB10 as a memory-bandwidth-bound decode box serving a hybrid-attention dense 27B

One sentence to carry through the rest of this HLD: **DGX Spark is a memory-bandwidth-bound decode box with unified memory, serving a dense-hybrid 27B whose per-token KV is unusually small (16 standard-attention layers out of 64, 32 KB/token at FP8) and whose DeltaNet recurrent-state pressure is a second optimization target alongside attention KV.**

That framing drives §4 (the serving-backend contract makes bandwidth-saturation the throughput target, with a three-dimensional latency SLO because TTFT and TPOT behave very differently under hybrid attention), §5 (the auto-research agent is searching for the config where bandwidth is the only remaining bottleneck — not kernel launch overhead, not request admission, not attention-KV pressure, not DeltaNet-state pressure, not determinism-check overhead — *and* is tuning two kernel families, not one), and §6 (weight-update cost is measured in bandwidth cycles, not re-sharding complexity).

---

## 4. Serving Backend Contract

What the serving stack must deliver, independent of how the §5 auto-research agent happens to have tuned it. These are the invariants downstream LLDs code against.

### 4.1 Concurrency, throughput, and the three-dimensional latency SLO

- **Sustained concurrency.** The serving layer must support `N_concurrent` simultaneous Codex threads, where `N_concurrent` is the ceiling discovered by the §5 auto-research agent for the `(Qwen 3.5 27B, family, eval-workload)` tuple. Typical expected range per §3.11 / §3.12 math on hybrid attention: **20–60 concurrent threads** at realistic context lengths.
- **Throughput target.** Saturate the 273 GB/s memory-bandwidth ceiling (§3.3). The operational definition: sustained aggregate decode throughput across all active streams reaches ≥ 80% of the bandwidth-bound theoretical ceiling for the observed active-param byte count. Anything below 80% means something non-bandwidth is dominating; §5 treats that as a signal to keep searching.
- **Three-dimensional latency SLO (all three hard constraints).** A single per-turn latency ceiling is not enough on an agentic-coding workload with thinking enabled. The serving contract binds three p95 latency metrics independently, each sourced from the campaign's family spec:

  | Metric | Definition | Default family-spec ceiling (`L_*`) |
  |---|---|---|
  | **`TTFT_p95`** — time-to-first-token | Wall-clock from request admission to first `content`/`reasoning_content` token emitted. **Thinking tokens do not defer TTFT** — the first thinking token counts. | `L_ttft = 2 s` |
  | **`TPOT_p95`** — time-per-output-token (inter-token latency during steady-state decode) | Median-or-p95 gap between consecutive emitted tokens after warm-up, averaged across `content` and `reasoning_content`. | `L_tpot = 80 ms` (~12.5 tok/s per stream) |
  | **`TurnLatency_p95`** — per-turn end-to-end | Wall-clock from admission to final token of the assistant turn (thinking + response combined). | `L_turn = 30 s` at the family-spec `max_output_tokens` |

  All three must hold simultaneously for a configuration to be feasible. The §5.4 objective is `maximize N_concurrent` subject to all three hard constraints (plus the other §5.4 constraints). A configuration that hits `L_turn` by saving TTFT but blowing past `TPOT` is rejected, and vice versa.

- **Admission control.** When the active stream count would push any of the three latency metrics past its ceiling at the next request's requested context length + thinking budget, the request is queued (not dropped) and a queue-depth metric is emitted. §5's measurement harness uses queue depth as a primary signal. Admission is computed against all three ceilings, not just turn latency.

- **Vision tower disabled.** vLLM is launched with `--limit-mm-per-prompt image=0` (or the equivalent config-level flag). Image inputs in a `/v1/responses` request return a structured 4xx before any GPU work is issued. Re-enabling the vision tower is a v0.2 scope change (§11.12).

- **Thinking-mode contract (load-bearing). Request-level override is closed for campaign traffic.** Every `/v1/responses` request accepted by this serving layer runs with:
  - `chat_template_kwargs.enable_thinking = true` (server default, honored — see override policy below).
  - `extra_body.thinking_token_budget = family_spec.thinking_budget` (default 8192 reasoning tokens per turn — server default, honored — see override policy below).
  - `--reasoning-parser qwen3` enabled at vLLM launch so thinking content surfaces under `reasoning_content`, never leaking into `content`.

  Thinking tokens consume the same KV + DeltaNet state slots as response tokens, contribute to TPOT and TurnLatency (but not to TTFT — the first thinking token *is* the first token), and are tagged in telemetry with the LLD-SB-02 counters `tokens_reasoning_total` and `tokens_response_total` (see §4.5). The auto-research agent tunes `thinking_token_budget` as an L1 knob (§5.3.2) but never turns thinking off — that would change the model's behavior beyond what the family spec signed off on.

  **Override-rejection policy (the load-bearing part).** Per current vLLM reasoning-mode docs, request-level `chat_template_kwargs` override server defaults by default. That is not acceptable for this HLD: a client that flips `enable_thinking=false` or inflates `thinking_token_budget` at request time silently leaves the subset of traffic the tuned bundle's three-dim SLO actually covers, and the tuned `N_concurrent` ceiling stops being a guarantee. Therefore, for requests tagged `class=eval` or `class=rollout` (i.e., campaign traffic — see §4.3), the admission layer sitting in front of vLLM **rejects** any request that sets `chat_template_kwargs.enable_thinking` or `extra_body.thinking_token_budget` (regardless of value) with a structured 4xx (`error.code = thinking_override_forbidden`, `error.field` naming the offending field). Rejection happens before any GPU work is issued. Non-campaign classes (developer debug, ad-hoc smoke tests) are out of scope for this HLD; if a second class is ever introduced that needs override, it lives behind a separate admission path. The serving posture's thinking defaults — `enable_thinking=true`, `thinking_token_budget=<family spec>` — are part of the bundle-validity rule (§5.9 `serving_posture_pin`, §6.4) and cannot be changed without a re-tune.

### 4.2 Weight-update semantics — the trivial contract

Full detail in §6; summary here.

```
training process → writes new weights to /models/active/next/weights.pt
training process → atomic rename: /models/active/next → /models/active/current
serving layer    → inotify / polling detects rename, sends invalidate
serving layer    → drains or terminates in-flight requests (per §4.4)
serving layer    → flushes KV cache and prefix cache
serving layer    → reloads weights from /models/active/current
serving layer    → resumes accepting requests
```

Contract guarantees:

- **Atomicity.** Weight update is all-or-nothing, protected by the rename. Partial-write corruption is impossible.
- **No hot-swap-with-preserved-KV.** Old KV is unconditionally invalidated; there is no attempt to preserve in-flight decode state across a weight change. The design admits this makes a weight update briefly expensive (cold-path reload, ~9s internal NVMe per §3.13 — actual hot-reload is cheaper because weights are already mmapped).
- **Version tagging.** Every completed response carries a `weight_version_id` (git-short-SHA or monotonic counter) in its telemetry row. Eval reproducibility depends on this.
- **Back-pressure, not silent drop.** Requests received while a weight update is in progress are queued and replayed on resume, subject to three-dim-SLO-aware admission control (§4.1).

### 4.3 Eval-and-train-on-the-fly contract

Two distinct request classes share the serving process:

| Class | Source | Treatment |
|---|---|---|
| **Eval** | Benchmark runner + Codex threads executing family tasks | Higher priority. Latency-sensitive. Counts against `N_concurrent` ceiling. Tagged `class=eval`, `family_id`, `variant`. |
| **Rollout** | Online-RL rollout generator | Lower priority. Throughput-sensitive. May be preempted or queued when eval concurrency is high. Tagged `class=rollout`. |

Both classes hit the **same** `/v1/responses` endpoint and share weights + KV cache. The priority separation is an admission-control layer sitting in front of vLLM, not a second server. Reproducibility: every response is tagged with `weight_version_id`, `tuned_config_id`, `class`, `thinking_token_budget`, the `tokens_reasoning` / `tokens_response` counts, and the LLD-04 telemetry row ID. The auto-research agent measures both classes but optimizes primarily against the eval-class three-dim latency SLO (§4.1, §5.4), with a soft constraint that rollout throughput does not collapse.

### 4.4 Request-lifecycle and interrupt semantics

- **In-flight request on weight update.** Default: **finish-at-current-weights**. Rationale: partial responses tagged with a stale `weight_version_id` are simpler than mid-decode replay. Eval runs that need weight-consistent responses must issue all requests after a weight-update barrier.
- **In-flight request on server restart.** Terminated with an `integrity_flag = 1` trajectory row (per Training Flywheel HLD §3.1) and retry semantics handled by the caller. The serving layer does not implement in-flight request durability.
- **Request cancellation.** Clients may cancel in-flight requests; cancellation frees the KV slot eagerly. The measurement harness verifies cancellation does not leak KV.

### 4.5 Integration surface

The serving stack presents exactly one HTTP surface and one control plane:

| Surface | Endpoint | Consumer |
|---|---|---|
| Request plane | `POST /v1/responses` (OpenAI-compatible Responses API, native Qwen 3.5; thinking content surfaces under `reasoning_content` via `--reasoning-parser qwen3`) | Codex threads via benchmark runner; online-RL rollout generator |
| Request plane | `GET /v1/models` | Health + model introspection |
| Telemetry plane | `GET /metrics` (Prometheus-format, per LLD-04). Exposes vLLM-native histograms and counters, **not** pre-aggregated p95 series. The three latency dimensions from §4.1 land as vLLM histograms: `vllm:time_to_first_token_seconds` (TTFT), `vllm:inter_token_latency_seconds` (TPOT), `vllm:e2e_request_latency_seconds` (per-turn / end-to-end). p95s are derived client-side (in the harness, the campaign dashboards, or downstream alerting) via PromQL `histogram_quantile(0.95, sum by (le, instance) (rate(vllm_<metric>_seconds_bucket[5m])))`. Additional counters added by LLD-SB-02 — `tokens_reasoning_total` and `tokens_response_total` (labeled by `class`, `family_id`, `weight_version_id`, `tuned_config_id`) — are *not* native vLLM metrics; they are emitted by the LLD-SB-02 telemetry plane layered on top of `/metrics`. Queue depth, active-sequence count, prefix-cache hit rate, and aggregate throughput are exposed by vLLM's own metrics directly. | Auto-research measurement harness; campaign dashboards |
| Control plane | `POST /admin/invalidate` | Training process (invokes weight-update flow) |
| Control plane | `POST /admin/load_tuned_config` | Campaign bootstrap (loads the frozen tuned-config bundle from §5.9; bundle-validity check per §6.4 runs here) |

Launch-time invariants enforced at process boot (failing any of these aborts the launch rather than degrading silently):
- Vision tower disabled: `--limit-mm-per-prompt image=0`.
- Reasoning parser active: `--reasoning-parser qwen3`.
- vLLM version matches the pin recorded in the tuned-config bundle or, at first boot before any bundle is loaded, matches the HLD-v0.1 pin (`0.19.x`). On SM100+ (Blackwell) the `attention_backend` is NOT left to vLLM's auto-selection (which resolves to FlashInfer first, then FlashAttention) — the tuned bundle hard-pins an explicit backend per §5.9 / §6.4, and the first-boot default before any bundle is loaded is whatever vLLM auto-selects with accuracy screening left to L0-GatedAttn-a (§5.3.1.a).

Code-level mapping to the existing `src/lumo_flywheel_serving/` tree — `model_server.py` owns the request plane, `metrics.py` owns the telemetry plane, new admin endpoints land in `inference_proxy.py` — is detailed in §10's migration plan.

---

## 5. Auto-Research Agent — Design

The core of this HLD. Karpathy-style closed-loop research agent: propose → measure → iterate → freeze. Operates offline, pre-campaign, once per `(model, family, eval-workload)` tuple, and produces a frozen tuned-config bundle that the campaign's serving stack consumes unchanged.

### 5.1 When the agent runs

**One-time per `(model, family)` pair.** Take a family that already has its eval set defined, take the model checkpoint, run the agent once, produce the tuned-config bundle, done. Every subsequent campaign against that `(model, family)` pair consumes the same frozen bundle unchanged — no re-run, no continuous tuning, no per-campaign kickoff step. This is deliberately not a recurring process; treating it as one-off is what lets §5.7's safety rails and §5.8's stopping criteria be simple and strict.

Re-running is out of scope for v0.1 of this HLD. If the model, family, or serving stack ever changes in a way that plausibly invalidates the bundle (e.g., model-structure lock in §2 item 2 is reopened, or the vLLM version jumps across a major release), that is a v0.2 problem, not a v0.1 trigger. §11 tracks the open-question framing.

- **Duration.** Bounded by §5.8 stopping criterion — a wall-clock cap (default: 4 hours) and a diminishing-returns threshold. A single overnight slot is the expected envelope.
- **Isolation.** Runs on the same DGX Spark as production serving, but in a dedicated tuning slot — no eval, no RL traffic in parallel. Cold-start + per-iteration reset costs are budgeted in §5.7.

### 5.2 Inputs and outputs

**Inputs** (all frozen for the duration of the run):

- **Model checkpoint.** Qwen 3.5 27B FP8 weights at a specific `weight_version_id`. Hybrid-attention shape (16 × (3 DeltaNet + 1 Gated Attention) = 64 decoder blocks per the public Qwen3.5-27B-FP8 model card) is a hard-coded architectural fact, not a tunable.
- **Serving posture.** Vision tower disabled (`--limit-mm-per-prompt image=0`), reasoning parser `qwen3`, thinking-on by default. These are invariants the agent does not explore — they define the search space's walls, not its interior.
- **Family spec.** The benchmark family (or training-family proxy) the tuned bundle is for. Defines `L_ttft`, `L_tpot`, `L_turn` (the three-dim latency SLO per §4.1), the `thinking_token_budget` for this family (default 8192, family may override), expected context-length distribution, tool-call pattern, and turn-count distribution.
- **Eval-workload distribution.** Empirical distribution of request shapes (context length, output length, thinking-token usage, tool-call frequency) derived from a seed run of the family's eval set through the default-config serving stack. Captured once at the start of the agent's one-time run; not re-derived afterward.
- **Hardware profile.** The GB10 envelope from §3 is hard-coded; the agent does not rediscover hardware limits.
- **Baseline config.** The default, untuned vLLM 0.19.x launch configuration with the above serving posture — i.e., vLLM's out-of-the-box defaults for Qwen 3.5 27B FP8 hybrid-attention (which on SM100+ means vLLM's own auto-select chain — FlashInfer with TRTLLM first, FlashAttention with FA4 as the default FA version second — *not* a hard FA4 selection), no kernel overrides, and no request shaping. This is what the regression guard (§5.7) compares against. There is no prior tuned-config bundle to inherit from, because this is a one-time run per §5.1. The baseline's `vllm-default`-resolved attention backend is captured in the run trace so the §9.3.13 acceptance check can confirm the bundle's explicit pick is non-default.
- **Environment pin template.** The exact vLLM version, Triton version, CUTLASS version, Optuna version, and kernel-source SHAs the run will record in the output bundle — these are captured up-front from the live environment and become part of the bundle's identity (§5.9, §6.4).

**Outputs:**

- **Tuned-config bundle** (§5.9) — a versioned YAML artifact containing the selected vLLM flags, kernel choices, request-shaping parameters, LoRA-management policy, the objective value achieved, and pointers to the measurement traces that justify it.
- **Measurement traces** — raw LLD-04 `/metrics` deltas per iteration, stored alongside the bundle for post-hoc analysis.
- **Search trace** — the sequence of proposed configurations and their measured objective values, retained for audit and for any future v0.2 re-run that wants a hot start.

### 5.3 Action space

The action space is deliberately small, strictly typed, and **explored one layer at a time**. Every dimension has a reason-to-tune documented here; every dimension that isn't listed is excluded from the agent's search.

**Why layered and not joint.** A single loop that tunes all four dimensions at once has to discover cross-layer interactions from scratch (e.g., "FlashInfer pairs well with smaller `max_num_batched_tokens`") and wastes evaluations on combinations that are nonsensical before the lower layers are fixed. A layered traversal matches the dependency structure of the stack — kernels sit below vLLM config, vLLM config bounds what request shaping can enforce, request shaping is a policy layer that LoRA composes on top of — and reduces each layer's search space to something small enough to reason about. The trade is that coordinate descent can miss a cross-layer optimum; §5.5 covers the bounded re-open escape hatch that keeps that risk small.

**Why L0 is asymmetric.** L1/L2/L3 are knob-space — the action is picking values from a fixed config surface, and the optimizer's job is interpolating / searching over that surface efficiently. L0 is not knob-space only. On fresh silicon (GB10, a Blackwell variant with its own SM layout and FP8 path), the headroom is frequently inside the kernel implementation itself: block sizes, pipeline depth, fusion boundaries, even whether a custom Triton kernel replaces a stock op. L0 therefore has three sub-levels (selection → parameter-tune → mutation) rather than one flat action space — and the "research" framing of this HLD is load-bearing at the mutation sub-level, which is where an LLM-in-the-loop agent most clearly earns its name.

**Why L0 is split across two kernel families.** Qwen 3.5 27B's hybrid attention stack means L0 is tuning *two independent kernel pipelines* — Gated DeltaNet (linear attention, **48 layers**) and Gated Attention (standard attention, **16 layers**). These have disjoint kernel menus (FA4 / FlashInfer apply only to Gated Attention; chunked-delta / state-update kernels apply only to DeltaNet), disjoint autotune spaces, and disjoint mutation candidates. L0 runs the three sub-levels independently per family and concatenates the winners; the parity harness runs per-family against the same-family reference. §5.3.1 details the per-family structure and the parity gate that makes mutation safe enough to include.

**Layer stack** (bottom to top; traversal order L0 → L1 → L2 → L3):

```
L3  LoRA adapter management        ← highest layer; composes on top of everything
L2  Request shaping                ← policy layer over vLLM
L1  vLLM config                    ← interpreter; bounded by the L0 kernels beneath it
L0  Kernel selection (two families) ← bedrock; determines what L1 knobs are meaningful
    ├── L0-DeltaNet  (linear-attention kernels, 48 layers)
    └── L0-GatedAttn (standard-attention kernels, 16 layers)
```

Each layer is tuned with all lower layers **frozen** at their winning values. Each layer outputs the frozen choices plus a per-layer measurement trace that is retained in the bundle (§5.9) and feeds into the next layer's proposer.

#### 5.3.1 Layer 0 — Kernel tuning (two families × three sub-levels)

Tuned first. L0 has **two families** (Gated DeltaNet, Gated Attention) that are tuned independently, each through the same three sub-levels (selection → parameter-tune → mutation). The families share no kernel candidates, share no autotune spaces, and share no mutation references; the only thing they share is the overall traversal order (DeltaNet first, Gated Attention second — the DeltaNet path is the less mature kernel menu on Blackwell and benefits more from being tuned before the standard-attention path locks in) and the aggregated per-family winners feed L1 together.

At this layer vLLM runs with default config, shaping disabled, no LoRA, thinking-mode on (to exercise the realistic reasoning-token load), vision tower disabled. The measurement signal is pure kernel cost per family: decode step latency attributable to that family's layers, prefill throughput, determinism under fixed seed, and — for the mutation sub-level — numerical parity against that family's pinned reference kernel (§5.7 rail 9).

**Per-family structure.** Within each family F ∈ {DeltaNet, GatedAttn}, the agent runs L0-F-select → L0-F-autotune → L0-F-mutate. Appendix B lists per-family menus and per-family parameter-tune / mutation surfaces.

##### 5.3.1.a L0-select — Pick a kernel from each family's Appendix B menu

Two discrete search spaces, one per family. Kernel choice sets the ceiling for everything above.

The agent may pick any combination that Appendix B marks **supported** for that family; anything marked *experimental* or *blocked* is out of the action space for v0.1. Kernels that have demonstrably broken Responses-API streaming or determinism are removed from Appendix B, not left in the action space with a penalty.

| Family | Knob | Type | Options |
|---|---|---|---|
| **DeltaNet** | `deltanet_kernel` | enum | {triton-chunked-delta, triton-state-update-fused, triton-fallback, vllm-default-deltanet} |
| **GatedAttn** | `attention_backend` | enum | {flash-attn-4, flash-attn-3, flashinfer, triton, vllm-default} |
| (cross-family) | `torch_compile_mode` | enum | {off, default, reduce-overhead, max-autotune} |
| (cross-family) | `cuda_graph_capture` | enum | {off, decode-only, full} |
| (cross-family) | `fp8_gemm_kernel` | enum | {cutlass-default, cutlass-warpspecialized} |

The three "cross-family" knobs (torch-compile, CUDA graphs, FP8 GEMM) are single-valued across both families — we do not capture two different graph modes. If an empirical trade-off emerges that wants DeltaNet and Gated Attention to run under different compile / graph settings, that is a §11 open question, not a v0.1 action.

**Measure.** Per-family decode latency × prefill throughput × determinism-pass, Pareto-best. Overall L0-select winner = (DeltaNet winner, GatedAttn winner, shared compile/graph/GEMM winner).

**Output.** A two-tuple of kernel selections (one per family) plus the three cross-family knobs, frozen for L0-autotune and downstream.

##### 5.3.1.b L0-autotune — Parameter-tune each selected kernel

With each family's L0-select winner frozen, autotune its internal parameters (block sizes, tiling, pipeline depth, warp specialization). The inner optimizer is the kernel ecosystem's native tuner:

- **For Triton-based kernels** (both families have Triton candidates — DeltaNet's chunked-delta and state-update kernels; Gated Attention's `triton` backend and fused RMSNorm+QKV; any custom Triton ops L0c introduces): `@triton.autotune` with a per-kernel search space of `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, `num_stages` (plus `CHUNK_SIZE` for DeltaNet chunked-delta). We author the search space per kernel; Triton drives the search.
- **For CUTLASS FP8 GEMM** (`cutlass-default`, `cutlass-warpspecialized`, shared across both families): CUTLASS Python tuner over tile shapes, mainloop pipeline depth, and epilogue schedule for the `(M, N, K)` shapes Qwen 27B actually issues (prefill-heavy and decode-heavy, separately).
- **For vendor-library kernels** (FlashAttention 4, FlashAttention 3, FlashInfer — all GatedAttn-family): no autotune — the library's own heuristics own the parameters. L0-autotune is a no-op for these and completes in one iteration.

| Family / knob (per selected kernel) | Driven by |
|---|---|
| DeltaNet Triton `BLOCK_*`, `CHUNK_SIZE`, `num_warps`, `num_stages` | `@triton.autotune` |
| GatedAttn Triton `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, `num_stages` | `@triton.autotune` |
| CUTLASS tile shape, pipeline depth, epilogue schedule (cross-family) | CUTLASS Python tuner |
| Vendor-library internal params (FA4 / FA3 / FlashInfer) | Not tuned (library heuristics own it) |

**Why this sub-level is safe.** The op semantics are fixed — we are moving tile shapes under the same kernel contract. Numerical output is bit-exact (or tolerance-bounded by the library) across parameter choices. No parity gate required at this sub-level beyond the existing §5.7 determinism check.

**Measure.** Same metrics as L0-select, re-measured under the autotuned kernel, per family.

**Output.** An autotune-config per selected kernel (a dict of the winning tile / warp / stage / chunk values), serialized alongside the kernel choice in the bundle. The DeltaNet winner at its autotuned params and the GatedAttn winner at its autotuned params are **both** committed as the per-family reference kernels for the L0c parity harness.

##### 5.3.1.c L0-mutate — LLM edits or authors a kernel (per family)

This is the research sub-level, and the reason this HLD uses the "auto-research" framing rather than "auto-tune." With each family's L0-select + L0-autotune winner frozen as *that family's* reference, the LLM proposer may, independently per family:

- **Edit an existing Triton kernel.** DeltaNet family: re-lay out the chunk-wise delta-rule reduction, fuse the gating activation into the state update, change how the recurrent-state matmul is tiled across warps. GatedAttn family: change a fusion boundary (e.g., fuse RMSNorm into the attention QKV projection), add a software-pipeline stage the current kernel does not exploit, rework the RoPE application.
- **Author a new Triton kernel.** Propose a custom fused op the stock kernels do not offer (e.g., DeltaNet family: a fused gate-and-state-update kernel; GatedAttn family: a fused SwiGLU + down-proj, a fused KV-append + attention step tailored to GB10's SM count).
- **Propose a CUTLASS kernel variant** (shared FP8 GEMM). Parameterize a CUTLASS template with a schedule the shipped variants do not offer; a shared-GEMM mutation is scored against both families' throughput.

Every mutation is a diff against a committed reference kernel (same-family), tagged with the family it targets, attached to the per-iteration measurement trace, and frozen into the bundle as patch + build command (not as a live-patched binary). A shared-GEMM mutation is tagged `family: shared` and scored on the union objective.

**The parity gate is load-bearing, and runs per-family.** Before any mutation's latency numbers count, the mutated kernel must pass a numerical-parity harness (§5.7 rail 9): run both the reference kernel (same family, same autotune params) and the mutated kernel over a pinned set of held-out token inputs, compare outputs element-wise with a documented tolerance (`rtol=1e-3, atol=1e-3` for FP8 forward passes as a default; DeltaNet state-update outputs are checked in both the output-logit space *and* the recurrent-state space at end-of-chunk to catch state corruption that would otherwise manifest only many tokens later). A mutation that fails parity is rejected from the measurement loop entirely — it does not count as an iteration, does not enter the bundle, and is logged for the search trace. Without this gate, the agent will eventually propose a faster-but-wrong kernel and poison every eval and rollout downstream — the single most important reason this sub-level exists as a separately-gated stage rather than merged into L0-select. DeltaNet mutations in particular carry higher parity risk because recurrent state accumulates errors over the sequence; the DeltaNet parity fixture runs probes at both the 1st and the ~1024th output-token position to catch slow-drift bugs.

**Weight-sensitive mutation flag.** Some kernel mutations are numerically stable at the *reference* weight values but may drift at a later `weight_version_id` (e.g., if a mutation's tolerance margin narrows at a new weight scale). A mutation that the proposer suspects may be weight-sensitive is annotated `weight_sensitive: true` in the bundle; the bundle-validity rule (§6.4) uses this flag to decide whether a new `weight_version_id` under the same pinned environment requires a re-parity-check or just a metadata update.

**Measure.** Same metrics as L0-select/autotune — decode latency (family-attributable), prefill throughput, determinism. A mutation is promoted to that family's L0 winner only if it beats the autotuned selection on the §5.4 objective *and* passes parity.

**Acceptable outcome: no mutation wins, per family.** For many `(model, hardware)` combinations the autotuned stock kernels are already close to the bandwidth ceiling and no mutation beats them. L0-F-mutate is allowed to complete with no change to family F's winner; the bundle simply records "mutation search ran for family F, no improvement." This is a success, not a failure — and it is allowed independently per family (DeltaNet might yield a winning mutation while GatedAttn stays at stock, or vice versa).

**Output.** Per family, either (a) a `null` mutation entry (no-improvement case) or (b) a mutation diff + build command + parity attestation + optional `weight_sensitive` flag. Frozen for L1 through L3.

#### 5.3.2 Layer 1 — vLLM config (including thinking-mode knobs)

Tuned second, with L0 (both families) frozen. This is the layer with the most continuous knobs; it is where Bayesian optimization earns its keep.

| Knob | Type | Range | Why it matters |
|---|---|---|---|
| `max_num_seqs` | int | 4 … 96 | The N in §3.12 concurrency math. Directly sets the concurrent-stream ceiling. Upper range widened vs the pure-attention envelope because hybrid attention has cheaper per-token KV. |
| `max_num_batched_tokens` | int | 512 … 16384 | Per-step batching budget; controls prefill / decode interleaving. |
| `enable_chunked_prefill` | bool | {false, true} | Chunked prefill smooths tail latency under prefill-heavy traffic; interacts non-trivially with DeltaNet chunk boundaries (L0-DeltaNet `CHUNK_SIZE` autotune). |
| `enable_prefix_caching` | bool | {false, true} | With long shared prompts (family-scoped system messages), near-always-true — but verified empirically per family. Prefix caching covers both the GatedAttn KV *and* the DeltaNet recurrent state at the shared-prefix boundary. |
| `gpu_memory_utilization` | float | 0.70 … 0.95 | Trades KV + DeltaNet-state headroom for concurrency. Upper bound clamped to leave §6 weight-update headroom. |
| `max_model_len` | int | 8192 … 131072 | Clamped to the family's 99th-percentile context length plus margin, plus the reasoning-token budget headroom. |
| `kv_cache_dtype` | enum | {fp8_e5m2} | Locked to fp8_e5m2 per current `model_registry.yaml`; in the action space for future FP4 exploration (§11.2). Applies only to the 16 Gated Attention layers; DeltaNet recurrent state lives in its own dtype (fp16 on Blackwell today, fp8 is a §11.11 open question). |
| `thinking_token_budget` | int | 512 … 16384 | Per-turn reasoning-token cap exposed via `extra_body.thinking_token_budget`. Default 8192 from family spec. Larger budgets improve agentic-task quality but eat into TPOT headroom and TurnLatency ceilings. Agent tunes per family subject to the §4.1 three-dim SLO. |
| `thinking_parallel_streams` | bool | {false, true} | Whether to allow the reasoning phase to run under a higher per-sequence concurrency than the response phase. Default `false` (reasoning and response use the same batcher slot); `true` is a power-user knob tied to future vLLM features and is a §11 open question if not supported in the pinned vLLM version. |

**What L1 measures.** With the frozen L0 kernels underneath, measure all three latency dimensions (TTFT_p95, TPOT_p95, TurnLatency_p95), sustained concurrency, aggregate decode throughput, and reasoning-vs-response token mix under the synthetic driver at the family's workload distribution. Saturate the 273 GB/s bandwidth ceiling if possible; stop pushing when any of the three latency ceilings is violated.

**Output of L1.** A single vLLM-config tuple (including thinking-budget) + a Layer-1 measurement trace. Frozen for L2 and L3.

#### 5.3.3 Layer 2 — Request shaping

Tuned third, with L0 + L1 frozen. This is a policy layer — small discrete action space, tuned by policy comparison rather than Bayesian opt.

| Knob | Type | Range | Why it matters |
|---|---|---|---|
| `concurrency_cap_eval` | int | 1 … `max_num_seqs` | Hard ceiling on concurrent eval-class streams. Decoupled from vLLM's internal batcher so the admission layer can enforce family-specific bounds. |
| `concurrency_cap_rollout` | int | 0 … `max_num_seqs` | Ceiling on rollout-class streams. `eval + rollout ≤ max_num_seqs`. |
| `admission_queue_depth_max` | int | 0 … 512 | Beyond this, requests reject-with-retry rather than queue. |
| `per_request_kv_budget` | int (tokens) | `max_model_len / 4` … `max_model_len` | Per-request KV pre-reservation cap. Prevents one long-context request from starving N short-context requests. |
| `priority_preemption` | enum | {off, rollout-preempts, strict} | How aggressive the admission layer is about preempting rollout-class for eval-class. |

**What L2 measures.** With kernel + vLLM config frozen, drive mixed eval-and-rollout traffic and measure: (a) eval-class p95 under contention, (b) rollout-class throughput floor, (c) queue depth stability.

**Output of L2.** A single request-shaping policy + a Layer-2 measurement trace. Frozen for L3.

#### 5.3.4 Layer 3 — LoRA adapter management

Tuned last, with all lower layers frozen. Only meaningful when the campaign actually uses adapters — otherwise L3 picks the trivial "no adapters" policy and completes in one iteration.

| Knob | Type | Options | Why it matters |
|---|---|---|---|
| `adapter_mode` | enum | {static-merged, runtime-apply, hot-swap} | Merged adapters are faster to serve but expensive to rotate; runtime-apply is the inverse; hot-swap is a middle ground. |
| `max_loaded_adapters` | int | 0 … 8 | Resident adapter count. Each adapter adds ~200 MB (§3.13). |
| `adapter_eviction_policy` | enum | {lru, manual, pinned-set} | Which adapter leaves when `max_loaded_adapters` is hit. |
| `adapter_merge_on_freeze` | bool | {false, true} | Whether a "final" adapter should be merged into the base weights at campaign freeze, collapsing adapter cost to zero on subsequent eval runs. |

**What L3 measures.** With the full lower-layer stack frozen, measure per-adapter swap latency, adapter-aware cache-hit rate, and the concurrency impact of holding N adapters resident.

**Output of L3.** A LoRA policy + a Layer-3 measurement trace. Combined with the L0–L2 output, this is the final tuned-config bundle.

### 5.4 Objective function

**Primary objective:** maximize `sustained_concurrent_eval_threads` over a measurement window `W` (default: 30 minutes of synthetic campaign traffic), subject to the three-dimensional latency SLO from §4.1 holding simultaneously.

**Subject to hard constraints (all must hold; the three latency constraints are a conjunction, not a disjunction):**

1. `TTFT_p95_eval ≤ L_ttft` from the family spec (default 2 s).
2. `TPOT_p95_eval ≤ L_tpot` from the family spec (default 80 ms).
3. `TurnLatency_p95_eval ≤ L_turn` from the family spec (default 30 s at family's `max_output_tokens` with `thinking_token_budget` applied).
4. `scorer_determinism_check_pass_rate ≥ 99.9%` across the measurement window (§5.7 details).
5. `rollout_class_throughput ≥ 0.5 × rollout_baseline` — a floor to prevent the optimizer from eliminating rollout traffic.
6. `no_oom_events` in the measurement window.
7. `weight_update_invalidate_to_ready_time ≤ 15 s` — exercised at least twice during the window.
8. `reasoning_content_purity` = 100% — every token emitted as `reasoning_content` was produced during the thinking phase; no thinking tokens leaked into `content` (Qwen3 reasoning parser invariant).

**Tie-breakers** (when multiple configs meet all constraints at the same concurrency level): lower TTFT_p95 → lower TPOT_p95 → lower TurnLatency_p99 → higher prefix-cache hit rate → lower rollout-throughput impact.

The objective is deliberately *not* "maximize throughput." Throughput with no latency bound collapses to "batch everything, return slowly," which is wrong for an interactive Codex-thread workload. The three-dim latency conjunction is deliberate: on a thinking-on hybrid-attention model, TTFT is dominated by the prefill path through all 64 layers (48 DeltaNet + 16 GatedAttn), TPOT is dominated by the decode path through the same 64 layers but at very different arithmetic intensity (DeltaNet reads recurrent state; GatedAttn reads KV), and TurnLatency is their composition plus the reasoning budget. Collapsing to a single `L_ceiling` would let the optimizer trade TTFT for TPOT (or vice versa) in ways the family spec does not endorse.

### 5.5 Search strategy

Two nested loops — but unlike a flat joint optimization, the outer loop walks the **layer stack**, not a region index.

**Outer loop — layer traversal.** The agent advances L0 → L1 → L2 → L3 in order. Each layer runs to convergence before the next begins. After a layer converges, an LLM-in-the-loop proposer reviews the per-layer measurement trace against the prior layers' traces and makes one of two calls:

  (a) **Advance.** Freeze this layer's winning values, hand them to the next layer, proceed.
  (b) **Re-open exactly one prior layer.** If the LLM proposer identifies a cross-layer interaction the current layer cannot resolve without changing a lower layer's choice (e.g., "L1 concurrency ceiling is capped at 16 by the kernel choice L0 picked; a different kernel would let L1 reach 28"), it may re-open one prior layer, re-tune it, and re-traverse forward. **Each prior layer may be re-opened at most once per run**, bounded by a safety rail in §5.7. In practice re-open fires rarely; its purpose is to prevent coordinate descent from getting silently stuck, not to become a routine crutch.

**Inner loop — per-layer optimization method.** The method is matched to the layer's action-space shape. L0 has two families (DeltaNet, GatedAttn) × three sub-levels each — six distinct inner-loop invocations — because their action-space shapes differ sharply. Sub-levels within a family traverse in order; families themselves traverse DeltaNet first, then GatedAttn.

| Layer / family / sub-level | Action-space shape | Method | Budget |
|---|---|---|---|
| **L0-DeltaNet-a select** | Small discrete (~20 combinations across `deltanet_kernel` × compile × graph × FP8 GEMM) | Grid-sweep over Appendix-B-feasible cells, LLM-proposed order | Up to 8 evaluations |
| **L0-DeltaNet-b autotune** | Triton `BLOCK_*`, `CHUNK_SIZE`, `num_warps`, `num_stages` for the selected DeltaNet kernel | `@triton.autotune` — native tuner owns the inner loop | Up to 6 evaluations |
| **L0-DeltaNet-c mutate** | Open-ended: LLM-proposed DeltaNet kernel diffs, each must pass §5.7 rail 9 parity gate (output + recurrent-state) | LLM proposer + parity gate + autotune-on-acceptance | Up to 12 evaluations |
| **L0-GatedAttn-a select** | Small discrete (~15 combinations across `attention_backend` × compile × graph × FP8 GEMM; compile/graph/GEMM shared with the DeltaNet winner) | Grid-sweep over Appendix-B-feasible cells, LLM-proposed order | Up to 8 evaluations |
| **L0-GatedAttn-b autotune** | Triton `BLOCK_M/N/K`, `num_warps`, `num_stages` for the selected GatedAttn kernel; CUTLASS tile/pipeline/epilogue (shared); no-op for vendor libraries | `@triton.autotune` / CUTLASS Python tuner — native tuner owns the inner loop | Up to 6 evaluations |
| **L0-GatedAttn-c mutate** | Open-ended: LLM-proposed GatedAttn kernel diffs (and shared-GEMM diffs), each must pass §5.7 rail 9 parity gate | LLM proposer + parity gate + autotune-on-acceptance | Up to 12 evaluations |
| **L1 vLLM** | Mixed continuous (`max_num_seqs`, `gpu_memory_utilization`, `max_num_batched_tokens`, `thinking_token_budget`) + discrete (chunked-prefill, prefix-caching) | Bayesian opt (TPE via Optuna) on continuous knobs, bandit on discrete | Up to 36 evaluations |
| **L2 Request shaping** | Small discrete (policy variants) | Policy comparison across ~6 hand-curated candidate policies; LLM-proposed candidates | Up to 12 evaluations |
| **L3 LoRA** | Small discrete + trivial when no adapters are in use | Policy comparison; often resolved in 1 evaluation | Up to 8 evaluations |

Total evaluation ceiling across all four layers (sum of sub-level caps, both families counted): **108**, matching the compute-budget cap in §5.7 (bumped from 96 to absorb the second L0 family). Per-layer stopping (§5.8) means the ceiling is only hit in the worst case. Vendor-library selections short-circuit their family's autotune sub-level to 1 iteration.

**Open-source foundations — what we reuse, what is ours.** This HLD deliberately builds on OSS primitives rather than reinventing them. The inner optimizers and measurement harness are things the ecosystem already does well; what is ours is the outer LLM-in-the-loop proposer, the layer-stack traversal, the safety rails, and the parity gate for L0-mutate.

| Piece | Built on (OSS) | Ours |
|---|---|---|
| Measurement harness (§5.6) | vLLM 0.19.x `benchmarks/benchmark_serving.py` (+ `benchmarks/auto_tune.py` for L1 reference); `benchmark_serving.py`'s own separate TTFT / TPOT / end-to-end p50/p95/p99 reporting (computed from per-request latencies it records, not from /metrics); vLLM-native Prometheus histograms (`vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:e2e_request_latency_seconds`) consumed via `histogram_quantile()` for cross-run aggregates | The synthetic workload distribution sampler that mirrors the family's eval traffic; the three-dim SLO constraint-checker; the LLD-SB-02 `tokens_reasoning_total` / `tokens_response_total` counters |
| L1 inner optimizer | Optuna TPE (Bayesian opt); Nevergrad as a drop-in fallback | Action-space definition including thinking-budget knob, regression guard, integration with the outer layer traversal |
| L0b autotune (both families) | `@triton.autotune`; CUTLASS Python tuner | Per-kernel search-space definition for both DeltaNet and GatedAttn |
| L0c mutation proposer (both families) | — (this is the novel bit) | LLM proposer, per-family parity harness (DeltaNet state-aware), patch-and-rebuild plumbing, weight-sensitivity flagging, mutation-iteration accounting |
| Outer layer-stack traversal (§5.5) | — (the layered coordinate-descent + single-re-open pattern is our design) | The whole thing, including the per-family L0 traversal |

No existing framework is a drop-in "auto-research for LLM serving backends." Sakana AI Scientist and similar LLM-research-agent scaffolds exist but target ML-paper authoring, not serving-kernel tuning; worth reading for loop structure, not forking.

**Why this split.** Pure Bayesian opt over the full joint action space wastes iterations on locally-poor regions and on combinations whose lower layers are wrong. Pure LLM proposing is unreliable on continuous numerics like `max_num_seqs`. The layered split gives the LLM agency where reasoning wins (the kernel picker, the mutation proposer, the shaping-policy comparator, the re-open decision) and the numeric optimizer where it wins (the L1 continuous knobs, the per-kernel autotune spaces, with L0-select already fixed).

**Karpathy framing, honestly applied.** The LLM proposer reads each layer's measurement trace and explains the observed bottleneck in natural language before proposing the next candidate — this is the "reasoning about measurement" part, and it is especially load-bearing at L0c where the proposer is actually writing kernel diffs. The deterministic Optuna / Triton-autotune inner loops and the parity gate are what keep the overall run reproducible and correct across iterations.

### 5.6 Measurement harness

Built on vLLM's own benchmark tooling rather than rewritten from scratch.

- **Harness base.** Fork of vLLM 0.19.x `benchmarks/benchmark_serving.py` (request driver, latency bucketing, separate TTFT / TPOT / TurnLatency p50/p95/p99 reporters — the three-dim split is already first-class in upstream `benchmark_serving.py` as of 0.19) extended with the synthetic-workload sampler below. vLLM's own `benchmarks/auto_tune.py` is the reference implementation for the L1 vLLM-config sweep; we extend its loop shape to the layered traversal in §5.5 rather than mimicking its single-pass heuristic. vLLM version is pinned — an upstream change that breaks the harness API requires a v0.2-style re-tune per §6.4.
- **Telemetry substrate.** LLD-04 vLLM `/metrics` delta-sampling. No new native vLLM telemetry primitives invented in this HLD; the only net-new series are the LLD-SB-02 additions noted in §4.5 (`tokens_reasoning_total`, `tokens_response_total` counters, labeled by `class` / `family_id` / `weight_version_id` / `tuned_config_id`). The measurement harness consumes vLLM's own histograms (`vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:e2e_request_latency_seconds`), vLLM's own counters and gauges (aggregate throughput, prefix-cache hit rate, queue depth, active-sequence count), and the LLD-SB-02 reasoning / response token counters. **p95 values used in the §5.4 three-dim constraint check are computed by the harness** — either (a) directly from the per-request latencies it records via `benchmark_serving.py`'s internal bucketing (this is the canonical path; the fork already separates TTFT / TPOT / TurnLatency p50/p95/p99 from raw per-request timings), or (b) from the vLLM histograms via `histogram_quantile()` for cross-run aggregate views. The harness does not rely on `/metrics` to pre-aggregate p95 — it computes p95 itself, which matches vLLM's own metric shape (histograms, not quantiles).
- **Synthetic load generator.** A driver that mirrors the eval-workload distribution input (§5.2): samples context lengths, output lengths, thinking-token usage, tool-call patterns, and turn counts from the empirical distribution, and issues requests at a target concurrency level via the `benchmark_serving.py` request path. Each request carries the family's `thinking_token_budget` in `extra_body`; reasoning-content and response-content tokens are counted separately in the measurement trace. The driver is deterministic under a seed so iterations are comparable.
- **Inner optimizer.** Optuna (TPE sampler) drives the L1 sweep against the harness; Nevergrad is a drop-in fallback if Optuna's version pin becomes a problem. L0b uses `@triton.autotune` / CUTLASS Python tuner as the inner loop, per family. L0c uses the LLM proposer + per-family parity gate. L2 / L3 use policy comparison over a small candidate set.
- **Measurement window.** Default 30 minutes per iteration: 5 minutes warm-up (discarded), 25 minutes steady-state (scored). Shorter windows are allowed for early-iteration screening; the final top-K are always scored at the full window.
- **Per-iteration reset.** Between iterations the vLLM process is restarted with the new config (cold restart per §3.13 is budgeted). The prefix cache intentionally starts cold — warm-prefix behavior is measured *within* the iteration, not carried across iterations.
- **Parity fixtures, per family** (consumed by §5.7 rail 9). Two pinned fixtures, both held out from the workload-distribution capture:
  - **DeltaNet fixture** — ~64 `(prompt, output-token-count)` probes with cached reference outputs *and* cached reference recurrent-state snapshots at both early (~1st token) and mid-sequence (~1024th token) checkpoints, taken from the L0-DeltaNet-select+autotune winner. Catches both immediate output drift and slow recurrent-state corruption.
  - **GatedAttn fixture** — ~64 `(prompt, output-token-count)` probes with cached reference logits taken from the L0-GatedAttn-select+autotune winner. Standard logit-space comparison.
  Every L0c mutation runs against its own family's fixture before its latency is measured; a shared-GEMM mutation runs against both.

### 5.7 Safety rails

The agent runs unattended; these rails keep it from producing bad configs or breaking the machine.

1. **Compute-budget cap.** Wall-clock hard limit (default 8 hours — bumped from 4 → 6 → 8 because L0 now spans two kernel families with mutation compile overhead each); total-iteration hard limit (default 108 across all layers / sub-levels, per §5.5 budget table); per-iteration hard limit (default 40 minutes, with a separate 20-minute cap on per-mutation compile + parity check at L0c — bumped from 15 minutes because DeltaNet mutations tend to compile slower than GatedAttn mutations and both must fit the same per-mutation ceiling). The one-time run stops at the first cap it hits.
2. **Re-open budget.** Each prior layer may be re-opened by the outer loop at most once per run (§5.5). Second re-open attempts on the same layer are denied; the agent advances with whatever config that layer last produced. Sub-levels within L0 are not separately re-openable from outside L0 — L0 advances only once both families have run through their three sub-levels. *Within* L0, a family's earlier sub-level is never re-opened by a later sub-level of the *same* family, and a cross-family re-open (GatedAttn finding that DeltaNet's winner is blocking concurrency) counts as the one re-open allotted to L0.
3. **Regression guard against the default-config baseline.** The default, untuned vLLM configuration (§5.2) is measured on the synthetic driver at the start of the run and also at the end of each layer. The final bundle must match or beat that baseline on the §5.4 objective, or the run emits no bundle and the serving stack continues to use the default. No silent downgrade, no forced bundle.
4. **Per-layer and per-family regression guard.** Advancing from layer N to layer N+1 (and from L0 sub-level to L0 sub-level *within the same family*, and from family DeltaNet's final winner to family GatedAttn's starting point) requires the winner to be at least as good as the prior sub-level's / family's / layer's winner on the subset of the objective it can influence. A layer, sub-level, or family that cannot improve is frozen at the prior winner's values (not at a strictly-worse candidate). In particular, L0c may emit no mutation per family — see §5.3.1.c.
5. **Scorer-determinism check.** A held-out deterministic-scorer probe set is run at the end of each layer and every K=4 iterations within a layer. If scoring determinism drops below 99.9% agreement with the run's baseline measurement, the offending configuration is added to a deny-list for the remainder of the run.
6. **OOM handling.** Per-iteration OOM events abort the iteration, record the crash, mark the configuration `infeasible`, and continue. Three OOMs in a row within a single sub-level (DeltaNet-a, DeltaNet-b, DeltaNet-c, GatedAttn-a, GatedAttn-b, GatedAttn-c, L1, L2, or L3) abort that sub-level / layer (freezing it at prior winner) and advance. DeltaNet OOMs are distinguished from GatedAttn OOMs in the trace — a three-in-a-row count is scoped to the specific sub-level, not aggregated across L0.
7. **KV-cache-poisoning detection.** A probe request with known input and known expected output is issued at the start and end of each iteration. If the two responses diverge, the iteration is marked suspect and its measurement discarded. The probe covers both GatedAttn KV and DeltaNet recurrent state — divergence in either counts.
8. **Rollback path.** If something catastrophic happens mid-run, the serving stack reverts to the default-config baseline via the `/admin/load_tuned_config` control-plane endpoint (loading "no bundle" is equivalent to the default-config baseline). This is the same path §3.8 machine-restart resumability uses.
9. **Numerical-parity gate for L0c kernel mutations — per family, with family-specific fixtures.** Every candidate kernel mutation (§5.3.1.c) must pass a parity check against *its family's* pinned reference kernel (that family's L0-select + L0-autotune winner) before any latency measurement counts. The check runs the per-family `parity fixture` from §5.6 through both the reference and the mutated kernel:
   - **DeltaNet mutations** are checked in **both the output-logit space and the recurrent-state space**, at both the 1st output token (catches immediate corruption) and the ~1024th output token (catches slow state drift that would not show up on a short probe). Tolerance `rtol=1e-3, atol=1e-3` on logits; recurrent-state tolerance `rtol=5e-3, atol=5e-3` on end-of-chunk snapshots (looser because the state is an accumulated product, tightened on follow-up in §11.8).
   - **GatedAttn mutations** are checked in logit space only, `rtol=1e-3, atol=1e-3` on FP8 forward passes.
   - **Shared-GEMM mutations** must pass *both* fixtures — logit-space against the GatedAttn fixture and state-space against the DeltaNet fixture.
   A mutation that fails its family's parity is rejected without consuming an L0c iteration slot for that family, is logged to the search trace with the first diverging probe index and (for DeltaNet) the first diverging state channel, and the proposer is fed the failure so it can avoid the same mutation class. Without this rail, L0c cannot be enabled — it is the load-bearing safety invariant that makes kernel-mutation-as-auto-research tractable. The per-family split matters: a DeltaNet mutation that only passes a logit-space check can still have corrupted recurrent state that poisons every long rollout downstream.

### 5.8 Stopping criterion

Stopping is evaluated **per sub-level × per family** within L0, **per family** for L0 as a whole, **per layer** for L1–L3, and then for the overall run. Six sub-level stops fire within L0 (DeltaNet-a, DeltaNet-b, DeltaNet-c, GatedAttn-a, GatedAttn-b, GatedAttn-c) plus a cross-family "L0 overall" stop.

**Per-sub-level stop (L0 only, evaluated independently per family).** Each of L0-DeltaNet-a, L0-DeltaNet-b, L0-DeltaNet-c, L0-GatedAttn-a, L0-GatedAttn-b, L0-GatedAttn-c stops and advances when any of the following fire first:

- **Sub-level diminishing returns.** For the "-a select" and "-b autotune" sub-levels (both families): over the last K=4 evaluations within the sub-level, best-so-far per-family decode-latency metric improved by less than `ε = 2%`. For the "-c mutate" sub-levels (both families): over the last K=6 evaluations (mutation has higher variance), improvement less than `ε = 2%` *or* the LLM proposer declares no further productive mutation direction for this family.
- **Sub-level iteration cap.** Exhausted per §5.5 table: 8 for each "-a select", 6 for each "-b autotune", 12 for each "-c mutate" — per family.
- **Sub-level infeasibility.** All candidates in that family's sub-level failed the §5.7 per-layer/per-family regression guard, or (for "-c mutate" only) more than 80% of that family's mutation proposals were rejected by the §5.7 rail 9 parity gate — advance with the sub-level frozen at the prior winner for that family. For L0-F-c (either family) this is the "mutation search ran, no improvement for family F" outcome — success, not failure, and it is allowed independently per family.

**Per-family stop (L0 only).** A family F ∈ {DeltaNet, GatedAttn} stops and its winners become immutable for L1 when all three of F's sub-levels have stopped or the family-scoped iteration cap (26 per family = 8+6+12) is exhausted.

**Per-layer stop.** A layer stops and advances when any of the following fire first:

- **Per-layer diminishing returns.** Over the last K evaluations within the layer (K=4 for L2/L3, K=8 for L1; L0's stop is the conjunction of both families' stops — L0 does not advance until both DeltaNet and GatedAttn have stopped).
- **Per-layer iteration cap.** Exhausted per §5.5 table: 52 for L0 (26 per family × 2 families), 36 for L1, 12 for L2, 8 for L3.
- **Per-layer infeasibility.** All candidates in the layer failed the §5.7 regression guard or produced OOMs — advance with the layer frozen at prior winner values. For L0, infeasibility is per-family: if DeltaNet is infeasible but GatedAttn is feasible, L0 advances with DeltaNet frozen at the stock kernel and GatedAttn at its winner.

**Overall-run stop.** The run stops when any of the following fire first:

- **Wall-clock cap.** Default 8 hours across all layers combined (§5.7 rail 1).
- **Total-iteration cap.** Default 108 across all layers (§5.7 rail 1).
- **Layer stack complete.** L3 has converged — the normal termination path.
- **Hard infeasibility.** All four layers individually failed and the final bundle is identical to default — in which case no bundle is emitted.

At normal stop, the best feasible configuration per layer / family / sub-level is frozen and concatenated into the output bundle.

### 5.9 Output artifact — the tuned-config bundle

A single YAML blob persisted to `/data/tuned_configs/<family_id>/<weight_version_id>/<run_id>.yaml` on internal NVMe. Schema:

```yaml
tuned_config_bundle:
  bundle_id: <uuid>
  produced_at: <iso8601>
  model_id: qwen3.5-27b                              # HARD-PINNED in bundle-validity rule (§6.4)
  family_id: <family>                                # HARD-PINNED in bundle-validity rule (§6.4)
  weight_version_id: <sha>                           # METADATA — does NOT pin; re-validated via weight_sensitive flag (§6.4)
  workload_distribution_id: <hash>                   # METADATA — informational
  # Layer 0 is split across two kernel families. Each is independently tuned (§5.3.1).
  layer_0_deltanet:
    l0a_select:
      deltanet_kernel: <enum>       # see §5.3.1.a
    l0b_autotune:
      per_kernel_params:            # Triton BLOCK_*, CHUNK_SIZE, num_warps, num_stages
        BLOCK_M: <int>
        BLOCK_N: <int>
        BLOCK_K: <int>
        CHUNK_SIZE: <int>
        num_warps: <int>
        num_stages: <int>
    l0c_mutation:
      # Either the "no mutation won for DeltaNet" case …
      diff_ref:            null
      build_command:       null
      parity_attestation:  null
      weight_sensitive:    false
      # … or an accepted DeltaNet mutation:
      # diff_ref:           /data/tuned_configs/.../mutations/<mutation_id>.patch
      # build_command:      "cd kernels/deltanet && make mutation_<id>"
      # parity_attestation: { fixture_id: deltanet_v1, rtol_logit: 1e-3, atol_logit: 1e-3, rtol_state: 5e-3, atol_state: 5e-3, probes_passed: <int>, probes_total: <int>, first_divergence: null, state_checkpoints: [1, 1024] }
      # weight_sensitive:   <bool>   # proposer-annotated; gates re-parity-check on weight_version_id change (§6.4)
  layer_0_gatedattn:
    l0a_select:
      attention_backend: <enum>     # see §5.3.1.a — FORCE-SELECTED per §9.3.13; never `vllm-default` in a shipped bundle
    l0b_autotune:
      per_kernel_params:            # Triton BLOCK_*, num_warps, num_stages (no-op for FA4 / FA3 / FlashInfer)
        BLOCK_M: <int | null>
        BLOCK_N: <int | null>
        BLOCK_K: <int | null>
        num_warps: <int | null>
        num_stages: <int | null>
    l0c_mutation:
      diff_ref:            null
      build_command:       null
      parity_attestation:  null
      weight_sensitive:    false
      # Accepted GatedAttn mutation structure mirrors DeltaNet, but parity_attestation omits state_checkpoints
      # (logit-space only: { rtol: 1e-3, atol: 1e-3, probes_passed, probes_total, first_divergence: null })
  layer_0_shared:                   # cross-family L0-select knobs (§5.3.1.a)
    torch_compile_mode:   <enum>
    cuda_graph_capture:   <enum>
    fp8_gemm_kernel:      <enum>
    cutlass_autotune:                 # CUTLASS tile/pipeline/epilogue (shared FP8 GEMM)
      tile_shape:         <string>
      pipeline_depth:     <int>
      epilogue_schedule:  <string>
    shared_gemm_mutation:             # optional — a CUTLASS mutation scored against BOTH families
      diff_ref:            null
      build_command:       null
      parity_attestation:  null      # when present, must include BOTH deltanet_v1 AND gatedattn_v1 fixtures
      weight_sensitive:    false
  layer_1_vllm:      { ... §5.3.2 knobs, including thinking_token_budget ... }
  layer_2_shaping:   { ... §5.3.3 knobs ... }
  layer_3_lora:      { ... §5.3.4 knobs ... }
  # Per-layer provenance
  layer_traces:
    l0:
      deltanet:
        l0a: { iterations: <int>, best_objective: <float>, trace_ref: <path> }
        l0b: { iterations: <int>, best_objective: <float>, trace_ref: <path> }
        l0c: { iterations: <int>, mutations_proposed: <int>, mutations_parity_rejected: <int>, mutations_accepted: <int>, best_objective: <float>, trace_ref: <path> }
      gatedattn:
        l0a: { iterations: <int>, best_objective: <float>, trace_ref: <path> }
        l0b: { iterations: <int>, best_objective: <float>, trace_ref: <path> }
        l0c: { iterations: <int>, mutations_proposed: <int>, mutations_parity_rejected: <int>, mutations_accepted: <int>, best_objective: <float>, trace_ref: <path> }
      shared:
        shared_gemm_mutations_proposed: <int>
        shared_gemm_mutations_parity_rejected: <int>
        shared_gemm_mutations_accepted: <int>
      reopened: <bool>
    l1: { iterations: <int>, best_objective: <float>, trace_ref: <path>, reopened: <bool> }
    l2: { iterations: <int>, best_objective: <float>, trace_ref: <path>, reopened: <bool> }
    l3: { iterations: <int>, best_objective: <float>, trace_ref: <path>, reopened: <bool> }
  # Overall objective (measured on the fully-assembled bundle)
  objective:
    metric: sustained_concurrent_eval_threads_under_three_dim_slo
    value: <int>
    ttft_p95_ms:         <int>        # measured, must be ≤ L_ttft
    tpot_p95_ms:         <int>        # measured, must be ≤ L_tpot
    turn_latency_p95_ms: <int>        # measured, must be ≤ L_turn
    reasoning_content_purity: 1.0     # §5.4 constraint 8
    measurement_window_minutes: <int>
  search_trace_ref: <path to combined cross-layer search trace>
  baseline_bundle_id: <uuid | null>   # null for v0.1 one-time runs
  regression_guard:   { baseline_value: <int>, delta: <int> }
  safety_rails:       { ... boolean attestations per §5.7 ... }
  # Bundle-validity pin — used by §6.4 refusal rule at campaign bootstrap
  environment_pin:                                   # ALL HARD-PINNED by §6.4
    vllm_version:     <string>                       # e.g. "0.19.2"
    kernel_versions:
      flash_attn:     <string>
      flashinfer:     <string>
      deltanet_kernel_pkg: <string>
    triton_version:   <string>
    cutlass_version:  <string>
    optuna_version:   <string>
    cuda_version:     <string>
    driver_version:   <string>
  serving_posture_pin:                               # HARD-PINNED by §6.4 (launch-time invariants from §4.5)
    limit_mm_per_prompt_image: 0
    reasoning_parser: qwen3
    enable_thinking:  true                            # server default; admission rejects request-level override (§4.1)
    thinking_token_budget: <int>                      # server default (== family_spec.thinking_budget); admission rejects request-level override (§4.1)
    request_level_thinking_override_policy: reject    # MUST be "reject" for campaign traffic per §4.1 override-rejection policy
  mutation_artifacts_ref: <path | null>              # directory of accepted mutation .patch files (both families) + rejected-mutation log for this run
```

`environment_pin` + `serving_posture_pin` + `model_id` + `family_id` constitute the bundle-validity rule enforced at bootstrap — see §6.4. `weight_version_id` is metadata, not a pin; weight rotation re-validates only mutations whose `weight_sensitive: true` flag is set (plus the default-baseline regression guard).

Campaign bootstrap loads the bundle via `POST /admin/load_tuned_config` (§4.5). Bundles are immutable — a new tune produces a new bundle, never mutates an existing one.

---

## 6. Weight-Update Path — PyTorch Writes → KV Invalidate → Resume

Trivial by design. The entire section is about keeping it trivial.

### 6.1 The trivial-by-design contract

One sentence: **training writes weights to a staging path, renames atomically into place, signals the server, the server invalidates KV and reloads. That's it.** No hot-swap-with-preserved-KV, no partial re-shard, no gradient-aware KV rebuild, no streaming weight delta.

Why this is the contract, not a compromise:

- KV preservation across a weight change is mathematically wrong — old KV was computed by old weights, applying new-weight decode over old-weight KV produces garbage. Any design that "preserves" KV is either (a) also wrong or (b) also invalidating but hiding it behind a cache warm-up.
- Partial re-shard is a multi-GPU optimization; single-DGX-Spark (§3.9) removes the motivation.
- Every minute spent on non-trivial weight-update mechanics is a minute not spent on §5 (auto-research) or §9 (verification). The trade is obviously right.

### 6.2 Hook points

```
Training process:
  write  →  /models/active/next/weights.pt.tmp       (atomic; can fail partway without affecting serving)
  write  →  /models/active/next/config.json
  rename →  /models/active/next  →  /models/active/current   (atomic; new weights are visible or they aren't)
  POST   →  /admin/invalidate                                (notifies serving; body = weight_version_id)

Serving process:
  on /admin/invalidate:
    set state = UPDATING
    drain in-flight requests per §4.4 (finish-at-current-weights policy)
    flush KV cache
    flush prefix cache
    reload weights from /models/active/current
    warm up: issue N=4 warm-up requests with diverse shapes
    set state = READY, weight_version_id = <new>
```

The atomic rename is the key correctness primitive. `inotify` (preferred) or polling (fallback) detects the rename. The `/admin/invalidate` HTTP call is the faster path; the filesystem watch is the safety net if the HTTP call is missed.

### 6.3 Failure modes and recovery

| Failure | Detection | Recovery |
|---|---|---|
| Training writes corrupt weights | Reload-time state_dict mismatch | Abort reload, keep old weights, mark weight_version_id `rejected`, alert |
| OOM during reload | Allocator exception | Abort reload, keep old weights, mark config `infeasible`, rollback to last-known-good |
| Invalidate HTTP missed | Filesystem watch fires within 5 s | Fallback path invokes the same reload sequence |
| Determinism drift post-update | §5.7 probe at next eval iteration | Flag weight_version_id as `determinism-drift`; training pipeline decides whether to roll back |
| In-flight request mid-update | §4.4 finish-at-current-weights | Response tagged with old `weight_version_id`; caller sees timestamp + version and routes accordingly |

### 6.4 Tuned-config bundle validity rule

A single, explicit rule governs whether a previously-tuned bundle can be loaded against the currently-running serving stack. The serving backend applies this rule exactly once — at bootstrap, when `POST /admin/load_tuned_config` or the equivalent startup path resolves the bundle. If the rule refuses, the serving stack falls back to the default-config baseline; it does **not** silently degrade or partially apply a bundle.

**The rule, in one statement.** A bundle is valid if and only if **every** field below in the bundle matches the live serving process's corresponding attestation, exactly:

| Bundle field | Hard pin? | Source of truth at bootstrap |
|---|---|---|
| `model_id` | yes | live process's loaded model |
| `family_id` | yes | the family the campaign is bootstrapping |
| `environment_pin.vllm_version` | yes | `vllm.__version__` |
| `environment_pin.kernel_versions.flash_attn` | yes | installed package version |
| `environment_pin.kernel_versions.flashinfer` | yes | installed package version |
| `environment_pin.kernel_versions.deltanet_kernel_pkg` | yes | installed package version |
| `environment_pin.triton_version` | yes | `triton.__version__` |
| `environment_pin.cutlass_version` | yes | cutlass-python reported version |
| `environment_pin.optuna_version` | yes | `optuna.__version__` |
| `environment_pin.cuda_version` | yes | `torch.version.cuda` |
| `environment_pin.driver_version` | yes | `nvidia-smi` reported |
| `serving_posture_pin.limit_mm_per_prompt_image` | yes | launch arg parsed from process cmdline |
| `serving_posture_pin.reasoning_parser` | yes | launch arg parsed from process cmdline |
| `serving_posture_pin.enable_thinking` | yes | default chat-template value |
| `serving_posture_pin.thinking_token_budget` | yes | default chat-template / server-level value (must equal `family_spec.thinking_budget` — §4.1) |
| `serving_posture_pin.request_level_thinking_override_policy` | yes | admission-layer policy (MUST be `reject` for campaign traffic — §4.1) |
| `weight_version_id` | **no — metadata only** | see "weight rotation" below |
| `workload_distribution_id` | **no — metadata only** | informational; changing it does not refuse the bundle |

Exactly one rule. No soft overrides, no "close enough" matching on version strings, no implicit upgrades. A mismatch on any hard-pinned field refuses the bundle; the refusal is logged with the specific mismatching field(s) and the serving stack uses the default baseline until a re-tune produces a new bundle.

**Weight rotation (the one nuance).** The bundle's `weight_version_id` is metadata, not a hard pin — the common case (training publishes new weights, environment unchanged) must not invalidate the tuned bundle. However, kernel mutations tagged `weight_sensitive: true` (§5.3.1.c) are an exception: on a `weight_version_id` change, every `weight_sensitive: true` mutation in the bundle must re-pass its family's parity fixture against the new weights before the bundle is accepted. A re-parity-check failure downgrades the bundle in one of two ways:

1. The affected mutation is silently replaced by its family's L0-autotune winner (the pre-mutation reference) *and* the bundle is tagged `weight_rotation_downgraded: true`. Serving continues with the downgraded bundle; the §5.4 regression guard is re-run at next campaign boundary.
2. If the downgraded bundle fails the §5.4 baseline regression guard, the bundle is refused entirely and the serving stack falls back to the default baseline.

Mutations flagged `weight_sensitive: false` skip the re-parity-check. This keeps the common case — weights rotate, everything else held — from triggering a full re-tune, while the rare weight-sensitive mutation is isolated and handled specifically.

**What this rule deliberately does not allow.** Loading a bundle tuned under vLLM 0.19.2 into a vLLM 0.19.3 process. Loading a bundle whose `limit_mm_per_prompt_image` was 0 into a process that re-enabled the vision tower. Loading a bundle whose kernel-package versions drifted under `pip install -U`. Loading a bundle tuned under `request_level_thinking_override_policy: reject` into a process whose admission layer silently accepts request-level thinking overrides (or vice-versa) — the bundle's thinking posture must match the admission layer's enforcement. These are all refused. In a v0.1 one-time-run world the refusal is a *feature*, not a bug: the bundle captures a point-in-time (env × model × posture) artifact, and drift on any dimension invalidates the artifact.

**Scope.** If `model_registry.yaml` moves to a new model entirely, the bundle is refused on `model_id` mismatch and the serving stack falls back to the default-config baseline until a new one-time tuning run is commissioned (which is itself a v0.2 scope decision per §11.9). A weight update does **not** trigger any auto-research step — re-running is out of scope per §5.1.

### 6.5 Interaction with prefix caching and LoRA adapters

- **Prefix cache.** Keyed on `(weight_version_id, prompt_prefix_hash)`. On weight update, all entries with the old `weight_version_id` are invalidated. The next request after an update is always a cold-prefix request. Prefix caching covers both the GatedAttn KV entries *and* the DeltaNet recurrent-state snapshot at the shared-prefix boundary; either missing invalidates the cache line.
- **LoRA adapters.** Swapping a LoRA adapter does *not* invalidate base-weight KV or DeltaNet state. LoRA adapters are applied per-request (per §5.3.4), so the base-weight KV and recurrent state remain valid. A weight update of the *base* model, however, invalidates all adapter-augmented KV and state as well.
- **Tuned-config bundle.** Governed by §6.4 above — weight rotation is the metadata case; env / posture drift is refused.

---

## 7. Sequencing Across Sprints

Four-sprint delivery plan. Earlier sprints land infrastructure the later sprints need; no sprint is blocked on a sibling-HLD deliverable.

Sprint delivery matches the §5.3 layer stack, with L0's three sub-levels split across two sprints **and each sub-level spanning two kernel families (DeltaNet + GatedAttn)**. L0c mutation depends on a per-family parity-gate harness that only makes sense once L0a+L0b are committed as each family's reference. Sprint 0 ships L1 (with thinking-mode knob), Sprint 1 slides L0a+L0b for **both families** beneath it, Sprint 2 stacks L2 and L3 plus the weight-update hook, Sprint 3 lands L0c mutation (both families) + deprecation.

### 7.1 Sprint 0 — scaffold + L1 (vLLM config incl. thinking budget) only

- Stand up the measurement harness (§5.6) against the existing `src/lumo_flywheel_serving/` implementation (LLD-01/04-era code is adequate substrate). Fork vLLM 0.19.x `benchmarks/benchmark_serving.py` as the driver base — pin vLLM version per §6.4. Wire Optuna as the L1 inner optimizer. Confirm the driver separately reports TTFT / TPOT / TurnLatency p50/p95/p99 (computed from its own per-request latencies, *not* scraped from `/metrics`), per §4.1 three-dim SLO. Separately, confirm the Prometheus plane exposes vLLM-native histograms (`vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:e2e_request_latency_seconds`) and LLD-SB-02 counters (`tokens_reasoning_total`, `tokens_response_total`) and that a `histogram_quantile()`-derived p95 matches the harness's computed p95 within sampling noise.
- Serving-posture lockdown: launch vLLM with `--limit-mm-per-prompt image=0` (vision tower off — §4.5) and `--reasoning-parser qwen3` (thinking contract — §4.1). Write a pre-boot check that refuses to start if either is missing.
- Implement the §5.3.2 Layer 1 (vLLM config) action space only, **including `thinking_token_budget` and `thinking_parallel_streams`**. L0 (both families) / L2 / L3 held at their defaults — the agent is effectively single-layer for this sprint. L1 must honor `reasoning_content_purity = 100%` (§5.4 constraint 8) against the Qwen3 reasoning parser.
- Implement §5.7 safety rails 1–8 (rail 9 per-family parity gate lands in Sprint 3 with L0c — not needed before, and its per-family split is a Sprint-3 concern).
- Implement §6.4 bundle-validity rule (env + posture pinning) end-to-end so the Sprint-0 bundle round-trips through `POST /admin/load_tuned_config`.
- Produce a L1-only tuned-config bundle for Qwen 3.5 27B against a V1 family's workload; verify all three SLO dimensions (TTFT/TPOT/TurnLatency) hold at the chosen concurrency.
- Deliverable: §9.1 (contract endpoints), §9.3.1, §9.3.2, §9.3.3 (auto-research runs + produces bundle + safety rails), §9.3.8 (three-dim SLO enforcement), §9.3.9 (thinking contract + purity) pass. §9.5 resumability also lands in Sprint 0.

### 7.2 Sprint 1 — slide L0a (select) + L0b (autotune) for BOTH kernel families beneath L1

- Populate **per-family** Appendix B kernel compatibility matrices for Qwen 3.5 27B (empirically — attempt each cell, record PASS / FAIL-streaming / FAIL-determinism / FAIL-accuracy). Separate tables for DeltaNet family and GatedAttn family. For GatedAttn on Blackwell (SM100+): explicitly screen `flashinfer` against GitHub vllm-project/vllm #35138 on the family's workload (the `flashinfer` backend is first in vLLM's own auto-select priority but carries a known accuracy issue on Qwen3.5); empirically confirm which force-selectable backend wins (FA4 is the expected winner post-accuracy-screen, but must be measured) — a `vllm-default` dispatch is *not* an acceptable shipped value per §9.3.13. Confirm the DeltaNet Triton kernels meet the determinism bar.
- Enable §5.3.1.a Layer 0a (selection) and §5.3.1.b Layer 0b (parameter-tune) **for both families**:
  - **DeltaNet**: `deltanet_kernel` select over {triton-chunked-delta, triton-state-update-fused, triton-fallback, vllm-default-deltanet}; autotune via `@triton.autotune` over `BLOCK_*`, `CHUNK_SIZE`, `num_warps`, `num_stages`.
  - **GatedAttn**: `attention_backend` select over {flash-attn-4, flash-attn-3, flashinfer, triton, vllm-default}; autotune via `@triton.autotune` on the Triton candidates (no-op for vendor-library backends — FA4/FA3/FlashInfer own their internals). CUTLASS Python tuner handles the shared FP8 GEMM on tile / pipeline-depth / epilogue.
  - Cross-family (shared) knobs: `torch_compile_mode`, `cuda_graph_capture`, `fp8_gemm_kernel` — single values across both families.
- Implement the layer traversal loop (§5.5) with the per-family L0 traversal (DeltaNet first, then GatedAttn, then the shared CUTLASS autotune) and the single-re-open escape hatch from L1 back into L0 (which may re-open either family — §5.7 rail 2).
- Commit each family's L0-select + L0-autotune winner as **that family's reference kernel** for L0c's parity harness (Sprint 3 dependency). Author and pin the per-family parity fixtures now (DeltaNet fixture with end-of-chunk state snapshots at token 1 and 1024; GatedAttn fixture with logit-space outputs) so Sprint 3 does not have to invent them.
- Re-run the agent — during sprint development, replacing the Sprint-0 bundle is a dev-iteration act, not the ongoing re-tune that §5.1 rules out for production.
- Deliverable: §9.3.4 (kernel action space produces non-default bundle **for each family**) and §9.3.5 (stopping criterion fires, including per-family stops from §5.8) pass.

### 7.3 Sprint 2 — stack L2 + L3 + weight-update hook

- Implement §6 weight-update path end-to-end, including `/admin/invalidate`, filesystem-watch fallback, and §6.4 bundle-validity rule's `weight_sensitive` re-parity-check hook (scaffold only — §6.4 re-parity path activates once Sprint 3 lands a weight-sensitive mutation).
- Enable §5.3.3 Layer 2 (request shaping) and §5.3.4 Layer 3 (LoRA adapter management).
- Complete the full L0 (DeltaNet-a/b → GatedAttn-a/b → shared CUTLASS) → L1 → L2 → L3 traversal with the single-re-open escape hatch at every boundary. L0c (both families) remains gated off this sprint; the bundle records `layer_0_deltanet.l0c_mutation: null` and `layer_0_gatedattn.l0c_mutation: null`.
- Wire serving to Training Flywheel LLD-06 → LLD-11 weight-publication contract. Confirm prefix-cache and DeltaNet-state invalidation both fire on weight rotation (§6.5).
- Deliverable: §9.4 (weight-update path end-to-end) passes; the final tuned-config bundle is a full four-layer bundle (with both L0c entries null); the Training Flywheel HLD's weight-publication seam is live.

### 7.4 Sprint 3 — L0c mutation (both families) + per-family parity gate + Deprecation migration

- Build the **per-family** parity-gate harness (§5.7 rail 9): reuse the Sprint-1-authored DeltaNet and GatedAttn parity fixtures; implement the tolerance-check path per family (DeltaNet logit + recurrent-state at token 1 and 1024; GatedAttn logit-only); wire mutation-patch-and-rebuild plumbing for each family's kernel source tree plus the shared CUTLASS tree.
- Enable §5.3.1.c Layer 0c (mutation) **for both families** in the layer traversal, plus shared-GEMM mutations scored against both fixtures. LLM proposer writes kernel diffs against each family's reference kernel; parity gate fences every candidate before latency counts; the `weight_sensitive` flag is set per-mutation by the proposer.
- Wire §6.4's `weight_sensitive` re-parity-check path through to bundle-load time: when a new `weight_version_id` arrives, every `weight_sensitive: true` mutation in the live bundle is re-checked against the new weights; failing mutations are replaced by their family's L0-autotune winner and the bundle is tagged `weight_rotation_downgraded: true`.
- Expected outcome on first run: per family independently — one or two accepted mutations, several parity rejections, possibly a `null` L0c entry — all three are acceptable §5.8 outcomes. DeltaNet-c and GatedAttn-c may produce different outcomes (e.g., DeltaNet accepts a mutation, GatedAttn does not, or vice versa).
- Re-platform `src/lumo_flywheel_serving/` against the derived LLDs.
- Post DEPRECATED banners on LLD-01 / 02 / 03 / 04 / 13.
- Final verification pass: all §9 checklist items green, including §9.3.6 (**per-family** parity gate rejects a wrong mutation in each family), §9.3.7 (L0c sub-level ran for each family, outcome recorded), §9.3.10 (bundle-validity rule refuses a deliberately-drifted env), §9.3.11 (`weight_sensitive` re-parity-check path exercised).
- Deliverable: this HLD is fully implemented; next scope is FP4 exploration and/or additional model support (both gated on §11 open-question resolution).

---

## 8. What This HLD Locks Down For The Downstream LLD(s)

The downstream LLD series that replaces LLD-01/02/03/04/13. Names are placeholders; final numbering is decided by the implementer before LLDs are drafted.

- [ ] **LLD-SB-01** (replaces LLD-01): vLLM Serving Layer — new rev derived from §4 and §6 of this HLD.
- [ ] **LLD-SB-02** (replaces LLD-04): Telemetry & Measurement Plane — reused by §5.6 measurement harness; minor changes for `weight_version_id` / `tuned_config_id` tagging.
- [ ] **LLD-SB-03** (replaces LLD-03): Task Orchestrator — re-platformed for concurrent Codex threads, now with admission control (§4.1) and the two-class queue split (§4.3).
- [ ] **LLD-SB-04** (replaces LLD-02): Data Pool Manager — reused largely as-is; minor changes for concurrency-aware dispatch.
- [ ] **LLD-SB-05** (replaces LLD-13): Codex-Long Scenario Framework — reused; minor changes for the new orchestrator contract.
- [ ] **LLD-SB-06** (new): Auto-Research Agent — full implementation of §5, including the layer-stack traversal (L0 → L1 → L2 → L3 with single-re-open), L0's **two kernel families (DeltaNet, GatedAttn) × three sub-levels (select / autotune / mutate)** plus the shared cross-family knobs and CUTLASS autotune, the **per-family parity-gate harness** for L0c (§5.7 rail 9 — DeltaNet with recurrent-state snapshots at tokens 1 and 1024, GatedAttn with logit-space, shared-GEMM against both), the LLM proposer with `weight_sensitive` annotation, the **three-dim latency SLO** enforcement (TTFT_p95 / TPOT_p95 / TurnLatency_p95 conjunction, §4.1), the `reasoning_content_purity = 100%` constraint, the **thinking-mode knobs** (`thinking_token_budget`, `thinking_parallel_streams` in §5.3.2), the outer-loop integration with OSS inner optimizers (Optuna, `@triton.autotune`, CUTLASS Python tuner), the harness fork of vLLM 0.19.x `benchmarks/benchmark_serving.py` (pinned per §6.4), and the **per-family Appendix B kernel matrices**.
- [ ] **LLD-SB-07** (new): Weight-Update Hook — full implementation of §6, including the atomic-rename watcher, the `/admin/invalidate` handler, the rollback path, **the §6.4 bundle-validity rule enforcement at `POST /admin/load_tuned_config`, and the `weight_sensitive` re-parity-check path on weight rotation**.

Every LLD in this list carries a one-paragraph "What the HLD locks and what this LLD is free to choose" preamble, so the implementer agent and verifier agent have a clear seam between spec-level constraints and implementation judgment.

---

## 9. Verification Checklist (Pass/Fail, for the Implementer + Verifier Agent Pair)

Binary per item. Each item has a verifiable artifact the verifier agent can inspect without subjective judgment. PASS is only awarded when the artifact exists, is signed with a recent git SHA, and matches the described shape.

### 9.1 Contract satisfaction (§4)

- [ ] **9.1.1** `/v1/responses` endpoint exists and passes a representative Qwen 3.5 27B request with streaming enabled. Artifact: `curl` transcript or test-log record.
- [ ] **9.1.2** `/metrics` endpoint emits the required vLLM-native primitives and LLD-SB-02 additions. Specifically, the scrape must show: (a) vLLM histograms `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:e2e_request_latency_seconds` (with `_bucket`, `_count`, `_sum` series each — these are the shape `histogram_quantile()` consumes), (b) vLLM counters/gauges for aggregate throughput, prefix-cache hit rate, queue depth, and active-sequence count, (c) LLD-SB-02 counters `tokens_reasoning_total` and `tokens_response_total` with `class` / `family_id` / `weight_version_id` / `tuned_config_id` labels. Additionally, a `histogram_quantile(0.95, sum by (le, instance) (rate(vllm_time_to_first_token_seconds_bucket[5m])))` query against the scraped data returns a numeric value consistent with the harness's directly-computed TTFT p95 for the same window. Artifact: scrape log covering every required field + one PromQL-derived p95 cross-check.
- [ ] **9.1.3** `/admin/invalidate` and `/admin/load_tuned_config` endpoints accept valid payloads and reject invalid ones with structured errors. Artifact: test log.
- [ ] **9.1.4** Admission control queues rather than drops on three-dim SLO pressure (any of TTFT_p95, TPOT_p95, TurnLatency_p95 approaching its family-spec ceiling). Artifact: synthetic-load test log showing queue-depth growth under over-concurrency, zero drops, and rejection with structured retry once `admission_queue_depth_max` is hit.
- [ ] **9.1.5** Every response row in telemetry carries `weight_version_id`, `tuned_config_id`, `class`, and LLD-04 telemetry row ID. Artifact: sampled trajectory-row dump.

### 9.2 Hardware envelope attestation (§3)

- [ ] **9.2.1** Post-install microbenchmark run published: measured bandwidth, measured FP8 GEMM throughput, measured NVMe read. Artifact: benchmark log committed to `docs/hw-attestation/`. Deviations > 15% from §3 published figures raise a §11 risk rather than a fail.
- [ ] **9.2.2** `make resume` (or documented equivalent) restores serving to last-known-good after a full power cycle, in under 60 seconds. Artifact: recording of resume run.

### 9.3 Auto-research agent end-to-end (§5)

- [ ] **9.3.1** Agent runs on the measurement harness without OOM or determinism drift for at least one full iteration. Artifact: run log.
- [ ] **9.3.2** Agent produces a tuned-config bundle that beats the baseline on §5.4 objective. Artifact: bundle YAML with `regression_guard.delta > 0`.
- [ ] **9.3.3** §5.7 safety rails tested individually: regression guard trips on a synthetic regression; determinism check trips on an injected non-deterministic kernel; OOM-handling trips on a synthetic oversized config. Artifact: three separate test logs.
- [ ] **9.3.4** Kernel action space (§5.3.1 Layer 0) produces at least one bundle where **each family** force-selects an explicit (non-`vllm-default`) kernel and the bundle beats the vllm-default baseline. Artifact: bundle YAML referencing a concrete `deltanet_kernel` under `layer_0_deltanet.l0a_select` **and** a concrete `attention_backend` under `layer_0_gatedattn.l0a_select` (typically `flash-attn-4` on Blackwell after the FlashInfer accuracy screen, but any explicit non-default value is acceptable — §9.3.13 is the load-bearing acceptance item for the explicit-selection policy).
- [ ] **9.3.5** Stopping criterion (§5.8) fires on at least one of: wall-clock cap, total-iteration cap, diminishing returns, or per-family stop (DeltaNet or GatedAttn reaching its sub-level cap independently) — not only on hard infeasibility. Artifact: run log.
- [ ] **9.3.6** **Per-family** parity gate (§5.7 rail 9) rejects a deliberately-wrong kernel mutation in **each** family. (a) A unit-test DeltaNet mutation that deliberately perturbs the recurrent-state update is proposed to L0-DeltaNet-c; the parity harness reports divergence on the state snapshot at token 1024 (not at token 1 — demonstrating the slow-drift catch path) and the mutation is rejected without consuming an L0c iteration slot for DeltaNet. (b) A unit-test GatedAttn mutation that deliberately flips a sign in the RoPE path is proposed to L0-GatedAttn-c; the parity harness reports logit-space divergence on the first probe and the mutation is rejected. Artifact: two parity-harness test logs, one per family, each showing first-divergence probe index (and for DeltaNet, first-diverging state channel) and the rejection path.
- [ ] **9.3.7** L0c mutation sub-level actually executed **for each family** in at least one production run, producing (per family) either (a) an accepted mutation with that family's parity attestation, or (b) a `null` L0c entry with `mutations_proposed > 0` logged (i.e., the agent tried). Both outcomes are PASS per family — see §5.3.1.c. Artifact: bundle YAML `layer_traces.l0.deltanet.l0c` and `layer_traces.l0.gatedattn.l0c` plus `mutation_artifacts_ref`.
- [ ] **9.3.8** **Three-dim SLO enforcement (§4.1 / §5.4).** A synthetic workload that would violate `TPOT_p95 ≤ L_tpot` at a candidate's concurrency level is rejected by the objective check even if `TTFT_p95` and `TurnLatency_p95` both pass. Artifact: test log showing the rejected candidate's three-dim measurements and the constraint that tripped.
- [ ] **9.3.9** **Thinking contract & reasoning-content purity (§4.1, §5.4 constraint 8).** Every thinking token emitted during a measurement run surfaces under `reasoning_content`; zero thinking tokens leak into `content`. `thinking_token_budget` from the bundle is honored (stream truncation fires when the budget is exceeded, producing a `thinking_truncated` telemetry flag). Artifact: response-dump audit showing `reasoning_content_purity = 1.0` across ≥10k sampled streams, plus one budget-truncation test log.
- [ ] **9.3.9.a** **Admission-layer request-level thinking-override rejection (§4.1 override-rejection policy).** The admission layer rejects (before any GPU work) every request that attempts to override the serving-posture thinking contract. Specifically: (i) a request carrying `chat_template_kwargs.enable_thinking = false` is rejected with HTTP 4xx, structured error `code = thinking_override_forbidden`, and no tokens are generated; (ii) a request carrying `chat_template_kwargs.enable_thinking = true` against the same server default is likewise rejected (the rule rejects the *presence* of the override field, not its value, to avoid silent drift if the server default changes); (iii) a request carrying `extra_body.thinking_token_budget = <any>` is rejected with the same structured code; (iv) a well-formed request that sets neither field is accepted and produces a normal streamed response governed by the server-side `enable_thinking` / `thinking_token_budget` from the bundle. The rejection emits a telemetry row tagged `rejection_reason = thinking_override_forbidden` countable in the admission-control rejection metric. Artifact: four request/response transcripts (three rejections + one acceptance) plus the corresponding telemetry rows.
- [ ] **9.3.10** **Bundle-validity rule refusal (§6.4).** A bundle whose `environment_pin.vllm_version` does not match the live process's `vllm.__version__` is refused at `POST /admin/load_tuned_config` with a structured error naming the mismatching field; the serving stack falls back to the default-config baseline. A bundle whose `serving_posture_pin.limit_mm_per_prompt_image` is 0 loaded into a process launched without that flag is likewise refused. Artifact: two `/admin/load_tuned_config` test transcripts (env drift + posture drift), each showing the refusal payload and the fallback path.
- [ ] **9.3.11** **Weight-sensitive re-parity-check (§6.4).** A bundle containing a mutation flagged `weight_sensitive: true` is loaded; a subsequent `weight_version_id` rotation triggers the re-parity-check against the new weights; a failing re-parity-check downgrades the affected mutation to its family's L0-autotune winner and tags the bundle `weight_rotation_downgraded: true`. Artifact: bundle-state-before/after dump + re-parity-check log.
- [ ] **9.3.12** **Vision tower disabled at launch (§4.5).** The serving process refuses to start if `--limit-mm-per-prompt image=0` is missing from the launch command, and the pre-boot check emits a structured error naming the missing argument. Artifact: test log of the refused launch.
- [ ] **9.3.13** **`attention_backend` force-selected on Blackwell, not left to vLLM auto-select.** On an SM100+ target, `layer_0_gatedattn.l0a_select.attention_backend` in the shipped bundle resolves to a concrete, non-`vllm-default` value (one of `flash-attn-4`, `flash-attn-3`, `flash-attn-2`, `flashinfer`, `triton`) — never to `vllm-default` and never unset. The verifier additionally confirms: (a) the selected backend beat both `vllm-default` *and* the runner-up in the L0-GatedAttn-a trace on the §5.4 objective; (b) if the selected backend is `flashinfer`, the trace records an accuracy-screen pass against GitHub vllm-project/vllm #35138 on the family's workload; (c) if the selected backend is not `flash-attn-4` on SM100+, §11.10 carries an entry naming the reason. Artifact: bundle YAML with the non-default `attention_backend`, L0-GatedAttn-a trace showing the comparison, and (for `flashinfer` selection) the accuracy-screen log.

### 9.4 Weight-update path end-to-end (§6)

- [ ] **9.4.1** Training process writes weights via atomic rename; serving detects within 5 s and reloads. Artifact: log with timestamps.
- [ ] **9.4.2** In-flight request during weight update completes under `finish-at-current-weights`, tagged with the old `weight_version_id`. Artifact: response transcript + telemetry row.
- [ ] **9.4.3** Corrupt-weights scenario: serving aborts reload, retains old weights, marks new `weight_version_id` as `rejected`. Artifact: test log.
- [ ] **9.4.4** Weight-update-invalidate-to-ready wall-clock ≤ 15 s on 27 GB FP8. Artifact: timing log.

### 9.5 Resumability (§3.8)

- [ ] **9.5.1** `make resume` reloads last-known-good tuned-config bundle and last-known-good weights. Artifact: recorded session.
- [ ] **9.5.2** In-flight tasks interrupted by restart produce `integrity_flag = 1` trajectory rows. Artifact: sampled rows.
- [ ] **9.5.3** Prefix cache is cold post-restart — first request latency reflects cold-cache behavior, later requests reflect warm. Artifact: latency-vs-request-index log.

### 9.6 Deprecation migration (§10)

- [ ] **9.6.1** All five superseded LLDs (LLD-01/02/03/04/13) carry a DEPRECATED banner pointing to this HLD. Artifact: grep of `docs/` for the banner string.
- [ ] **9.6.2** `src/lumo_flywheel_serving/` retains day-one behavior until the re-platform sprint lands; a smoke test of the pre-re-platform path still passes. Artifact: CI log.
- [ ] **9.6.3** Rollback plan (§10.4) is documented in a runbook. Artifact: `docs/runbooks/serving-rollback.md` exists and covers the scenarios in §10.4.

---

## 10. Deprecation Migration — LLD-01 / 02 / 03 / 04 / 13

Five LLDs are superseded by this HLD. The goal of this section: produce a migration that does not lose functionality and does not require a flag-day cutover.

### 10.1 What gets a DEPRECATED banner and when

On HLD approval (pre-Sprint 0), each of LLD-01, LLD-02, LLD-03, LLD-04, LLD-13 receives a banner at the top of the doc:

```
> Status: DEPRECATED — superseded by HLD-Serving-Backend-AutoResearch-v0_1.md.
> Retained for historical context and as reference for the <Sprint 3> re-platforming pass.
> Do not start new work against this LLD. New implementations derive from the HLD and the LLD-SB-* series.
```

### 10.2 Reading order for new implementers

1. This HLD (primary).
2. `HLD-Training-Flywheel-LLD06-11-v0_1.md` (for the seams in §1.4).
3. `HLD-Family-Test-Requirements-v0_1.md` (for the family-spec inputs to §5.2).
4. Derived LLD-SB-* series, once drafted.
5. The deprecated LLD-01/02/03/04/13 are context only — useful for understanding the current `src/lumo_flywheel_serving/` code, not for specifying new work.

### 10.3 Code-level migration path

`src/lumo_flywheel_serving/` today (summary per the codebase read):

| File | Role today | Target after re-platform |
|---|---|---|
| `model_server.py` | vLLM launch + request handling | LLD-SB-01; request-plane handler with explicit admission-control seam |
| `inference_proxy.py` | FastAPI proxy + Responses API bridging | LLD-SB-01; hosts new admin endpoints |
| `registry.py` | ModelConfig dataclass | Extended: tuned-config-bundle loader |
| `task_orchestrator.py` | Codex CLI driver | LLD-SB-03; two-class queue + admission control |
| `metrics.py` | vLLM `/metrics` delta-sampling | LLD-SB-02; minor changes for new tagging fields |
| `cli.py` | CLI entry point | Adds `make resume`, `load-tuned-config`, `auto-research run` subcommands |

Migration proceeds file-by-file across sprints; no flag day. Old endpoints remain live until the last caller migrates.

### 10.4 Rollback plan if the new serving backend regresses

At any sprint boundary, rollback is:

1. Revert `src/lumo_flywheel_serving/` to the sprint's tag.
2. Load the last-known-good tuned-config bundle via `/admin/load_tuned_config` (supported by both old and new code paths — the bundle format is forward-compatible; old code ignores unknown fields).
3. If corruption is suspected, `make resume --from-baseline` loads the factory-default bundle (no tuning) and a known-good `weight_version_id`.

Every rollback path is exercised at least once as part of §9.6.3.

---

## 11. Open Questions & Risks

Tracked explicitly so they don't become silent assumptions.

### 11.1 Revisit trigger for model-structure lock (Qwen 3.5 27B → MoE)

Qwen 3.5 27B is locked for v0.1. The MoE roster (35B-A3B, 80B-A3B, 122B-A10B) is out of scope. The revisit trigger is: (a) a measured campaign ceiling on Qwen 27B where concurrency is no longer the bottleneck *and* quality is saturated, or (b) a Training Flywheel outcome indicating MoE quality wins justify the RL / SFT complexity. When either fires, open a v0.2 of this HLD — not a patch.

### 11.2 FP4 serving maturity

vLLM's FP4-on-Blackwell kernel support is a 2026 moving target. Tracked as a risk, not a plan. If it matures during Sprint 1/2, the agent's action space extends naturally — `kv_cache_dtype` in Layer 1 (§5.3.2) gains an `fp4` option, and a new `weight_dtype` dimension lands in Layer 0 (§5.3.1); if FP4 does not mature, FP8 remains the only supported precision.

### 11.3 Coherence with the Training Flywheel HLD's weight-publication contract

The seam at §1.4 / §6 depends on both HLDs agreeing on the path. Risk: a change in one HLD's path without the other. Mitigation: the Training Flywheel HLD cross-references this HLD's §6 (and vice versa); any weight-publication path change requires paired updates.

### 11.4 Thermal throttling under sustained load

GB10 is rated for sustained operation, but the 24/7 campaign posture is unusual for a developer-class device. Tracked risk: thermal throttling erodes the 273 GB/s bandwidth ceiling over hour-long windows. Mitigation: the §9.2.1 post-install benchmark is re-run quarterly; material deviation triggers a §11 entry.

### 11.5 Responses-API drift

Qwen 3.5 supports Responses API natively today. If a future Qwen patch release breaks some corner of the API (tool streaming, parallel tool calls, reasoning-content format), the serving stack will detect it via determinism-check failure (§5.7). Recovery: pin the Qwen checkpoint until the regression is fixed upstream.

### 11.6 Auto-research over-fitting to the synthetic load generator

The agent measures against a synthetic mirror of the eval-workload distribution (§5.6). Because the run is one-time per §5.1, any drift between the mirror and real campaign traffic is baked into the frozen bundle for the lifetime of that `(model, family)` pair. Mitigation: the first hour of the first real campaign using the bundle samples a small measurement window against real traffic and logs the objective gap. A > 10% gap is recorded as a v0.2 prompt — not an automatic re-tune trigger, because automatic re-tuning is explicitly out of scope per §5.1.

### 11.7 OSS dependency pins — Optuna, Triton, CUTLASS, vLLM benchmark harness

The auto-research agent leans on several OSS inner optimizers and harness pieces (Optuna TPE for L1; `@triton.autotune` for L0b on Triton kernels; CUTLASS Python tuner for L0b on CUTLASS GEMM; vLLM `benchmarks/benchmark_serving.py` as the request driver). Each is a version-pinned dependency in `environment_pin` (§5.9). Risk: a minor-release API break in any of these invalidates the bundle's reproducibility — a re-run against a newer Optuna or a newer Triton may produce subtly different winners. Mitigation: bundles record the exact versions they were tuned against; at campaign bootstrap the environment-pin check refuses to load a bundle whose pinned OSS versions do not match the live environment, falling back to the default-config baseline. Upgrading any of these OSS pieces requires a v0.2-style re-tune and a new bundle, not an in-place patch.

### 11.8 Parity-gate tolerance selection

§5.7 rail 9 uses `rtol=1e-3, atol=1e-3` for FP8 forward-pass parity as a default. These numbers are judgment calls, not derived: FP8 itself has ~2–3 decimal digits of precision, so tighter tolerances produce false rejections and looser tolerances let meaningfully-wrong kernels through. Risk: a family whose eval-score is unusually sensitive to logit-tail distribution (e.g., tight greedy-decode tie-breaking, or scorers that hash on top-K probabilities) may need tighter tolerances than the default; conversely, a family that only cares about top-1 token under nucleus sampling can probably tolerate looser. Mitigation: the parity fixture is per-`(model, family)` and the tolerance is bundled into the fixture spec, not hard-coded into the agent. The first campaign using a bundle samples eval-score drift between the reference kernel and any accepted mutation; a > 0.5% drift retroactively tightens the tolerance and triggers a v0.2 prompt.

### 11.9 One-time-run fragility

The one-time framing is a deliberate simplification, not a claim that the bundle is durable forever. Risk: the `(model, family)` pair changes in a way that silently invalidates the bundle (a new vLLM release changes kernel behavior under the same flag names, Qwen 3.5 patches that shift the KV-cache shape, a Triton minor bump changes autotune-winner ordering, etc.). Mitigation: §6.4 bundle-validity rule — every bundle hard-pins vLLM version, kernel-package versions, Triton, CUTLASS, Optuna, CUDA, driver, and the serving-posture invariants (vision-off, Qwen3 reasoning parser, thinking-on). At campaign bootstrap, the serving stack refuses to load a bundle whose pins do not match the live environment, and falls back to the default-config baseline — which is worse than the tuned bundle, but correct. Lifting this into an automatic re-tune is a v0.2 scope item.

### 11.10 Hybrid-attention kernel maturity on Blackwell

Qwen 3.5 27B's hybrid DeltaNet + Gated Attention layout is less battle-tested than pure-attention transformers on Blackwell (SM100+). The Gated Attention path has FA4 as a mature default, but the DeltaNet path depends on Triton kernels (chunked-delta, state-update-fused) whose Blackwell-specific tuning is fresh — GB10 silicon plus Qwen 3.5's exact `(24 Q / 4 KV / head_dim=256)` + DeltaNet `(d_k=128, d_v=256)` shapes are not a hot path in the OSS ecosystem as of v0.1. Risk: (a) a DeltaNet Triton kernel passes determinism checks but silently runs 1.5× slower than its theoretical peak because no one has tuned `CHUNK_SIZE` for Blackwell's exact L1/L2 topology; (b) a vLLM minor release changes the DeltaNet dispatch path and Appendix B's "supported" marks go stale overnight. Mitigation: §7.2 Sprint 1 empirically walks the full DeltaNet Appendix B matrix (PASS/FAIL-streaming/FAIL-determinism) before committing a reference kernel; L0-DeltaNet-c mutation (§5.3.1.c) is the explicit escape valve if stock kernels are bandwidth-underutilizing. Upgrading the DeltaNet kernel package is a §6.4 bundle refusal — it forces a re-tune rather than silent drift.

### 11.11 DeltaNet recurrent-state dtype

§3.12 concurrency math assumes DeltaNet recurrent state lives in fp16 on Blackwell; §5.3.2 kv_cache_dtype table notes fp8 for the recurrent state is an open question. Risk: keeping DeltaNet state in fp16 while compressing GatedAttn KV to fp8 means DeltaNet state is the dominant per-sequence memory line on long contexts (~16 MB per stream at 131k context), tightening the concurrency envelope beyond what §3.12's math already accounts for. Dropping DeltaNet state to fp8 ~halves that footprint but re-opens §5.7 rail 9: the recurrent-state parity tolerance must be revalidated (fp8 recurrent state accumulates error faster than fp8 KV because the state is a continuous product, not a sparse lookup). Mitigation: v0.1 ships DeltaNet state in fp16. A v0.2 prompt is triggered if Sprint 2 measurements show DeltaNet state is the dominant allocator line at target concurrency; in that case, a dedicated `deltanet_state_dtype` action-space entry is added to L0-DeltaNet-a with a tightened per-fixture tolerance.

### 11.12 Vision-tower-off as a launch-time invariant

§4.5 requires `--limit-mm-per-prompt image=0` (or equivalent) to be set at vLLM launch, which disables the Qwen 3.5 vision tower for coding workloads. Risk: a future runbook or operator re-enables the tower (e.g., for a multimodal eval campaign) against a bundle tuned for the vision-off posture — silently changing the per-turn latency envelope and invalidating the bundle's `TTFT_p95` measurements (vision preprocessing adds ~80 ms prefill). Mitigation: §6.4 bundle-validity rule hard-pins `serving_posture_pin.limit_mm_per_prompt_image` to 0; launching without the flag fails the bundle-load check (§9.3.12) and falls back to the default-config baseline. If a multimodal campaign is ever commissioned, it requires a fresh auto-research run with the vision tower enabled — v0.2 scope.

### 11.13 Thinking-token-budget tuning risk

§5.3.2 exposes `thinking_token_budget` as an L1 knob (512 … 16384, default 8192). Risk: the optimizer may converge on a thinking budget that hits the TurnLatency ceiling with almost no headroom, leaving no slack for families whose reasoning quality actually needs more thinking tokens. Mitigation: the family spec's `L_turn` is the ceiling, not a target; the tie-breaker chain in §5.4 favors *lower* TurnLatency_p99 after constraints are met, which pushes the optimizer toward a smaller budget at the same quality. A `thinking_budget_regret` v0.2 prompt fires if post-freeze eval shows budget-capped turns correlate with quality loss above a family-defined threshold. Separately, `thinking_parallel_streams` = true is a §11 open question if the pinned vLLM version does not implement it — the bundle-validity rule ignores it in that case and the serving stack operates as if false.

### 11.14 Bundle-validity rule strictness vs. operational flexibility

§6.4 refuses a bundle on any env or posture mismatch — no soft overrides. Risk: in practice this means `pip install -U` on any transitive dependency of Triton / CUTLASS silently bumps a pinned version and invalidates every previously-tuned bundle, forcing a re-tune. For a single-machine, one-time-run world this is the right trade (strict pinning > silent drift), but the operational surface area of "which dependency bumps force a re-tune" needs to be visible. Mitigation: `environment_pin` includes the full dependency set (§5.9); a pre-upgrade script should dry-run the bundle-validity check against a staged environment before a live `pip install -U` is committed. A v0.2 prompt fires if the observed re-tune frequency exceeds one per calendar quarter per `(model, family)` pair — at that point the pin granularity should be revisited.

---

## Appendix A — Worked Example: Tuning qwen3.5-27b for a V1 Eval Campaign

A single concrete walkthrough so the implementer and verifier agents have something end-to-end to match their code against. This walkthrough reflects the two-family L0 structure (DeltaNet + GatedAttn) and the three-dim SLO conjunction.

**Scenario.** Campaign kickoff for a V1 family (per `HLD-Family-Test-Requirements-v0_1.md`). Model: Qwen 3.5 27B FP8 (dense, hybrid-attention — 16 × (3 DeltaNet + 1 GatedAttn) = 64 decoder blocks per the public Qwen3.5-27B-FP8 model card, vision tower disabled, thinking mode on), `weight_version_id = abc1234`. Family: long-context coding tasks with moderate tool-call frequency. Family SLO from family spec: `L_ttft = 2 s`, `L_tpot = 80 ms`, `L_turn = 30 s`, `thinking_token_budget = 8192`. Expected context-length distribution peaks at 16k, 99th percentile at 48k.

**Inputs the agent receives.** Model checkpoint pinned at `abc1234`. Family spec loaded (including three-dim SLO + thinking budget). Workload distribution sampled from a one-hour seed run of the family's eval set through default-config serving (~2000 requests, thinking-on). Default-config baseline measured. Hardware profile per §3 (hard-coded). Launch-time invariants verified: `--limit-mm-per-prompt image=0`, `--reasoning-parser qwen3`, vLLM 0.19.2 pinned.

**Baseline measurement.** Default vLLM 0.19.2 config with no kernel overrides, thinking-on at budget=8192: `sustained_concurrent_eval_threads = 14` at `TTFT_p95=1.6s`, `TPOT_p95=74ms`, `TurnLatency_p95=28s`, `reasoning_content_purity=1.0`. All three SLO dimensions meet constraints. Regression guard armed.

**Layer 0-DeltaNet — Kernel family A (DeltaNet linear attention).**

*L0-DeltaNet-a (select).* 4 iterations.
- it1: `vllm-default-deltanet` (stock Triton dispatch) — baseline replication. 14 concurrent.
- it2: `triton-chunked-delta` — 15 concurrent, TPOT_p95=70ms. DeltaNet decode-step latency drops ~8%.
- it3: `triton-state-update-fused` — 16 concurrent, TPOT_p95=67ms. Best DeltaNet kernel so far. Also fixes the cross-family `torch_compile_mode=reduce-overhead`, `cuda_graph=decode-only`, `fp8_gemm_kernel=cutlass-warpspecialized` for shared eval.
- it4: `triton-fallback` — regression; reject.
- **Converges on `triton-state-update-fused`.**

*L0-DeltaNet-b (autotune).* 3 iterations.
- it1: `@triton.autotune` over `BLOCK_M ∈ {32, 64, 128}`, `CHUNK_SIZE ∈ {64, 128, 256}`, `num_warps ∈ {4, 8}`, `num_stages ∈ {2, 3}`. Winner: `BLOCK_M=64, CHUNK_SIZE=128, num_warps=8, num_stages=3`. 17 concurrent at TPOT_p95=64ms.
- it2: Narrow search around winner — ~1% improvement, within ε.
- it3: Diminishing returns; converge.
- **DeltaNet reference kernel committed** for L0c parity harness (autotuned `triton-state-update-fused`). Reference logits + end-of-chunk state snapshots at token 1 and 1024 cached.

*L0-DeltaNet-c (mutate).* 5 proposals, 4 iteration slots consumed.
- prop1: Proposer reads DeltaNet trace, notes that the state-update matmul tiles across warps wastefully at `head_dim=256`. Proposes a **re-tiling** patch that reduces warp-level register pressure. Parity gate (both logit + state at token 1 and 1024): all probes pass. Latency: 18 concurrent at TPOT_p95=61ms. Accepted.
- prop2: Proposer fuses the gating activation into the state update. Parity gate: state at token 1024 diverges on channel 47 (`rtol` exceeded by 8e-3 — slow drift confirmed). Rejected without consuming a slot. Proposer told "gate-fusion was the failing class."
- prop3: Proposer proposes larger `num_stages=5` pipeline depth. Parity passes, but no speedup. Regression-rejected.
- prop4: Proposer proposes **fused gate-and-state-update** without the prop2 shortcut (separate accumulator buffer). Parity passes at all checkpoints. Latency: 19 concurrent at TPOT_p95=59ms. Accepted — best DeltaNet so far.
- prop5: No improvement direction. Proposer declares done.
- **L0-DeltaNet converges.** Final DeltaNet winner: `triton-state-update-fused` + autotuned + prop1 + prop4 patches.

**Layer 0-GatedAttn — Kernel family B (standard gated attention, 24Q / 4KV / head_dim=256).**

*L0-GatedAttn-a (select).* 5 iterations — note the backend is *force-selected* per §9.3.13, so `vllm-default` is scored as a baseline and every non-default candidate is scored explicitly.
- it1: `vllm-default` (which on this vLLM 0.19.2 / Blackwell build auto-resolves to `flashinfer` with TRTLLM — first in the Blackwell auto-select priority) — 19 concurrent (DeltaNet winner in place), but the `flashinfer`-on-Qwen3.5 accuracy screen (GitHub vllm-project/vllm #35138) fails two probes in the determinism probe set. Scored as baseline; `vllm-default` is *not* a candidate for the bundled winner.
- it2: `flashinfer` force-selected (same backend as the auto-select resolution, measured independently to confirm the accuracy fingerprint) — same 19 concurrent, same two accuracy-probe failures. Rejected from the action space on accuracy grounds; logged with the #35138 reference.
- it3: `flash-attn-4` (force-selected — the expected winner on SM100+ after FlashInfer is rejected) — 22 concurrent at TPOT_p95=56ms, TTFT_p95=1.3s. Accuracy screen clean. Large jump over baseline.
- it4: `flash-attn-3` — 21 concurrent; §11.10 entry would be required if this were the pick (it is not).
- it5: `triton` — 19 concurrent, regression; reject.
- **Force-selects `flash-attn-4`.** (Matches §9.3.13: non-`vllm-default` explicit pick; accuracy screen clean; the runner-up that would have required §11.10 entry did not win.)

*L0-GatedAttn-b (autotune).* 1 iteration.
- FA4 is a vendor-library kernel — no Triton autotune surface. CUTLASS Python tuner runs against the shared FP8 GEMM for GatedAttn's dominant `(M, N, K)` shapes; selects a tile shape not in CUTLASS's default heuristic for GB10. 23 concurrent at TPOT_p95=54ms.
- **GatedAttn reference kernel committed** for L0c parity harness (FA4 + autotuned shared CUTLASS GEMM). Reference logits cached.

*L0-GatedAttn-c (mutate).* 4 proposals, 3 iteration slots consumed.
- prop1: Proposer proposes a **shared-GEMM split-epilogue** — one epilogue for prefill-heavy shapes, one for decode-heavy. Tagged `family: shared` (CUTLASS). Parity gate runs BOTH fixtures: DeltaNet state-probe (since GatedAttn's shared GEMM is also in DeltaNet's call path) and GatedAttn logit-probe. Both pass. Latency: 24 concurrent at TPOT_p95=52ms. Accepted.
- prop2: Proposer proposes fusing RoPE rotation into the QKV projection for GatedAttn. Parity gate: probe #7 diverges at token 6 (`rtol` exceeded by 2.3e-3 on a single logit). Rejected without consuming a slot; proposer told "GatedAttn RoPE-fusion was the failing class."
- prop3: Proposer proposes a **fused KV-append + attention step** exploiting the 24:4 Q:KV ratio more aggressively. Patch compiles. Parity passes at `rtol=1e-3, atol=1e-3` across all 64 probes. Latency: 24 concurrent at TPOT_p95=51ms. Marginal; accepted on p99 tie-break. Flagged `weight_sensitive: true` — proposer notes the fusion's tolerance margin narrows at smaller weight-scale.
- prop4: Proposer proposes an alt decode-tiling scheme. No improvement; regression-rejected.
- **L0-GatedAttn converges.** Final GatedAttn winner: FA4 + autotuned shared CUTLASS + prop1 (shared-GEMM) + prop3 (GatedAttn-specific, `weight_sensitive: true`).

**L0 totals.** DeltaNet: 4 + 3 + 4 = 11 iterations. GatedAttn: 5 + 1 + 3 = 9 iterations (extra -a iteration for the FlashInfer-accuracy screen per §9.3.13). Combined L0: 20 iterations (plus 2 parity-rejected proposals that didn't consume slots, plus 1 accuracy-rejected FlashInfer selection that is logged but counted as a consumed -a slot since it scored against the workload). Per-family stops fired before hitting the 52-iteration L0 cap. Both families contributed: DeltaNet improved TPOT_p95 by ~15% (74 → 59 ms when GatedAttn was still at stock), GatedAttn FA4 alone was the single biggest lever (moved concurrency 19 → 22).

**Layer 1 — vLLM config (including thinking-budget), L0 frozen.**

- L1-it1: Default vLLM config with the L0 kernel stack applied — 24 concurrent, TTFT_p95=1.2s, TPOT_p95=51ms, TurnLatency_p95=22s. Baseline for L1.
- L1-it2 through L1-it9: Optuna TPE over `max_num_seqs ∈ [16, 72]`, `gpu_memory_utilization ∈ [0.75, 0.92]`, `max_num_batched_tokens ∈ [2048, 12288]`, `thinking_token_budget ∈ [4096, 12288]`. Best at iteration 7: `max_num_seqs=48, gpu_memory_utilization=0.88, max_num_batched_tokens=8192, thinking_token_budget=8192` → 38 concurrent, TPOT_p95=62ms, TurnLatency_p95=26s.
- L1-it10: Add `enable_chunked_prefill=true` (DeltaNet CHUNK_SIZE=128 from L0b autotune — no boundary conflict). 40 concurrent, TPOT_p95=63ms.
- L1-it11: Optimizer tries `thinking_token_budget=10240` — TurnLatency_p95 pushes to 31s, violates SLO, reject.
- L1-it12 through L1-it14: Narrow around winner, diminishing returns.
- **L1 output:** `max_num_seqs=48, gpu_memory_utilization=0.88, max_num_batched_tokens=8192, enable_chunked_prefill=true, enable_prefix_caching=true, max_model_len=49152, kv_cache_dtype=fp8_e5m2, thinking_token_budget=8192, thinking_parallel_streams=false`. 40 concurrent, TTFT_p95=1.4s, TPOT_p95=63ms, TurnLatency_p95=26s, `reasoning_content_purity=1.0`. 14 iterations.

**Layer 2 — Request shaping (L0 + L1 frozen).**

- L2-it1: Default shaping — 40 eval concurrent, but rollout starved at 0.3× baseline. Rollout-floor violated.
- L2-it2: `concurrency_cap_eval=37, concurrency_cap_rollout=3, priority_preemption=rollout-preempts` — passes.
- L2-it3: `concurrency_cap_eval=40, concurrency_cap_rollout=2, admission_queue_depth_max=64, per_request_kv_budget=40k` — passes, rollout at 0.6×. Best so far.
- L2-it4: Re-open trigger — proposer notes with shaping enforcing rollout floor, L1's `max_num_seqs=48` has headroom. Re-opens L1 once (§5.7 rail 2); L1 re-tunes to `max_num_seqs=52`; returns to L2.
- L2-it5 through L2-it7: With L1 re-tune, converge at 44 eval + 3 rollout.
- **L2 output:** `concurrency_cap_eval=44, concurrency_cap_rollout=3, admission_queue_depth_max=64, per_request_kv_budget=40k, priority_preemption=rollout-preempts`. 7 iterations.

**Layer 3 — LoRA (L0 + L1 + L2 frozen).**

- L3-it1: Campaign has no LoRA adapters for V1 family. Trivial: `adapter_mode=static-merged, max_loaded_adapters=0, adapter_eviction_policy=lru, adapter_merge_on_freeze=false`. 1 iteration.
- **L3 converges immediately.**

**Final bundle.** 44 eval + 3 rollout concurrent, TTFT_p95=1.5s, TPOT_p95=65ms, TurnLatency_p95=27s, all three SLO dimensions satisfied, `reasoning_content_purity=1.0`, rollout floor satisfied, determinism-probe 100% pass. Parity attestations recorded for DeltaNet prop1+prop4, GatedAttn prop3 (with `weight_sensitive: true`), shared-GEMM prop1. All safety rails green. Bundle persisted to `/data/tuned_configs/v1-family/abc1234/<run-id>.yaml` with `mutation_artifacts_ref` pointing at four accepted patches + rejected-mutation log. `environment_pin` captures vLLM 0.19.2, Triton, CUTLASS, Optuna, CUDA, driver versions; `serving_posture_pin` captures vision-off, qwen3 reasoning parser, thinking-on. Campaign bootstrap loads via `POST /admin/load_tuned_config` — §6.4 validity rule passes (live env matches pins).

**Totals.** 42 iterations counted (L0 DeltaNet: 11, L0 GatedAttn: 9, L1: 14, L2: 7, L3: 1). Parity-rejected mutations (2 total) did not count per §5.7 rail 9; the FlashInfer accuracy-screen rejection is logged against §9.3.13 and consumed one -a slot. Wall-clock ~5h30m, well within the 8-hour cap. One L1 re-open triggered by L2. Improvement trajectory: baseline 14 → L0-DeltaNet 19 → L0-GatedAttn 24 → L1 40 → L2 44 concurrent. The two-family L0 split earned its keep: GatedAttn alone (the "legacy" pure-attention lever) contributed +5 concurrent over DeltaNet-only; DeltaNet alone contributed +5 over baseline. Combined L0 lift was +10, with L1 the largest single lever (+16) and L2 refining the admission-control boundary (+4).

**What the verifier agent checks.** All §9.3 items against this run's artifacts. The bundle's `layer_0_deltanet`, `layer_0_gatedattn`, `layer_0_shared`, `layer_1_vllm`, `layer_2_shaping`, `layer_3_lora` are the primary artifacts for §9.3.2; the per-family L0 traces under `layer_traces.l0.deltanet` and `layer_traces.l0.gatedattn` are the primary artifact for §9.3.4 (non-default kernel picked in **each** family), §9.3.5 (per-family stops fired at diminishing returns), §9.3.6 (per-family parity gate rejected at least one mutation — DeltaNet prop2 on state-drift, GatedAttn prop2 on logit-divergence), and §9.3.7 (L0c sub-level actually ran and produced accepted-mutation entries per family). §9.3.8 is checked against the TPOT_p95=65ms / TTFT_p95=1.5s / TurnLatency_p95=27s values in the `objective` section. §9.3.9 is checked against `reasoning_content_purity=1.0` and the 8192 thinking-budget honor. §9.3.10/§9.3.11 are exercised via separate env-drift and weight-rotation fixtures rather than this run's artifact.

---

## Appendix B — Kernel Notes for Qwen 3.5 27B

Two kernel families, two matrices. Qwen 3.5 27B's 64 decoder blocks alternate 3 Gated DeltaNet layers per 1 Gated Attention layer (16 × (3 DeltaNet + 1 GatedAttn), per the public Qwen3.5-27B-FP8 model card), so the §5.3.1.a Layer 0a action space is split into two independent selection problems, each with its own menu, autotune surface, and mutation candidates. The cross-family knobs (`torch_compile_mode`, `cuda_graph_capture`, `fp8_gemm_kernel`) and the shared CUTLASS FP8 GEMM apply to both families.

**vLLM version is pinned** per §6.4 — these matrices are accurate for vLLM 0.19.x on CUDA 12.x against GB10 Blackwell (SM100+). Any vLLM minor bump requires re-walking both matrices before a bundle built on the prior pin can be trusted.

### B.1 Gated DeltaNet family — linear-attention kernels (48 of 64 layers, 3/4 of all layers)

DeltaNet has no per-token KV cache; the recurrent state is accumulated across a chunk and updated per chunk. This changes the kernel selection problem: the "attention backend" is a specialized chunk-wise Triton kernel menu rather than FA4/FlashInfer. Only Triton candidates exist today.

| Kernel (`deltanet_kernel`) | FP8 weight GEMM | DeltaNet state dtype | Streaming Responses API | Determinism (scorer-probe passes) | L0b autotune surface | L0c mutation candidate | In action space |
|---|---|---|---|---|---|---|---|
| `triton-chunked-delta` | ✓ | fp16 (fp8 §11.11 open) | ✓ | ✓ | Full — `BLOCK_M`, `BLOCK_N`, `CHUNK_SIZE`, `num_warps`, `num_stages` | **Yes** — full source, mutation via patch-and-rebuild | **Yes** |
| `triton-state-update-fused` | ✓ | fp16 | ✓ | ✓ | Full — `BLOCK_M`, `CHUNK_SIZE`, `num_warps`, `num_stages` (no `BLOCK_N` — state-matmul is fixed-width at `head_dim=256`) | **Yes** — prime mutation target (state-update fusion has unfused headroom) | **Yes** |
| `triton-fallback` | ✓ | fp16 | ✓ | ✓ | Limited — reference implementation, `@triton.autotune` over `BLOCK_M`, `num_warps` only | No (kept as correctness reference) | **Yes** (slowest; fallback only) |
| `vllm-default-deltanet` | ✓ | per vLLM dispatch | ✓ | ✓ | Same as whatever vLLM dispatches to | Transitive | **Yes** (baseline) |
| `flash-linear-attn` (candidate) | — | — | — | — | — | — | **No** (insufficient Qwen 3.5 DeltaNet coverage at v0.1) |

**DeltaNet autotune notes.** `@triton.autotune` owns the inner loop for all three Triton candidates. The `CHUNK_SIZE` knob interacts non-trivially with `enable_chunked_prefill` at L1 — both families share awareness of chunk boundaries. Search-space authoring is per-kernel in LLD-SB-06 and committed alongside the kernel sources.

**DeltaNet mutation notes.** `triton-state-update-fused` is the prime L0c mutation target — the state-update matmul tiles wastefully at `head_dim=256` on GB10, and gate-fusion variants have non-trivial room (see Appendix A L0-DeltaNet-c prop1/prop4). Recurrent-state parity is the hard gate: §5.7 rail 9 checks the state at both token 1 and token 1024 to catch slow drift. A mutation that passes logit-only parity but corrupts state is the highest-risk DeltaNet failure class; the parity fixture is explicitly designed to catch it.

### B.2 Gated Attention family — standard attention kernels (16 of 64 layers, 1/4 of all layers, 24Q / 4KV / head_dim=256)

Standard attention. On Blackwell (SM100+) vLLM's own auto-selection priority is **FlashInfer first, then FlashAttention** (with FA4 as the default FA *version* when FlashAttention is selected) — FA4 is not the default *backend*. FlashInfer on Qwen3.5 on Blackwell carries a known accuracy issue (GitHub vllm-project/vllm #35138). For reproducibility this HLD **does not use vLLM auto-selection**; §5.3.1.a L0-GatedAttn-a force-selects the empirical winner per family spec and the selection is hard-pinned in the bundle (§5.9). The "In action space" column lists candidates that L0-GatedAttn-a may pick among; it does not annotate any one of them as "the default."

Rows are listed in the order vLLM's Blackwell auto-selection would consider them (FlashInfer → FlashAttention → Triton → vllm-default), not in order of expected performance. The expected winner on this hardware is FA4 after accuracy screening, but that must be measured at L0-GatedAttn-a and pinned in the bundle — not assumed.

| Kernel (`attention_backend`) | FP8 weight GEMM | FP8 KV cache | Paged attention | Streaming Responses API | Determinism (scorer-probe passes) | L0b autotune surface | L0c mutation candidate | In action space |
|---|---|---|---|---|---|---|---|---|
| `flashinfer` | ✓ | ✓ | ✓ (superior paged-KV for mixed prefill/decode; uses TRTLLM on Blackwell) | ✓ | ⚠ known accuracy issue on Qwen3.5 on Blackwell (GitHub vllm-project/vllm #35138) — determinism check MUST be re-verified at L0-GatedAttn-a before this kernel is picked | Partial — Triton fused ops it delegates to are autotunable via `@triton.autotune` | Partial — delegated Triton ops are mutation candidates; core attention kernel is not | **Yes** (first candidate in vLLM auto-select priority on SM100+; blocked from acceptance until #35138 is screened on the exact workload) |
| `flash-attn-4` | ✓ | ✓ (fp8_e5m2) | ✓ | ✓ | ✓ (default FA *version* on SM100+ when FA is selected) | None (library-owned — Blackwell-specific FA4) | No (vendor-owned binary) | **Yes** (expected winner on SM100+ after the FlashInfer accuracy screen; must be explicitly force-selected) |
| `flash-attn-3` | ✓ | ✓ | ✓ | ✓ | ✓ | None (library-owned) | No (vendor-owned binary) | **Yes** (fallback if FA4 fails acceptance; requires §11.10 entry if picked on Blackwell) |
| `flash-attn-2` | ✓ | ✓ | ✓ | ✓ | ✓ | None (library-owned) | No | **Yes** (pre-Hopper fallback; §11.10 entry required if picked on Blackwell) |
| `triton` | ✓ | ✓ | ✓ | ✓ | ✓ | Full — `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, `num_stages` | **Yes** — full source, mutation via patch-and-rebuild | **Yes** (slower than FA4 on GB10; kept for fallback and as the secondary GatedAttn L0c target) |
| `vllm-default` | ✓ | ✓ | ✓ | ✓ | ✓ | Same as whatever vLLM dispatches to | Transitive | **Yes as a measurement baseline only** — force-select is mandatory for the bundled winner (§5.9), so `vllm-default` is never the committed value in a shipped bundle; it is the row the baseline regression guard in §5.7 compares against |
| `xformers` | partial | — | — | — | — | — | — | **No** (insufficient FP8 coverage on Blackwell at v0.1) |

**GatedAttn autotune notes.** Triton candidates use `@triton.autotune`. FA4/FA3/FlashInfer are vendor-library; L0b is a no-op for those (`l0b_autotune` records `{vendor_owned: true}`).

**GatedAttn mutation notes.** RoPE-fusion into QKV projection is a classic target but has historically failed parity (single-logit divergence by token 6 — see Appendix A L0-GatedAttn-c prop2). Fused KV-append + attention-step exploiting the 24:4 Q:KV ratio is a lower-risk direction. Mutations should be proposed with `weight_sensitive: true` when their tolerance margin is small enough that a different weight scale could push a passing parity into failing — see §5.3.1.c and §6.4 re-parity-check path.

### B.3 Shared / cross-family knobs

These apply uniformly across both families — a single selection governs both the DeltaNet layers and the GatedAttn layers.

| Knob | Options | L0b surface | L0c mutation | Notes |
|---|---|---|---|---|
| `torch_compile_mode` | `off`, `default`, `reduce-overhead`, `max-autotune` | — | — | `max-autotune` has long compile warm-up that may clash with the 5-minute harness warm-up window; agent may pick it but should prefer `reduce-overhead` by default. |
| `cuda_graph_capture` | `off`, `decode-only`, `full` | — | — | `full` captures both prefill and decode; expensive to rebuild on config change. `decode-only` is usually right. |
| `fp8_gemm_kernel` | `cutlass-default`, `cutlass-warpspecialized` | Tile shape, mainloop pipeline depth, epilogue schedule (via CUTLASS Python tuner) | **Yes** — CUTLASS template variants can be re-instantiated with novel schedules; a shared-GEMM mutation must pass **both** family fixtures (DeltaNet state + GatedAttn logit) per §5.7 rail 9 | Warp-specialized is Blackwell-preferred for FP8 dense GEMM per NVIDIA guidance. |

**L0b autotune note (shared CUTLASS).** Autotune search spaces for the shared FP8 GEMM are authored per-`(M, N, K)`-shape-bucket (prefill-heavy and decode-heavy separately) and recorded in `layer_0_shared.cutlass_autotune` of the bundle.

**L0c mutation note (shared CUTLASS).** Shared-GEMM mutations are tagged `family: shared` and must pass both the DeltaNet state-aware fixture *and* the GatedAttn logit fixture before their latency is scored. Shared mutations are scored on the combined objective — they must beat the reference on the union of both families' latency contributions, not just one. See Appendix A L0-GatedAttn-c prop1 for a shared-GEMM example.

### B.4 Exclusions and re-entry

Kernels excluded from the action space for v0.1 — and the reason each is excluded — are listed above (`xformers`, `flash-linear-attn`). Re-entering a kernel requires a PR against this appendix plus §9.3.4-style verification that the kernel passes streaming and determinism checks on the campaign workload, in the appropriate family.

**General L0c mutation principle.** Triton kernels are the primary mutation target because the source is in-tree and a proposer diff can be compiled and parity-tested in minutes. CUTLASS template re-instantiations (novel schedules, non-default epilogues) are the secondary target; these fit the §5.7 rail 1 20-minute per-mutation cap. Vendor-library kernels (FA4, FA3, FlashInfer core) are not mutation candidates — we do not fork them in this HLD; if a performance wall hits at that layer, the escape hatch is to ship a Triton replacement in the same family and have L0-F-select pick it, not to patch the vendor library. The per-family parity harness (§5.7 rail 9) uses each family's L0-F-select + L0-F-autotune winner at commit time (end of Sprint 1) as that family's reference — any mutation is judged against its family's reference, not against `vllm-default`.

---
