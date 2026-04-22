# HLD · Serving Backend & Auto-Research Optimization Loop

> Codex-Bench · High-Level Design
> Scope: The serving-and-optimization stack that runs on a single DGX Spark — vLLM-based serving, kernel-level tuning, request shaping, an auto-research agent that optimizes serving config offline before each campaign, and the weight-update path that connects online RL / SFT to live serving.
> Status: DRAFT v0.1 — supersedes LLD-01 (vLLM Serving Layer), LLD-02 (Data Pool Manager), LLD-03 (Task Orchestrator), LLD-04 (Latency Telemetry Capture), LLD-13 (Codex-Long Scenario Framework). Those five LLDs are retained in `docs/` with DEPRECATED banners once this HLD is approved; implementation in `src/lumo_flywheel_serving/` continues to run until the re-platforming pass derived from this HLD lands.
> Target hardware: 1× NVIDIA DGX Spark (GB10 Superchip) — no second machine, no cluster assumption anywhere in this doc.
> Target model: [Qwen 3.5 27B](https://huggingface.co/Qwen/Qwen3.5-27B), served from FP8 weights. Locked as the single model in scope for this HLD. Model-structure alternatives (MoE variants in `model_registry.yaml`) are explicitly out of scope; their rows appear in §3.10 as envelope reference only. Revisit trigger in §11 open questions.
> Responses API: Qwen 3.5 supports the OpenAI-compatible Responses API natively via vLLM's `/v1/responses` endpoint; no separate compatibility gate is required. Any future model added to the roster must re-open this decision.
> Audience: coding-agent implementer + verifier pair. Every load-bearing requirement lands in §9's binary pass/fail checklist.

---

## Changelog

| Version | Change |
|---|---|
| v0.1 | Initial draft. Model locked to Qwen 3.5 27B (FP8). Establishes hardware envelope for single DGX Spark, the serving-backend contract, the auto-research agent design (action space: vLLM config, request shaping, kernel selection, LoRA management), the weight-update path (PyTorch writes → KV invalidate → resume), the machine-restart resumability contract, the pass/fail verification checklist for the implementer + verifier agent pair, and the deprecation migration plan for LLD-01 / 02 / 03 / 04 / 13. Explicitly out of scope: Responses-API compatibility gating (Qwen 3.5 supports it natively) and model-structure alternatives (single-model lock per above). |

---

## 1. Purpose & Scope

This HLD defines what a single DGX Spark must deliver to run the Codex-Bench training flywheel end-to-end — fast enough that the bottleneck becomes the model, not the serving stack. It is the architectural parent of every LLD in the serving + orchestration path going forward.

### 1.1 What is in scope

- The vLLM serving layer that exposes Qwen 3.5 27B (FP8) via an OpenAI-compatible `/v1/responses` endpoint to Codex threads.
- The auto-research agent that tunes the serving layer offline, pre-campaign, per `(model, family, eval-workload)` tuple — producing a frozen tuned-config bundle consumed at campaign time.
- The auto-research action space: vLLM config knobs, request shaping (concurrency cap, admission control, per-request KV budget), attention-kernel selection, and LoRA merging / adapter management.
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
2. **Model is locked to Qwen 3.5 27B (FP8).** Single model in scope for this HLD. The 27 GB weight footprint, FP8 precision, dense (non-MoE) architecture, and native Responses-API support are all load-bearing assumptions that ripple into §4 (backend contract), §5 (auto-research agent action space), §6 (weight-update path), and §9 (verification checklist). Re-evaluation trigger in §11.1.
3. **Auto-research is offline and pre-campaign.** Not continuous, not during-training. Runs once per `(model, family, eval-workload)` tuple, outputs a frozen tuned-config bundle consumed by the serving stack during the campaign. This is the Karpathy closed-loop research-agent framing: propose a config → measure on the fixed workload → iterate → produce a locked artifact. Cadence detail in §5.1.
4. **Weight-update path is intentionally trivial.** PyTorch writes model weights → KV cache invalidated → next request served against fresh weights. No hot-swap-while-preserving-KV, no partial re-shard, no gradient-aware KV rebuild. Simplicity is the feature; see §6.
5. **Kernel selection is a first-class action-space dimension for the auto-research agent.** Attention backend (FA-2 / FA-3 / FlashInfer / Triton / vLLM-default), `torch.compile` settings, and CUDA / Triton variants are all things the agent can tune within a pinned compatibility matrix — see §5.3.3 and Appendix B. Kernels that demonstrably break Responses-API streaming or determinism are excluded from the action space, not tuned against.
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

**273 GB/s is the single number that shapes every optimization decision downstream.** For autoregressive decode, each generated token requires reading the full set of active parameters from memory exactly once. The ceiling on per-stream decode throughput is:

```
tokens/sec ≤ bandwidth / active_param_bytes_read_per_token
```

At FP8 (1 byte per param) for Qwen 27B dense: `273 GB/s / 27 GB ≈ 10.1 tokens/sec per single stream` — memory-bound, single decode, no batching.

Batching and paged KV amortize this: N concurrent streams that share weight reads approach `273 / (27 + N × KV_per_step)` — single-digit milliseconds amortized per stream at realistic N. In practice KV contention, prefill interleaving, and decode-vs-prefill scheduling all eat into the ideal. §5.6 (measurement harness) is what distinguishes "the model is bandwidth-bound as expected" from "something else is slowing it down."

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

| Model | Total params | FP8 weight | Headroom (KV + activations) | Verdict |
|---|---|---|---|---|
| **Qwen 3.5 27B dense (locked)** | **27 B** | **~27 GB** | **~101 GB** | **Fits very comfortably** |
| Qwen 3.5 35B-A3B MoE | 35 B total, 3 B active | ~35 GB | ~93 GB | Fits (reference only; not in scope) |
| Qwen 3 Coder Next 80B-A3B MoE | 80 B total, 3 B active | ~80 GB | ~48 GB | Fits tight (reference only; not in scope) |
| Qwen 3.5 122B-A10B MoE | 122 B total, 10 B active | ~122 GB | ~6 GB | Does not fit at FP8 (reference only; not in scope) |

Numbers are weight-footprint only; activations + workspace + CUDA graphs + PyTorch overhead consume an additional ~4-8 GB.

### 3.11 KV-cache-budget math (concrete)

For Qwen 3.5 27B at FP8 KV cache, per-token KV footprint ≈ `2 × num_layers × num_kv_heads × head_dim × bytes`. Using approximate Qwen 27B architecture (64 layers × 8 KV heads × 128 head dim × 1 byte): **~128 KB per token**. At 131,072 max-context, one fully-filled sequence is ~16 GB of KV cache alone.

Implication: with 27B weights resident (~27 GB) + OS / CUDA / activations (~8 GB), ~93 GB remains for KV. Theoretical ceiling of ~5 concurrent fully-filled 131k-context streams — but real workloads never run max-context. With `max_model_len` tuned to campaign needs (say, 32k average), effective concurrency lands in the ~15-40 concurrent Codex-thread range. The exact number is what the §5 auto-research agent is for.

### 3.12 Concurrency-ceiling math

Decode latency per step on GB10 is dominated by memory bandwidth. For a dense model serving N concurrent streams, each doing one decode step:

```
decode_step_latency ≈ (active_param_bytes + N × KV_bytes_per_step) / bandwidth
                    ≈ (27 GB + N × ~64 KB) / 273 GB/s
                    ≈ ~99 ms + N × negligible
```

Single-stream decode is ~99 ms per token (~10 tok / s). Adding concurrent streams is **almost free** until KV pressure or prefill-vs-decode scheduling bites. That is the central fact the auto-research agent is exploiting: GB10 is very good at concurrent decode for small-to-medium dense models, and the optimization target is finding the `N` at which something other than bandwidth starts dominating.

### 3.13 Cold-start and weight-loading costs

| Action | Source → target | Approx. wall-clock |
|---|---|---|
| Load 27 GB FP8 weights (Qwen 27B) | Internal NVMe → unified memory | ~9 s |
| Load 27 GB FP8 weights | External NVMe → unified memory | ~27 s |
| Swap LoRA adapter (~200 MB) | Internal → unified memory | < 1 s |
| Full cold restart (CUDA init + vLLM warmup + first-request compile) | — | ~45-60 s for 27B dense |

**Policy.** Cold restart is acceptable as an occasional event (< 1 per hour steady-state). Frequent restarts would dominate campaign wall-clock — which is why §6's weight-update path is "PyTorch writes → KV invalidate → resume," not "cold restart per weight update."

### 3.14 Implications for training

- **QLoRA SFT on Qwen 27B FP8.** Fits cleanly — base weights (~27 GB) + LoRA adapters (< 1 GB) + optimizer state + gradients well within 128 GB.
- **Full SFT on Qwen 27B.** Does not fit — FP16 master weights (~54 GB) + Adam optimizer state (~108 GB) + gradients exceed 128 GB. Blocked without FSDP, which requires a cluster we do not have.
- **DAPO RL on 27B dense.** Fits via QLoRA. Policy + reference co-residency would not fit at full FT — the Training Flywheel HLD prescribes reference-free DPO-style losses or offloaded reference.

### 3.15 Summary — GB10 as a memory-bandwidth-bound decode box

One sentence to carry through the rest of this HLD: **DGX Spark is a memory-bandwidth-bound decode box with unified memory, and every serving optimization must honor that constraint.**

That framing drives §4 (the serving-backend contract makes bandwidth-saturation the throughput target, not kernel utilization), §5 (the auto-research agent is searching for the config where bandwidth is the only remaining bottleneck — not kernel launch overhead, not request admission, not KV pressure, not determinism-check overhead), and §6 (weight-update cost is measured in bandwidth cycles, not re-sharding complexity).

---

## 4. Serving Backend Contract

What the serving stack must deliver, independent of how the §5 auto-research agent happens to have tuned it. These are the invariants downstream LLDs code against.

### 4.1 Concurrency and throughput targets

- **Sustained concurrency.** The serving layer must support `N_concurrent` simultaneous Codex threads, where `N_concurrent` is the ceiling discovered by the §5 auto-research agent for the `(Qwen 3.5 27B, family, eval-workload)` tuple. Typical expected range per §3.11 / §3.12 math: **15–40 concurrent threads** at realistic context lengths.
- **Throughput target.** Saturate the 273 GB/s memory-bandwidth ceiling (§3.3). The operational definition: sustained aggregate decode throughput across all active streams reaches ≥ 80% of the bandwidth-bound theoretical ceiling for the observed active-param byte count. Anything below 80% means something non-bandwidth is dominating; §5 treats that as a signal to keep searching.
- **Per-thread latency ceiling `L_ceiling`.** A configurable ceiling on per-turn p95 latency, sourced from the campaign's family spec (not the serving layer). Serving must never sacrifice `L_ceiling` compliance for throughput — i.e., the objective in §5.4 is concurrency *subject to* `L_ceiling`, not throughput at any latency.
- **Admission control.** When the active stream count would push the next request past `L_ceiling` at its requested context length, the request is queued (not dropped) and a queue-depth metric is emitted. §5's measurement harness uses queue depth as a primary signal.

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
- **Back-pressure, not silent drop.** Requests received while a weight update is in progress are queued and replayed on resume, subject to `L_ceiling`-aware admission control.

### 4.3 Eval-and-train-on-the-fly contract

Two distinct request classes share the serving process:

| Class | Source | Treatment |
|---|---|---|
| **Eval** | Benchmark runner + Codex threads executing family tasks | Higher priority. Latency-sensitive. Counts against `N_concurrent` ceiling. Tagged `class=eval`, `family_id`, `variant`. |
| **Rollout** | Online-RL rollout generator | Lower priority. Throughput-sensitive. May be preempted or queued when eval concurrency is high. Tagged `class=rollout`. |

Both classes hit the **same** `/v1/responses` endpoint and share weights + KV cache. The priority separation is an admission-control layer sitting in front of vLLM, not a second server. Reproducibility: every response is tagged with `weight_version_id`, `tuned_config_id`, `class`, and the LLD-04 telemetry row ID. The auto-research agent measures both classes but optimizes primarily against the eval-class objective, with a soft constraint that rollout throughput does not collapse (§5.4).

### 4.4 Request-lifecycle and interrupt semantics

- **In-flight request on weight update.** Default: **finish-at-current-weights**. Rationale: partial responses tagged with a stale `weight_version_id` are simpler than mid-decode replay. Eval runs that need weight-consistent responses must issue all requests after a weight-update barrier.
- **In-flight request on server restart.** Terminated with an `integrity_flag = 1` trajectory row (per Training Flywheel HLD §3.1) and retry semantics handled by the caller. The serving layer does not implement in-flight request durability.
- **Request cancellation.** Clients may cancel in-flight requests; cancellation frees the KV slot eagerly. The measurement harness verifies cancellation does not leak KV.

### 4.5 Integration surface

The serving stack presents exactly one HTTP surface and one control plane:

| Surface | Endpoint | Consumer |
|---|---|---|
| Request plane | `POST /v1/responses` (OpenAI-compatible Responses API, native Qwen 3.5) | Codex threads via benchmark runner; online-RL rollout generator |
| Request plane | `GET /v1/models` | Health + model introspection |
| Telemetry plane | `GET /metrics` (Prometheus-format, per LLD-04) | Auto-research measurement harness; campaign dashboards |
| Control plane | `POST /admin/invalidate` | Training process (invokes weight-update flow) |
| Control plane | `POST /admin/load_tuned_config` | Campaign bootstrap (loads the frozen tuned-config bundle from §5.9) |

Code-level mapping to the existing `src/lumo_flywheel_serving/` tree — `model_server.py` owns the request plane, `metrics.py` owns the telemetry plane, new admin endpoints land in `inference_proxy.py` — is detailed in §10's migration plan.

---

## 5. Auto-Research Agent — Design

The core of this HLD. Karpathy-style closed-loop research agent: propose → measure → iterate → freeze. Operates offline, pre-campaign, once per `(model, family, eval-workload)` tuple, and produces a frozen tuned-config bundle that the campaign's serving stack consumes unchanged.

### 5.1 When the agent runs

- **Trigger.** Campaign kickoff, before any eval or training request hits the serving stack with the new configuration. Also triggered when a new family is introduced, when the eval-workload distribution changes materially, or when a kernel / vLLM version bump invalidates the current tuned-config bundle.
- **Cadence.** Not continuous. Not during training. A single campaign may run hundreds of training steps and tens of thousands of eval requests against one frozen tuned-config bundle; the agent is re-invoked only on the triggers above.
- **Duration.** Bounded by §5.8 stopping criterion — a wall-clock cap (default: 4 hours) and a diminishing-returns threshold. A typical run should complete within a single overnight slot.
- **Isolation.** The agent runs on the same DGX Spark as production serving, but in a dedicated tuning slot (no eval / RL traffic). Cold-start + per-iteration reset costs are budgeted in §5.7.

### 5.2 Inputs and outputs

**Inputs** (all frozen for the duration of the run):

- **Model checkpoint.** Qwen 3.5 27B FP8 weights at a specific `weight_version_id`.
- **Family spec.** The benchmark family (or training-family proxy) the tuned bundle is for. Defines `L_ceiling`, expected context-length distribution, tool-call pattern, turn-count distribution.
- **Eval-workload distribution.** Empirical or model-derived distribution of request shapes (context length, output length, tool-call frequency) representative of the campaign. Obtained either from a prior campaign's telemetry (hot start) or from a small seed run (cold start).
- **Hardware profile.** The GB10 envelope from §3 is hard-coded; the agent does not rediscover hardware limits each run.
- **Baseline config.** The current best tuned-config bundle, if one exists. The agent's regression guard (§5.7) requires the new run to match or beat the baseline on the stated objective.

**Outputs:**

- **Tuned-config bundle** (§5.9) — a versioned YAML artifact containing the selected vLLM flags, kernel choices, request-shaping parameters, LoRA-management policy, the objective value achieved, and pointers to the measurement traces that justify it.
- **Measurement traces** — raw LLD-04 `/metrics` deltas per iteration, stored alongside the bundle for post-hoc analysis.
- **Search trace** — the sequence of proposed configurations and their measured objective values; feeds the next run's hot-start.

### 5.3 Action space

The action space is deliberately small and strictly typed. Every dimension has a reason-to-tune documented here; every dimension that isn't listed is excluded from the agent's search.

#### 5.3.1 vLLM config knobs

| Knob | Type | Range | Why it matters |
|---|---|---|---|
| `max_num_seqs` | int | 4 … 64 | The N in §3.12 concurrency math. Directly sets the concurrent-stream ceiling. |
| `max_num_batched_tokens` | int | 512 … 16384 | Per-step batching budget; controls prefill / decode interleaving. |
| `enable_chunked_prefill` | bool | {false, true} | Chunked prefill smooths tail latency under prefill-heavy traffic. |
| `enable_prefix_caching` | bool | {false, true} | With long shared prompts (family-scoped system messages), near-always-true — but verified empirically per family. |
| `gpu_memory_utilization` | float | 0.70 … 0.95 | Trades KV headroom for concurrency. Upper bound clamped to leave §6 weight-update headroom. |
| `max_model_len` | int | 8192 … 131072 | Clamped to the family's 99th-percentile context length plus margin. |
| `kv_cache_dtype` | enum | {fp8_e5m2} | Locked to fp8_e5m2 per current `model_registry.yaml`; in the action space for future FP4 exploration (§11.2). |

#### 5.3.2 Request shaping

| Knob | Type | Range | Why it matters |
|---|---|---|---|
| `concurrency_cap_eval` | int | 1 … `max_num_seqs` | Hard ceiling on concurrent eval-class streams. Decoupled from vLLM's internal batcher so the admission layer can enforce family-specific bounds. |
| `concurrency_cap_rollout` | int | 0 … `max_num_seqs` | Ceiling on rollout-class streams. `eval + rollout ≤ max_num_seqs`. |
| `admission_queue_depth_max` | int | 0 … 512 | Beyond this, requests reject-with-retry rather than queue. |
| `per_request_kv_budget` | int (tokens) | `max_model_len / 4` … `max_model_len` | Per-request KV pre-reservation cap. Prevents one long-context request from starving N short-context requests. |
| `priority_preemption` | enum | {off, rollout-preempts, strict} | How aggressive the admission layer is about preempting rollout-class for eval-class. |

#### 5.3.3 Kernel selection

Constrained by Appendix B's compatibility matrix. The agent may pick any (row, column) that Appendix B marks **supported**; anything marked *experimental* or *blocked* is out of the action space for v0.1.

| Knob | Type | Options |
|---|---|---|
| `attention_backend` | enum | {flash-attn-2, flash-attn-3, flashinfer, triton, vllm-default} |
| `torch_compile_mode` | enum | {off, default, reduce-overhead, max-autotune} |
| `cuda_graph_capture` | enum | {off, decode-only, full} |
| `fp8_gemm_kernel` | enum | {cutlass-default, cutlass-warpspecialized} |

Kernels that have demonstrably broken Responses-API streaming or determinism are not candidates — they are removed from Appendix B, not left in the action space with a penalty.

#### 5.3.4 LoRA merging strategy / adapter management

| Knob | Type | Options | Why it matters |
|---|---|---|---|
| `adapter_mode` | enum | {static-merged, runtime-apply, hot-swap} | Merged adapters are faster to serve but expensive to rotate; runtime-apply is the inverse; hot-swap is a middle ground. |
| `max_loaded_adapters` | int | 0 … 8 | Resident adapter count. Each adapter adds ~200 MB (§3.13). |
| `adapter_eviction_policy` | enum | {lru, manual, pinned-set} | Which adapter leaves when `max_loaded_adapters` is hit. |
| `adapter_merge_on_freeze` | bool | {false, true} | Whether a "final" adapter should be merged into the base weights at campaign freeze, collapsing adapter cost to zero on subsequent eval runs. |

### 5.4 Objective function

**Primary objective:** maximize `sustained_concurrent_eval_threads_at_L_ceiling` over a measurement window `W` (default: 30 minutes of synthetic campaign traffic).

**Subject to hard constraints:**

1. `p95_turn_latency_eval ≤ L_ceiling` from the family spec.
2. `scorer_determinism_check_pass_rate ≥ 99.9%` across the measurement window (§5.7 details).
3. `rollout_class_throughput ≥ 0.5 × rollout_baseline` — a floor to prevent the optimizer from eliminating rollout traffic.
4. `no_oom_events` in the measurement window.
5. `weight_update_invalidate_to_ready_time ≤ 15 s` — exercised at least twice during the window.

**Tie-breakers** (when multiple configs meet all constraints at the same concurrency level): lower p99 latency → higher prefix-cache hit rate → lower rollout-throughput impact.

The objective is deliberately *not* "maximize throughput." Throughput with no latency bound collapses to "batch everything, return slowly," which is wrong for an interactive Codex-thread workload.

### 5.5 Search strategy

The agent uses a **two-tier propose → measure → iterate** loop:

1. **LLM-in-the-loop proposer (outer loop).** A Codex-driven research agent reads the prior iteration's measurement trace, explains the observed bottleneck in natural language, and proposes a configuration delta with justification. This matches Karpathy's framing: the agent is *reasoning* about the measurement, not running a blind sweep. The LLM proposer has access to Appendix B's compatibility matrix, §3's hardware envelope, and this HLD.
2. **Bandit / Bayesian-opt executor (inner loop).** Inside each proposed configuration region, a Bayesian-opt routine (Gaussian process over the continuous knobs; bandit over the discrete ones) fine-tunes numeric knobs (`max_num_seqs`, `gpu_memory_utilization`, `max_num_batched_tokens`) within the region. Default implementation: `optuna` with a TPE sampler.

The outer loop caps the number of configuration regions explored per run (default: 12). The inner loop caps per-region evaluations (default: 8). Total evaluation count is therefore bounded at 96 × per-iteration cost — which, at ~2 minutes of synthetic traffic per iteration, fits within the 4-hour wall-clock cap.

**Rationale for this split.** Pure Bayesian opt over the full action space wastes iterations on locally-poor regions; pure LLM proposing is unreliable on continuous numerics. The two-tier split gives the LLM agency where reasoning helps (discrete kernel choices, shaping policies, regime detection) and the numeric optimizer where it helps (smooth continuous knobs).

### 5.6 Measurement harness

- **Telemetry substrate.** LLD-04 vLLM `/metrics` delta-sampling. No new telemetry primitives invented in this HLD. The measurement harness consumes TTFT, throughput, cache-hit rate, per-turn latency, queue depth, and active-sequence count.
- **Synthetic load generator.** A driver that mirrors the eval-workload distribution input (§5.2): samples context lengths, output lengths, tool-call patterns, and turn counts from the empirical distribution, and issues requests at a target concurrency level. Must be deterministic under a seed so iterations are comparable.
- **Measurement window.** Default 30 minutes per iteration: 5 minutes warm-up (discarded), 25 minutes steady-state (scored). Shorter windows are allowed for early-iteration screening; the final top-K are always scored at the full window.
- **Per-iteration reset.** Between iterations the vLLM process is restarted with the new config (cold restart per §3.13 is budgeted). The prefix cache intentionally starts cold — warm-prefix behavior is measured *within* the iteration, not carried across iterations.

### 5.7 Safety rails

The agent runs unattended; these rails keep it from producing bad configs or breaking the machine.

1. **Compute-budget cap per run.** Wall-clock hard limit (default 4 hours); iteration-count hard limit (default 96); per-iteration hard limit (default 40 minutes).
2. **Regression guard against the current best config.** The baseline tuned-config bundle is re-measured on the synthetic driver at the start of each run. The new run's output must match or beat the baseline on the §5.4 objective, or the run aborts and the baseline is retained. No silent downgrade.
3. **Scorer-determinism check.** A held-out deterministic-scorer probe set is run every K iterations (default K=4). If scoring determinism drops below 99.9% agreement with the baseline run, the iteration is marked non-deterministic and its configuration is added to a deny-list for the remainder of the run.
4. **OOM handling.** Per-iteration OOM events abort the iteration, record the crash, mark the configuration `infeasible`, and continue. Three OOMs in a row abort the entire run.
5. **KV-cache-poisoning detection.** A probe request with known input and known expected output is issued at the start and end of each iteration. If the two responses diverge, the iteration is marked suspect and its measurement discarded.
6. **Rollback path.** On any unrecoverable failure, the serving stack returns to the last-known-good tuned-config bundle via the `/admin/load_tuned_config` control-plane endpoint. This is the same path §3.8 machine-restart resumability uses.

### 5.8 Stopping criterion

The run stops when any of the following fire first:

- **Wall-clock cap.** Default 4 hours.
- **Iteration cap.** Default 96 total inner-loop evaluations (12 regions × 8 per region).
- **Diminishing returns.** Over the last N=8 iterations, the best-so-far objective improved by less than `ε = 2%` — signal that the search has converged.
- **Hard infeasibility.** Three consecutive OOMs or three consecutive determinism-check failures — the machine or the search region is unhealthy, abort and retain baseline.

At stop, the best feasible configuration is frozen as the output bundle.

### 5.9 Output artifact — the tuned-config bundle

A single YAML blob persisted to `/data/tuned_configs/<family_id>/<weight_version_id>/<run_id>.yaml` on internal NVMe. Schema:

```yaml
tuned_config_bundle:
  bundle_id: <uuid>
  produced_at: <iso8601>
  weight_version_id: <sha>
  model_id: qwen3.5-27b
  family_id: <family>
  workload_distribution_id: <hash>
  vllm_config:       { ... §5.3.1 knobs ... }
  request_shaping:   { ... §5.3.2 knobs ... }
  kernel_selection:  { ... §5.3.3 knobs ... }
  lora_policy:       { ... §5.3.4 knobs ... }
  objective:
    metric: sustained_concurrent_eval_threads_at_L_ceiling
    value: <int>
    L_ceiling_ms: <int>
    measurement_window_minutes: <int>
  measurement_trace_ref: <path to raw trace>
  search_trace_ref:      <path to search trace>
  baseline_bundle_id:    <uuid | null>
  regression_guard:      { baseline_value: <int>, delta: <int> }
  safety_rails:          { ... boolean attestations per §5.7 ... }
```

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

### 6.4 Interaction with prefix caching and LoRA adapters

- **Prefix cache.** Keyed on `(weight_version_id, prompt_prefix_hash)`. On weight update, all entries with the old `weight_version_id` are invalidated. The next request after an update is always a cold-prefix request.
- **LoRA adapters.** Swapping a LoRA adapter does *not* invalidate base-weight KV. LoRA adapters are applied per-request (per §5.3.4), so the base-weight KV remains valid. A weight update of the *base* model, however, invalidates all adapter-augmented KV as well.
- **Tuned-config bundle.** A weight update does **not** automatically re-trigger §5 auto-research. The bundle carries the `weight_version_id` it was tuned against; if the new weights are in the same family and scale, the bundle is carried forward. If `model_registry.yaml` moves to a new model entirely, auto-research is re-triggered per §5.1.

---

## 7. Sequencing Across Sprints

Four-sprint delivery plan. Earlier sprints land infrastructure the later sprints need; no sprint is blocked on a sibling-HLD deliverable.

### 7.1 Sprint 0 — auto-research scaffold + first tuned config for qwen3.5-27b

- Stand up the measurement harness (§5.6) against the existing `src/lumo_flywheel_serving/` implementation (LLD-01/04-era code is adequate substrate).
- Implement the §5.3.1 vLLM-config-knobs action space only; skip request shaping, kernel selection, and LoRA management for now.
- Implement §5.7 safety rails (all of them — they are cheap and necessary from day one).
- Produce the first tuned-config bundle for Qwen 3.5 27B against a V1 family's workload.
- Deliverable: §9 verification checklist items §9.1 (contract endpoints), §9.3.1, §9.3.2, §9.3.3 (auto-research runs + produces bundle + safety rails) pass. §9.5 resumability also lands in Sprint 0 since it is cheap to implement.

### 7.2 Sprint 1 — kernel-selection action-space enabled

- Populate Appendix B kernel compatibility matrix for Qwen 3.5 27B (empirically — attempt each cell, record PASS / FAIL-streaming / FAIL-determinism).
- Enable §5.3.3 kernel action-space dimensions in the agent.
- Re-run auto-research for the V1 family with the expanded action space; expect modest concurrency gains.
- Deliverable: §9.3.4 (kernel action space produces non-default bundle) and §9.3.5 (stopping criterion fires) pass.

### 7.3 Sprint 2 — LoRA adapter management + weight-update hook

- Implement §6 weight-update path end-to-end, including `/admin/invalidate` and filesystem-watch fallback.
- Implement §5.3.4 LoRA adapter action-space.
- Wire serving to Training Flywheel LLD-06 → LLD-11 weight-publication contract.
- Deliverable: §9.4 (weight-update path end-to-end) passes; the Training Flywheel HLD's weight-publication seam is live.

### 7.4 Sprint 3 — Deprecation migration + re-platform

- Re-platform `src/lumo_flywheel_serving/` against the derived LLDs.
- Post DEPRECATED banners on LLD-01 / 02 / 03 / 04 / 13.
- Final verification pass: all §9 checklist items green.
- Deliverable: this HLD is implemented; next scope is FP4 exploration and/or additional model support (both gated on §11 open-question resolution).

---

## 8. What This HLD Locks Down For The Downstream LLD(s)

The downstream LLD series that replaces LLD-01/02/03/04/13. Names are placeholders; final numbering is decided by the implementer before LLDs are drafted.

- [ ] **LLD-SB-01** (replaces LLD-01): vLLM Serving Layer — new rev derived from §4 and §6 of this HLD.
- [ ] **LLD-SB-02** (replaces LLD-04): Telemetry & Measurement Plane — reused by §5.6 measurement harness; minor changes for `weight_version_id` / `tuned_config_id` tagging.
- [ ] **LLD-SB-03** (replaces LLD-03): Task Orchestrator — re-platformed for concurrent Codex threads, now with admission control (§4.1) and the two-class queue split (§4.3).
- [ ] **LLD-SB-04** (replaces LLD-02): Data Pool Manager — reused largely as-is; minor changes for concurrency-aware dispatch.
- [ ] **LLD-SB-05** (replaces LLD-13): Codex-Long Scenario Framework — reused; minor changes for the new orchestrator contract.
- [ ] **LLD-SB-06** (new): Auto-Research Agent — full implementation of §5, including the two-tier proposer/executor split and Appendix B kernel matrix.
- [ ] **LLD-SB-07** (new): Weight-Update Hook — full implementation of §6, including the atomic-rename watcher, the `/admin/invalidate` handler, and the rollback path.

Every LLD in this list carries a one-paragraph "What the HLD locks and what this LLD is free to choose" preamble, so the implementer agent and verifier agent have a clear seam between spec-level constraints and implementation judgment.

---

## 9. Verification Checklist (Pass/Fail, for the Implementer + Verifier Agent Pair)

Binary per item. Each item has a verifiable artifact the verifier agent can inspect without subjective judgment. PASS is only awarded when the artifact exists, is signed with a recent git SHA, and matches the described shape.

### 9.1 Contract satisfaction (§4)

- [ ] **9.1.1** `/v1/responses` endpoint exists and passes a representative Qwen 3.5 27B request with streaming enabled. Artifact: `curl` transcript or test-log record.
- [ ] **9.1.2** `/metrics` endpoint emits LLD-04 fields (TTFT, throughput, cache-hit rate, per-turn latency, queue depth). Artifact: scrape log showing every required field.
- [ ] **9.1.3** `/admin/invalidate` and `/admin/load_tuned_config` endpoints accept valid payloads and reject invalid ones with structured errors. Artifact: test log.
- [ ] **9.1.4** Admission control queues rather than drops on `L_ceiling` pressure. Artifact: synthetic-load test log showing queue-depth growth under over-concurrency, zero drops.
- [ ] **9.1.5** Every response row in telemetry carries `weight_version_id`, `tuned_config_id`, `class`, and LLD-04 telemetry row ID. Artifact: sampled trajectory-row dump.

### 9.2 Hardware envelope attestation (§3)

- [ ] **9.2.1** Post-install microbenchmark run published: measured bandwidth, measured FP8 GEMM throughput, measured NVMe read. Artifact: benchmark log committed to `docs/hw-attestation/`. Deviations > 15% from §3 published figures raise a §11 risk rather than a fail.
- [ ] **9.2.2** `make resume` (or documented equivalent) restores serving to last-known-good after a full power cycle, in under 60 seconds. Artifact: recording of resume run.

### 9.3 Auto-research agent end-to-end (§5)

- [ ] **9.3.1** Agent runs on the measurement harness without OOM or determinism drift for at least one full iteration. Artifact: run log.
- [ ] **9.3.2** Agent produces a tuned-config bundle that beats the baseline on §5.4 objective. Artifact: bundle YAML with `regression_guard.delta > 0`.
- [ ] **9.3.3** §5.7 safety rails tested individually: regression guard trips on a synthetic regression; determinism check trips on an injected non-deterministic kernel; OOM-handling trips on a synthetic oversized config. Artifact: three separate test logs.
- [ ] **9.3.4** Kernel action space (§5.3.3) produces at least one bundle that selects a non-default kernel and beats the vllm-default baseline. Artifact: bundle YAML referencing non-default `attention_backend`.
- [ ] **9.3.5** Stopping criterion (§5.8) fires on at least one of: wall-clock cap, iteration cap, or diminishing returns — not only on hard infeasibility. Artifact: run log.

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

vLLM's FP4-on-Blackwell kernel support is a 2026 moving target. Tracked as a risk, not a plan. If it matures during Sprint 1/2, the auto-research agent's action space (§5.3.1 `kv_cache_dtype` and a new `weight_dtype` dimension) extends naturally; if it does not, FP8 remains the only supported precision.

### 11.3 Coherence with the Training Flywheel HLD's weight-publication contract

The seam at §1.4 / §6 depends on both HLDs agreeing on the path. Risk: a change in one HLD's path without the other. Mitigation: the Training Flywheel HLD cross-references this HLD's §6 (and vice versa); any weight-publication path change requires paired updates.

### 11.4 Thermal throttling under sustained load

GB10 is rated for sustained operation, but the 24/7 campaign posture is unusual for a developer-class device. Tracked risk: thermal throttling erodes the 273 GB/s bandwidth ceiling over hour-long windows. Mitigation: the §9.2.1 post-install benchmark is re-run quarterly; material deviation triggers a §11 entry.

### 11.5 Responses-API drift

Qwen 3.5 supports Responses API natively today. If a future Qwen patch release breaks some corner of the API (tool streaming, parallel tool calls, reasoning-content format), the serving stack will detect it via determinism-check failure (§5.7). Recovery: pin the Qwen checkpoint until the regression is fixed upstream.

### 11.6 Auto-research over-fitting to the synthetic load generator

The agent measures against a synthetic mirror of the eval-workload distribution. Risk: the mirror drifts from reality and the tuned bundle is optimal for the mirror, not the campaign. Mitigation: the measurement-window objective is re-measured against a small sample of real campaign traffic in the first hour of each campaign; > 10% objective gap triggers a short re-tune.

### 11.7 Regression-guard false positives

If the baseline tuned-config bundle is stale (from an older Qwen checkpoint or older vLLM), it may pass the regression guard for a worse reason than true parity. Mitigation: baseline bundles older than 30 days are automatically demoted to reference-only at the start of a new auto-research run, and the run produces an unconditional new bundle if no valid baseline exists.

---

## Appendix A — Worked Example: Tuning qwen3.5-27b for a V1 Eval Campaign

A single concrete walkthrough so the implementer and verifier agents have something end-to-end to match their code against.

**Scenario.** Campaign kickoff for a V1 family (per `HLD-Family-Test-Requirements-v0_1.md`). Model: Qwen 3.5 27B FP8, `weight_version_id = abc1234`. Family: long-context coding tasks with moderate tool-call frequency; `L_ceiling` from family spec = 30 seconds per turn at p95; expected context-length distribution peaks at 16k, 99th percentile at 48k.

**Inputs the agent receives.** Model checkpoint pinned at `abc1234`. Family spec loaded. Workload distribution sampled from a prior campaign's telemetry (5000 real trajectories). Baseline tuned-config bundle from the last campaign on the same family. Hardware profile per §3 (hard-coded).

**Iteration 1 (LLM proposer region: baseline re-validation).** Re-run baseline against the synthetic driver for 30 minutes. Confirm baseline reproduces: `sustained_concurrent_eval_threads = 18` at `p95_turn_latency = 27 s`. Regression guard armed.

**Iteration 2 (region: vLLM-config sweep).** Proposer reasoning: "Baseline uses `max_num_seqs=24`, `gpu_memory_utilization=0.82`. §3.11 math says 32k-average-context KV budget allows ~30 concurrent. Proposing push on `max_num_seqs` and `gpu_memory_utilization`." Inner-loop Bayesian opt: 8 evaluations. Best: `max_num_seqs=28, gpu_memory_utilization=0.88`. Result: 22 concurrent at p95=28s. Regression guard passes (+4 vs baseline).

**Iteration 3 (region: chunked-prefill ablation).** Proposer: "Latency tail is prefill-dominated on long-context requests; try `enable_chunked_prefill=true` with a tighter `max_num_batched_tokens`." Best: 23 concurrent at p95=27s. Passes.

**Iteration 4 (region: kernel selection).** Proposer: "Appendix B marks flashinfer as supported for Qwen 3.5 27B FP8 with deterministic flag; its paged-KV handling wins on our workload's prefill:decode mix. Try flashinfer + cuda-graph-decode-only." Determinism probe passes. Result: 26 concurrent at p95=26s. Passes.

**Iteration 5 (region: LoRA policy).** Campaign has no LoRA adapters for this family. Proposer: "Skip `adapter_mode` tuning; leave `adapter_mode=static-merged` and `max_loaded_adapters=0`." No measurement needed; policy decided.

**Iteration 6–9 (inner-loop fine-tune on the best region).** Bayesian opt narrows around iteration 4's config. Best: `max_num_seqs=29, gpu_memory_utilization=0.87, enable_chunked_prefill=true, attention_backend=flashinfer, cuda_graph_capture=decode-only`. Result: 28 concurrent at p95=26s.

**Iteration 10 (region: request-shaping knobs).** Proposer: "Admission queueing policy can smooth queue-depth spikes. Try `concurrency_cap_eval=28, concurrency_cap_rollout=1, admission_queue_depth_max=64, per_request_kv_budget=40k`." Result: 28 concurrent, p95=25s, rollout throughput holds at 0.8× baseline. Passes rollout-floor constraint.

**Iteration 11–12 (diminishing returns check).** Proposer attempts two more regions (`max_num_batched_tokens` variants; `priority_preemption=strict`). Neither improves over iteration 10. Stopping criterion: diminishing-returns threshold met (< 2% improvement over last 8 iterations).

**Final bundle.** `sustained_concurrent_eval_threads = 28`, p95 = 25s, rollout-floor satisfied, all safety rails green. Bundle persisted to `/data/tuned_configs/v1-family/abc1234/<run-id>.yaml`. Campaign bootstrap loads it via `POST /admin/load_tuned_config`.

**Wall-clock.** ~3h15m across 12 iterations. Well within the 4-hour cap.

**What the verifier agent checks.** All §9.3 items against this run's artifacts; the bundle is the primary artifact for §9.3.2.

---

## Appendix B — Kernel Notes for Qwen 3.5 27B

A simplified kernel-selection reference for the §5.3.3 action space. Each row is a kernel; columns indicate whether the kernel is in the agent's action space for Qwen 3.5 27B FP8 on GB10, and why.

| Kernel | FP8 weight GEMM | FP8 KV cache | Paged attention | Streaming Responses API | Determinism (scorer-probe passes) | In action space |
|---|---|---|---|---|---|---|
| `flash-attn-2` | ✓ | ✓ (fp8_e5m2) | ✓ | ✓ | ✓ | **Yes** |
| `flash-attn-3` | ✓ | ✓ | ✓ | ✓ | ✓ (Blackwell support pending final upstream) | **Yes** (gated on §9 benchmark run confirming) |
| `flashinfer` | ✓ | ✓ | ✓ (superior paged-KV for mixed prefill/decode) | ✓ | ✓ | **Yes** |
| `triton` | ✓ | ✓ | ✓ | ✓ | ✓ | **Yes** (slower than flash variants on GB10; kept for fallback) |
| `vllm-default` | ✓ | ✓ | ✓ | ✓ | ✓ | **Yes** (the baseline) |
| `xformers` | partial | — | — | — | — | **No** (insufficient FP8 coverage on Blackwell at v0.1) |

Companion knobs:

- `torch_compile_mode` ∈ {`off`, `default`, `reduce-overhead`, `max-autotune`}. All four are in the action space; `max-autotune` has a long compile warm-up that inflates cold-start cost and may clash with the 5-minute harness warm-up window — the agent is free to pick it but should prefer `reduce-overhead` as a default.
- `cuda_graph_capture` ∈ {`off`, `decode-only`, `full`}. `full` captures both prefill and decode; expensive to rebuild on vLLM config change. `decode-only` is usually the right choice for this workload.
- `fp8_gemm_kernel` ∈ {`cutlass-default`, `cutlass-warpspecialized`}. Warp-specialized variant is Blackwell-preferred for FP8 dense GEMM per NVIDIA guidance; kept in the action space, agent picks empirically.

Kernels excluded from the action space for v0.1 — and the reason each is excluded — are listed in the change log of this appendix. Re-entering a kernel requires a PR against this appendix plus §9.3.4-style verification that the kernel passes streaming and determinism checks on the campaign workload.

---
