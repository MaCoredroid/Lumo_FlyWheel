# Serving Auto-Research v0.2 — L0 Kernel Action Plan

**Parent documents.** `docs/HLD-Serving-Backend-AutoResearch-v0_1.md` (parent HLD), `docs/HLD-Serving-Backend-AutoResearch-v0_1-SubSpec-AutoResearchAgent.md` (agent sub-spec v0.1.11), `docs/HLD-Serving-Backend-AutoResearch-v0_1-HardeningPlan.md` (hardening plan v0.1.5-plan). This action plan is the v0.2 follow-on after the H1 round (`reports/serving-auto-research-hardening-h1-2026-04-25.md`) returned **Case β — L1 lacks defensible headroom on this hardware × model combination, so kernel work is the next investment**.

**Driving observations.** Two independent hardened rounds — Sprint-0 on proposal-ranking-only + H1 on multi-family-v5 — both placed the L1 winner within the noise floor of the default config (within +3% and within −5% respectively, against a noise floor of ~25%). Parent §11.10 estimated kernel work has order-of-magnitude more headroom than L1 config. v0.2 commits to that direction with concrete scope and a non-negotiable correctness substrate.

**Operator directive (2026-04-26).** "Go from L0a → L0b → L0c quickly, closer to Karpathy autoresearch. But if we write kernels we have to make sure correctness." This plan honors both: aggressive Karpathy-style cycle time at L0c (LLM proposes a kernel `.patch`, compile, test, keep/discard, repeat), with a parity-gate primitive that no mutation bypasses. The parity gate is what separates kernel work from training-loop work — Karpathy's `val_bpb` self-corrects when training breaks; a faster-but-wrong kernel runs faster *and* corrupts every downstream eval silently.

---

## 0. Scope and non-goals

**In scope (v0.2).**
- L0a kernel selection — pick `attention_backend` (FA4/FA3/FA2/FlashInfer/Triton) for GatedAttn, `deltanet_kernel` (chunked-delta / state-update-fused / ...) for DeltaNet, plus `torch_compile_mode`, `cuda_graph_capture`, `fp8_gemm_kernel`. Pure config-space search over ~20 combinations.
- L0b kernel autotune — `@triton.autotune` over `BLOCK_M/N/K`, `CHUNK_SIZE`, `num_warps`, `num_stages` for the chosen Triton kernel; CUTLASS Python tuner for the FP8 GEMM. Standard upstream tooling.
- L0c kernel mutation — Karpathy-style: LLM proposes a `.patch` against the chosen kernel source, compile + parity-check + measure. Per-attempt watchdog timeout enforced; no soft cycle-time estimate.
- Heavy single-family workload — `responses-sdk-adapter-cutover` (decode-heavy, 4-turn trajectory replayed per measurement).
- Cross-family generalization check at decision time — staleness check on P1-pre-captured sibling holdouts + 16-probe parity-vs-§2.2.0-reference correctness gate per (sibling × mutated kernel target) + paired n=5 baseline + n=5 winner per parity-passing sibling (up to 80 measurements across 8 siblings). Per-sibling Welch-t consumes only contemporaneous pairs. Sibling holdouts themselves are captured upstream in P1, not at decision time.
- Parity-fixture infrastructure (per-family probe set with state checkpoints, weight-sensitive flagging, refusal-on-divergence semantics).

**Out of scope (deferred to v0.3+).**
- Multi-family kernel tuning. We pick one heavy family for v0.2; if the kernel winner generalizes (cross-family check passes), v0.3 expands. If not, v0.3 is per-family kernel tuning with separate fixtures.
- Authoring new families. The `wire_api_missing` cleanup for the 19 excluded families is parallel work, not gated on this plan.
- L1 / L2 search re-runs. v0.1 hardening already established L1 has no defensible headroom; v0.2 doesn't re-test that.
- Production GPU-fleet rollout. v0.2 produces a kernel-tuned bundle for the single heavy family; rollout strategy is a v0.3 question.

**Relationship to the v0.1 substrate.** Everything the v0.1 hardening plan built — `replay-round`, statistical machinery (n=5 baseline, Welch-t CI, noise floor, confidence derivation), worktree isolation, BOOTSTRAP commit, trailer contract, watchdog, thinking probe — carries forward unchanged. v0.2 adds three new CLI subcommands (`tune-kernel-select`, `tune-kernel-autotune`, `mutate-kernel`) and one new primitive (the parity fixture). The candidate proposer at L0a/b is deterministic grid + autotune; at L0c it's an LLM in the Karpathy spawn-per-iteration pattern.

---

## 0.5 Hardware grounding — GB10 specifics + bandwidth-bound thesis (v0.3.0)

The v0.2.x plan focused on attention + FP8 GEMM as the kernel-tuning surface. v0.3.0 expands the surface to all kernel categories called out in the @TheAhmadOsman framing — MatMul, Attention, RMSNorm, KV cache, Quantized linear, Sampling, and Fused. The expansion is grounded in concrete GB10/DGX Spark physical facts that change the relative priority of these categories.

### 0.5.1 GB10 / DGX Spark hardware facts (load-bearing for this plan)

| Property | Value | Source / note |
|---|---|---|
| GPU architecture | Blackwell (sm_100/sm_101 family) | NVIDIA DGX Spark product page |
| Unified memory | **128 GB LPDDR5x** | NVIDIA spec (NOT 192 GB — v0.2.x had this wrong) |
| Aggregate memory bandwidth | **~273 GB/s** (LPDDR5x-8533, 256-bit) | NVIDIA spec; ~30× LOWER than B200's HBM3e |
| FP4 tensor peak | 1 PFLOP sparse / 500 TFLOPS dense | NVIDIA marketing |
| FP8 tensor peak | ~500 TFLOPS dense (inferred) | Not separately published; halved from FP4 |
| NVLink-C2C (CPU↔GPU) | ~600 GB/s | Consistent with Grace-Blackwell C2C; not separately documented for GB10 |
| 5th-gen Tensor Cores | Native FP8/FP6/FP4, MXFP block scaling | Blackwell architecture |
| TMA (Tensor Memory Accelerator) | Supported (Hopper-era feature retained) | Async bulk copies with swizzle |
| Thread Block Clusters / DSMEM | Supported | Hopper-era |
| `tcgen05` PTX (new Blackwell tensor-core MMAs) | Supported, exposed by CUTLASS 3.6+ | Blackwell-specific |
| 2-CTA MMA shape | Supported on Blackwell | Favors 256×256-class GEMM tiles vs Hopper |

**Undocumented for GB10 specifically** (treat as uncertain; verify on-device before tuning):
- SM count, register file per SM, shared memory per SM, L1/L2 cache sizes.
- GB10-specific FA4 perf table (FA4 is **beta/experimental** on Blackwell as of late 2025; vLLM `FLASH_ATTN_VERSION=4` is gated behind an env flag and not the default — see open-question §10.6).
- GB10-specific CUTLASS perf table (sm_100 kernels work but tile schedulers are tuned for B200 SM counts).

### 0.5.2 Bandwidth-bound thesis (the core insight)

**GB10 is severely bandwidth-bound for inference.** 273 GB/s vs ~500 TFLOPS FP8 dense gives an arithmetic intensity threshold of ~1800 FLOPS/byte to be compute-bound. Most decode-path kernels (norm, residual, sampler, KV write, small-batch GEMM) are nowhere near that — they are bandwidth-bound by 1–2 orders of magnitude. This reframes the optimization priority:

1. **Every saved memory pass is high-leverage.** A kernel that fuses `attn_output + residual + rmsnorm + fp8_quant` into one pass eliminates 2–3 full-tensor reads/writes per layer × 64 layers × thousands of decode tokens. On a 273 GB/s memory system each saved pass is roughly 10 µs/layer in steady-state decode — hundreds of microseconds per token.
2. **Tile sizes for GB10 should NOT be inherited from B200 defaults.** Larger tiles improve compute reuse but increase per-launch overhead and register pressure; on a smaller-SM-count GB10 the tradeoff differs.
3. **Norms and samplers, individually small, become non-trivial when fused.** RMSNorm alone is ~1–2% of decode; fused into a quant-epilogue path it pulls 8–15% e2e on bandwidth-bound hardware.
4. **DeltaNet's recurrent-state writeback is bandwidth-bound, not compute-bound.** Tile size matters less than write-coalescing and async-copy pipelining.

This is the v0.3.0 thesis: **kernel categories that reduce memory traffic outrank kernel categories that improve raw FLOPS** on GB10. Section §0.6 ranks the categories accordingly.

### 0.6 Kernel category priority order — provisional, gated on P3a roofline (v0.3.1)

The §0.5.2 bandwidth-bound thesis predicts the priority order below. **It is provisional**: AR.54 priority enforcement does NOT activate until phase P3a (mandatory roofline probe, see §7.2) has measured per-category Tpot share, DRAM throughput, SM occupancy, tensor-core utilization, and kernel launch count on the actual GB10 + Qwen 3.5 27B FP8 + heavy workload. If the measured roofline contradicts the predicted priority, this section is rewritten and the executable target list in §0.7 is re-narrowed.

**Predicted priority** (Qwen 3.5 27B FP8 hybrid, layout `16 × (3×DeltaNet + 1×GatedAttn)` per HF model card):

| Rank | Category | Predicted reason | v0.3.1 status |
|---|---|---|---|
| 1 | **DeltaNet Triton kernels** (48/64 layers) | Dominant by layer count; recurrent-state writeback bandwidth-bound; default `BLOCK_*`/`num_warps` mis-tuned for GB10 | **executable L0c target** |
| 2 | **Fused norm+residual+quant epilogues** (Triton sidecar) | Triton sidecar fuses **`+residual → RMSNorm → FP8 quantize`** into one kernel — eliminates 1–2 memory passes per layer × 64 layers × thousands of tokens. **Caveat (P0.4 fix):** sidecar runs *after* the GEMM and cannot eliminate the GEMM-output write itself; eliminating that write requires CUTLASS/CuTe epilogue integration deferred to v0.3.3+. P3a must measure **actual launch-count and pass-count reduction** before this row's priority is enforced. Magnitude ("3–8% e2e" reduced from v0.3.0's overclaimed "8–15%") is a hypothesis pending P3a. | **executable L0c target — Triton-sidecar surface only**, NOT CUTLASS |
| 3 | **Sampling (top-k/top-p)** | Qwen3.5-27B-FP8 padded vocab is **248,320** (per HF model card, larger than v0.3.0's 152K estimate); thinking-heavy decode amplifies | **gated L0c target** — runs only if P3a measures sampler share ≥ 3–5% of Tpot |
| 4 | **GatedAttn backend selection + paged KV** (16 layers) | Long-context thinking amplifies KV traffic; affects only 16/64 layers | **L0a-only**, NOT L0c (FA4/FA3/FA2/FlashInfer are vendor; Triton GatedAttn is rarely the L0a winner) |
| 5 | **Quantized Linear (GEMM) backend selection** | Backend choice is high-leverage; CUTLASS schedulers are B200-tuned. Mutation surface depends on which backend is shipped as a Triton kernel | **L0a backend selection only** in v0.3.1; Triton-mutation deferred to v0.3.2 if shipping kernels are mutable |

**Categories explicitly NOT pursued as standalone L0c targets in v0.3.1:**
- **RMSNorm standalone.** Standalone RMSNorm uplift is small (~1–2% of decode); the leverage is through fused epilogues (#2). RMSNorm is reachable inside the fused-epilogue Triton kernel; no separate L0c round.
- **KV-cache standalone mutation.** PagedAttention C++ is not source-mutable; only the Triton paged path is, and KV-cache rarely dominates Tpot. Stays as L0a config knobs (`kv_cache_block_size`, `kv_cache_dtype`).
- **FP8 GEMM Triton mutation (former P7f).** Removed from v0.3.1 as unreachable: the L0a action space exposes `fp8_gemm_kernel ∈ {cutlass, cublas}` only; there is no Triton FP8 GEMM in vLLM's shipping path that is straightforward to mutate. Defer to v0.3.2 if we add a Triton FP8 GEMM to the action space first.

### 0.7a v0.3.3 EXECUTABLE scope (impl-agent recommendation, narrowed from v0.3.2)

v0.3.2 specified architectural intent (composite multi-family workload, multi-target L0c with fused_epilogue + sampling). v0.3.3 narrows to what's **actually executable today** given on-disk fixture data, vLLM shipping paths, and L0b-empirical-winner findings:

| Aspect | v0.3.2 architectural intent | v0.3.3 EXECUTABLE today | Deferred to |
|---|---|---|---|
| Workload | Composite 9-family with full-agent-flow trajectories (8–30 turns) | **Single-family heavy** (`responses-sdk-adapter-cutover`, 4-turn) — what was captured | v0.3.4+ for full-flow capture across other 8 families |
| Base stack | Hand-authored fixed (FA3 + Triton sidecar + Triton sampler + bf16) | **L0b-empirical-winner bundle (`4866bc3f`)** with `actually_resolved_kernel_selection` pinned (vllm-default routing + fp8_e5m2 KV) | n/a — empirical anchor |
| L0c targets | `{deltanet, fused_epilogue, sampling}` | **`{deltanet}` only** | v0.3.4+ for fused_epilogue + sampling, gated on Triton-sidecar / Triton-sampler wiring verification |
| Pre-L0c phases | P1, P2, P2b, P5, P3a | P1, P2, P2b, P5, **P5b (fp8 KV purity)**, P3a | n/a — P5b is new in v0.3.3 |

**Architectural intent is preserved in the doc**, NOT demoted. v0.3.4 will reinstate fused_epilogue + sampling as executable targets once their wiring prerequisites are verified. The §0.5.2 bandwidth-bound thesis still drives priority order; what changed is the proof-of-execution gate.

### 0.7 What v0.3.1 adds at-a-glance

- **§3.1 L0a action space**: keeps v0.2.x's 5 knobs; adds 4 new knobs (`kv_cache_block_size`, `kv_cache_dtype`, `sampling_kernel`, `epilogue_fusion_mode`). Drops `rmsnorm_kernel` (fused inside epilogue). FP8 GEMM stays `{cutlass, cublas}` — no Triton.
- **§4 L0b autotune**: stays at DeltaNet only. v0.3.0's expansion to RMSNorm/sampling autotune was not coherent (no concrete autotune surfaces defined) and is dropped.
- **§5 L0c mutation**: `--kernel-target` accepts **three executable values** in v0.3.1: `deltanet`, `fused_epilogue`, `sampling` (sampling gated on roofline). GatedAttn remains in the schema but is L0a-only unless a Triton GatedAttn kernel is L0a-selected.
- **§7 Phase DAG re-ordered**: **L0c runs FIRST** against `vllm-default` baseline (per operator note: prior L0a/L0b rounds showed no headroom against baseline, so L0c is the unexplored surface). Order is now P1/P2/P2b/P5 setup → **P3a mandatory roofline probe** → P7a/P7e/P7d L0c rounds → P9/P10 → optional later P3/P6 if L0c results suggest config-space headroom on top.
- **§5.5 iteration_brief.md**: now includes embedded DGX Spark hardware notes + speedup tips so the L0c agent doesn't waste cycles on online research per iteration.
- **§2.2 parity fixtures**: §2.2.3 (RMSNorm), §2.2.5 (KV cache mutation), §2.2.7 (FP8 GEMM Triton) removed. §2.2.6 (Fused epilogue) keeps but Triton-sidecar shape only.
- **§9 verification**: AR.49 (RMSNorm), AR.52 (FP8 GEMM Triton) dropped. AR.53 (kernel-target coverage) now enumerates the 3 executable targets. AR.54 (priority-order audit) gated on P3a roofline pass.
- **§10 open questions**: §10.6 FA4 GB10 status, §10.7 CUTLASS GB10 tile schedulers, §10.8 bandwidth thesis verification (now answered by P3a). New §10.9 multi-instance memory math (per P0.6).

The L0a/b/c contract, parity-fixture infrastructure, BOOTSTRAP/Candidate/Rescreen/FINALIZE commit shape, three-cap structure, paired-A/B baselines, composite-bundle identity, and partition-completeness invariants from v0.2.x ALL carry forward unchanged. v0.3.1 is narrower than v0.3.0 (3 executable targets vs 6) and re-ordered (L0c first, not last).

---

## 1. Workload commitment — composite multi-family with full-agent-flow trajectories (v0.3.2)

### 1.0 Architectural shift from v0.3.1 (operator directive)

v0.3.1 committed to a **single heavy family** (`responses-sdk-adapter-cutover`, 4-turn trajectory) with v0.2.x's "promote heavy → multi-family at P10" architecture. v0.3.2 inverts: **the workload itself is multi-family from the start**, with each family contributing a *full-agent-flow trajectory* (8–30 turns of real agent work, not 4-turn thinking-only snippets). The L0c bundle's `workload_distribution_id` is then the multi-family-composite id directly — no separate "heavy bundle → composite at P10" promotion phase.

This:
- Removes the L0c-target-vs-multi-family-promotion lock (any of `deltanet`, `fused_epilogue`, `sampling` can multi-family-promote, since the bundle is for the composite workload from the start).
- Collapses the v0.2.x/v0.3.1 sibling-fixture phase (P9 step b) — fixtures are captured against the composite, covering all per-family shapes natively.
- Surfaces the operator's "not just several turns" directive: trajectories are full agent flows, captured via codex driving each family's eval set to completion or to a defensible cap.
- Increases workload-construction work upfront (multi-family full-flow capture is a multi-day codex-driven effort) but eliminates the P9/P10 fixture-promotion fragility.

### 1.1 Composite workload definition

`benchmark_blueprints/workloads/multi-family-v5-l0c-composite/workload.yaml`. Includes 1–9 component families from the v0.1.5 included pool, sorted alphabetically by `family_id`. Each family contributes a **full-agent-flow trajectory** captured against `vllm-default` baseline.

**Component-family selection (v0.3.2 starting set):**

| Family | Variant | Why included | Trajectory turn count target |
|---|---|---|---|
| `responses-sdk-adapter-cutover` | v5 | Decode-heavy thinking-heavy (4096+512+512+4096 thinking) — already captured per v0.3.1 §2.1 | 4 (existing) → 12+ (full-flow expansion in v0.3.2 P1) |
| `codex-provider-rollover` | v5 | Multi-step config-migration; tool-use heavy | 8–20 |
| `codex-skill-runtime-v2-split` | v5 | Code-and-test trajectory; longer flows | 12–25 |
| `esm-plugin-loader-modernization` | v5 | Module refactor with reasoning chains | 10–20 |
| `nightly-regression-watch` | v5 | Iterative debugging trajectory | 15–30 |
| `objective-driven-repo-improvement` | v5 | Open-ended improvement task; very long | 20–30 |
| `policy-aware-request-resolution` | v5 | Compliance reasoning + branching | 8–18 |
| `release-manifest-v2-modernization` | v5 | Manifest schema migration | 10–18 |
| `sqlalchemy-2-session-modernization` | v5 | DB session migration; iterative | 12–22 |

**Full-agent-flow capture criteria** (per family):
- Trajectory runs to either (a) family eval-set completion (success or terminal failure) OR (b) a 30-turn cap, whichever is first.
- Captured against `vllm-default` baseline + thinking-probe row-3 verified at every turn.
- `reasoning_tokens` non-zero at expected turns; `output_tokens` non-zero at expected turns.
- Per-trajectory metadata records: `turn_count`, `total_thinking_tokens`, `total_output_tokens`, `total_prompt_tokens`, `terminated_reason ∈ {completed, failed, cap_reached}`.

**Per-family holdout slice** (v0.1.5 stratified-split style, baked into workload):
- Each family's full-flow trajectory is split 3:1 seed:holdout at trajectory boundary (NOT mid-turn).
- Seed slice is used for L0c measurement; holdout slice is used for §7.2 P9 per-family generalization check.
- Both slices are captured at workload-build time, both committed to main, both included in `workload_distribution_id` via the §6.6 canonical hash.

**Dropping v0.3.1's "heavy family" framing.** §2.1 onward refers to the composite as **the workload** — the v0.3.1 heavy-family pseudo-bundle pattern is replaced by direct composite identity. The §1.X heavy-trajectory description below is the *first variant* in the composite, retained for v0.3.2 P1's incremental expansion.

---

## 1.X (v0.3.1 carry-forward) First-variant trajectory — `responses-sdk-adapter-cutover`

**Why this family.** Among the 9 included pool families:
- **Reasoning-heavy:** `seed_trace_v5.jsonl` rows show `thinking_tokens` of 4096, 512, 512, 4096 — exclusively thinking, zero response tokens. Pure decode-side load on Qwen3.5's hybrid attention path (DeltaNet recurrent state update + GatedAttn KV read). Exercises both kernel families v0.2 cares about.
- **Trajectory shape stresses both prefill and decode.** Turns 0 and 3 are short prompt (58 tokens) → long output (4096 thinking tokens) — pure decode-bound. Turns 1 and 2 are mid prompt (~1.2k tokens) → mid output (512 thinking tokens) — mixed. Mix is realistic for a multi-event API-migration task.
- **v5 verification matrix exists** (`verification_matrix_v5.md`) — the family is hardened and flywheel-ready.
- **In the existing 9-family included pool** — the v0.1.5-plan thinking-fixed capture pipeline already works for it.

**Trajectory sampling.** The captured `seed_trace_v5.jsonl` has only 4 turns. For n=20 candidate measurements per round, **each measurement replays the full 4-turn trajectory once** (paired-sample design): n=20 measurements = 20 replays × 4 turns = 80 individual harness invocations per candidate. This is the same shape as v0.1 `replay-round`'s baseline replays — no new sampling math needed.

**Cross-family validation pool (post-L0c generalization check).** The other 8 included families: `codex-provider-rollover`, `codex-skill-runtime-v2-split`, `esm-plugin-loader-modernization`, `nightly-regression-watch`, `objective-driven-repo-improvement`, `policy-aware-request-resolution`, `release-manifest-v2-modernization`, `sqlalchemy-2-session-modernization`. The cross-family round (P9) runs three sub-steps: (a) verify pre-captured per-sibling holdout traces are still valid against the live serving stack, (b) run a 16-probe parity-vs-§2.2.0-reference correctness gate **per (sibling × mutated kernel target)** against the kernel-mutated stack — in dual-mutation rounds (P7 + P8 both produced winners) each sibling is checked against both DeltaNet and GatedAttn, in single-mutation rounds only the mutated kernel is checked, (c) paired n=5 `vllm-default` baseline + n=5 kernel-mutated winner per parity-passing sibling — up to 80 measurements (8 × 10), fewer if parity excluded any siblings.

**Sibling holdouts pre-exist on disk.** Sibling holdouts are NOT captured during P9 — that would require modifying P9's BOOTSTRAP commit after bootstrap, which the ledger model forbids. Instead, **P1 (workload descriptor + capture) is extended to capture sibling holdouts up front.** Each of the 8 sibling families gets a fresh holdout capture against `vllm-default` baseline (same semantic as the heavy family's §2.1 holdout), written to `benchmark_blueprints/families/<sibling_fid>/holdout_trace_v5.jsonl` and committed to main BEFORE any L0 round bootstraps. By the time P9 runs, these files already exist; P9 only verifies they're still valid against the current serving stack (via thinking-probe re-check).

---

## 2. Workstream 1 — Heavy single-family workload + parity fixture

### 2.1 Workload descriptor

New file: `benchmark_blueprints/workloads/responses-sdk-adapter-cutover-heavy/workload.yaml`. Schema follows the v0.1.5-plan composite descriptor:

```yaml
family_id: responses-sdk-adapter-cutover-heavy   # composite-family pseudo-id (parent §5.9 amendment)
workload_distribution_id: <sha256>               # canonical hash per parent §5.9 P1-YY procedure
workload_distribution_id_hardening_version: v2-l0-kernel-heavy
capture_date: <iso8601>
source_family: responses-sdk-adapter-cutover
trajectory_source: benchmark_blueprints/families/responses-sdk-adapter-cutover/seed_trace_v5.jsonl
trajectory_turns: 4
samples_per_measurement: 1   # one full-trajectory replay per measurement (4 individual turns each)
seed_trace_ref: seed_trace.jsonl       # full 4-turn trajectory, captured fresh against fixed vLLM
holdout_trace_ref: holdout_trace.jsonl # second independent capture, same prompts, different seed
nominal_ttft_ms: 2000
nominal_tpot_ms: 80
nominal_turn_ms: 30000
target_concurrency: 4
thinking_probe_ref: reports/thinking-probe-<yyyymmdd>.md
thinking_probe_outcome: row-3   # required (thinking fires when asked)
# Per-kernel parity fixture references (consumed by L0a smoke phase + L0c rounds).
# All files MUST exist before any L0c round runs (built against reference baseline — see §2.2.0).
# v0.3.1: full map covering all v0.3.1 kernel targets (P0.2 fix).
parity_fixture_refs:
  deltanet:        parity_fixture/deltanet_v1.yaml         # §2.2.2
  gatedattn:       parity_fixture/gatedattn_v1.yaml        # §2.2.1 — schema-only target in v0.3.1
  fused_epilogue:  parity_fixture/fused_epilogue_v1.yaml   # §2.2.6 — Triton-sidecar parity (4 checkpoints)
  sampling:        parity_fixture/sampling_v1.yaml         # §2.2.4 — token-id determinism + KL divergence
  # NOTE: rmsnorm and fp8_gemm_triton fixtures are NOT in v0.3.1 (targets dropped per §0.6).
  # NOTE: kv_cache fixture is NOT in v0.3.1 — KV cache is L0a-only, no L0c mutation target.
```

### 2.2 Per-family parity fixture (the correctness substrate)

**The non-negotiable primitive.** A kernel mutation that runs faster but produces wrong output is worse than no mutation. The parity fixture is what catches that case before its latency is recorded. The fixture is also a **correctness gate for L0a kernel selection**: a combination that passes determinism but fails parity-against-reference is a deterministically-wrong kernel (e.g. FlashInfer on Qwen3.5 per #35138). Determinism without correctness is not safe to measure.

**Two fixture files** at `benchmark_blueprints/families/responses-sdk-adapter-cutover/parity_fixture/`:

#### 2.2.0 Reference baseline — what the fixture is captured against

Fixtures are captured against an **externally-trusted reference baseline**, not against any round's winner. The reference baseline is the parent §2 Decision 2 / §11.10 forcing pattern:

| Knob | Reference value | Source of trust |
|---|---|---|
| `attention_backend` (GatedAttn) | `flash-attn-4` (forced) | Parent §2 Decision 2; the v0.1 hardening winner; explicitly recommended over FlashInfer's #35138-affected path |
| `deltanet_kernel` | `triton-chunked-delta-v2` (vLLM stable default) | vLLM upstream's tested default for Qwen3.5-style hybrid attention |
| `fp8_gemm_kernel` | `cublas` (vLLM stable default) | vLLM upstream's tested default |
| `torch_compile_mode` | `default` | Avoids autotune-driven non-determinism in fixture capture |
| `cuda_graph_capture` | `off` | Avoids capture-replay correctness ambiguity in fixture |

This reference baseline produces what we treat as ground truth. **No kernel — including the L0a winner — is allowed to disagree with it within the fixture's tolerances.** A kernel that disagrees is wrong, regardless of how fast it runs.

The reference baseline is itself audited: the parity fixture's first-build run is reproduced 3 times bit-identically (same input → same logits) before fixture is sealed. If the reference itself is non-deterministic, fixture capture fails — that's a halt-and-surface condition (P1's `fixture_reference_nondeterministic`).

**Consequence for §6.3 weight rotation:** the fixture re-builds when `weight_version_id` rotates. It does **not** rebuild when an L0a/L0b/L0c winner changes. The fixture is round-independent.

#### 2.2.1 GatedAttn fixture (logit-space)

```yaml
# parity_fixture/gatedattn_v1.yaml
fixture_id: responses-sdk-adapter-cutover-gatedattn-v1
generated_at: <iso8601>
generated_against:
  vllm_version: <version>
  weight_version_id: <sha>
  reference_baseline:                       # §2.2.0 externally-trusted reference, NOT a round winner
    attention_backend: flash-attn-4
    deltanet_kernel: triton-chunked-delta-v2
    fp8_gemm_kernel: cublas
    torch_compile_mode: default
    cuda_graph_capture: off
  reference_reproducibility_runs: 3         # reference captured 3× bit-identically before sealing
probe_count: 64
probe_token_lengths: [16, 64, 256, 1024]    # mix of short and long prompts
probe_input_ref: probes_input.jsonl
reference_logits_ref: gatedattn_reference_logits.npz
tolerances:
  rtol_logit: 1.0e-3
  atol_logit: 1.0e-3
parity_check_method: per_token_logit_compare   # see §6.2
```

`probes_input.jsonl` is 64 hand-picked (prompt, output_token_count) pairs sampled to cover GatedAttn's relevant `(M, N, K)` shapes. `gatedattn_reference_logits.npz` is the reference logits at every output position for every probe, captured by running the §2.2.0 reference baseline 3× and confirming bit-identity before sealing.

#### 2.2.2 DeltaNet fixture (logit + recurrent-state)

```yaml
# parity_fixture/deltanet_v1.yaml
fixture_id: responses-sdk-adapter-cutover-deltanet-v1
generated_at: <iso8601>
generated_against:
  vllm_version: <version>
  weight_version_id: <sha>
  reference_baseline:                       # §2.2.0 externally-trusted reference, NOT a round winner
    attention_backend: flash-attn-4
    deltanet_kernel: triton-chunked-delta-v2
    fp8_gemm_kernel: cublas
    torch_compile_mode: default
    cuda_graph_capture: off
  reference_reproducibility_runs: 3         # reference captured 3× bit-identically before sealing
probe_count: 64
probe_token_lengths: [16, 64, 256, 1024, 4096]    # 4096 included to stress recurrent-state decay
probe_input_ref: probes_input.jsonl
reference_logits_ref: deltanet_reference_logits.npz
reference_state_snapshots_ref: deltanet_reference_state.npz
state_checkpoints_at_token: [1, 1024]    # per parent §5.6 — catches slow drift
tolerances:
  rtol_logit: 1.0e-3
  atol_logit: 1.0e-3
  rtol_state: 5.0e-3
  atol_state: 5.0e-3
parity_check_method: logit_plus_state_compare   # see §6.2
```

**Why state-snapshot checkpoints matter.** A bad DeltaNet kernel mutation can pass logit parity at token 1 (output looks fine) but corrupt the recurrent state, which then accumulates error and produces wrong logits by token 1024+. Parent §5.6 calls this "slow drift" and explicitly requires the 1024-token checkpoint. Logit-only parity at token 1 is necessary but **not sufficient** for DeltaNet.

#### 2.2.3 ~~RMSNorm fixture~~ (REMOVED in v0.3.1; subsumed into §2.2.6 fused-epilogue)

The standalone RMSNorm fixture from v0.3.0 is removed. RMSNorm correctness is now verified inside the fused-epilogue 4-checkpoint compare (§2.2.6 checkpoint 2: `post_norm`). Standalone RMSNorm L0c target dropped per §0.6.

<details>
<summary>v0.3.0 schema (kept here for archival reference only — NOT in v0.3.2 active scope)</summary>

```yaml
# parity_fixture/rmsnorm_v1.yaml
fixture_id: <family>-rmsnorm-v1
generated_against:
  reference_baseline: <§2.2.0>
  reference_reproducibility_runs: 3
probe_count: 32                              # smaller surface than attention; norm shapes are simpler
probe_input_tensors_ref: norm_input_tensors.npz   # input tensors at norm boundary (post-residual)
reference_norm_output_ref:  rmsnorm_reference_output.npz
reference_downstream_logits_ref: rmsnorm_reference_downstream_logits.npz   # logits 32 layers downstream — catches norm errors that compound
tolerances:
  rtol_norm_output: 5.0e-4   # tighter than logit (norm operates on smaller-magnitude tensors)
  atol_norm_output: 5.0e-4
  rtol_downstream_logit: 1.0e-3
  atol_downstream_logit: 1.0e-3
parity_check_method: norm_output_plus_downstream_logit
```

A mutated RMSNorm kernel passes parity iff: (a) its output matches reference at every probe within `(rtol_norm_output, atol_norm_output)`, AND (b) feeding that output through the rest of the model produces logits within `(rtol_downstream_logit, atol_downstream_logit)` of reference. The downstream check catches small norm errors that compound across the remaining layers — analogous to the DeltaNet state-snapshot check at token 1024.

</details>

#### 2.2.4 Sampling fixture (token-id determinism + KL divergence) — v0.3.1+ (gated)

```yaml
# parity_fixture/sampling_v1.yaml
fixture_id: <family>-sampling-v1
generated_against:
  reference_baseline: <§2.2.0>
  reference_reproducibility_runs: 3
probe_count: 64                              # 64 (logits, sampling_params, seed) tuples
probe_input_logits_ref: sampling_input_logits.npz   # 64 × vocab_size FP32 logits tensors
probe_sampling_params_ref: sampling_params.jsonl    # per-probe (top_k, top_p, temperature, seed)
reference_sampled_token_ids_ref: reference_token_ids.jsonl   # 64 token IDs from reference run
reference_distribution_ref: reference_sampling_distribution.npz   # 64 × vocab post-top-k-top-p softmax distributions
tolerances:
  token_id_must_match: true                  # sampled token MUST be byte-equal to reference (deterministic given seed)
  kl_divergence_max: 1.0e-4                  # post-top-k-top-p distribution KL(reference || mutated) <= threshold
parity_check_method: token_id_plus_kl_divergence
```

A mutated sampling kernel passes parity iff: (a) for every probe, the sampled token ID is byte-identical to the reference (deterministic test, easiest to verify), AND (b) the post-top-k-top-p softmax distribution has KL divergence from reference ≤ `kl_divergence_max`. Token-id-only would catch boolean correctness; KL-divergence catches subtle precision drift in the top-k/top-p selection that wouldn't change the sampled token but would shift probability mass.

#### 2.2.5 ~~KV cache fixture~~ (REMOVED in v0.3.1; KV cache is L0a-only, no L0c mutation target)

KV cache stays as L0a config knobs (`kv_cache_block_size`, `kv_cache_dtype`) per §0.6 #4. The standalone KV-cache fixture from v0.3.0 is not used in v0.3.1+. KV-cache correctness for L0a smoke parity is checked by the existing GatedAttn fixture (§2.2.1) since the GatedAttn forward pass exercises both KV write and KV read paths.

<details>
<summary>v0.3.0 schema (kept here for archival reference only — NOT in v0.3.2 active scope)</summary>

```yaml
# parity_fixture/kv_cache_v1.yaml
fixture_id: <family>-kv-cache-v1
generated_against:
  reference_baseline: <§2.2.0>
  reference_reproducibility_runs: 3
probe_count: 32
probe_token_lengths: [16, 256, 4096]         # span short to long-context regimes
probe_input_ref: kv_input_sequences.jsonl
reference_kv_state_after_write_ref: kv_state_post_write.npz   # full KV cache state after each token write
tolerances:
  rtol_kv_state: 1.0e-3   # tighter for FP16/BF16 paths
  atol_kv_state: 1.0e-3
  rtol_kv_state_fp8: 5.0e-3   # looser for FP8 KV path
  atol_kv_state_fp8: 5.0e-3
parity_check_method: kv_state_post_write_compare
```

KV-cache mutations are not a primary L0c target in v0.3.0 (paged-attention C++ is not source-mutable; only the Triton paged path is, and it's secondary). This fixture is included for completeness and for L0a parity-vs-reference checking when `kv_cache_dtype` changes. It catches FP8 KV scale drift before any KV-dependent measurement runs.

</details>

#### 2.2.6 Fused-epilogue fixture (intermediate-tensor parity at fusion boundary) — v0.3.1+ (active)

```yaml
# parity_fixture/fused_epilogue_v1.yaml
fixture_id: <family>-fused-epilogue-v1
generated_against:
  reference_baseline: <§2.2.0>
  reference_reproducibility_runs: 3
probe_count: 32
probe_input_attn_output_ref: attn_output_input.npz
probe_input_residual_ref: residual_input.npz
reference_post_residual_add_ref: post_residual.npz       # checkpoint 1: after residual add
reference_post_norm_ref: post_norm.npz                   # checkpoint 2: after RMSNorm
reference_post_quant_ref: post_quant.npz                 # checkpoint 3: after FP8 quantize
reference_downstream_logits_ref: downstream_logits.npz   # checkpoint 4: 32 layers downstream
tolerances:
  rtol_at_each_checkpoint: 1.0e-3
  atol_at_each_checkpoint: 1.0e-3
parity_check_method: four_checkpoint_compare
```

A mutated fused-epilogue kernel must match reference at all four checkpoints. Each checkpoint isolates a fusion boundary; a mutation that fuses incorrectly will fail the first checkpoint after the broken boundary, surfacing the precise step that's wrong. This is the **v0.3.0 thesis target** (priority #3) and gets the most thorough fixture.

#### 2.2.7 ~~FP8 GEMM Triton fixture~~ (REMOVED in v0.3.1; target was unreachable per P0.1)

FP8 GEMM Triton mutation target was dropped because vLLM's shipping FP8 GEMM path is CUTLASS/cuBLAS, not Triton. No Triton FP8 GEMM in the L0a action space → no L0c surface to mutate. Re-add when v0.3.3+ either lands a Triton FP8 GEMM in the action space or admits CUTLASS-side mutation.

<details>
<summary>v0.3.0 schema (kept here for archival reference only — NOT in v0.3.2 active scope)</summary>

```yaml
# parity_fixture/fp8_gemm_triton_v1.yaml
fixture_id: <family>-fp8-gemm-triton-v1
generated_against:
  reference_baseline: <§2.2.0>
  reference_reproducibility_runs: 3
probe_count: 32
probe_input_a_ref: gemm_input_a.npz   # FP8 input matrices
probe_input_b_ref: gemm_input_b.npz
reference_gemm_output_ref: gemm_reference_output.npz
reference_downstream_logits_ref: downstream_logits.npz
tolerances:
  rtol_gemm_output: 2.0e-3   # looser than other kernels — FP8 native rounding has stronger inherent error
  atol_gemm_output: 2.0e-3
  rtol_downstream_logit: 1.0e-3
  atol_downstream_logit: 1.0e-3
parity_check_method: gemm_output_plus_downstream_logit
```

Looser GEMM-output tolerances reflect FP8's inherent ~3-bit mantissa rounding; the downstream-logit tolerance stays tight because the model's logits are computed in FP16/BF16 reductions and shouldn't drift more than 1e-3 from any correct FP8 GEMM path.

</details>

### 2.3 Parity-fixture build pipeline

New script: `scripts/build_parity_fixture.py`. Run **once before L0a** against the §2.2.0 reference baseline. Re-built only when `weight_version_id` rotates. **Not** rebuilt when an L0a/L0b/L0c winner changes — the fixture is the round-independent ground truth that all three rounds are evaluated against.

### 2.4-pre Per-sibling parity fixtures (for P9 cross-family correctness gate)

P9's per-sibling parity probe (§7.2 P9 step b) requires a per-sibling parity fixture. These are **smaller versions** of the heavy-family fixtures — same structure, same reference baseline (§2.2.0), 16 probes per sibling instead of 64. Captured at the same time as the heavy-family fixture (P2b's scope expands to all 9 families).

Path convention: `benchmark_blueprints/families/<sibling_fid>/parity_fixture/{deltanet,gatedattn}_v1.yaml` and `.npz` companion files.

```yaml
# Example: benchmark_blueprints/families/codex-provider-rollover/parity_fixture/deltanet_v1.yaml
fixture_id: codex-provider-rollover-deltanet-v1
generated_at: <iso8601>
generated_against:
  vllm_version: <version>
  weight_version_id: <sha>
  reference_baseline:                       # SAME §2.2.0 reference as heavy fixture
    attention_backend: flash-attn-4
    deltanet_kernel: triton-chunked-delta-v2
    fp8_gemm_kernel: cublas
    torch_compile_mode: default
    cuda_graph_capture: off
  reference_reproducibility_runs: 3
probe_count: 16                             # smaller than heavy's 64 — fast cross-family check
probe_token_lengths: [16, 64, 256, 1024]
probe_input_ref: probes_input.jsonl         # 16 probes sampled from sibling's seed_trace_v5.jsonl
reference_logits_ref: deltanet_reference_logits.npz
reference_state_snapshots_ref: deltanet_reference_state.npz
state_checkpoints_at_token: [1, 1024]
tolerances:
  rtol_logit: 1.0e-3
  atol_logit: 1.0e-3
  rtol_state: 5.0e-3
  atol_state: 5.0e-3
parity_check_method: logit_plus_state_compare
```

Same shape for `gatedattn_v1.yaml` (logit-only, no state checkpoint).

`build_parity_fixture.py` accepts a `--family-id` flag and a `--probe-count` flag. P2b runs it once with `--family-id responses-sdk-adapter-cutover --probe-count 64` (heavy), then 8 times with `--family-id <sibling> --probe-count 16` for the siblings. Each invocation captures + validates reproducibility + writes the fixture pair.

**Why this is safe to capture in P2b:** the §2.2.0 reference baseline is identical across all 9 families — kernel selection doesn't depend on the workload — so a single reference vLLM lifecycle can serve all 9 fixture captures back-to-back. Reproducibility-3-times is verified per family.

### 2.4 Step-level verification — Workstream 1

| Step | PASS | FAIL — retry by agent | FAIL — escalate to human |
|---|---|---|---|
| Author `workload.yaml` | YAML schema-validates against parent §5.9 composite-family descriptor; `workload_distribution_id` matches canonical hash | n/a — deterministic file write | hash mismatch persists after re-derivation → bug in canonical hash function or descriptor schema drift |
| Capture heavy-family `seed_trace.jsonl` + `holdout_trace.jsonl` | Both files present; both pass thinking-probe row-3 (thinking fires when asked); `reasoning_tokens` non-zero | Re-run capture once if vLLM transient (OOM, timeout) | 2 consecutive captures yield `reasoning_tokens == 0` despite `<thinking>` content present → vLLM Responses API regression, halt and surface |
| Capture per-sibling holdouts (8 files, one per sibling family) at `benchmark_blueprints/families/<sibling_fid>/holdout_trace_v5.jsonl` | All 8 files present; each passes thinking-probe row-3; each captured against `vllm-default` baseline (recorded in capture metadata); files committed to main before any L0 round bootstraps | Re-run capture once per family on transient vLLM failure | A sibling family's capture repeatedly fails thinking-probe row-3 → that sibling cannot participate in P9; halt P1 and surface; human decides whether to drop the sibling from `component_families` upstream or fix the workload-stack issue |
| Build parity fixture (`build_parity_fixture.py`) against §2.2.0 reference baseline | Both `.yaml` + `.npz` files emitted; reference run reproduces logits bit-identically across 3 invocations on same vLLM | Re-run once on transient | Reference run is non-deterministic across runs (logits differ across same-input invocations) → kernel determinism is broken at the *baseline*; halt and surface — no point building a fixture that can't itself be reproduced |
| Fixture binds to weight version | `generated_against.weight_version_id` matches live `serving stack weight_version_id` | n/a | Mismatch detected → block all downstream rounds (L0a/b/c); surface for human decision (re-capture vs roll back weights) |
| Fixture exists BEFORE L0a runs | `parity_fixture/{deltanet,gatedattn}_v1.{yaml,npz}` all present at L0a round bootstrap | n/a | L0a refuses to bootstrap if any fixture file is missing; surface as `l0a_precondition_missing_fixture` |

---

## 3. Workstream 2 — L0a kernel selection

### 3.1 Action space (v0.3.0 expanded)

The v0.2.x action space (5 attention/GEMM/compile knobs, 180 combos) carries forward unchanged. v0.3.0 adds five new knobs covering RMSNorm, KV cache, sampling, and epilogue fusion. The grid grows from 180 to ~3,840 combos before pruning; the smoke phase's determinism + parity-vs-reference culls (§3.2) typically eliminate >90% of combos before any latency is measured, so the wall-clock impact is bounded by the number of survivors, not the grid size.

#### 3.1.1 v0.2.x knobs (unchanged)

| Knob | Candidates | Notes |
|---|---|---|
| `attention_backend` (GatedAttn) | `flash-attn-4`, `flash-attn-3`, `flash-attn-2`, `flashinfer`, `triton` | Per parent §2 Decision 2 + §11.10. FlashInfer has the #35138 accuracy issue on Qwen3.5 — must pass purity probe explicitly. **FA4 is beta on Blackwell** (vLLM gated behind `FLASH_ATTN_VERSION=4` env flag); GB10-specific perf table not published — see §10.6. |
| `deltanet_kernel` | `triton-chunked-delta-v1`, `triton-chunked-delta-v2`, `triton-state-update-fused` | Triton-only; vLLM doesn't ship FA-style kernels for linear attention. |
| `torch_compile_mode` | `default`, `reduce-overhead`, `max-autotune` | Cross-cutting. |
| `cuda_graph_capture` | `on`, `off` | On may trade compile time for steady-state speed. |
| `fp8_gemm_kernel` | `cutlass`, `cublas` | The FP8 GEMM under both DeltaNet and GatedAttn. CUTLASS tile schedulers are tuned for B200 SM counts and may underperform on GB10 — see §10.7. |

Subtotal: 5 × 3 × 3 × 2 × 2 = **180** v0.2.x combos.

#### 3.1.2 v0.3.1 new knobs (RMSNorm dropped, FP8 GEMM stays at `{cutlass, cublas}`)

| Knob | Candidates | Notes |
|---|---|---|
| `kv_cache_block_size` | `8`, `16`, `32` | PagedAttention block size. Smaller = less internal fragmentation; larger = better attention compute/memory ratio. Affects only the 16 GatedAttn layers (DeltaNet has no KV cache). Per vLLM PagedAttention design (https://docs.vllm.ai/en/latest/design/paged_attention/). |
| `kv_cache_dtype` | `fp16`, `bf16`, `fp8_e4m3`, `fp8_e5m2` | FP8 KV cache halves bandwidth pressure on the 16 GatedAttn layers. Requires explicit purity probe — FP8 KV scales can drift from reference logits beyond rtol/atol. |
| `sampling_kernel` | `flashinfer-sampling-from-probs`, `triton-top-k-top-p`, `torch-native-top-k`, `vllm-v1-triton-sampler` | Qwen3.5-27B-FP8's padded vocab is **248,320** (per HF model card, larger than v0.3.0's 152K estimate); top-k/top-p over that vocab is non-trivial. v1-engine Triton sampler is L0c-mutable; FlashInfer is vendor-only. |
| `epilogue_fusion_mode` | `none`, `attn_out+residual`, `attn_out+residual+rmsnorm`, `attn_out+residual+rmsnorm+fp8_quant` | Predicted #2 priority per §0.6 (provisional, P3a-gated). Each level fuses one more pass into a **Triton sidecar kernel** (NOT CUTLASS — see §5.2 and P0.5 fix). `attn_out+residual+rmsnorm+fp8_quant` is the v0.3.1 thesis target. Triton fused kernel must be the path; combinations that require CUTLASS-side epilogue support are not in v0.3.1 scope. |

Subtotal: 3 × 4 × 4 × 4 = **192** v0.3.1 combos from the new knobs.

**Total grid arithmetic (made explicit per P1 fix):**
- v0.2.x knobs: `attention_backend(5) × deltanet_kernel(3) × torch_compile_mode(3) × cuda_graph_capture(2) × fp8_gemm_kernel(2)` = **180** combos.
- v0.3.1 new knobs: `kv_cache_block_size(3) × kv_cache_dtype(4) × sampling_kernel(4) × epilogue_fusion_mode(4)` = **192** combos.
- Cartesian product: 180 × 192 = **34,560** raw combos.
- Static incompatibility pruning per §3.1.3 reduces to ≈ **5,000–8,000** eligible combos before smoke phase. Range, not a single number — actual count depends on which incompatibilities the matrix file declares.
- Smoke-phase determinism + parity-vs-reference culls then eliminate >90% before any latency is measured. Post-smoke survivor count is typically ≤ a few hundred, the same regime as v0.2.x.

#### 3.1.3 Combinatorial pruning hints (kernel_compatibility_matrix.yaml)

`benchmark_blueprints/kernel_compatibility_matrix.yaml` is checked into git, included in the round's BOOTSTRAP commit, and applied before smoke phase. Documented incompatibilities:

- `epilogue_fusion_mode != none` AND `fp8_gemm_kernel = cublas` → cuBLAS exposes limited epilogue surface; eliminate. `elimination_reason: epilogue_fusion_unsupported_on_backend`.
- `kv_cache_dtype ∈ {fp8_e4m3, fp8_e5m2}` AND `attention_backend = triton` → no Triton path for FP8 KV in vLLM main as of late 2025; eliminate. Reconsider when vLLM PR series lands.
- `epilogue_fusion_mode != none` AND `attention_backend ∈ {flash-attn-2, flash-attn-3, flash-attn-4, flashinfer}` → these are vendor attention kernels with their own epilogues; the Triton-sidecar fused-epilogue replaces the GEMM-output → residual → norm → quant chain that is downstream of the *attention output projection*, not downstream of the attention kernel itself. The combination is admissible. (No elimination; flagged here for clarity.)
- `sampling_kernel = flashinfer-sampling-from-probs` AND P7d L0c-Sampling round exists → P7d cannot mutate FlashInfer (vendor); P7d skipped per its own precondition. (No grid elimination; precondition handled in §5.2.)
- All other combinations: enumerated and run through the smoke phase. The matrix file is part of the round's BOOTSTRAP commit, so the eligible-combination set is auditable post-round.

### 3.2 Search strategy — deterministic grid + correctness gate

L0a is **not** an LLM-in-the-loop search. It's a finite, enumerable, deterministic grid sweep. The search strategy is three phases in order:

1. **Smoke phase** — two checks per combo, both required to pass:
   - **Determinism probe**: 4 sample inputs, run each twice on the same combo, logits must be bit-identical across the two runs. Catches non-deterministic kernels.
   - **Parity-vs-reference probe**: same 4 sample inputs run against the combo, logits must agree with the §2.2.0 reference fixture within `(rtol=1e-3, atol=1e-3)` for GatedAttn-only logits, and additionally state-snapshot at token 1 and 1024 within `(rtol=5e-3, atol=5e-3)` for DeltaNet kernels. Catches deterministic-but-wrong kernels (the FlashInfer #35138 class). A combo that passes determinism but fails parity is recorded with `elimination_reason: parity_diverges_from_reference` and `first_diverging_probe_index` populated.
   - This means **any L0a candidate measured in the screen phase is already known to be correct against the externally-trusted reference**. L0a cannot pick a kernel that "wins" by being wrong.
2. **Screen phase**: n=5 baseline replays at Screen profile + n=2 measurements per surviving combo at Screen profile.
3. **Top-K rescreen**: n=4 measurements at Screen profile per top 8 combinations. Picks single winner per kernel family (one `attention_backend`, one `deltanet_kernel`).

### 3.3 Parallelization

L0a is the workstream where parallelism pays back the most. The P2 driver therefore discovers the **maximum viable fanout on the current hardware** under repo-approved serving settings, then records that value as `l0a_parallel_fanout` for P3 scheduling. A four-vLLM-instance fanout on DGX Spark/GB10 at `gpu_memory_utilization: 0.2` remains the optimization target, but it is not a hard pass/fail requirement.

**vLLM fanout caveat.** GB10 has **128 GB unified LPDDR5x memory** (not 192 GB; this was a v0.2.x error corrected in v0.3.0) at ~273 GB/s aggregate bandwidth. A 4-way attempt at 4 × 0.2 = 0.8 utilization is below the 0.92 single-instance L1 winner, but each instance has only ~25.6 GB to work with and kernel-tuning rounds can carry extra compile/KV pressure. P2 probes candidate fanouts from the 4-way target downward until it finds the largest value whose endpoints start, accept fixture traffic, and satisfy the concurrency correctness check. Rejected fanouts record memory evidence (`gpu_memory_utilization`, free/used memory before and after startup, OOM/bind/timeout traces) plus a structured halt/rejection reason. If the maximum viable fanout is 1, P2 can pass as serial-only with explicit evidence, provided the single endpoint is healthy and deterministic. If the maximum viable fanout is 0, P2 blocks downstream phases. **Bandwidth implication:** GB10's 273 GB/s is ~30× lower than B200-class HBM3e — every saved memory pass is high-leverage; epilogue fusion (norm + residual + quant) becomes a top-tier optimization, see §0.5.

For every discovered `l0a_parallel_fanout > 1`, concurrent fixture results must equal sequential single-instance results byte-for-byte before that fanout is usable. A mismatch halts even if all instances start, because the parallel harness would be measuring different numbers than serial execution.

### 3.4 New CLI subcommand

```
lumoserve auto-research tune-kernel-select \
  --workload-file benchmark_blueprints/workloads/responses-sdk-adapter-cutover-heavy/workload.yaml \
  --action-space-file kernel_search/l0a_action_space.yaml \
  --baselines 5 \
  --screen-measurements-per-combo 2 \
  --rescreen-top-k 8 \
  --rescreen-measurements-per-candidate 4 \
  --parallel-instances auto \
  --round-root output/auto_research \
  [--harness real|synthetic]
```

`--parallel-instances auto` reads P2's discovered `l0a_parallel_fanout`.

Effects, in order:
1. Bootstrap a round in the same way `replay-round` does (worktree, BOOTSTRAP commit, codex-home bypassed — no agent in L0a). Verify both `parity_fixture_refs.deltanet` and `parity_fixture_refs.gatedattn` files exist; refuse to bootstrap with `HALT_REASON: l0a_precondition_missing_fixture` if either is missing.
2. **Smoke phase — TWO checks per combo, both required to pass before any measurement:**
   a. Serialize each combo's `kernel_select.yaml`, dispatch to a vLLM instance.
   b. **Determinism probe** (4 sample inputs, run twice each): logits bit-identical across the two runs. Combos that fail get `elimination_reason: nondeterministic`.
   c. **Parity-vs-reference probe** (same 4 inputs run once against the combo, compared to the §2.2.0 reference fixture loaded from `parity_fixture_refs[<kernel_target>]`): logits within `(rtol=1e-3, atol=1e-3)` for GatedAttn; additionally state-snapshot at token 1 and 1024 within `(rtol=5e-3, atol=5e-3)` for DeltaNet. Combos that pass determinism but fail this get `elimination_reason: parity_diverges_from_reference` + `first_diverging_probe_index` + `tolerance_overshoot`.
   d. Both checks ran for every combo; both must PASS for the combo to advance to step 4. Catches deterministic-but-wrong kernels (FlashInfer #35138 class) before any latency is measured.
3. Write `eliminated.tsv` with `elimination_reason` + `first_diverging_probe_index` for every culled combo. Per §3.5 AR.38 + AR.38b.
4. Run the screen phase: n=5 baselines + n=2 measurements per surviving combo. All measurements committed via the same `Candidate-UUID` trailer + `Fixture-Mode` machinery as `replay-round`.
5. Run the rescreen phase on top-8 combos: n=4 Screen measurements each.
6. Pick the winner by `objective_mean` over rescreen rows (per v0.1.5-plan §4.2 algorithm).
7. Re-run determinism + parity check on the winner specifically (AR.39); refuse to write the bundle if either fails.
8. Finalize → bundle records `layer_0_winner.attention_backend`, `layer_0_winner.deltanet_kernel`, etc. The kernel choices become the *baseline* for L0b autotune.

L0a result is not a final bundle — it's an intermediate substrate. The bundle's `round_provenance.round_type: l0a_select_only` marks it as such. v0.1.5-plan's §11.1 precondition refuses to load it as production; only L0b/c rounds use it as their baseline.

### 3.5 Verification (extends v0.1.5-plan AR list)

- **9.3.AR.38 L0a smoke-phase determinism culls.** `eliminated.tsv` lists every combo that failed determinism, with `elimination_reason: nondeterministic` + `first_diverging_probe_index` recorded. Surviving-combo count = total − eliminated. Artifact: count match.
- **9.3.AR.38b L0a smoke-phase parity-vs-reference culls.** `eliminated.tsv` additionally lists every combo that passed determinism but failed parity-against-reference, with `elimination_reason: parity_diverges_from_reference` + `first_diverging_probe_index` + `tolerance_overshoot` recorded. Survivors are determinism-clean AND parity-clean. Artifact: count match + per-row reason audit.
- **9.3.AR.39 L0a winner is determinism-clean AND parity-clean against reference.** Final winner combination passes 64-probe determinism check + 64-probe parity check against §2.2.0 fixture at the end of the round (independent of the smoke-phase culls). Artifact: explicit determinism log + parity_check.json with `pass: true`.
- **9.3.AR.40 L0a bundle marked intermediate.** `round_provenance.round_type == l0a_select_only`; production-mode `bootstrap-round` (for L1/L2 rounds) refuses to load it. Artifact: synthetic test.

### 3.6 Step-level verification — Workstream 2 (L0a)

| Step | PASS | FAIL — retry by agent | FAIL — escalate to human |
|---|---|---|---|
| Hardware-aware vLLM fanout discovery works against synthetic fixture | P2 discovers the maximum viable fanout `N >= 1` under repo-approved settings; records memory evidence and rejection reasons for every attempted fanout above `N`; records `l0a_parallel_fanout: N` for P3. If `N > 1`, N concurrent fixture requests return results equal to sequential single-instance results. If `N == 1`, the single endpoint is healthy and deterministic, and P3 runs serial-only. | Restart up to 3 times if an instance fails to bind port or a startup probe hits a transient timeout; then retry the same fanout before stepping down. | `N == 0` (no healthy single-instance endpoint under repo-approved settings) → halt. Any `N > 1` fanout whose concurrent results differ from sequential single-instance results → halt — concurrent harness produces different numbers than serial; surface for diagnosis. A fanout below 4 due only to memory/contention is not terminal if the recorded lower fanout passes. |
| Smoke phase produces `eliminated.tsv` (determinism + parity-vs-reference) | All 180 combos exercised; eliminated rows have `elimination_reason` ∈ {`nondeterministic`, `parity_diverges_from_reference`} + `first_diverging_probe_index`; survivor count > 0 | n/a — exhaustive sweep | Survivor count == 0 (every combo failed determinism or parity) → backbone unstable OR fixture itself is wrong; surface for diagnosis (likely Triton/CUDA toolchain regression OR fixture not captured against intended reference) |
| Smoke phase: parity-vs-reference culls capture deterministic-but-wrong combos | At least the FlashInfer combos that hit #35138 are eliminated by `parity_diverges_from_reference`, NOT by `nondeterministic` (FlashInfer's bug is stable-but-wrong, not noisy) | n/a | FlashInfer combos pass parity → either #35138 was fixed (good news, audit fixture) OR fixture is too loose (bad news, tighten tolerances); halt and surface |
| Screen phase n=5 baselines complete | All 5 baselines emit measurement traces; `inconsistent_baseline` flag NOT set per parent §9.3.x | Re-run individual baseline if vLLM transient | 2+ baselines flagged `inconsistent_baseline` even on retry → workload is not stable on this stack at all; halt and surface |
| Screen phase per-combo measurements | Every surviving combo has n=2 measurements committed with `Candidate-UUID` trailers; `commits_count == surviving_combos × 2` | Auto-retry one transient failure per combo | Same combo fails 2 times in a row on transient → mark combo as `wedged`; do not auto-promote; record in BLOCKED.md and continue with rest of grid; surface at round end if any wedged |
| Rescreen top-8 → winner pick | Winner has `objective_mean` strictly greater than runner-up by ≥ noise_floor in rescreen rows | Auto-retry rescreen of inconclusive top combos | Top-2 rescreen rows are within noise_floor → no defensible single winner; round emits `ROUND_INCONCLUSIVE`; surface so human picks the substrate (FA4 vs FA3 vs Triton) for L0b/c by hand |
| Bundle finalize | `round_provenance.round_type == l0a_select_only` + AR.38–40 artifacts present | n/a | AR.38/39/40 fail at finalize → bundle does NOT write; round halts; surface |

---

## 4. Workstream 3 — L0b kernel autotune

### 4.1 Action space — per kernel

For the **DeltaNet Triton kernel chosen at L0a**, the autotune surface is:

| Param | Range | Notes |
|---|---|---|
| `BLOCK_M` | `[16, 32, 64, 128]` | Triton autotune-supported. |
| `BLOCK_N` | `[16, 32, 64, 128]` | |
| `BLOCK_K` | `[32, 64, 128, 256]` | |
| `CHUNK_SIZE` | `[64, 128, 256, 512, 1024]` | DeltaNet-specific; affects both compute time and recurrent-state memory. |
| `num_warps` | `[2, 4, 8, 16]` | |
| `num_stages` | `[1, 2, 3, 4, 5]` | Pipeline depth. |

For the **GatedAttn kernel chosen at L0a** (only meaningful if winner was Triton; FA4/FA3/FA2/FlashInfer are vendor-tuned):

| Param | Range | Notes |
|---|---|---|
| `BLOCK_M` | `[16, 32, 64, 128]` | |
| `BLOCK_N` | `[16, 32, 64, 128]` | |
| `BLOCK_K` | `[32, 64, 128, 256]` | |
| `num_warps` | `[2, 4, 8, 16]` | |
| `num_stages` | `[1, 2, 3, 4]` | |

### 4.2 Search strategy — `@triton.autotune` + Welch-t

Triton's native autotuner picks based on first-run timings. Two known weaknesses we work around:

1. **First-run variance.** First-run timings are GPU-cold; Triton's autotuner can lock in the wrong winner. Mitigation: 5-warmup runs before autotune begins, recorded but discarded.
2. **Single-config-mode bias.** Default Triton autotune assumes static `(M, N, K)` shapes. Our workload has `(prompt_len, output_len)` variability across the 4-turn trajectory. Mitigation: parameterize autotune over the `(M, N, K)` shape distribution captured in the workload's `seed_trace_v5.jsonl` rather than over a single representative shape.

### 4.3 New CLI subcommand

```
lumoserve auto-research tune-kernel-autotune \
  --workload-file benchmark_blueprints/workloads/responses-sdk-adapter-cutover-heavy/workload.yaml \
  --base-bundle output/tuned_configs/responses-sdk-adapter-cutover-heavy/<sha>/<l0a-bundle.yaml> \
  --kernel-target {deltanet, gatedattn, fp8_gemm}     # one at a time
  --base-measurements 5 \              # paired-A/B: re-measure L0a baseline IN THIS ROUND
  --autotune-budget-minutes 60 \
  --measurement-rescreens 4 \
  --round-root output/auto_research \
  [--harness real|synthetic]
```

Effects, in order:
1. Load the L0a bundle to recover the chosen kernel selections.
2. **Paired-A/B baseline.** Run n=`--base-measurements` Screen measurements of the L0a winner kernel selection in the same round, in the same vLLM process lifecycle, immediately before autotune starts. These baseline measurements are committed with `Candidate-UUID` trailers + `Measurement-Role: l0a_baseline_remeasured` so they're auditable. **Welch-t comparison MUST use these contemporaneous baseline rows, not the L0a bundle's prior `objective_mean`** — otherwise the comparison crosses time windows / cache states / GPU thermal conditions.
3. Wrap the chosen kernel with `@triton.autotune` over the §4.1 parameter ranges.
4. Drive a workload replay; Triton autotune picks winner per `(M, N, K)` shape it sees.
5. Once autotune stabilizes (no new winner picked in 10 consecutive replays): freeze the winning configs.
6. Run n=`--measurement-rescreens` Screen measurements with the frozen autotune winners → confidence derivation per v0.1.5-plan §4.2 against the paired baseline (step 2).
7. Finalize bundle marks `round_type: l0b_autotune`; bundle's `layer_0_*.l0b_autotune.per_kernel_params` populated; bundle records both `paired_baseline_objective_mean` and `autotune_winner_objective_mean` so future readers can see the pair.

### 4.4 Verification

- **9.3.AR.41 L0b autotune frozen, not still running.** Bundle's `layer_0_*.l0b_autotune.frozen_at: true`. Trace records the warmup count + the stable-window count (10 replays without new winner). Artifact: trace inspection.
- **9.3.AR.41b L0b paired-A/B baseline measured in same round.** Round commits include n=`--base-measurements` rows with `Measurement-Role: l0a_baseline_remeasured` trailer. Welch-t for the L0b vs L0a comparison uses these contemporaneous rows, NOT the L0a bundle's prior `objective_mean`. Artifact: trailer grep + Welch-t input audit.
- **9.3.AR.42 L0b winner determinism-clean AND parity-clean.** Same 64-probe determinism check as L0a + parity-vs-reference check against §2.2.0 fixture, against the frozen winners. Artifact: determinism log + parity_check.json with `pass: true`.

### 4.5 Step-level verification — Workstream 3 (L0b)

| Step | PASS | FAIL — retry by agent | FAIL — escalate to human |
|---|---|---|---|
| `@triton.autotune` wiring against chosen DeltaNet kernel | Autotune compiles + runs; emits per-shape winners; logits bit-identical to pre-autotune kernel for same input | n/a | Autotune wiring changes logits even before tuning (autotune wrapper itself is non-correctness-preserving) → halt; surface — `@triton.autotune` integration is broken |
| Stable-window detection (10 replays without new winner) | Reaches stable window before `autotune-budget-minutes`; `frozen_at: true` written | Extend budget once if stable window not reached and budget exhausted | Stable window never reached even after extension → autotune is oscillating; surface — likely shape distribution is too wide to autotune over jointly |
| Frozen-winner determinism probe | 64-probe determinism PASS against the frozen autotune winners | Re-run determinism probe once on transient | Determinism FAIL after autotune freeze → autotune chose a non-deterministic config; halt; surface for human review (autotune weighting bug or Triton bug) |
| L0b winner > L0a winner | Welch-t CI strictly above zero at `confidence: defensible` | n/a — pure statistical outcome | L0b CI within noise floor of L0a → write `ROUND_NULL_RESULT` bundle; surface — autotune found no headroom, human decides whether to ship L0a as v0.2 reference or proceed to L0c with L0a winner as base |

---

## 5. Workstream 4 — L0c kernel mutation (the Karpathy core)

### 5.1 The Karpathy pattern, applied to kernels

Karpathy's `autoresearch` lets one agent edit `train.py`, run, check `val_bpb`, keep on improvement, discard on regression. ~12 experiments/hour. We apply the same shape with three differences:

| Karpathy `autoresearch` | v0.2 `mutate-kernel` |
|---|---|
| Agent edits `train.py` (Python source) | Agent edits **one chosen kernel source file** (e.g., `kernels/deltanet/chunked_delta.py` or its CUDA equivalent). One file at a time per round. |
| Metric: `val_bpb` (model-derived; self-corrects on training breakage) | Metric: `eval_throughput`, gated by **parity check** against the §2.2 fixture. A faster-but-wrong kernel cannot keep — parity is checked **before** latency counts. |
| Loop ends when human interrupts | Loop ends when iteration cap reached, OR 3-in-a-row parity failures (proposer is stuck), OR 3-in-a-row compile failures (proposer is broken). Same Karpathy "never stop until terminal" loop. |
| Single GPU | Single GPU + isolated build directory per attempt (no kernel `.so` cross-contamination). |

**The parity gate is the load-bearing primitive.** Every mutation runs against §2.2 fixture **before** its latency is measured. A mutation that fails parity:
- Does NOT count as an iteration toward the iteration cap (Karpathy "this experiment was a nothing-burger" semantics).
- IS recorded in `mutations_rejected.tsv` with the first-diverging probe index + tolerance margin so the proposer learns from rejections.
- Triggers an `inconsistent_rescreen`-equivalent flag on the agent's transcript so the proposer knows not to re-propose the same edit.

### 5.2 Per-kernel-target L0c strategy (v0.3.1 narrowed — 3 executable targets)

L0c runs **per kernel target**. v0.3.1 narrows from v0.3.0's six targets to **three executable targets** plus one schema-only target (GatedAttn). The narrowing addresses P0.1 (fp8_gemm_triton was unreachable), P0.5 (fused_epilogue Triton-vs-CUTLASS contradiction), and the operator directive ("standalone RMSNorm not in scope; mutation surface should be tight"). The Karpathy spawn-per-iteration loop, three-cap structure, paired-A/B baseline, and watchdog rails are unchanged across targets — only the kernel-source-path, parity fixture, and `iteration_brief.md` template differ.

**Sequencing inversion (v0.3.1/v0.3.2).** Per operator note: prior v0.1/v0.2 rounds confirmed L0a/L0b have no defensible headroom against `vllm-default` baseline within noise floor. v0.3.1+ therefore runs **L0c FIRST against `vllm-default` baseline directly** — not against an L0a/L0b winner. The paired-A/B baseline measured in each L0c round is `vllm-default` resolution. L0a/L0b are deferred (and remain as v0.3.3+ exploration phases if any L0c result suggests config-space headroom on top of the kernel mutation).

**v0.3.3 base stack — L0b-empirical-winner bundle, NOT hand-authored (impl-agent amendment).**

v0.3.2 specified a hand-authored fixed base stack (FA3 + Triton sidecar + Triton sampler + bf16 KV). Implementation-agent review of the on-disk L0b winner bundles found:
- 22 L0b winner bundles on disk total.
- The 8 FA4 bundles have **identical synthetic objective values** (1.18125) — synthetic-harness fingerprint, not real measurements.
- The vllm-default bundles have real measured values varying 0.0157–1.286.
- The most-recent real-harness L0b winner is **`4866bc3f`** (`obj=0.0171, ROUND_NULL_RESULT`) — empirically grounded.

Comparing the v0.3.2 hand-authored stack against `4866bc3f`:

| Knob | `4866bc3f` (measured) | v0.3.2 (hand-authored) | Empirically validated? |
|---|---|---|---|
| `attention_backend` | **`vllm-default`** | `flash-attn-3` | 4866bc3f: yes (L0b real winner). v0.3.2: no — FA3 picked for "stable/explicit" reasoning |
| `deltanet_kernel` | `triton-chunked-delta-v2` | `triton-chunked-delta-v2` | both: yes (same) |
| `fp8_gemm_kernel` | `cublas` | `cublas` | both: yes (same) |
| `kv_cache_dtype` | **`fp8_e5m2`** | `bf16` | 4866bc3f: yes (production-deployed). v0.3.2: bf16 is "conservative" not measured-better |
| `epilogue_fusion_mode` | (default — likely `none`) | sidecar enabled | 4866bc3f: status quo. v0.3.2: requires unwired Triton sidecar |
| `sampling_kernel` | (default) | `vllm-v1-triton-sampler` | 4866bc3f: status quo. v0.3.2: requires Triton sampler wiring not verified |

**v0.3.3 amends to anchor on `4866bc3f`** rather than the hand-authored guess. The reasoning:
- **Empirical > inferred.** `4866bc3f` is where L0b actually settled across multiple real-harness rounds. The v0.3.2 stack was an architectural-coherence argument, not a measurement.
- **Captured fixture data validates against `4866bc3f`.** ~14 GB of parity fixture data was already captured against this L0b-winner stack (vllm-default routing + fp8_e5m2 KV). Amending the doc to bf16 would invalidate this data and require ~5h of re-capture work for no measured benefit.
- **`vllm-default` and `fp8_e5m2` are production-deployed today** — `flash-attn-3` and `bf16` are theoretical-better-defaults that haven't been measured to win.

**v0.3.3 base stack specification (replaces the v0.3.2 fixed table):**

```yaml
# benchmark_blueprints/workloads/multi-family-v5-l0c-composite/base_stack.yaml
base_stack_source: l0b_empirical_winner_bundle
base_bundle_path:  output/tuned_configs/.../4866bc3f/bundle.yaml
base_bundle_objective_mean: 0.0171
base_bundle_round_outcome: ROUND_NULL_RESULT     # i.e., L0b found no headroom; L0c starts from this stack
actually_resolved_kernel_selection:               # MUST be captured at fixture-build time
  attention_backend: <resolved by vLLM at base_bundle build time>
  deltanet_kernel:   triton-chunked-delta-v2
  fp8_gemm_kernel:   cublas
  kv_cache_dtype:    fp8_e5m2
  kv_cache_block_size: <resolved>
  torch_compile_mode: <resolved>
  cuda_graph_capture: <resolved>
  vllm_version:       <pinned at capture>
  weight_version_id:  <pinned at capture>
```

**`actually_resolved_kernel_selection` is load-bearing.** vLLM's `vllm-default` routing changes across versions, env vars, and Blackwell-specific selection logic. The symbolic alias `vllm-default` is NOT stable identity — what matters is what it resolved to at fixture-capture time. Recording the resolved values at capture pins the base stack to a concrete, reproducible configuration.

Mutators (P7a in v0.3.3, P7e/P7d in v0.3.4+ when wired) build on top of `actually_resolved_kernel_selection`, NOT on top of `vllm-default` symbolic. If the resolved values change between rounds (e.g., vLLM upgrade), the round must re-capture or refuse to bootstrap.

| `--kernel-target` | Source path | Parity-check semantic | Skip condition | §0.6 rank |
|---|---|---|---|---|
| `deltanet` | `kernels/deltanet/chunked_delta.py` (Triton) | logit + state-snapshot at token 1 and 1024 (§2.2.2) | never skipped | #1 |
| `fused_epilogue` | `kernels/fused/attn_residual_norm_quant.py` (**Triton sidecar — no CUTLASS path in v0.3.1**) | 4-checkpoint compare: post-residual-add, post-norm, post-quant, downstream-logit (§2.2.6) | `epilogue_fusion_mode == none` (no fusion to mutate). NO CUTLASS-side precondition — Triton-sidecar always available regardless of `fp8_gemm_kernel` | #2 |
| `sampling` (gated) | `kernels/sampling/triton_sampler.py` | sampled-token-id parity (deterministic given seed) + post-top-k-top-p KL divergence (§2.2.4) | (a) L0a winner is `flashinfer-sampling-from-probs` or `torch-native-top-k` (vendor / non-Triton); OR (b) **P3a roofline measures sampler share < 3% of Tpot** (sampler-cost-too-small-to-justify-mutation gate per §0.6) | #3 |
| `gatedattn` *(schema-only in v0.3.1)* | `kernels/gatedattn/<chosen_triton>.py` (Triton) | logit-only (§2.2.1) | **almost always skipped in practice**: L0a winner is vendor (FA4/FA3/FA2/FlashInfer) for ~all combinations the smoke phase admits; round only runs if a Triton GatedAttn kernel is the L0a winner AND P3a roofline shows attention-output share > 5% of Tpot | (rarely actionable) |

**Removed from v0.3.1** (was in v0.3.0): `rmsnorm` (subsumed into `fused_epilogue`; no standalone round), `fp8_gemm_triton` (P7f was unreachable per P0.1; no Triton FP8 GEMM in vLLM shipping path). To re-add either, v0.3.2 must (a) add a mutable shipping kernel to the L0a action space, (b) define a roofline justification for it, and (c) author the corresponding parity fixture.

**Skip-condition handling.** A skipped target produces no L0c round (no `ROUND_NULL_RESULT` artifact, no rounds counted toward §8.4). Skip metadata is recorded in `output/p7_skipped_targets.tsv` with structured reason: `l0a_winner_is_vendor`, `epilogue_fusion_disabled`, `roofline_share_below_threshold`, or `triton_kernel_unavailable`. The composite descriptor's `kernel_mutations` list reflects which targets actually ran AND produced winners — the same mutated-kernel-set invariants from v0.2.x apply (AR.48d nested eligibility check ensures `parity_fixture_refs[family].keys() == set(kernel_mutations[].kernel_target)`).

**Parallelizability across targets.** L0c rounds for `deltanet`, `fused_epilogue`, and (when ungated) `sampling` are independent — they mutate disjoint source files, use disjoint parity fixtures, and produce disjoint candidate commits. They can run in parallel on the same GB10 (constrained by §3.3 multi-instance limits) or sequentially.

**Priority-driven ordering when GPU time is constrained.** §0.6 priority order dictates: `deltanet` → `fused_epilogue` → `sampling` (if ungated). Per AR.54, GPU-budget-skipped targets must be a suffix of this priority. A round that ran `sampling` but skipped `fused_epilogue` due to budget is invalid.

### 5.3 New CLI subcommand

```
lumoserve auto-research mutate-kernel \
  --workload-file benchmark_blueprints/workloads/responses-sdk-adapter-cutover-heavy/workload.yaml \
  --base-stack-resolution {vllm_default, reference_baseline, bundle} \    # v0.3.2: 'vllm_default' is the v0.3.1/v0.3.2 default (L0c-first ordering — no L0b winner exists yet). 'bundle' takes a path to an L0a/L0b bundle and is the v0.3.3+ path when L0a/L0b run.
  --base-bundle-path <bundle.yaml>                                        # required iff --base-stack-resolution=bundle. ignored otherwise.
  --kernel-target {deltanet, fused_epilogue, sampling, gatedattn} \    # v0.3.1: 3 executable + 1 schema-only (gatedattn rarely runs)
  --kernel-source-path kernels/deltanet/chunked_delta.py \
  --parity-fixture benchmark_blueprints/families/responses-sdk-adapter-cutover/parity_fixture/deltanet_v1.yaml \
  --base-measurements 5 \                # paired-A/B: re-measure L0b baseline IN THIS ROUND
  --accepted-iteration-cap 12 \          # caps parity-passing measured candidates
  --total-attempt-cap 36 \               # caps total spawn count regardless of accept/reject
  --round-timeout-hours 12 \             # round-level wall-clock cap (parent §5.7 rail compatible)
  --round-root output/auto_research
```

**Three-cap semantics (P1.4 fix).** The Karpathy "never-stop-until-terminal" loop has three independent terminal conditions, evaluated after each iteration:
- `accepted_iteration_cap` — count of mutations that **passed parity AND were measured**. Default 12. Hitting this cap is the "successful round terminates" path.
- `total_attempt_cap` — count of total spawn-and-evaluate attempts regardless of accepted/rejected outcome. Default 36 (3× accepted cap). Hitting this cap means too many rejected mutations relative to accepted ones — round terminates with `HALT_REASON: total_attempt_cap_reached` and surfaces. Without this cap, an alternating fail/pass proposer could spin forever.
- `round_timeout_hours` — round-level wall-clock cap (independent of parent §5.7 per-attempt watchdog). Hitting this cap terminates with `HALT_REASON: round_timeout`.

The 3-in-a-row halt conditions (`proposer_stuck`, `compile_failures_3x`) remain in addition; they fire faster than the cap conditions when the proposer is clearly broken. Agent infrastructure failures are separate: `agent_rate_limited` / `agent_unavailable` terminate the round as blocked but do **not** write synthetic mutation-rejection rows unless a real `mutation.patch` + `parity_check.json` already exists on disk.

Effects, in order:
1. Bootstrap a round (worktree, BOOTSTRAP commit, codex-home, iteration_brief.md generated for L0c).
2. **Paired-A/B baseline (resolved per `--base-stack-resolution`).** Run n=`--base-measurements` Screen measurements of the resolved base stack in the same round, same vLLM process lifecycle, before any mutation is proposed. The base stack and `Measurement-Role` trailer are determined by `--base-stack-resolution`:
   - `vllm_default` (v0.3.1/v0.3.2 default — L0c-first ordering): runs `vllm-default` resolution; trailer `Measurement-Role: vllm_default_baseline_remeasured`.
   - `reference_baseline`: runs the §2.2.0 externally-trusted reference stack (forced FA4 + Triton DeltaNet defaults + cuBLAS + default torch_compile + cuda_graph off); trailer `Measurement-Role: reference_baseline_remeasured`.
   - `bundle` (v0.3.3+ path): loads `--base-bundle-path`, runs that bundle's resolved stack; trailer `Measurement-Role: <round_type>_baseline_remeasured` (e.g., `l0b_autotune_baseline_remeasured` if the bundle is from L0b). All accepted-mutation Welch-t comparisons use these rows.
3. For each attempt while `accepted < accepted_iteration_cap` AND `total_attempts < total_attempt_cap` AND `wall_clock < round_timeout_hours`:
   a. Spawn ONE `codex exec` process. Brief tells codex: read `kernels/deltanet/chunked_delta.py`, read `mutations_rejected.tsv` (prior failures), propose a `.patch` diff, write it to `candidates/<NNN>/mutation.patch`.
   b. `total_attempts += 1` (increments unconditionally — every spawn counts toward total cap).
   c. Apply the patch in an isolated build directory: `git -C <build_dir> apply mutation.patch`.
   d. Compile: `cd <build_dir> && make kernels/deltanet/<kernel>.so`. Timeout 20 min per parent §5.7 rail 1.
   e. Run parity check (§6.2). If FAIL: write `mutations_rejected.tsv` row with first-diverging probe + tolerance, do NOT increment `accepted`, return to step 3a.
   f. If parity PASS: run n=2 Screen-profile measurements against the workload. Record in `results.tsv`. `accepted += 1`.
   g. If improvement above noise floor against the paired baseline: keep, advance best_so_far. If regression: discard.
   h. `commit-candidate` with the mutation patch + measurement trace. Trailers: `Candidate-UUID`, `Mutation-Hash: <sha of patch>`, `Measurement-Role: l0c_candidate`, `Signed-off-by`.
4. After loop terminates: rescreen top-K accepted mutations at full Screen-profile measurement (n=4 each).
5. Holdout validation (parity + measurement) on the rescreen winner.
6. Finalize: bundle records `layer_0_*.l0c_mutation.diff_ref` pointing at the winning patch + `parity_attestation` block + `weight_sensitive` flag + `paired_baseline_objective_mean` + `terminal_condition` (which of the three caps fired, or `accepted_cap_reached`).

### 5.4 Cycle-time engineering — keep agent unblocked

Triton kernels recompile fast but CUDA can be slow. Three decisions to keep the loop tight:
- **Default `kernel_target: deltanet`** — Triton kernels (faster compile than CUDA).
- **Cache compiled kernels by `Mutation-Hash`** — re-running an already-tested mutation skips compile.
- **Parallel compile + parity-probe pipeline** — while attempt N is measured, attempt N+1's compile runs.

Per-attempt budget is enforced as a hard timeout per parent §5.7 watchdog rails (compile timeout 20 min; total per-attempt timeout 30 min). Attempts exceeding the timeout are killed and recorded in `mutations_rejected.tsv` with `rejection_reason: timeout`.

### 5.5 What the agent's `iteration_brief.md` says

Per v0.1.11 §5.2, brief is short and instructs the agent. v0.2 brief for L0c (composed by `mutate-kernel` at bootstrap):

```markdown
You are an autonomous kernel-research agent for iteration {{iteration}} of round {{round_id}}.

# Your one job
Propose ONE mutation to {{kernel_source_path}} that is faster than the current
best on the workload, AND passes the parity gate at
{{parity_fixture_path}}.

# Hard rules
- Edit ONLY {{kernel_source_path}}. No other file.
- Do not change the kernel's input/output signature.
- Do not change tile or grid sizes outside the autotune surface (those
  belong to L0b, not L0c).
- Read mutations_rejected.tsv. Mutations identical to a prior rejection
  by patch hash are immediately rejected without re-running. Read the
  rejection reasons (first_diverging_probe, tolerance_overshoot) and
  propose something genuinely different.
- Your mutation MUST pass parity. Latency is irrelevant if parity fails.

# Parity contract
- Logit-space tolerance: {{rtol_logit}} / {{atol_logit}}
- (DeltaNet only) State-snapshot tolerance: {{rtol_state}} / {{atol_state}}
- Recurrent-state checkpoints at: {{state_checkpoints_at_token}}

# Procedure
1. Read {{kernel_source_path}}, mutations_rejected.tsv, results.tsv (best_so_far).
2. Write your proposal to {{iteration_dir}}/mutation.patch.
3. Run: lumoserve auto-research apply-and-test --round-id {{round_id}} \
     --iteration {{iteration}} --kernel-target {{kernel_target}}
4. Read the result. If parity fails, write a one-line note to BLOCKED.md
   explaining what you'll try next. Do NOT propose the same edit again.
5. Exit 0.

# What you do NOT do
- You do not call finalize-round. Python does that.
- You do not run measurement directly. The CLI does that.
- You do not write any file except mutation.patch and BLOCKED.md.

# DGX Spark hardware notes (embedded — do NOT do online research)
The system you are tuning for is NVIDIA DGX Spark with the GB10 Superchip.
These facts are pre-researched and load-bearing for kernel decisions:
- 128 GB unified LPDDR5x memory at ~273 GB/s aggregate bandwidth.
  This is ~30x lower BW than B200 HBM3e. Most decode-path kernels are
  bandwidth-bound, NOT compute-bound. Eliminating memory passes (fused
  epilogues, in-place ops, async copy pipelining) is high-leverage.
- Blackwell architecture (sm_100/sm_101). 5th-gen Tensor Cores with
  native FP8/FP6/FP4. TMA (Tensor Memory Accelerator), Thread Block
  Clusters / DSMEM, and `tcgen05` PTX MMAs are available.
- 2-CTA MMA shape on Blackwell favors 256x256-class GEMM tiles vs
  Hopper's 128x256. CUTLASS bundled tile schedulers target B200 SM
  counts; GB10 has fewer SMs and may want different tile sizes —
  empirically discover.
- SM count, register file per SM, shared memory per SM, L1/L2 cache
  sizes are NOT publicly documented for GB10. Do not assume B200
  values. If your mutation is sensitive to register pressure, prefer
  conservative tile sizes and let L0b autotune (later phase) widen.
- FA4 is beta on Blackwell (vLLM gated behind FLASH_ATTN_VERSION=4
  env flag). FA3 is the production-stable path.

# DGX Spark speedup tips for L0c agents
For DeltaNet (recurrent-state update kernel):
- Recurrent-state writeback is the dominant memory traffic. Coalescing
  writes and using cp.async / TMA bulk copies saves more than tile
  reshape.
- num_warps=4 is rarely the right answer on GB10's small-SM-count
  profile; try num_warps=2 or num_warps=8 for narrow tiles.
- chunk_size affects compute vs recurrent-state-memory tradeoff;
  smaller chunks = more launches, larger chunks = more state held
  in SMEM/registers.

For fused_epilogue (Triton sidecar kernel attn_residual_norm_quant.py):
- The chain is: attn_output_proj_GEMM_output -> +residual -> RMSNorm
  -> FP8_quantize. Each arrow is a memory pass at default. Fusing
  saves 2-3 memory passes per layer x 64 layers x thousands of tokens.
- The Triton kernel takes the GEMM output as input (already in SMEM
  ideally) and writes the FP8-quantized output once. RMSNorm reduction
  is a within-tile reduction; SMEM-only.
- Watch out for: numerical precision in the RMSNorm sqrt-rsqrt step
  (FP32 reduction required for stability), and FP8 quantization scale
  computation (per-token vs per-tensor matters).

For sampling (Triton sampler — only if P3a roofline shows >=3% Tpot share):
- Qwen3.5-27B-FP8 padded vocab is 248,320 (NOT 152K). top-k over this
  vocab is non-trivial.
- FlashInfer ships fused top-k/top-p with rejection sampling that
  avoids explicit sorting. Beating that requires either better
  rejection-sampling iteration or fundamentally different algorithm.
- KL divergence between mutated and reference distributions is the
  load-bearing parity check, not just sampled-token-id equality.

# References (do not browse; these are the sources behind the embedded facts above)
- NVIDIA DGX Spark spec: 128 GB LPDDR5x / 273 GB/s aggregate BW.
- NVIDIA Blackwell architecture: 5th-gen TC, FP8/FP4 native, tcgen05.
- LMSYS DGX Spark inference benchmark: confirms LPDDR5x BW is the
  practical inference bottleneck on this box.
- vLLM PagedAttention design doc: BLOCK_SIZE 8/16/32 affects KV
  layout and fragmentation.
- HF Qwen3.5-27B-FP8 model card: 64-layer hybrid layout
  (16 x (3 GatedDeltaNet -> FFN + 1 GatedAttn -> FFN)),
  padded vocab 248,320.
- FlashAttention repo: FA4 beta on Blackwell as of late 2025.
- flash-linear-attention repo: source of the DeltaNet Triton kernels
  vLLM calls.
```

### 5.6 Verification

- **9.3.AR.43 Parity gate ran (or didn't, with structured reason) for every attempt, before measurement.** Every iteration directory `candidates/<NNN>/` MUST contain `parity_check.json` with one of these structured shapes:
  - **(a) Parity ran AND passed** → `{pass: true, reason: ran_passed, ...}` AND `measurement_trace.json` exists in the same directory.
  - **(b) Parity ran AND failed** → `{pass: false, reason: parity_logit_diverged | parity_state_diverged | intermittent_parity, first_diverging_probe: ..., tolerance_overshoot: ...}` AND no `measurement_trace.json` exists.
  - **(c) Parity did NOT run because a pre-parity step failed** → `{pass: false, reason: patch_apply_failed | compile_timeout | compile_nvcc_error | sandbox_setup_failed, error_detail: <string>}` AND no `measurement_trace.json` exists.

  Illegal combinations (verifier refuses):
  - `parity_check.json` missing entirely (every spawned attempt must record outcome).
  - `pass: false` of any reason AND `measurement_trace.json` exists (faster-but-wrong leakage).
  - `pass: true` AND `measurement_trace.json` missing (pass without measurement is a dropped result).

  Artifact: per-iteration `parity_check.json` schema check + `measurement_trace.json` presence check matrix.
- **9.3.AR.44 Correct kernel-family fixture used.** `parity_fixture_refs.deltanet` and `parity_fixture_refs.gatedattn` are **path strings** in `workload.yaml` (per §2.1 schema), not objects. Verifier procedure:
  1. Read `workload.yaml`; resolve `fixture_path = parity_fixture_refs[<kernel_target>]` (a path string).
  2. Load the fixture yaml at `fixture_path`; read its top-level `fixture_id` field.
  3. Assert this `fixture_id` equals `parity_check.json.fixture_id`.

  Artifact: trace cross-check across workload yaml → fixture yaml → parity_check.json.
- **9.3.AR.45 DeltaNet mutations check both probes (token 1 + token 1024).** For DeltaNet rounds, `parity_check.json.checkpoints_checked == [1, 1024]`. Logit-only check is insufficient. Artifact: trace inspection.
- **9.3.AR.46 Rejected mutations don't count toward iteration cap.** `results.tsv` row count = passed-parity count, not total-attempts count. `mutations_rejected.tsv` contains the rejected attempts. Artifact: row count match.
- **9.3.AR.47 Mutation patch hash matches trailer; patch applies to declared base.** Two-part check:
  - (a) `sha256(candidates/<NNN>/mutation.patch)` equals the commit's `Mutation-Hash` trailer value (the trailer is the hash of the patch file itself, not of any source file).
  - (b) The patch applies cleanly via `git apply --check` against the declared `kernel_source_path` at the round's `BOOTSTRAP` commit's tree. A patch that hash-matches its trailer but no longer applies to the declared base is rejected.

  Artifact: per-commit hash recompute + `git apply --check` dry-run.
- **9.3.AR.48 Weight-sensitive flag honored.** Mutations whose proposer-annotated `weight_sensitive: true` are flagged in the bundle's `layer_0_*.l0c_mutation.weight_sensitive`. Artifact: bundle field check.
- **9.3.AR.48b L0c three-cap structure recorded.** Bundle's `layer_0_*.l0c_mutation.terminal_condition` ∈ {`accepted_cap_reached`, `total_attempt_cap_reached`, `round_timeout`, `proposer_stuck`, `compile_failures_3x`, `intermittent_parity_observed`, `agent_rate_limited`, `agent_unavailable`}. Counter values `accepted_count`, `total_attempt_count`, `wall_clock_minutes` recorded. For kernel-level terminal conditions, accepted_count + (count of `mutations_rejected.tsv` rows) = total_attempt_count exactly. For agent-infrastructure terminal conditions, `mutations_rejected.tsv` includes only attempts that produced a real `mutation.patch` + controller-readable verdict artifact before the agent failed. Artifact: counter audit.
- **9.3.AR.48c L0c paired-A/B baseline measured in same round.** Round commits include n=`--base-measurements` rows with `Measurement-Role: <stack>_baseline_remeasured` trailer where `<stack>` matches `--base-stack-resolution` (`vllm_default`, `reference_baseline`, or the bundle's `round_type` for `bundle`). All accepted-mutation Welch-t comparisons use these contemporaneous rows, NOT any prior round's `objective_mean`. v0.3.1/v0.3.2 L0c-first rounds use `vllm_default_baseline_remeasured`. Artifact: trailer grep + Welch-t input audit.
- **9.3.AR.48c2 P9 paired sibling baselines.** For each parity-passing sibling family, the P9 round commits include n=5 measurements with `Measurement-Role: sibling_baseline` (running `vllm-default` resolution) AND n=5 measurements with `Measurement-Role: sibling_winner` (running the kernel-tuned bundle). Per-sibling Welch-t comparisons consume only contemporaneous (sibling_baseline, sibling_winner) pairs — never historical `objective_mean`. Total per round: up to 80 measurement commits (8 parity-passing siblings × 10), fewer if any sibling failed parity in step (b). Artifact: trailer grep + per-family pair audit.
- **9.3.AR.48c3 P1 sibling holdout pre-capture (verified at P9).** For each of the 8 sibling families, P1 produced `benchmark_blueprints/families/<sibling_fid>/holdout_trace_v5.jsonl` and committed it to main. P9's step-(a) staleness check verifies for each: file present; thinking-probe row-3 still passes; `weight_version_id` at capture matches current. Heavy family's holdout is the §2.1 capture. Artifact: P1 capture log + P9 staleness-check log per family.
- **9.3.AR.48c4 P9 sibling parity probe (per sibling × per mutated kernel target) + complete partition + composite exclusion.** A sibling passes P9 parity ONLY if it passes against **every** mutated kernel target in `kernel_mutations`. Concretely: for each sibling family AND each `kernel_target` ∈ `{m.kernel_target for m in kernel_mutations}` (so dual-mutation rounds check both DeltaNet AND GatedAttn per sibling), `parity_check_sibling_<sibling_fid>_<kernel_target>.json` exists with `pass: true|false`, `probe_count: 16`, `fixture_id` matching `benchmark_blueprints/families/<sibling_fid>/parity_fixture/<kernel_target>_v1.yaml`. The TSV-derivation rule is mechanical, with canonical row shapes that byte-equality reconstruction relies on:

  - **`sibling_parity_passes.tsv`** — single column header `family_id`. One row per sibling that passed parity for ALL mutated kernel targets. Rows sorted by `family_id` ascending.

  - **`sibling_parity_failures.tsv`** — header `family_id\tfailing_kernel_target\tfirst_diverging_probe_index\ttolerance_overshoot`. **One row per (sibling, failing_kernel) pair** — a sibling that fails both DeltaNet AND GatedAttn produces TWO rows, not one. Rows sorted lexicographically by `(family_id, kernel_target)` ascending. The partition's failure set is the projection `set(failures.tsv.family_id)` (drops duplicates; a sibling on two rows still maps to one partition entry).

  Any other shape (e.g. one-row-per-sibling with a singular `failing_kernel_target`, or unsorted rows, or rows with multi-value `failing_kernel_target` fields) makes byte-equality reconstruction impossible and surfaces as `sibling_parity_partition_invalid` at P10.

  The §8.4 outcome arithmetic uses `len(sibling_parity_passes)` as denominator (equivalently, `8 - len(set(failures.tsv.family_id))`).

  **Partition completeness invariant** (auditable at P10, derived from the per-(sibling, kernel) artifacts directly — does NOT trust the TSVs as primary):
  - **Coverage:** `set(passes.family_id) ∪ set(failures.family_id) == set(SIBLING_FAMILY_IDS)` where `SIBLING_FAMILY_IDS` is the fixed 8-family pool from §1. No sibling is silently omitted from both TSVs.
  - **Disjointness:** `set(passes.family_id) ∩ set(failures.family_id) == ∅`. A sibling cannot simultaneously pass and fail.
  - **Pass-side derivation correctness:** for every sibling in `passes.family_id`, AND for every `kernel_target` in `kernel_mutations`: `parity_check_sibling_<sibling>_<kernel>.json` exists AND has `pass: true`. (No false sibling slipped into passes.)
  - **Fail-side derivation correctness:** for every sibling in `failures.family_id`, there exists at least one `kernel_target` in `kernel_mutations` with `parity_check_sibling_<sibling>_<kernel>.json.pass == false`. (No spurious siblings in failures.)
  - Closes the loophole where a faulty P9 could omit a sibling from both TSVs (silently shrinking S without raising the broad-fail halt) or omit a failing sibling from `failures` (slipping past the ≥2 halt). The verifier reconstructs the partition from the per-(sibling, kernel) artifacts and demands byte-equality with the TSVs.

  **Composite exclusion invariant** (auditable at P10): every family in `sibling_parity_failures.tsv` is ABSENT from the composite descriptor's `component_families`; AND for every (sibling, mutated_kernel_target) pair in the composite, a `parity_check_sibling_<fid>_<kernel>.json` with `pass: true` exists in P9's artifact directory. Verifier: `set(composite.component_families) ∩ set(parity_failures.family_id) == ∅` AND for every sibling in `component_families` ∖ {heavy} and every kernel_target in `kernel_mutations`: `parity_check_sibling_<sibling>_<kernel>.json.pass == True`.

  Artifact: per-(sibling, kernel) parity_check artifacts + parity_passes/failures TSVs + partition-completeness audit (coverage + disjointness + pass-side + fail-side) + composite-exclusion audit. Any failure → `sibling_parity_partition_invalid` (P9 artifacts malformed) or `composite_descriptor_includes_parity_failed_sibling`/`composite_descriptor_component_families_mismatch_p9` depending on which step failed.
- **9.3.AR.48c4b P2b per-sibling parity fixtures exist.** P2b extended scope produces `benchmark_blueprints/families/<sibling_fid>/parity_fixture/{deltanet,gatedattn}_v1.yaml` (+ companion `.npz` files) for each of the 8 sibling families, captured against §2.2.0 reference baseline, reproducibility-3-times verified. Probe_count: 16 per sibling. Artifact: 8 × 2 = 16 fixture files present at expected paths; each fixture's `generated_against.weight_version_id` matches live serving stack.
- **9.3.AR.48c5 GatedAttn mutation base_kernel is Triton.** If the composite descriptor's `kernel_mutations` list includes an entry with `kernel_target: gatedattn`, then `base_kernel` MUST be a Triton kernel identifier (matches regex `^triton-`). Vendor kernels (`flash-attn-*`, `flashinfer`) are not source-mutable, so a `gatedattn` mutation entry against a vendor base is internally inconsistent with P8's precondition (P8 ran ⇒ L0a GatedAttn winner was Triton). Verifier rejects with `composite_descriptor_gatedattn_base_invalid`. Artifact: schema check.
- **9.3.AR.48d Composite bundle identity at P10 promotion.** P10 mints a composite descriptor per §6.6 schema. Verifier (per §6.6.4):
  - (a0) **`component_families` matches P9's parity-passes set exactly.** Verifier loads `output/p9_round_<id>/sibling_parity_passes.tsv` and asserts `set(composite.component_families) == {"responses-sdk-adapter-cutover"} ∪ set(parity_passes.family_id)`. P10 cannot drop a parity-passing sibling (which would silently shrink the promoted workload after P9's decision) AND cannot include a parity-failing one. Mismatch → `composite_descriptor_component_families_mismatch_p9`. This is the load-bearing anchor: every other AR.48d step assumes `component_families` reflects P9's eligible set; this step proves that assumption.
  - (a) **Per-family input audit.** For every path in `seed_trace_assembly.per_family_seed_paths` and `holdout_trace_assembly.per_family_holdout_paths`: file exists, size > 0, last byte is `b'\n'`. Any failure → `composite_descriptor_missing_per_family_trace` or `composite_descriptor_input_not_newline_terminated`.
  - (b) **Composite trace re-derivation.** Re-runs `assemble_composite_trace` (§6.6.3, newline-safe) and confirms output byte-equals the file at `seed_trace_ref`. Same for holdout.
  - (c) **Per-mutation patch hash recompute.** For every entry in `kernel_mutations`: `sha256(<patch_path>) == kernel_mutations[i].patch_hash`. Mismatch on any entry fails the AR.
  - (d) **Per-family parity fixture content hash recompute (§6.6.6 canonical manifest).** For every entry in `parity_fixture_refs.<family>.<kernel>`: `fixture_content_hash(yaml_path)` (per §6.6.6 — yaml + every referenced blob in sorted-key delimiter-framed order) `== content_hash`. Mismatch on any entry → `composite_descriptor_parity_fixture_content_drift`. Missing referenced blob → `composite_descriptor_parity_fixture_blob_missing`. This binds bundle identity to the **complete** fixture contents (yaml + probes + reference logits + reference state snapshots if applicable) that justified `component_families`, not just the yaml plus a single npz.
  - (e) **Mutations list canonicalization.** `kernel_mutations` list is sorted by `kernel_target`; yaml-hash stable across re-derivations.
  - (f) **Parity-fixture eligibility consistency — outer + nested.** Two-level invariant:
    - **Outer:** `set(parity_fixture_refs.keys()) == set(component_families)`. No included family is missing a fixture block; no excluded family has one.
    - **Nested:** for every `family in component_families`, `set(parity_fixture_refs[family].keys()) == set(m.kernel_target for m in kernel_mutations)`. Every mutated kernel target has a per-family fixture entry; no extra entries for non-mutated kernel targets (since their content_hash would not be load-bearing for bundle correctness). In dual-mutation rounds (`kernel_mutations` has both `deltanet` and `gatedattn`), every family must carry both fixture entries; in single-mutation rounds, every family must carry exactly that one entry. Mismatch → `composite_descriptor_parity_fixture_eligibility_mismatch`.
  - (g) **Parent §5.9 recompute.** Recomputes `workload_distribution_id` via **parent §5.9 canonical procedure unchanged** — no second hash algorithm — and confirms it matches the descriptor's declared field.
  - (h) **Evidence preservation.** Heavy-family-keyed bundle from P7/P8 is preserved as evidence at `composite_bundle.evidence_refs[]` and is NOT independently loadable for sibling families.

  Artifact: per-step recomputation log; mismatches surface as the matching halt code.

- **9.3.AR.50 P7d (Sampling) parity contract.** When P7d runs (gated on P3a roofline showing sampler share ≥ 3% of Tpot), every `candidates/<NNN>/parity_check.json` records token-id determinism (every probe's sampled token-id byte-equal to reference) AND post-top-k-top-p KL divergence ≤ `kl_divergence_max`. Both must pass for parity. Token-id-only is insufficient because it can paper over distribution drift that doesn't change the sampled token but changes generation diversity over many tokens. Artifact: token-id-equality count (must equal probe_count) + KL-divergence audit per probe.
- **9.3.AR.51 P7e (Fused-epilogue, Triton-sidecar) parity contract.** When P7e runs, every `candidates/<NNN>/parity_check.json` records all four checkpoints from §2.2.6 (post-residual-add, post-norm, post-quant, downstream-logit). Mutations may pass earlier checkpoints and fail later ones; the parity gate fails on first failure but records which checkpoint failed so the proposer can isolate the broken fusion boundary. The mutation surface is **Triton sidecar only** (`kernels/fused/attn_residual_norm_quant.py`) — no CUTLASS-side epilogue mutation in v0.3.1. Artifact: per-iteration four-checkpoint matrix (each cell records pass/fail).
- *(AR.49 [P7c RMSNorm] removed in v0.3.1 — RMSNorm dropped as standalone L0c target; subsumed into P7e.)*
- *(AR.52 [P7f FP8 GEMM Triton] removed in v0.3.1 — target unreachable; dropped per §0.6.)*
- **9.3.AR.53 v0.3.1 kernel-target skip semantics.** For each `--kernel-target` in the v0.3.1 set `{deltanet, fused_epilogue, sampling, gatedattn}` (3 executable + 1 schema-only), exactly one of: (a) a P7x round produced a winning mutation in `kernel_mutations`, OR (b) a P7x round produced `ROUND_NULL_RESULT`, OR (c) the target was skipped per its precondition and recorded in `output/p7_skipped_targets.tsv` with a structured reason (`l0a_winner_is_vendor`, `epilogue_fusion_disabled`, `roofline_share_below_threshold`, `triton_kernel_unavailable`). No target is silently absent. Verifier audits that `set(kernel_mutations[].kernel_target) ∪ set(round_null_result_targets) ∪ set(skipped_targets.target) == {4 v0.3.1 targets}`. Artifact: per-target census + complete-coverage check.
- **9.3.AR.54 Priority-order audit — gated on P3a roofline.** AR.54 enforcement does NOT activate until P3a roofline (§7.2) has been completed AND its measured priority order has been compared against §0.6's predicted order. If they match, AR.54 enforces: GPU-budget-skipped targets must be a suffix of §0.6's order (i.e., higher-priority targets run before lower-priority ones get dropped). If they don't match, §0.6 is rewritten with the measured order before AR.54 activates. Round artifact `output/p3a_roofline_<id>.json` is the load-bearing input — its absence blocks AR.54 enforcement. Artifact: priority-order-source audit (predicted-vs-measured) + GPU-budget-skipped-suffix audit (when applicable).

### 5.7 Step-level verification — Workstream 4 (L0c)

| Step | PASS | FAIL — retry by agent | FAIL — escalate to human |
|---|---|---|---|
| `apply-and-test` CLI smoke against synthetic mutation | A no-op patch (e.g., comment-only) compiles + passes parity + measures within budget | n/a | No-op patch fails parity → fixture or check function is broken; halt; surface — every downstream attempt would fail |
| Iteration's mutation patch is parseable and applies cleanly | `git apply --check mutation.patch` returns 0; resulting source compiles | Re-spawn codex with a brief amendment that names the apply error | 3 consecutive iterations produce un-applicable patches → proposer is misconfigured (e.g., wrong base sha); halt round; surface for diagnosis |
| Compile step | `.so` produced within compile timeout (20 min); ldd resolves cleanly | Auto-kill on timeout; record in `mutations_rejected.tsv`; **does not** count toward iteration cap | 3 consecutive compile failures (timeout or NVCC error) on different patches → toolchain regression or CUDA driver issue; halt round; surface |
| Parity probe | Logit (and state, for DeltaNet) within tolerance for all 64 probes | Fail-record + continue (parity rejection is the expected case) | **Intermittent parity** — same mutation passes once and fails once across 3 probe runs → race condition; reject mutation **and** flag round transcript with `intermittent_parity_observed`; if 2 different mutations show this in same round, halt and surface — backbone non-determinism leaking through |
| Measurement post-parity | n=2 Screen measurements complete; `measurement_trace.json` written | Re-run the n=2 once on transient | 2 consecutive transient failures → vLLM/GPU instability; halt round; surface |
| 3-in-a-row parity-failure halt | Round halts with `HALT_REASON: proposer_stuck` written; up to that point's results preserved | n/a | Round halts via this path → surface to human; not a bug, just a signal proposer brief or model needs adjustment |
| Round finalize (L0c) | Bundle records winning mutation patch + parity_attestation + weight_sensitive flag; AR.43–48 pass | n/a | Any of AR.43–48 fail → bundle does NOT write; round halts; surface |

---

## 6. Correctness substrate — the parity gate as a first-class primitive

### 6.1 Why this section exists

The parity gate is what makes v0.2 different from "Karpathy autoresearch but for kernels." Karpathy's training loop self-corrects: a broken model produces a worse `val_bpb`, the agent discards it, no harm. Kernel work doesn't self-correct: a broken kernel produces a faster wrong answer, the agent might mistakenly keep it, every downstream eval becomes corrupt silently. The whole apparatus of v0.2 hinges on this gate working.

### 6.2 Parity-check semantics

Two check types depending on kernel family:

#### 6.2.1 Logit-space compare (GatedAttn + DeltaNet first-token)

```python
def logit_parity(reference_logits: np.ndarray, mutated_logits: np.ndarray, rtol: float, atol: float) -> dict:
    if reference_logits.shape != mutated_logits.shape:
        return {"pass": False, "reason": "shape_mismatch"}
    diff = np.abs(reference_logits - mutated_logits)
    tol = atol + rtol * np.abs(reference_logits)
    failing = diff > tol
    if not failing.any():
        return {"pass": True}
    first_failing_pos = np.argwhere(failing)[0]
    return {
        "pass": False,
        "first_diverging_probe": int(first_failing_pos[0]),
        "first_diverging_token": int(first_failing_pos[1]),
        "max_diff": float(diff[failing].max()),
        "tolerance_overshoot": float((diff[failing] - tol[failing]).max()),
    }
```

#### 6.2.2 State-snapshot compare (DeltaNet only, in addition to logit)

```python
def state_parity(reference_state: np.ndarray, mutated_state: np.ndarray, rtol: float, atol: float, checkpoint_token: int) -> dict:
    # Same shape as logit_parity but operates on the recurrent-state tensor.
    # Token 1: catches immediate state corruption.
    # Token 1024: catches slow-drift corruption that logit-only misses.
    ...
```

#### 6.2.3 Sampling parity — token-id determinism + KL divergence (v0.3.1+, P7d only)

```python
def sampling_parity(
    reference_token_ids: list[int],
    mutated_token_ids: list[int],
    reference_distribution: np.ndarray,   # shape (probe_count, vocab_size), post-top-k-top-p softmax
    mutated_distribution: np.ndarray,
    kl_divergence_max: float,
) -> dict:
    # (a) Token-id byte-equality check.
    if reference_token_ids != mutated_token_ids:
        first_diff = next(i for i, (a, b) in enumerate(zip(reference_token_ids, mutated_token_ids)) if a != b)
        return {"pass": False, "reason": "token_id_diverged", "first_diverging_probe": first_diff}
    # (b) KL divergence per probe; max across probes must be <= threshold.
    epsilon = 1e-12
    p = reference_distribution + epsilon
    q = mutated_distribution + epsilon
    kl_per_probe = (p * (np.log(p) - np.log(q))).sum(axis=-1)
    max_kl = float(kl_per_probe.max())
    if max_kl > kl_divergence_max:
        return {
            "pass": False, "reason": "kl_divergence_exceeded",
            "max_kl": max_kl, "kl_threshold": kl_divergence_max,
            "first_diverging_probe": int(np.argmax(kl_per_probe)),
        }
    return {"pass": True, "max_kl": max_kl}
```

A mutated sampling kernel passes parity iff both checks pass. Token-id-only would catch boolean correctness but miss distribution drift that wouldn't change the sampled token; KL divergence catches the latter and is what AR.50 audits.

#### 6.2.4 Fused-epilogue parity — four-checkpoint compare (v0.3.1+, P7e only)

```python
def fused_epilogue_parity(
    reference_post_residual_add: np.ndarray,
    mutated_post_residual_add: np.ndarray,
    reference_post_norm: np.ndarray,
    mutated_post_norm: np.ndarray,
    reference_post_quant: np.ndarray,
    mutated_post_quant: np.ndarray,
    reference_downstream_logits: np.ndarray,
    mutated_downstream_logits: np.ndarray,
    rtol: float, atol: float,
) -> dict:
    # Four checkpoints, evaluated in fusion-step order. Fail-on-first reports which
    # fusion boundary broke so the proposer can isolate.
    for name, (ref, mut) in [
        ("post_residual_add", (reference_post_residual_add, mutated_post_residual_add)),
        ("post_norm",         (reference_post_norm, mutated_post_norm)),
        ("post_quant",        (reference_post_quant, mutated_post_quant)),
        ("downstream_logit",  (reference_downstream_logits, mutated_downstream_logits)),
    ]:
        result = logit_parity(ref, mut, rtol=rtol, atol=atol)  # reused; tensor-comparison shape is the same
        if not result["pass"]:
            return {"pass": False, "reason": "fused_epilogue_checkpoint_fail",
                    "failed_checkpoint": name, **result}
    return {"pass": True, "checkpoints_passed": ["post_residual_add", "post_norm", "post_quant", "downstream_logit"]}
```

A mutated fused-epilogue kernel passes parity iff all four checkpoints pass. The order matters: a mutation that fuses incorrectly will fail the first checkpoint after the broken boundary, surfacing the precise step that's wrong (P7e iteration_brief uses this to give the proposer targeted feedback).

A DeltaNet mutation passes parity iff **all three** pass: logit at every token, state at token 1, state at token 1024.

### 6.3 Parity-fixture refresh on weight rotation

The fixture is captured against the §2.2.0 reference baseline and is **round-independent** — it does NOT rebuild when an L0a/L0b/L0c winner changes, because the reference baseline is externally trusted and stable. The fixture rebuilds **only** on weight rotation, per parent §6.4 weight-sensitive re-parity-check: if `weight_version_id` rotates, the fixture must be re-generated against the new weights. Otherwise the fixture's reference logits are stale and any mutation's pass/fail is meaningless.

`scripts/build_parity_fixture.py` is therefore re-run on every weight rotation. The fixture file's `generated_against.weight_version_id` field is the trigger — every L0a/L0b/L0c CLI refuses to bootstrap a round if its workload's fixture's `weight_version_id` doesn't match the current serving stack.

### 6.4 Two failure modes the parity gate doesn't catch (and what we do)

Even with the gate working, two correctness-adjacent failure modes exist:

**(a) Race conditions between threads.** Triton kernels with custom warp-level operations can have CUDA-thread-level races. These pass parity 99% of the time and fail occasionally. Mitigation: run parity probes 3 times per mutation; flag as `intermittent_parity_failure` if even one of the three runs disagrees. Reject the mutation.

**(b) Silent precision degradation.** A mutation might pass `rtol=1e-3, atol=1e-3` but degrade output quality on harder downstream tasks. The parity gate is a *technical* correctness primitive, not a *quality* primitive. Mitigation: at finalize-round, the kernel-tuned bundle MUST run the parent §11.5 live family gate (codex-driven evaluation against the family's full eval set). A parity-passing kernel that fails the live family gate gets `ROUND_BUNDLE_REJECTED: live_gate_failed`. The bundle goes to disk for post-mortem; it does not promote to production.

### 6.5 Verification

Covered in §3.5 (AR.38–40), §4.4 (AR.41–42), §5.6 (AR.43–48). The cross-cutting parity-gate properties land in §5.6's AR.43 and AR.45 specifically.

### 6.6 Composite-bundle identity at P10 — through parent §5.9 unchanged

P10 mints a new descriptor + bundle for the multi-family kernel-tuned production artifact. The descriptor is shaped so that **parent §5.9's `compute_workload_distribution_id(descriptor_path)` works unmodified** — no second hash algorithm, no parent amendment. The trick: concatenate the per-family trace files into a single `seed_trace_ref` / `holdout_trace_ref` pair, and place the mutation patch hash inside the descriptor body so it falls under the `yaml_hash` portion automatically.

#### 6.6.1 Composite descriptor schema

Path: `benchmark_blueprints/workloads/multi-family-v5-l0-kernel-tuned/workload.yaml`.

**Per-family seed traces already exist; sibling holdouts are pre-captured by P1.** Every included family has `seed_trace_v5.jsonl` from the v0.1.5-plan multi-family-v5 work. Only the heavy family had a holdout captured originally (§2.1) — sibling holdouts do NOT pre-exist from v0.1. **P1 (workload descriptor + capture) is therefore extended to capture sibling holdouts up front**, before any L0 round bootstraps: for each of the 8 sibling families, fresh capture against `vllm-default` (same semantic as heavy family's §2.1 holdout — second independent capture, same prompts, different seed), written to `benchmark_blueprints/families/<sibling_fid>/holdout_trace_v5.jsonl` and committed to main. By the time P9 runs, these files already exist on disk; P9 only verifies they're still valid against the live serving stack (staleness check). **No commit-to-BOOTSTRAP-after-bootstrap pattern is used** — the ledger model is preserved. P10 finalize verifies every path in `per_family_seed_paths` and `per_family_holdout_paths` resolves OR refuses to mint with `HALT_REASON: composite_descriptor_missing_per_family_trace`.

**Multiple winning mutations.** P7 (DeltaNet) and P8 (GatedAttn) can both produce winning mutations. The descriptor uses a `kernel_mutations` **list**, not a singular block. The list is canonicalized by sorting on `kernel_target` so yaml-hash is stable across re-derivations. If only one of P7/P8 produced a winner, the list has one element.

```yaml
family_id: multi-family-v5-l0-kernel-tuned   # composite-family pseudo-id (parent §5.9 amendment)
workload_distribution_id: <sha256>           # parent §5.9 canonical procedure unchanged
workload_distribution_id_hardening_version: v2-l0-kernel-tuned
capture_date: <iso8601>
component_families:                          # heavy + parity-passing siblings only. SIZE IS DYNAMIC.
                                              # P10 reads sibling_parity_passes.tsv from P9 and lists
                                              # ONLY those families (plus heavy) here. Parity-failing
                                              # siblings are explicitly EXCLUDED — the bundle is not
                                              # loadable for them. With 0 sibling parity failures the list
                                              # has 9 entries; with 1 failure it has 8; with ≥2 failures
                                              # the round halts before P10 (cross_family_correctness_broad_fail).
                                              # See §6.6.5. The example below shows the all-9-pass case.
  - responses-sdk-adapter-cutover            # heavy family (kernel-tuned against) — always present
  - codex-provider-rollover                  # parity-passing sibling. If a sibling failed parity in P9,
  - codex-skill-runtime-v2-split             # its line would be ABSENT here AND from
  - esm-plugin-loader-modernization          # seed_trace_assembly.per_family_seed_paths AND from
  - nightly-regression-watch                 # holdout_trace_assembly.per_family_holdout_paths AND from
  - objective-driven-repo-improvement        # parity_fixture_refs (so its content_hash doesn't enter
  - policy-aware-request-resolution          # the composite's identity).
  - release-manifest-v2-modernization
  - sqlalchemy-2-session-modernization
seed_trace_ref: composite_seed_trace.jsonl       # newline-separated concatenation of seed traces for the families in component_families.
                                                  # Promotable composite size ∈ {8, 9}: 9 if zero sibling parity failures,
                                                  # 8 if exactly one sibling failed parity. ≥2 sibling parity failures
                                                  # halts before P10 (cross_family_correctness_broad_fail), so P10 never
                                                  # mints a composite smaller than 8.
holdout_trace_ref: composite_holdout_trace.jsonl # newline-separated concatenation of holdout traces for the same set
seed_trace_assembly:
  ordering: alphabetical_by_family_id            # deterministic — matters for hash stability
  separator: single_lf                           # exactly one \n between component file contents (§6.6.3)
  per_family_seed_paths:
    responses-sdk-adapter-cutover:  benchmark_blueprints/families/responses-sdk-adapter-cutover/seed_trace_v5.jsonl
    codex-provider-rollover:        benchmark_blueprints/families/codex-provider-rollover/seed_trace_v5.jsonl
    codex-skill-runtime-v2-split:   benchmark_blueprints/families/codex-skill-runtime-v2-split/seed_trace_v5.jsonl
    esm-plugin-loader-modernization: benchmark_blueprints/families/esm-plugin-loader-modernization/seed_trace_v5.jsonl
    nightly-regression-watch:       benchmark_blueprints/families/nightly-regression-watch/seed_trace_v5.jsonl
    objective-driven-repo-improvement: benchmark_blueprints/families/objective-driven-repo-improvement/seed_trace_v5.jsonl
    policy-aware-request-resolution: benchmark_blueprints/families/policy-aware-request-resolution/seed_trace_v5.jsonl
    release-manifest-v2-modernization: benchmark_blueprints/families/release-manifest-v2-modernization/seed_trace_v5.jsonl
    sqlalchemy-2-session-modernization: benchmark_blueprints/families/sqlalchemy-2-session-modernization/seed_trace_v5.jsonl
holdout_trace_assembly:
  ordering: alphabetical_by_family_id
  separator: single_lf
  per_family_holdout_paths:
    responses-sdk-adapter-cutover:  benchmark_blueprints/families/responses-sdk-adapter-cutover/holdout_trace_v5.jsonl
    codex-provider-rollover:        benchmark_blueprints/families/codex-provider-rollover/holdout_trace_v5.jsonl
    codex-skill-runtime-v2-split:   benchmark_blueprints/families/codex-skill-runtime-v2-split/holdout_trace_v5.jsonl
    esm-plugin-loader-modernization: benchmark_blueprints/families/esm-plugin-loader-modernization/holdout_trace_v5.jsonl
    nightly-regression-watch:       benchmark_blueprints/families/nightly-regression-watch/holdout_trace_v5.jsonl
    objective-driven-repo-improvement: benchmark_blueprints/families/objective-driven-repo-improvement/holdout_trace_v5.jsonl
    policy-aware-request-resolution: benchmark_blueprints/families/policy-aware-request-resolution/holdout_trace_v5.jsonl
    release-manifest-v2-modernization: benchmark_blueprints/families/release-manifest-v2-modernization/holdout_trace_v5.jsonl
    sqlalchemy-2-session-modernization: benchmark_blueprints/families/sqlalchemy-2-session-modernization/holdout_trace_v5.jsonl
kernel_mutations:                                # list, sorted by kernel_target. Body field — falls under yaml_hash
  - kernel_target: deltanet                      # only present if P7 produced a winner
    patch_path:  layer_0_mutations/deltanet_winning_mutation.patch
    patch_hash:  <sha256_of_patch_file>          # binds bundle identity to this mutation
    base_kernel: triton-chunked-delta-v2
    parity_attestation_ref: parity_attestations/p7_run_<round_id>.json
  - kernel_target: gatedattn                     # only present if P8 ran AND produced a winner.
                                                  # P8 ONLY runs when the L0a GatedAttn winner is `triton`,
                                                  # because vendor kernels (FA4/FA3/FA2/FlashInfer) are not source-mutable.
                                                  # base_kernel here MUST be a Triton kernel identifier, NEVER a vendor kernel.
    patch_path:  layer_0_mutations/gatedattn_winning_mutation.patch
    patch_hash:  <sha256_of_patch_file>
    base_kernel: triton-gatedattn-<variant>      # e.g., triton-gatedattn-flash-style-v1
    parity_attestation_ref: parity_attestations/p8_run_<round_id>.json
parity_fixture_refs:                              # per-family, BODY field — falls under yaml_hash.
                                                  # Includes content_hash so a fixture file change forces a
                                                  # workload_distribution_id change (binds bundle identity to
                                                  # the actual fixture contents that justified component_families).
                                                  # ONLY parity-passing siblings appear (consistent with component_families).
                                                  # content_hash is a CANONICAL MANIFEST HASH over yaml + every
                                                  # referenced blob. See §6.6.6 for the exact procedure.
  responses-sdk-adapter-cutover:                  # heavy
    deltanet:
      path:         benchmark_blueprints/families/responses-sdk-adapter-cutover/parity_fixture/deltanet_v1.yaml
      content_hash: <sha256_fixture_manifest_per_6_6_6>   # canonical manifest hash, NOT yaml+npz alone
    gatedattn:
      path:         benchmark_blueprints/families/responses-sdk-adapter-cutover/parity_fixture/gatedattn_v1.yaml
      content_hash: <sha256_fixture_manifest_per_6_6_6>
  codex-provider-rollover:
    deltanet:
      path:         benchmark_blueprints/families/codex-provider-rollover/parity_fixture/deltanet_v1.yaml
      content_hash: <sha256_fixture_manifest_per_6_6_6>
    gatedattn:
      path:         benchmark_blueprints/families/codex-provider-rollover/parity_fixture/gatedattn_v1.yaml
      content_hash: <sha256_fixture_manifest_per_6_6_6>
  # ... one block per parity-passing sibling, sorted alphabetically by family_id for hash stability.
  # Parity-failing siblings have NO entry here (consistent with their absence from component_families).
nominal_ttft_ms: 2000
nominal_tpot_ms: 80
nominal_turn_ms: 30000
target_concurrency: 4
```

#### 6.6.2 Why parent §5.9 hashes this unmodified

Parent §5.9's `compute_workload_distribution_id(descriptor_path)`:
1. Reads `seed_trace_ref` and `holdout_trace_ref` from the descriptor.
2. Hashes both files (here: the concatenated composite trace files).
3. Nulls the `workload_distribution_id` field; canonicalizes yaml; hashes that.
4. Hashes the composite of the three.

Because every mutation patch's hash is a body field inside `kernel_mutations[].patch_hash`, all of them are included in step 3's `yaml_hash` automatically. A different mutation patch on either kernel target → different `patch_hash` → different `yaml_hash` → different `workload_distribution_id`. This is the right semantic — kernel mutations bind the bundle's identity, and bundles produced by single-mutation rounds (DeltaNet-only or GatedAttn-only) are identifiably different from dual-mutation rounds.

**Per-family per-mutated-kernel parity fixture content hashes bind identity.** Each entry in `parity_fixture_refs.<family>.<kernel>.content_hash` is a body field. If a fixture's content changes (new probes, retuned tolerances, recapture against rotated weights) OR a sibling is added/removed from the parity-passing set OR the set of mutated kernel targets changes, the descriptor's `parity_fixture_refs` block changes, `yaml_hash` changes, `workload_distribution_id` changes. The nested invariant `parity_fixture_refs[family].keys() == set(kernel_mutations[].kernel_target)` (per family, AR.48d step f) ensures that in dual-mutation rounds both kernels' fixtures are bound, and in single-mutation rounds only the mutated kernel's fixture is bound — closing the loophole where a fixture could be modified without changing bundle id, or where one mutated kernel's fixture binding could be silently absent.

A different family ordering, a different sibling family added/removed, a different fixture path or content — all of these flow through the existing hash because they're either body fields or affect the trace file contents.

#### 6.6.3 Composite-trace assembly contract (newline-safe)

Raw byte concatenation of JSONL files is **unsafe**: if any source file lacks a trailing newline, its last record fuses with the first record of the next file into a single invalid JSONL line. The assembler explicitly enforces newline boundaries:

```python
def assemble_composite_trace(per_family_paths: dict[str, Path], ordering: str, separator: str) -> bytes:
    """Build a deterministic, newline-safe concatenation of per-family JSONL traces.

    Contract:
    - Every source file MUST end with exactly one b'\\n'. If a file does not,
      the assembler raises ValueError (caught by P10 finalize -> HALT_REASON:
      composite_descriptor_input_not_newline_terminated).
    - Component file contents are concatenated in deterministic order.
    - Between two consecutive component files, the assembler inserts the
      `separator` token (currently only "single_lf" is supported -> b'\\n').
      Combined with the per-file trailing-newline requirement, this guarantees
      records are never fused and the boundary is never ambiguous.
    """
    if ordering != "alphabetical_by_family_id":
        raise ValueError(f"unsupported ordering: {ordering}")
    if separator != "single_lf":
        raise ValueError(f"unsupported separator: {separator}")

    family_ids = sorted(per_family_paths.keys())
    chunks: list[bytes] = []
    for idx, fid in enumerate(family_ids):
        path = per_family_paths[fid]
        if not path.exists():
            raise FileNotFoundError(f"per_family path missing: {fid} -> {path}")
        data = path.read_bytes()
        if not data.endswith(b"\n"):
            raise ValueError(
                f"per_family input {fid} ({path}) must end with b'\\n'; refusing to "
                f"concatenate to avoid record fusion"
            )
        chunks.append(data)
        # Insert single-LF boundary marker between files (NOT after last file).
        # Combined with the trailing-newline requirement, the boundary between
        # records[fid_n.last] and records[fid_n+1.first] is exactly two b'\\n':
        # one from fid_n's terminator, one from the separator. JSONL parsers
        # treat blank lines as no-ops so this is parser-safe and unambiguous.
        if idx != len(family_ids) - 1:
            chunks.append(b"\n")
    return b"".join(chunks)
```

This must be byte-deterministic across re-builds — alphabetical-by-family_id ordering, mandatory trailing-newline check, explicit separator. The assembly is checked into git as a script (`scripts/assemble_composite_trace.py`) so re-derivation is reproducible.

**Pre-flight check at P10 finalize.** Before calling `assemble_composite_trace`, P10 runs a separate audit step that opens each per-family path read-only, confirms file size > 0 and last byte is `\n`. Any failure here halts P10 with the structured reason above — surfaces to human, never silently emits a corrupted composite trace.

#### 6.6.4 Capture-mints / bootstrap-verifies (parent §5.9)

P10 mints `workload_distribution_id`, every `kernel_mutations[].patch_hash`, every `parity_fixture_refs.<family>.<kernel>.content_hash`, and the assembled composite trace files at finalize. Production-time `bootstrap-round` (and `/admin/load_tuned_config`) **verify** by:
1. For each entry in `kernel_mutations`: recompute `patch_hash = sha256(patch_path_contents)` — must match.
2. For each entry in `parity_fixture_refs.<family>.<kernel>`: recompute `content_hash` via §6.6.6's canonical manifest procedure (yaml + every referenced blob, sorted-key, delimiter-framed) — must match.
3. For both `seed_trace_assembly` and `holdout_trace_assembly`: recompute the composite trace via `assemble_composite_trace`, byte-equal the result against what `seed_trace_ref` / `holdout_trace_ref` resolves to.
4. Recompute `workload_distribution_id` via parent §5.9 — must match descriptor's declared field.

Any mismatch in steps 1–3 → `ROUND_BUNDLE_REJECTED: composite_bundle_hash_mismatch`. Missing per-family input → `composite_descriptor_missing_per_family_trace`. Per-family input not newline-terminated → `composite_descriptor_input_not_newline_terminated`. All three surface to human.

#### 6.6.5 Dynamic composite sizing — parity-failing siblings excluded

A composite bundle MUST NOT include a sibling family that failed P9's parity probe. That sibling has been demonstrated to produce wrong output under the kernel mutation — loading the bundle for that sibling in production would silently corrupt downstream evals.

**P10 build procedure for `component_families`:**
1. **Partition-completeness audit (run BEFORE reading TSVs as authoritative).** Per AR.48c4, reconstruct the parity partition from per-(sibling, kernel) artifacts in `output/p9_round_<id>/`:
   - For each sibling × each `kernel_target` in `kernel_mutations`, load `parity_check_sibling_<sibling>_<kernel>.json`.
   - Compute `derived_passes = {sibling : every per-kernel artifact has pass=true}`, `derived_failures = {sibling : at least one per-kernel artifact has pass=false}`.
   - Verify: `derived_passes ∪ derived_failures == set(SIBLING_FAMILY_IDS)` (coverage); `derived_passes ∩ derived_failures == ∅` (disjointness); both `derived_passes` and `derived_failures` byte-equal the corresponding TSVs (after canonical sort by family_id).
   - Any sub-check fails → halt with `sibling_parity_partition_invalid`. P10 does NOT proceed to step 2.
2. Read `output/p9_round_<id>/sibling_parity_passes.tsv` and `output/p9_round_<id>/sibling_parity_failures.tsv` (now audited as ground-truth-derived in step 1).
3. `component_families = ["responses-sdk-adapter-cutover"] + sorted(parity_passing_siblings)`.
4. Build `seed_trace_assembly.per_family_seed_paths` and `holdout_trace_assembly.per_family_holdout_paths` over `component_families` only — no entries for parity-failing siblings.
5. `assemble_composite_trace` is called over only the included families. Composite trace files are smaller than a hypothetical 9-family composite would be.
6. Parent §5.9 canonical hash runs over the descriptor as built. The resulting `workload_distribution_id` reflects the actual eligible-family set.
7. Final audit: P10 verifies `set(component_families) ∩ set(sibling_parity_failures.tsv["family_id"]) == ∅` — empty intersection. Any overlap → halt with `composite_descriptor_includes_parity_failed_sibling`.

**Halt path summary:** step 1 can halt with `sibling_parity_partition_invalid`; step 7 can halt with `composite_descriptor_includes_parity_failed_sibling`. Other steps are deterministic functions of audited-as-ground-truth inputs and cannot independently halt at this stage (their downstream verifications happen in AR.48d at descriptor finalize).

**§8.4 outcome arithmetic uses parity-passing denominator.** The advance threshold is `I ≥ ⌈0.75 × S⌉` where `S = len(parity_passing_siblings)`. Concretely: S=8 → advance needs I≥6 (`⌈6⌉`); S=7 → advance needs I≥6 (`⌈5.25⌉ = 6`). With 2+ parity failures (S ≤ 6) the round halts before P10 — no composite is minted, so the threshold is never evaluated for those cases. So in practice: the only two promotable cases are S=8 needing 6 improvers and S=7 needing 6 improvers.

This makes the contract auto-pruning and reviewer-clean: a single sibling parity failure does NOT cause the failed sibling to ship in the composite, and does NOT necessarily kill the round — the kernel can still be promotable for the remaining parity-passing families.

#### 6.6.6 Fixture content_hash — canonical manifest hash

A parity fixture is a yaml descriptor PLUS one or more referenced data blobs (`probe_input_ref`, `reference_logits_ref`, and for DeltaNet `reference_state_snapshots_ref`). Hashing only the yaml + a single npz companion misses the rest of the fixture's content, and changing `probes_input.jsonl` or one of the npz files would not change the descriptor's `content_hash`.

`content_hash` is therefore defined as a canonical manifest hash over the yaml + every file it references, in deterministic key order:

```python
def fixture_content_hash(fixture_yaml_path: Path) -> str:
    """Canonical manifest hash over a parity fixture and every referenced blob.

    Hashed inputs (in this exact order, with delimiters):
      1. fixture yaml file bytes
      2. for each ref_key in REFERENCED_KEYS (sorted alphabetically):
           if ref_key in fixture: hash the resolved file's bytes,
           with a delimiter framing of: b'\\x00' + ref_key.encode() + b'\\x00'
    Missing optional keys (e.g. reference_state_snapshots_ref on a GatedAttn fixture)
    are skipped — their absence is implicitly captured because the yaml itself
    (which is included first) records which keys were defined.
    """
    # P0.5 fix (v0.3.2): schema-specific referenced-key lists. Each fixture schema declares
    # its own set of *_ref keys. The function picks the right list by reading the fixture's
    # `parity_check_method` field. New schemas extend this map; missing schemas raise.
    REFERENCED_KEYS_BY_SCHEMA = {
        # DeltaNet (§2.2.2): logit + recurrent-state-snapshot
        "logit_plus_state_compare": (
            "probe_input_ref", "reference_logits_ref", "reference_state_snapshots_ref",
        ),
        # GatedAttn (§2.2.1): logit-only
        "per_token_logit_compare": (
            "probe_input_ref", "reference_logits_ref",
        ),
        # Sampling (§2.2.4): token-id determinism + KL divergence
        "token_id_plus_kl_divergence": (
            "probe_input_logits_ref", "probe_sampling_params_ref",
            "reference_sampled_token_ids_ref", "reference_distribution_ref",
        ),
        # Fused epilogue (§2.2.6): four checkpoints
        "four_checkpoint_compare": (
            "probe_input_attn_output_ref", "probe_input_residual_ref",
            "reference_post_residual_add_ref", "reference_post_norm_ref",
            "reference_post_quant_ref", "reference_downstream_logits_ref",
        ),
    }
    fixture = yaml.safe_load(fixture_yaml_path.read_text())
    base_dir = fixture_yaml_path.parent
    method = fixture.get("parity_check_method")
    if method not in REFERENCED_KEYS_BY_SCHEMA:
        raise ValueError(
            f"unknown parity_check_method: {method!r}. Add it to "
            f"REFERENCED_KEYS_BY_SCHEMA before this fixture can be content-hashed."
        )
    keys = REFERENCED_KEYS_BY_SCHEMA[method]
    h = hashlib.sha256()
    h.update(fixture_yaml_path.read_bytes())
    for key in sorted(keys):
        if key in fixture and fixture[key]:
            ref_path = base_dir / fixture[key]
            if not ref_path.exists():
                raise FileNotFoundError(f"fixture {fixture_yaml_path} references {key}={fixture[key]} which does not exist")
            h.update(b"\x00" + key.encode("ascii") + b"\x00")
            h.update(ref_path.read_bytes())
    return h.hexdigest()
```

Why this shape:
- **YAML first**: any change to the descriptor itself (tolerances, probe_count, generated_against fields) flows in immediately.
- **Sorted referenced keys**: deterministic ordering across runs; adding a new reference key in the schema is forward-compatible without breaking existing hashes.
- **Delimiter framing** (`b'\x00' + key + b'\x00'`): prevents content of one referenced file from being indistinguishable from prefix-matched content of another — a common manifest-hashing pitfall.
- **Missing-keys handled**: GatedAttn fixtures don't have `reference_state_snapshots_ref`; the yaml records that absence, and the hash skips the missing key. A future schema change that adds a key automatically extends the hash.

P10 mints `content_hash` per fixture by calling this function. `bootstrap-round` and `/admin/load_tuned_config` recompute it the same way and demand byte-equality. Mismatch → `composite_descriptor_parity_fixture_content_drift`. Missing referenced blob (e.g., the npz companion was deleted) → `composite_descriptor_parity_fixture_blob_missing`.

---

## 7. Sequencing and dependencies

The plan is a DAG, not a calendar. Each phase has a hard precondition and a verification gate; coding agents drive each phase to completion at whatever pace they take. No phase is "skipped because we're behind"; phases are skipped only when their precondition is unsatisfiable (e.g., L0c-GatedAttn skipped if the L0a winner is a vendor kernel).

### 7.1 Phase ordering (dependency DAG)

```
P1 Workload descriptor + capture  ──┐
                                    ├──> P2b Parity fixture vs §2.2.0 reference ──> P3 L0a real round ──> P6 L0b real round ──> P7 L0c-DeltaNet ──> P9 Cross-family ──> P10 Composite bundle finalize
P2 Hardware-aware vLLM fanout  ─────┘                                                                                       ──> P8 L0c-GatedAttn (conditional) ──┘
                                                                                                P5 (parallel) L0c plumbing (apply-and-test + mutate-kernel CLIs against synthetic fixture)
```

**v0.3.1 phase DAG (L0c-first ordering — L0a/L0b deferred):**

```
P1 Workload + capture (heavy + 8 sibling holdouts)   ──┐
P2 Multi-instance vLLM driver (empirical 1–4 count)  ──┤
                                                       ├──> P3a Roofline probe (vllm-default) ──> P7a L0c-DeltaNet (vs vllm-default)   ──┐
P2b Parity fixtures (heavy + sibling, 9 families)    ──┘                                       ──> P7e L0c-FusedEpilogue (vs vllm-default) ──┤──> P9 Cross-family ──> P10 Composite
                                                                                               ──> P7d L0c-Sampling (gated on P3a share)  ──┘                          (optional later: P3 L0a + P6 L0b
                              P5 (parallel) L0c plumbing (apply-and-test + mutate-kernel CLIs against synthetic fixture)                                              if any L0c result hints config-space headroom)
```

**Key changes from v0.3.0-plan:**
1. **L0c runs first against `vllm-default`**, not against an L0a/L0b winner (operator note: prior L0a/L0b rounds confirmed empty against baseline; L0c is the unexplored surface).
2. **P3a roofline probe is mandatory before P7a/e/d** — measures per-category Tpot share, DRAM throughput, SM occupancy, tensor-core utilization, kernel launch count. Outputs gate AR.54 priority enforcement and the `sampling` skip condition.
3. **P3 (L0a) and P6 (L0b) are deferred to v0.3.2 exploration**, after L0c results are in. They are not in v0.3.1's executable scope.
4. **L0c executable target set narrowed to `{deltanet, fused_epilogue, sampling (gated)}`** — see §5.2. RMSNorm and FP8 GEMM Triton dropped.

Key change from v0.2.1-plan (still applicable): the parity fixture (P2b) is built **before** L0a/L0c, against an externally-trusted reference baseline (§2.2.0), not against any round winner. Smoke phase (when L0a runs in v0.3.2) is a determinism + correctness gate rather than determinism-only.

### 7.2 Phase preconditions and exit gates

| Phase | Precondition | Exit gate (PASS to advance) | Halt-and-surface conditions |
|---|---|---|---|
| **P1** workload descriptor + capture (§2.1, §2.2 schema) | Heavy family hardened (already true); 8 sibling families hardened (already true per v0.1.5-plan) | **Heavy artifacts:** `workload.yaml` schema-valid; `seed_trace.jsonl` + `holdout_trace.jsonl` both pass thinking-probe row-3; canonical `workload_distribution_id` reproducible. **Sibling artifacts (load-bearing for P9):** all 8 of `benchmark_blueprints/families/<sibling_fid>/holdout_trace_v5.jsonl` exist on main, each captured against `vllm-default` baseline (capture metadata records `vllm-default + weight_version_id`), each passes thinking-probe row-3 at capture time. `git log --diff-filter=A` against main shows 8 new sibling holdout files added by P1. Without these, P9 cannot proceed. | Heavy capture yields `reasoning_tokens == 0` after 2 retries → halt with `capture_thinking_zero`. Sibling capture yields `reasoning_tokens == 0` for any family after 2 retries → halt with `sibling_holdout_capture_failed`. Any sibling holdout file missing from main at end of P1 → P1 PASS gate is not satisfied; downstream phases cannot precondition on P1 |
| **P2** hardware-aware vLLM fanout discovery (§3.3) — empirical instance-count discovery (v0.3.1 fix per P0.6) | None | Driver attempts to launch up to 4 vLLM instances at decreasing `gpu_memory_utilization` values (0.4, 0.3, 0.2, 0.15); records the maximum viable fanout `N ≥ 1` for which all `N` instances start AND produce concurrent fixture results equal to sequential single-instance results. Attempted fanouts above `N` have memory evidence and structured rejection reasons; `l0a_parallel_fanout: N` (also `instance_count_max`) recorded in `output/p2_instance_capacity.json` and consumed by all downstream phases. Four-way fanout is the DGX Spark/GB10 optimization target, not the pass gate. If `N == 1`, the single endpoint must be healthy and deterministic and later L0a scheduling runs serial-only. | `N == 0` (no single instance starts at any utilization) → halt (`vllm_driver_no_viable_fanout`). Any `N > 0` concurrent result differs from sequential single-instance result → halt (`multi_instance_concurrent_diverges`). Fanout < 4 due only to memory/contention is recorded, not terminal |
| **P2b** parity fixture capture against §2.2.0 reference baseline | P1 & P2 PASS | Heavy fixture (64 probes) written at `benchmark_blueprints/families/responses-sdk-adapter-cutover/parity_fixture/{deltanet,gatedattn}_v1.{yaml,npz}`. **Per-sibling fixtures** (16 probes each) written at `benchmark_blueprints/families/<sibling_fid>/parity_fixture/{deltanet,gatedattn}_v1.{yaml,npz}` for all 8 siblings. §2.2.0 reference baseline reproduces bit-identically 3 times per family before each fixture is sealed | Reference itself non-deterministic → halt (`fixture_reference_nondeterministic`); the entire L0 thesis is invalid until fixed. Sibling fixture capture fails for any family → halt (`sibling_fixture_capture_failed`); P9 cannot run without per-sibling fixtures |
| **P3a** roofline probe — v0.3.1 mandatory | P1 & P2 & P2b PASS | Run `vllm-default` resolution against the heavy workload at `target_concurrency=4` for 5 minutes; collect via Nsight Compute / Nsight Systems (or NVIDIA `nvbandwidth`/`ncu` equivalent on DGX Spark): per-category Tpot share (DeltaNet, GatedAttn, FFN/Linear, sampler, norm, KV write/read), DRAM read/write throughput (% of 273 GB/s peak), SM occupancy, tensor-core utilization, kernel launch count per token. Output: `output/p3a_roofline_<round_id>.json`. **Gates downstream:** (a) AR.54 priority enforcement activates if measured priority order matches §0.6 prediction; otherwise §0.6 is rewritten with measured order before AR.54 activates; (b) `sampling` L0c target is ungated only if measured sampler share ≥ 3% of Tpot. | Roofline run produces `Tpot_share` summing to <70% of decode time → unaccounted overhead is large; halt with `roofline_unaccounted_overhead`. Nsight tooling unavailable on DGX Spark → halt with `roofline_tooling_unavailable`; surface for human (alternative: nvprof, vLLM internal profiler) |
| **P5** (parallel to P3a) L0c plumbing — `apply-and-test` + `mutate-kernel` CLIs (§5.3) | P2 & P2b PASS | Synthetic-fixture L0c round uses P2's discovered fanout, passes: no-op patch path PASS-PASS, broken patch path FAIL-recorded | No-op patch fails the synthetic fixture → halt |
| **P5b** fp8_e5m2 KV purity attestation — v0.3.3 (impl-agent recommendation) | P2b PASS AND base stack uses `kv_cache_dtype: fp8_e5m2` (per `actually_resolved_kernel_selection`) | Run a small probe set: 16 short-prompt-short-output requests captured against fp8_e5m2 KV vs the same probes captured against bf16 KV. Verify divergence stays within fixture tolerances (`rtol_logit=1e-3`, `atol_logit=1e-3`) — i.e., fp8 KV does NOT introduce divergence beyond the parity-gate's normal noise floor. If it does, any L0c mutation against this base would be debugging quantization noise as if it were kernel divergence. ~10 minutes of vLLM time. Output: `output/p5b_fp8_kv_purity_<round_id>.json`. **Skipped if** base stack's `kv_cache_dtype` is bf16/fp16 (no FP8 noise risk to validate). | fp8_e5m2 KV introduces divergence beyond `rtol/atol` on >0 of 16 probes → halt with `fp8_kv_purity_violation`; kernel mutation work cannot proceed against this fixture set. Fix path: re-capture parity fixture against bf16 KV, OR loosen tolerance with explicit justification recorded in BLOCKED.md |
| **P7a** L0c-DeltaNet real round (§5.2) — **v0.3.3 EXECUTABLE; baselined against L0b-empirical-winner (`4866bc3f`)** | P2b, P5, P5b (if applicable), P3a PASS | Bundle records winning mutation OR records `ROUND_NULL_RESULT` (no parity-passing mutation beat the paired-A/B baseline); AR.43–48c PASS. Paired baseline is the resolved values from `4866bc3f` (vllm-default routing pinned at `actually_resolved_kernel_selection`, fp8_e5m2 KV); `Measurement-Role: l0b_empirical_winner_baseline_remeasured`. **This is the only L0c target executable in v0.3.3.** | All L0c §5.7 escalation rows |
| **P7e** L0c-FusedEpilogue real round (§5.2) — **DEFERRED to v0.3.4+ pending wiring verification** | NOT executable in v0.3.3. Requires: (a) Triton sidecar `kernels/fused/attn_residual_norm_quant.py` exists in vLLM shipping path; (b) `epilogue_fusion_mode` knob is wired to actually toggle the sidecar at vLLM init time; (c) the §2.2.6 fixture re-captured against the L0b-empirical-base with sidecar enabled. | When executable: same shape as P7a but uses §2.2.6 fixture (4-checkpoint compare) and AR.51 verifier. Triton-sidecar mutation surface only. | n/a in v0.3.3 |
| **P7d** L0c-Sampling real round (§5.2) — **DEFERRED to v0.3.4+ pending wiring verification** | NOT executable in v0.3.3. Requires: (a) `vllm-v1-triton-sampler` in the shipping path with mutable Triton source at known location; (b) `sampling_kernel` knob wired to vLLM init; (c) §2.2.4 fixture captured against the L0b-empirical-base with Triton sampler enabled; (d) P3a roofline measures sampler share ≥ 3% of Tpot. | When executable: same shape as P7a but uses §2.2.4 fixture (token-id determinism + KL divergence) and AR.50 verifier. | n/a in v0.3.3 |
| ~~**P3** L0a real round~~ | DEFERRED to v0.3.2 | Not in v0.3.1 executable scope (operator note: prior L0a rounds showed no headroom against baseline) | n/a |
| ~~**P6** L0b real round~~ | DEFERRED to v0.3.2 | Not in v0.3.1 executable scope | n/a |
| ~~**P7b** L0c-GatedAttn~~ | NOT in v0.3.1 executable scope | Schema-only target — runs only if a Triton GatedAttn kernel is the L0a winner AND P3a measures attention-output share > 5% Tpot. Both unlikely. | n/a |
| ~~**P7c** L0c-RMSNorm~~ | DROPPED in v0.3.1 | RMSNorm is subsumed into `fused_epilogue` Triton sidecar. No standalone round. | n/a |
| ~~**P7f** L0c-FP8GEMMTriton~~ | DROPPED in v0.3.1 | Was unreachable (P0.1: L0a action space had no `triton` FP8 GEMM choice). Defer to v0.3.2 if a mutable Triton FP8 GEMM is added. | n/a |
| **P9** cross-family generalization (§8.4) | At least one of P7a/P7b/P7c/P7d/P7e/P7f (whichever ran; each may be skipped per its own precondition) produced a winning mutation; P1 produced sibling holdouts; P2b produced sibling parity fixtures for every kernel target represented in `kernel_mutations` | Three sub-steps in order, all in the same round / same vLLM lifecycle: **(a) Sibling holdout staleness check** — for each of the 8 sibling families, verify the pre-existing holdout (captured by P1) is still valid against the live serving stack: file present, thinking-probe row-3 still passes, `weight_version_id` at capture matches current. **(b) Sibling parity probe — per sibling × per mutated kernel target** — for each sibling AND each `kernel_target` in `kernel_mutations` (DeltaNet AND GatedAttn in dual-mutation rounds), run a 16-probe parity-vs-§2.2.0-reference check using the per-(sibling, kernel) fixture from §2.4-pre. A sibling is `parity_pass` only if it passes for ALL mutated kernel targets; if it fails on any one, it goes to `sibling_parity_failures.tsv` with `failing_kernel_target` recorded. **(c) Paired throughput measurement** — for **only parity-passing** siblings: n=5 `vllm-default` baseline + n=5 kernel-mutated winner, with `Measurement-Role: sibling_baseline` / `sibling_winner` trailers. Per-sibling Welch-t from contemporaneous pairs only. P9 produces `sibling_parity_passes.tsv` (the eligible-for-composite list) and `sibling_parity_failures.tsv` (the excluded list, for audit). P10 reads `sibling_parity_passes.tsv` and builds the composite **only over heavy + the parity-passing siblings**. The §8.4 outcome is computed over parity-passing siblings only — denominator is `len(parity_passes)`, not 8. | Sibling holdout fails staleness check → halt with `sibling_holdout_stale_or_invalid`. `inconsistent_baseline` fires repeatedly → halt with `sibling_baseline_inconsistent_repeated`. ≥ 2 siblings fail parity → halt with `cross_family_correctness_broad_fail` (kernel is workload-overfit on correctness and not safely promotable in any composite shape) |
| **P10** production-bundle finalize | P9 satisfies the **advance predicate**: `R == 0 AND I ≥ ⌈0.75 × S⌉` (the §8.4 advance row, NOT row 1 — the §8.4 table is ordered with overfit/regression rows first; a literal "row 1" reference would point at the overfit reframe-scope outcome). Equivalently: zero regressing siblings AND ≥6 of 8 (S=8) or ≥6 of 7 (S=7) improving | A **new composite descriptor + bundle** is minted per §6.6 (composite-family pseudo-id, newline-safe concatenated trace files, list of mutation patch hashes inside descriptor body, **`component_families` = heavy ∪ parity-passing siblings only — parity-failing siblings excluded**, per-family seed/holdout paths reference P1-captured holdouts + existing v0.1.5 seed_trace_v5). `workload_distribution_id` is computed by **parent §5.9 canonical procedure unchanged** — no second hash algorithm. `round_type: l0_kernel_tuned`; live family gate (parent §11.5) PASS. The heavy-family pseudo-bundle from P7/P8 is NOT directly promoted — it's the *evidence* the composite bundle was minted from. | Live family gate FAIL → composite bundle goes to disk only; missing per-family trace inputs OR non-newline-terminated inputs OR hash recompute mismatch OR composite includes a sibling listed in `sibling_parity_failures.tsv` OR partition-completeness audit fails → halt with the matching §9.1 code; heavy-family bundle is preserved as evidence |

### 7.3 Dependencies (rationale)

**v0.3.1+ L0c-first ordering (active):**
- **Parity fixture (P2b) → L0c:** fixture must exist before L0c so per-mutation parity gating works (§5.6 AR.43). Without this, L0c rounds cannot bootstrap. Fixture is captured against the externally-trusted §2.2.0 reference baseline, not against any round's winner.
- **L0c (v0.3.1+) baselines against `vllm-default` directly.** The paired-A/B baseline is measured in the same L0c round with `Measurement-Role: vllm_default_baseline_remeasured`. No L0a/L0b winner is required.
- **P3a roofline → AR.54 priority:** AR.54 priority enforcement does not activate until P3a measures actual per-category Tpot share; if the measurement contradicts §0.6's predicted order, §0.6 is rewritten before AR.54 activates.

**v0.3.3+ L0a → L0b → L0c ordering (deferred, included for documentation):**
- **L0a → L0b:** L0b would need the L0a winner kernel choice as its base. L0b's Welch-t comparison would use a paired baseline measured in the same L0b round (AR.41b), not the L0a bundle's prior `objective_mean`.
- **L0b → L0c:** L0c mutations would sit on top of L0b-autotuned baselines via `--base-stack-resolution=bundle --base-bundle-path <l0b-bundle>`. v0.3.1/v0.3.2 do not run this path because prior rounds confirmed L0a/L0b have no defensible headroom.
- **L0c → P9:** P9 advances if AT LEAST ONE of P7a–P7f produced a winning mutation (the rounds that ran; each is skippable per its own precondition). Skipped rounds don't block.
- **All → composite bundle (P10):** a winning mutation does NOT promote as the heavy-family-keyed pseudo-bundle; P10 mints a new composite-family bundle whose `workload_distribution_id` hashes over the trace files for `component_families` (heavy + parity-passing siblings; size dynamic per P9 outcome) + every mutation patch hash + every per-family parity fixture content hash.
- **All → live family gate:** every kernel-tuned composite bundle that finalizes runs the parent §11.5 live family gate before promotion.

---

## 8. Decision points and exit criteria

These are the gates a human looks at to decide "advance / hold / reframe". Each row makes the action explicit. There is no "this is concerning, we should think about it" outcome — every row is one of: **advance**, **halt-and-diagnose**, **ship-as-is-and-stop**, or **reframe-scope**.

### 8.1 After L0a (gate on phase P3) — **DEFERRED to v0.3.3+** (P3 is not in v0.3.1/v0.3.2 executable scope; this section preserved for the documentation of the eventual L0a decision gate)

| Observed outcome | Classification | Action |
|---|---|---|
| L0a winner is `flash-attn-4 + triton-chunked-delta-v2 + …` and beats `vllm-default` by ≥ noise_floor at `confidence: defensible` | advance | Proceed directly to P6 (L0b). Parity fixture is already in place from P2b — no per-winner re-capture |
| L0a winner equals `vllm-default` resolution (no improvement) | halt-and-diagnose | Surface to human: either action space genuinely has no headroom, OR smoke-phase pruned the real winner, OR `vllm-default` resolution already lands on the same combo. Human reviews `eliminated.tsv` and decides reframe vs ship-L0a-as-reference |
| L0a winner is `flashinfer + …` | conditional advance | Trigger parent §9.3.13 flashinfer accuracy-screen. PASS → advance; FAIL → force-select FA4 manually and advance with that as winner |
| Round halts (any §3.6 escalation row triggered) | halt-and-diagnose | Round transcript surfaced; human decides whether the halt-cause is fixable or terminal |

### 8.2 After L0b (gate on phase P6) — **DEFERRED to v0.3.3+** (P6 is not in v0.3.1/v0.3.2 executable scope; preserved for documentation)

| Observed outcome | Classification | Action |
|---|---|---|
| L0b winners beat L0a by ≥ noise_floor at `confidence: defensible` | advance | Proceed to P7 (L0c-DeltaNet) |
| L0b winners equal L0a (autotune found no better params than kernel defaults) | ship-as-is-or-reframe | Surface: human decides between (a) ship L0a bundle as v0.2 reference artifact + skip L0c, or (b) proceed to L0c on the L0a base anyway with lower expectations |
| L0b winners regress vs L0a | halt-and-diagnose | Should be impossible (autotune searches a superset). If observed → autotune wiring or measurement is broken; halt |
| Round halts (any §4.5 escalation row triggered) | halt-and-diagnose | Surface |

### 8.3 After L0c (gate on phases P7 and optionally P8)

| Observed outcome | Classification | Action |
|---|---|---|
| At least one L0c mutation beats L0b winner by ≥ 2× noise_floor at `confidence: defensible` AND parity attestation present | advance | Proceed to P9 (cross-family generalization) |
| Zero accepted mutations across both L0c rounds (`ROUND_NULL_RESULT`) | ship-as-is | Ship L0b bundle as v0.2 production artifact for heavy family; v0.3 is the next investment decision (custom CUDA, larger mutation budget, different proposer) |
| ≥ 30% of mutations rejected by parity gate | inspect-but-not-halt | Round result still valid. Human reviews proposer's iteration_brief and rejected-patch reasons; may tighten brief or change proposer model for a re-run |
| Round halts via `proposer_stuck` or `compile_failures_3x` (§5.7) | halt-and-diagnose | Surface — proposer brief or model needs adjustment, not a system bug |
| `intermittent_parity_observed` flag set on round transcript | halt-and-diagnose | Surface — backbone non-determinism leaked through the gate; this is a correctness emergency not a kernel-tuning question |

### 8.4 After cross-family generalization (gate on phase P9)

**Per-sibling Welch-t requires paired contemporaneous baseline.** "Beats `vllm-default` by ≥ noise_floor" means: for each parity-passing sibling family the round captured n=5 `vllm-default` baseline measurements + n=5 kernel-mutated measurements **in the same round**, and the Welch-t CI for (mutated − baseline) is strictly above noise_floor at `confidence: defensible`. The classification rows below assume this paired structure.

**Denominator is `len(parity_passing_siblings)` ∈ {7, 8}.** With P parity failures (P ∈ {0, 1}), `S = 8 − P`. With P ≥ 2 the round halts before §8.4 is consulted (`cross_family_correctness_broad_fail`), so §8.4 only ever sees S=8 or S=7.

Let `I = improved-vs-vllm-default count among those S siblings`, `R = regressed-vs-vllm-default count among those S siblings`.

**Decision precedence:** rows are evaluated top-to-bottom; the first matching row wins. The R-band rows handle 0, low-positive (1–2), and overfit ranges separately so every (I, R) pair maps to exactly one row.

| Observed outcome | Classification | Action |
|---|---|---|
| `R ≥ ⌈0.375 × S⌉` — i.e. **≥3 of 8** OR **≥3 of 7** (`⌈2.625⌉ = 3`) | reframe-scope | Workload-overfit on throughput regardless of how many siblings improved. Bundle stays exploratory only; v0.3 scope is shape-aware per-family kernel selection |
| `0 < R < ⌈0.375 × S⌉` — any non-zero but not-overfit regression count (1 or 2 regressing siblings) | reframe-scope | At least one sibling regresses against `vllm-default`. Bundle is correctness-clean (parity passed) but performance-mixed across the workload pool. Ships as **heavy-family-only reference artifact + per-cluster v0.3 scoping note** so we can later co-tune the regressing sibling's cluster. Multi-family promotion is unsafe for performance regressions even if not many siblings regress |
| `R == 0` AND `I ≥ ⌈0.75 × S⌉` — i.e. **6 of 8** (S=8) OR **6 of 7** (S=7, since `⌈5.25⌉ = 6`) | advance | Proceed to P10 — **mint composite descriptor + bundle** per §6.6 over `component_families = ["responses-sdk-adapter-cutover"] + parity_passing_siblings`. `family_id: multi-family-v5-l0-kernel-tuned`, `workload_distribution_id` via parent §5.9 canonical procedure unchanged. Heavy-family bundle from P7/P8 is preserved as evidence; it is NOT what loads in production for sibling families |
| `R == 0` AND `⌈0.5×S⌉ ≤ I < ⌈0.75×S⌉` — i.e. **4 or 5 of 8** OR **4 or 5 of 7** | reframe-scope | Generalization is partial — not strong enough to ship as multi-family bundle, not weak enough to declare workload-overfit. v0.3 scope: either (a) per-cluster kernel tuning where families with similar shape distributions get co-tuned, or (b) re-run P9 with a larger n per sibling to tighten CIs. Bundle ships as heavy-family-only reference artifact in the meantime |
| `R == 0` AND `I < ⌈0.5×S⌉` — i.e. **≤ 3 of 8** OR **≤ 3 of 7** | reframe-scope | Bundle ships as heavy-family-only reference artifact; v0.3 scope is per-family kernel tuning |
| ≥ 2 siblings fail the P9 step-(b) parity probe (round already halted before §8.4) | halt-and-diagnose | `cross_family_correctness_broad_fail` — kernel mutation is not safe across the workload mix. Surface; do NOT promote any bundle |
| Sibling family runs hit thinking-probe failures or other workload-stack issues | halt-and-diagnose | Not a kernel issue; surface so workload-stack root cause is fixed before generalization is re-evaluated |

**Coverage proof.** Every (I, R) pair with S ∈ {7, 8} maps to exactly one row: (a) `R ≥ ⌈0.375×S⌉` row catches all overfit-regression cases; (b) `0 < R < ⌈0.375×S⌉` row catches small-regression cases regardless of I; (c) the three `R == 0` rows partition the non-regression cases by I-band. No (I, R) pair is uncovered.

---

## 9. Verification — extends v0.1.5-plan AR list (AR.38–54)

All new items are covered in §3.5, §4.4, §5.6 with full artifact specs. Items break into eight groups:

- **L0a determinism + parity-vs-reference + intermediate-bundle marking:** AR.38, AR.38b, AR.39, AR.40
- **L0b autotune frozen + paired baseline + determinism + parity-vs-reference:** AR.41, AR.41b, AR.42
- **L0c parity-gate properties + cap structure + paired baseline:** AR.43, AR.44, AR.45, AR.46, AR.47, AR.48, AR.48b, AR.48c (the load-bearing items — AR.43 is most important; the parity gate ran *before* measurement counted)
- **P1 sibling holdout pre-capture (verified at P9):** AR.48c3
- **P2b per-sibling parity fixtures:** AR.48c4b
- **P9 cross-family paired baselines + per-(sibling × kernel) parity probe + composite exclusion:** AR.48c2, AR.48c4
- **GatedAttn mutation invariant + P10 composite bundle identity (incl. canonical fixture content hash per §6.6.6):** AR.48c5, AR.48d
- **v0.3.1+ kernel-target expansion (Sampling + Fused-epilogue + coverage + priority-order):** AR.50, AR.51, AR.53, AR.54. *(AR.49 RMSNorm and AR.52 FP8-GEMM-Triton dropped per §0.6 — targets removed.)*

### 9.1 Named halt conditions (catalogue)

Every escalation row in §2.4, §3.6, §4.5, §5.7 writes one of these named codes into the round transcript. A human looking at a halted round greps `HALT_REASON:` to find the diagnosis entrypoint. Codes are stable so dashboards can count them.

| `HALT_REASON` code | Phase | What it means in one line |
|---|---|---|
| `capture_thinking_zero` | P1 | `reasoning_tokens == 0` after 2 retries — vLLM Responses API regression |
| `fixture_reference_nondeterministic` | P1, P2b | Reference run produces different logits across same-input invocations — backbone determinism broken |
| `fixture_weight_mismatch` | P2b, P3, P6, P7a–P7f | `generated_against.weight_version_id` ≠ live serving weights — every round that consumes the fixture rejects on mismatch |
| `multi_instance_concurrent_diverges` | P2 | 4-instance concurrent results ≠ sequential single-instance results |
| `multi_instance_insufficient_memory` | P2 | **0 instances start** at the lowest probed `gpu_memory_utilization` (0.15) — not a count-below-4 issue (P2 accepts `N ≥ 1` per v0.3.1 P0.6 fix); raised only when the model can't fit at any utilization. The accepted range is `1 ≤ N ≤ 4` |
| `l0a_precondition_missing_fixture` | P3 | L0a refused to bootstrap — parity fixture file(s) not present at expected path |
| `l0a_parity_fail_winner` | P3 | Combo passed determinism but failed parity-vs-reference — deterministic-but-wrong; eliminated mid-smoke |
| `flashinfer_passes_parity` | P3 | FlashInfer combo unexpectedly passed parity — either #35138 fixed (audit fixture) or fixture too loose (tighten tolerances) |
| `smoke_zero_survivors` | P3 | Every L0a combo failed determinism OR parity — toolchain/Triton/CUDA regression OR fixture itself is wrong |
| `inconsistent_baseline_repeated` | P3, P6 | 2+ baselines flagged inconsistent on retry — workload not stable on this stack |
| `wedged_combos` | P3 | One or more grid combos failed measurement 2× on transient — record + continue, surface at round end |
| `rescreen_inconclusive` | P3 | Top-2 rescreen rows within noise_floor — no defensible single winner; human picks substrate |
| `autotune_wrapper_breaks_logits` | P6 | `@triton.autotune` integration changes logits even before tuning — wrapper bug |
| `autotune_oscillating` | P6 | Stable window never reached — shape distribution too wide to autotune jointly |
| `autotune_winner_nondeterministic` | P6 | Frozen autotune winner fails 64-probe determinism |
| `noop_patch_fails_parity` | P5, P7 | Synthetic no-op patch fails the parity check — fixture or check function broken |
| `mutation_unappliable_3x` | P7a–P7f | 3 consecutive iterations produce un-applicable patches — proposer misconfigured |
| `compile_failures_3x` | P7a–P7f | 3 consecutive compile failures on different patches — toolchain regression |
| `intermittent_parity_observed` | P7a–P7f | Same mutation passes parity once and fails once — race condition leaking through gate |
| `proposer_stuck` | P7a–P7f | 3-in-a-row parity failures — brief or proposer model needs adjustment (NOT a system bug) |
| `agent_rate_limited` | P7a–P7f | Agent provider refused the attempt due to rate/usage limits — infrastructure block, not a kernel rejection |
| `agent_unavailable` | P7a–P7f | Agent binary/timeout/non-rate-limit exit prevented a valid proposal cycle — infrastructure block, not a compile failure |
| `total_attempt_cap_reached` | P7a–P7f | L0c spawn count hit `--total-attempt-cap` before reaching `--accepted-iteration-cap` — too many rejected mutations relative to accepted; surface for human review |
| `round_timeout` | P7a–P7f | L0c round-level wall-clock exceeded `--round-timeout-hours` — proposer too slow OR cycle-time engineering broken; surface |
| `measurement_transient_2x` | P7a–P7f | 2 consecutive measurement transient failures post-parity — vLLM/GPU instability |
| `sibling_baseline_inconsistent_repeated` | P9 | Per-sibling `inconsistent_baseline` flag fires repeatedly even on retry — sibling workload not stable on this stack; halt cross-family round |
| `sibling_holdout_capture_failed` | P1 | A sibling family's holdout capture failed thinking-probe row-3 OR `reasoning_tokens == 0` — workload stack regression on that sibling; halt P1 |
| `sibling_holdout_stale_or_invalid` | P9 | A sibling holdout that pre-existed from P1 fails P9's staleness check (file missing, thinking-probe regression, weight_version_id mismatch) — halt P9; human decides whether to recapture or drop the sibling |
| `sibling_fixture_capture_failed` | P2b | A sibling parity fixture capture failed reproducibility-3-times check — fixture cannot be trusted, halt P2b |
| `sibling_parity_fail` | P9 | Single-sibling parity probe failed (recorded, sibling excluded from composite by §6.6.5); round continues — NOT a halt code, listed here for grep completeness |
| `sibling_parity_partition_invalid` | P10 (audited from P9 artifacts) | `sibling_parity_passes.tsv` ∪ `sibling_parity_failures.tsv` does not cover the 8-sibling pool exactly once, OR the TSVs disagree with the per-(sibling, kernel) parity_check artifacts. Either P9 silently omitted a sibling from both TSVs (shrinking S) or omitted a failing sibling from `failures` (slipping past the ≥2 halt). P10 cannot trust the partition as written; halt and surface — this is a P9 code-path bug |
| `cross_family_correctness_broad_fail` | P9 | ≥ 2 siblings fail the parity probe — kernel mutation is workload-overfit on correctness, not safely promotable in any composite shape; halt and surface |
| `composite_descriptor_includes_parity_failed_sibling` | P10 | P10 attempted to mint a composite whose `component_families` intersects `sibling_parity_failures.tsv` — exclusion invariant violated; halt and surface (this is a P10 code-path bug, never user error) |
| `composite_descriptor_component_families_mismatch_p9` | P10 | `set(component_families)` ≠ `{heavy} ∪ set(sibling_parity_passes.family_id)` — P10 either dropped a parity-passing sibling (silent shrinkage) or included a parity-failing one. Halt and surface; this is a P10 code-path bug |
| `composite_descriptor_parity_fixture_content_drift` | P10 | A `parity_fixture_refs.<family>.<kernel>.content_hash` does not match recomputed §6.6.6 canonical manifest hash (yaml + every referenced blob in sorted-key delimiter-framed order) — fixture file mutated since P10 minted, OR descriptor's content_hash was wrong; bundle identity is invalid; halt |
| `composite_descriptor_parity_fixture_blob_missing` | P10 | A fixture yaml references a blob (probes_input.jsonl, reference logits npz, reference state npz) that does not resolve on disk — manifest hash cannot be computed; halt |
| `sampling_kl_divergence_exceeds_threshold` | P7d | Sampling mutation passes token-id determinism but fails KL-divergence check on post-top-k-top-p distribution — distribution drift that would harm long-horizon generation diversity; mutation rejected (NOT a round halt) |
| `fused_epilogue_checkpoint_fail` | P7e | Fused-epilogue mutation fails one of the four §2.2.6 checkpoints; the failing checkpoint name is recorded so the proposer can isolate the broken fusion boundary; mutation rejected (NOT a round halt) |
| `kernel_target_coverage_incomplete` | P10 (audited) | The union of {kernel_mutations targets, ROUND_NULL_RESULT targets, skipped_targets.target} does not equal the v0.3.1 target set `{deltanet, fused_epilogue, sampling, gatedattn}` — a target is silently absent; halt |
| `gpu_budget_skip_violates_priority_order` | P10 (audited) | The set of GPU-budget-skipped targets is not a suffix of the §0.6 priority order (post-P3a-roofline) — a higher-priority target was skipped while a lower-priority target ran; halt |
| `roofline_unaccounted_overhead` | P3a | Roofline run produces `Tpot_share` summing to <70% of decode time — unaccounted overhead is large; halt and surface |
| `roofline_tooling_unavailable` | P3a | Nsight Compute / Nsight Systems / `ncu` not available on DGX Spark — alternative profiler needed before P3a can complete; halt and surface |
| `fp8_kv_purity_violation` | P5b | fp8_e5m2 KV introduces divergence beyond fixture tolerances on >0 of 16 probes vs bf16 KV — kernel mutation work against this base would be debugging quantization noise; halt and surface. Fix paths: re-capture parity fixture against bf16 KV, OR loosen tolerance with explicit justification, OR change base stack's `kv_cache_dtype` |
| `actually_resolved_kernel_selection_drift` | P7a | A round bootstrap finds `actually_resolved_kernel_selection` from the live vLLM stack does NOT match the values pinned in `base_stack.yaml` (e.g., vLLM upgrade changed `vllm-default` routing). Round refuses to bootstrap; surface for human (re-pin or re-capture) |
| `composite_descriptor_parity_fixture_eligibility_mismatch` | P10 | Two-level invariant violated. **Outer:** `set(parity_fixture_refs.keys()) != set(component_families)` — descriptor has a fixture block for an excluded family, OR is missing one for an included family. **Nested:** for some `family in component_families`, `set(parity_fixture_refs[family].keys()) != set(m.kernel_target for m in kernel_mutations)` — family is missing a per-mutated-kernel fixture entry (e.g., dual-mutation round but family carries only `deltanet`), OR carries an entry for a non-mutated kernel target. Halt code surfaces with `level: outer` or `level: nested` plus the offending family/kernel pair |
| `composite_descriptor_gatedattn_base_invalid` | P10 | `kernel_mutations` entry with `kernel_target: gatedattn` has a vendor `base_kernel` (FA-style or FlashInfer) — internally inconsistent with P8's Triton-only precondition |
| `live_gate_failed` | P10 | Composite bundle passed parity but failed parent §11.5 live family gate — bundle goes to disk, no promotion |
| `composite_bundle_hash_mismatch` | P10 | Composite bundle's recomputed `workload_distribution_id` does not match its declared field — capture-mints/bootstrap-verifies violation; surface |
| `composite_descriptor_missing_per_family_trace` | P10 | A path in `per_family_seed_paths` or `per_family_holdout_paths` does not resolve to an existing file — composite cannot be assembled |
| `composite_descriptor_input_not_newline_terminated` | P10 | A per-family trace input does not end with `b'\n'` — assembler refuses to fuse records across boundaries; surface and fix per-family file before re-trying |

For each `HALT_REASON`, the round transcript writes:
1. The code (greppable).
2. The artifact path that triggered it (e.g., `eliminated.tsv` row index, `parity_check.json` path, fixture file path).
3. The phase identifier (P1–P10) where it triggered.
4. A `next_actions` block with the 1–3 things a human typically does to diagnose (read X, compare to Y, check Z).

---

## 10. Open questions

### 10.1 Hardware-aware vLLM fanout correctness

P2 discovers the maximum viable fanout for the current hardware instead of requiring four instances. Open: when the discovered fanout is greater than 1, do we need to serialize kernel compile (one instance compiles, others wait) or can compile parallelize without contaminating latency and memory evidence?

### 10.2 Triton cache invalidation across mutations

Triton caches compiled kernels keyed by source hash. A mutation's compiled `.so` could shadow a sibling mutation's compiled `.so` if the cache key collides. Mitigation in §5.4 is "cache by `Mutation-Hash`" — but Triton's own internal cache might still collide. Open: do we need to clear `~/.triton/cache` between mutations to be safe?

### 10.3 LLM proposer model choice for L0c

v0.1 sub-spec §2.4 pinned `gpt-5.4 high` for the L1 codex. L0c proposes kernel patches — a different reasoning task (more like writing CUDA than choosing config knobs). Open: is `gpt-5.4 high` still the right pin, or do we want `gpt-5.4 high` + a kernel-domain-specific system message? Or a different model entirely (e.g., a code-specialized variant)?

### 10.4 What if L0a returns FA4 + nothing else changes?

Default vLLM auto-select on Blackwell already resolves to FlashInfer-first / FA-second. If our L0a winner is "FA4 + Triton DeltaNet defaults + torch_compile=default" and that beats `vllm-default`, the win is just from forcing FA4 explicitly (not letting auto-select pick FlashInfer with #35138 issues). Open: is that a "real" L0 win or just a config-pin disguised as kernel work? Probably the former — it's the sub-spec's §9.3.13 requirement landing — but worth naming.

### 10.5 Per-mutation compile sandbox

Each L0c attempt builds the kernel into an isolated build directory to prevent cross-mutation contamination. Open: how do we enforce `LD_LIBRARY_PATH` / `PYTHONPATH` isolation so mutation N+1 doesn't accidentally load mutation N's `.so`? Probably one isolated venv per round, but this is a cycle-time concern.

### 10.6 GB10-specific FA4 / CUTLASS perf tables (v0.3.0)

FA4 is **beta/experimental on Blackwell** as of late 2025; the vLLM `FLASH_ATTN_VERSION=4` path is gated behind an env flag and not the default. NVIDIA has not published a GB10-specific FA4 perf table. CUTLASS 3.6+ has Blackwell support but bundled tile schedulers are tuned for B200 SM counts, not GB10's smaller-SM-count workstation profile. Open: do we (a) ship an L0a winner that pins `attention_backend=flash-attn-3` for stability and let L0c-DeltaNet/Fused-Epilogue carry the kernel uplift, or (b) pin FA4 if smoke-phase parity passes and accept the beta-status risk? The plan's L0a smoke phase will eliminate FA4 if it fails parity-vs-reference; if it passes, the question becomes whether to gate it behind a `--allow-beta-backend` CLI flag for production bundles.

### 10.7 GB10 SM count + register pressure for fused epilogues

GB10's SM count, register file size per SM, and shared memory per SM are not publicly documented. The plan's L0a smoke phase exercises every viable combination, but L0b autotune ranges (especially for Triton fused epilogues which need register-rich tiles) may need to be empirically discovered rather than copied from B200 defaults. Open: should L0b for `fused_epilogue` start from a wider grid than other targets (e.g., test both 64×64 and 128×128 register-tile shapes) to compensate for missing hardware specs?

### 10.8 Bandwidth-bound thesis verification

§0.5.2 claims GB10 is bandwidth-bound at the 273 GB/s level and that fused epilogues are highest-leverage. Open: the v0.3.0 plan assumes this; before committing to the priority order in §0.6, P3 (L0a) should produce roofline measurements per kernel category to confirm bandwidth-bound vs compute-bound — if the actual roofline is different from §0.5.2's prediction, §0.6 priority order needs to be revised. Add a roofline-probe sub-step to P3.

---

## 11. Changelog

- **v0.3.3-plan (2026-04-27)** — **Impl-agent amendment: anchor on L0b-empirical-winner, narrow v0.3.3 EXECUTABLE scope to single-target DeltaNet, preserve architectural intent for v0.3.4+.** Operator presented impl-agent analysis: 22 L0b winner bundles on disk; FA4 bundles all show synthetic-fingerprint identical objective values (1.18125); vllm-default bundles show real measured variance; the most-recent real-harness L0b winner is **`4866bc3f`** (obj=0.0171, ROUND_NULL_RESULT) with `vllm-default` routing + `fp8_e5m2` KV. Comparing to v0.3.2's hand-authored fixed stack (FA3 + bf16 KV + Triton sidecar + Triton sampler) found that the hand-authored stack diverges from `4866bc3f` on attention_backend, kv_cache_dtype, and epilogue_fusion_mode/sampling_kernel — all without measurement to back the divergence. Impl agent's argument: **empirical > architectural-coherence guess**, and ~14 GB of parity fixture data was already captured against the L0b-empirical-base — amending the doc preserves it. Three concrete amendments: (1) §5.2 base-stack table replaced with **L0b-empirical-winner anchor** (`base_bundle_path: 4866bc3f/bundle.yaml`, `actually_resolved_kernel_selection` recorded at fixture-build time so vllm-default's symbolic alias is pinned to concrete resolved values that don't drift across vLLM versions). (2) New phase **P5b fp8_e5m2 KV purity attestation**: 16-probe comparison vs bf16 KV before any L0c round, ensures kernel-mutation divergence ≠ FP8 quantization noise. New halt code `fp8_kv_purity_violation`. (3) **v0.3.3 executable L0c scope narrowed to `{deltanet}` only.** P7e (fused_epilogue) and P7d (sampling) deferred to v0.3.4+ pending verification that (a) the Triton sidecar / Triton sampler exist in vLLM shipping path with mutable source at known location, (b) the corresponding knob wires to vLLM init, (c) the §2.2.6 / §2.2.4 fixtures are re-captured against the L0b-empirical-base. The v0.3.2 architectural intent (composite multi-family workload, multi-target L0c, §0.5.2 bandwidth-bound thesis prioritizing fused_epilogue) is **preserved in the doc**, NOT demoted — only the executable gate moved. New §0.7a "v0.3.3 EXECUTABLE scope" table makes the architectural-intent-vs-executable-today distinction explicit. New halt code `actually_resolved_kernel_selection_drift` for vLLM-version-induced base-stack drift. **Net effect:** v0.3.3 ships single-target DeltaNet against the empirical L0b base, preserves the captured 14 GB fixture data, and explicitly carries multi-target / composite-workload / fused-epilogue work to v0.3.4 when wiring prerequisites are buildable.
- **v0.3.2-plan (2026-04-27)** — Architectural reframe: composite-multi-family workload from start + reviewer P0/P1 fixes. (operator directive) Pivoted from v0.3.1's "heavy family + late-stage multi-family promotion at P10" to **composite multi-family workload from the start**: workload spans variants from multiple families with full-agent-flow trajectories (8–30 turns, captured via codex against each family's eval set, NOT v0.3.1's 4-turn thinking-only snippets). Bundle's `workload_distribution_id` IS the multi-family identity directly — no separate "heavy bundle → composite" promotion phase. Closes the L0c-target-vs-multi-family-promotion lock the reviewer flagged: any of `deltanet`, `fused_epilogue`, `sampling` can multi-family-promote because the bundle is for the composite workload from the start, not heavy-then-promoted. Each family's full-flow trajectory is split 3:1 seed:holdout at trajectory boundary; both slices included in `workload_distribution_id` via §6.6 canonical hash; per-family holdout slice is what P9 cross-family check uses (v0.1.5 stratified-split style). Adds §1.0 architectural shift, §1.1 composite workload definition with 9-family table + full-agent-flow capture criteria + per-family holdout slice contract. **Reviewer P0/P1 fixes also landed:** (P0.1 P9/P10 fused/sampling fixture gap) Resolved architecturally by composite-from-start — fixtures cover all per-family shapes natively. (P0.2 L0c-first baseline residue) `--base-bundle` renamed to `--base-stack-resolution {vllm_default, reference_baseline, bundle}` + `--base-bundle-path`; `Measurement-Role` becomes `vllm_default_baseline_remeasured` for L0c-first ordering; AR.48c reworded to match. (P0.3 fused/sampling depend on deferred L0a) v0.3.2 commits to a **fixed base stack** (FA3 + Triton DeltaNet v2 + Triton sidecar epilogue + Triton sampler + cuBLAS FP8 + bf16 KV) so P7e/P7d preconditions don't reference an unresolved L0a winner. v0.3.3 replaces the fixed stack with the actual L0a winner. (P0.4 fused_epilogue overclaim) Revised §0.6 row #2: Triton sidecar fuses `+residual → RMSNorm → FP8 quant` but cannot eliminate the GEMM-output write itself (that requires CUTLASS/CuTe epilogue integration deferred to v0.3.3+). Magnitude downgraded from "8–15% e2e" to "3–8% e2e" pending P3a. (P0.5 fixture content_hash schema-blind) `fixture_content_hash` rewritten as a schema-aware function: `REFERENCED_KEYS_BY_SCHEMA` map keyed by `parity_check_method`. Each schema declares its referenced-key list; unknown schemas raise. New schemas (sampling, fused_epilogue) bind their full referenced-blob set into bundle identity. (P1 cleanup) §2.2.3 RMSNorm, §2.2.5 KV cache, §2.2.7 FP8 GEMM Triton fixture sections marked REMOVED with archival `<details>` tags. §6.2 parity semantics gains §6.2.3 (sampling — token-id determinism + KL divergence) and §6.2.4 (fused-epilogue — four-checkpoint fail-on-first). §7.3 dependency rationale split into v0.3.1+ active (L0c-first) and v0.3.3+ deferred (L0a→L0b→L0c) sections. §8.1 and §8.2 marked DEFERRED to v0.3.3+. §9 verification group description updated to drop AR.49/AR.52. `multi_instance_insufficient_memory` halt code reworded for the `N ≥ 1` empirical-discovery semantics.
- **v0.3.1-plan (2026-04-27)** — Reviewer P0/P1 pass on v0.3.0 + operator sequencing inversion. Six fixes + scope narrowing + L0c-first reordering: (P0.1 fp8_gemm_triton unreachable) Dropped P7f entirely. L0a action space's FP8 GEMM stays `{cutlass, cublas}` — no Triton path. AR.52 removed. Re-add deferred to v0.3.2 if a mutable Triton FP8 GEMM enters vLLM shipping. (P0.2 workload.yaml fixture map) `parity_fixture_refs` now a full map covering all v0.3.1 targets `{deltanet, gatedattn, fused_epilogue, sampling}`. (P0.3 sibling fixtures) Heavy-family-only for non-DeltaNet/GatedAttn targets in v0.3.1; sibling-fixture work for fused_epilogue / sampling deferred to v0.3.2. (P0.4 L0b autotune undefined) Dropped expanded L0b. v0.3.1 keeps L0b for DeltaNet only, and even that is deferred (see L0c-first below). (P0.5 fused_epilogue Triton-vs-CUTLASS contradiction) Picked **Triton sidecar** as the sole mutation surface in v0.3.1. P7e precondition no longer requires CUTLASS; only requires `epilogue_fusion_mode != none`. CUTLASS-side fused-epilogue mutation deferred to v0.3.2. (P0.6 4-instance claim) P2 phase changed from "4 instances required" to **empirical instance-count discovery**: try 4 → 3 → 2 → 1 at decreasing utilization; accept `N ≥ 1`; halt only if no instance starts. `output/p2_instance_capacity.json` consumed by all downstream phases. (P0.7 bandwidth thesis numeric claim) AR.54 priority-order enforcement gated on **new P3a roofline phase** that measures actual per-category Tpot share, DRAM throughput, SM occupancy, tensor-core utilization, kernel launch count via Nsight Compute / Nsight Systems. If measured priority differs from §0.6's prediction, §0.6 is rewritten before AR.54 activates. **Scope narrowing:** v0.3.1 has 3 executable L0c targets `{deltanet, fused_epilogue, sampling-gated}` + 1 schema-only (`gatedattn`). RMSNorm dropped (subsumed into fused_epilogue). FP8 GEMM Triton dropped (unreachable). KV cache dropped as L0c (stays as L0a config knobs). **Sequencing inversion (operator note):** v0.2 / v0.3.0 ran L0a → L0b → L0c. v0.3.1 inverts to **L0c first against `vllm-default` baseline** (L0a/L0b have already been shown empty against baseline; L0c is the unexplored surface). New phase order: P1/P2/P2b/P5 setup → **P3a mandatory roofline probe** → P7a/P7e/P7d L0c rounds → P9/P10. Original P3 (L0a) and P6 (L0b) deferred to v0.3.2 exploration only. **§5.5 iteration_brief.md** now embeds DGX Spark hardware notes + per-target speedup tips so the L0c agent does NOT do online research per iteration — facts are pre-researched and load-bearing. New halt codes `roofline_unaccounted_overhead`, `roofline_tooling_unavailable`. Removed halt code `rmsnorm_downstream_logit_diverges` (target dropped). HF model card correction: Qwen3.5-27B-FP8 padded vocab is **248,320**, NOT 152K. Hardware corrections from v0.3.0 (128 GB LPDDR5x at 273 GB/s, FA4 beta on Blackwell) carry forward unchanged.
- **v0.3.0-plan (2026-04-27)** — Scope expansion: kernel categories beyond attention + FP8 GEMM. Triggered by @TheAhmadOsman x.com post enumerating the kernel surface (MatMul, Attention, RMSNorm, KV cache, Quantized linear, Sampling, Fused) and grounded in GB10/DGX Spark physical facts. Six substantive additions: (1) **Hardware grounding §0.5.** GB10 = Blackwell + 128 GB LPDDR5x at 273 GB/s (NOT 192 GB — v0.2.x had this wrong). FP8 peak ~500 TFLOPS dense. FA4 is beta on Blackwell; CUTLASS tile schedulers are B200-tuned and may underperform on GB10. SM count / register file / cache sizes not publicly documented — must be empirically discovered. (2) **Bandwidth-bound thesis §0.5.2.** GB10 is severely bandwidth-bound at ~30× lower memory BW than B200; this reframes the priority order: kernel categories that reduce memory traffic (fused epilogues) outrank categories that improve raw FLOPS. (3) **Priority order §0.6** ranks five kernel categories by expected throughput uplift on Qwen 3.5 27B FP8 + GB10 + thinking-heavy decode: FP8 GEMM > DeltaNet > Fused norm+residual+quant epilogues > GatedAttn+KV > Sampling. (4) **§3.1 L0a action space expanded** with five new knobs (`rmsnorm_kernel`, `kv_cache_block_size`, `kv_cache_dtype`, `sampling_kernel`, `epilogue_fusion_mode`). Grid grows from 180 to ~3,840 combos; smoke-phase pruning unchanged. Added `kernel_compatibility_matrix.yaml` for documenting incompatible combinations (e.g., cuBLAS + epilogue fusion). (5) **§5.2 L0c kernel targets** generalized from {`deltanet`, `gatedattn`} to six targets {`deltanet`, `gatedattn`, `rmsnorm`, `sampling`, `fused_epilogue`, `fp8_gemm_triton`}, each with its own per-target parity-check semantic in §6.2 and per-target parity fixture schema in §2.2.3–§2.2.7. **`fused_epilogue` is the v0.3.0 thesis target** (priority #3 per §0.6, projected 8–15% e2e on bandwidth-bound GB10). (6) **Phase DAG §7** generalizes P7/P8 to P7a–P7f, one round per kernel target, parallelizable across independent kernel targets. New AR items AR.49–AR.54 cover the new kernel targets + skip-coverage + priority-order audits. New halt codes for per-target parity failures (`rmsnorm_downstream_logit_diverges`, `sampling_kl_divergence_exceeds_threshold`, `fused_epilogue_checkpoint_fail`) and round-coverage invariants (`kernel_target_coverage_incomplete`, `gpu_budget_skip_violates_priority_order`). New §10.6/§10.7/§10.8 open questions track FA4-on-GB10 status, GB10 register pressure for fused-epilogue tile sizing, and the bandwidth-bound thesis verification via P3 roofline probes. The L0a/b/c contract, parity-fixture infrastructure, BOOTSTRAP/Candidate/Rescreen/FINALIZE shape, three-cap structure, paired-A/B baselines, composite-bundle identity (parent §5.9 unchanged), partition-completeness invariants, and all v0.2.x AR items carry forward unchanged. v0.3.0 is purely additive scope.
- **v0.2.12-plan (2026-04-26)** — Eleventh reviewer P1/P2 pass. Three fixes: (P1.1) P10 phase-table precondition was "P9 lands in §8.4 row 1 (generalizes)" — but v0.2.10 reordered §8.4 with overfit/regression rows first, so "row 1" now points at the overfit reframe-scope outcome. Replaced with the explicit advance predicate `R == 0 AND I ≥ ⌈0.75 × S⌉` (zero regressing siblings AND ≥6 of 8 or ≥6 of 7 improving) so the precondition cannot drift if §8.4 is reordered again. (P1.2) §6.6.5 P10 build procedure restructured with the partition-completeness audit as **step 1** (run BEFORE the TSVs are read as authoritative). The audit reconstructs `derived_passes` and `derived_failures` from the per-(sibling, kernel) parity_check artifacts and demands byte-equality with the committed TSVs. Steps 2–7 are now downstream of the audit — an implementer following the procedure literally cannot skip the partition check. Halt path summary added: step 1 → `sibling_parity_partition_invalid`; step 7 → `composite_descriptor_includes_parity_failed_sibling`. (P2.1) Specified canonical row shapes for both TSVs to make byte-equality reconstruction deterministic. `sibling_parity_passes.tsv`: single column `family_id`, sorted. `sibling_parity_failures.tsv`: **one row per (sibling, failing_kernel) pair**, with `(family_id, kernel_target)` ascending sort. A dual-kernel failure produces two rows, not one with a multi-value field. The partition's failure set is the projection over `family_id`. Closes the loophole where dual-kernel failure shape ambiguity could make AR.48c4's byte-equality reconstruction non-reproducible.
- **v0.2.11-plan (2026-04-26)** — Tenth reviewer P1/P2 pass. One fix: (P1.1) Closed the partition-completeness gap. AR.48d(a0) anchors `component_families` to `sibling_parity_passes.tsv`, but a faulty P9 could have omitted a sibling from both `passes.tsv` and `failures.tsv` (silently shrinking S) or omitted a failing sibling from `failures.tsv` (slipping past the ≥2 broad-fail halt) — and AR.48c4 was trusting the TSVs as primary rather than deriving them from the per-(sibling, kernel) artifacts. Added an explicit partition-completeness invariant with four sub-checks: **coverage** (passes ∪ failures == 8-sibling pool), **disjointness** (passes ∩ failures == ∅), **pass-side derivation correctness** (every sibling in passes has all per-kernel artifacts with `pass: true`), **fail-side derivation correctness** (every sibling in failures has at least one per-kernel artifact with `pass: false`). The verifier reconstructs the partition from the per-(sibling, kernel) parity_check artifacts and demands byte-equality with the TSVs. New halt code `sibling_parity_partition_invalid` for any sub-check failure. The TSVs are now derived data audited against ground-truth artifacts, not primary trusted data.
- **v0.2.10-plan (2026-04-26)** — Ninth reviewer P1/P2 pass. Three fixes: (P1.1) §8.4 had uncovered `(I, R)` pairs — outcomes like S=7, I=6, R=1 didn't match any row (advance requires R==0; partial/weak rows require R==0; overfit row requires R≥3). Restructured table with explicit decision precedence top-to-bottom, with R-bands handled before I-bands: row 1 is `R ≥ ⌈0.375×S⌉` (overfit), row 2 is the new `0 < R < ⌈0.375×S⌉` (small-regression → reframe-scope, ships heavy-family-only with per-cluster v0.3 note), rows 3–5 are the `R == 0` partition by I-band. Added explicit coverage proof showing every (I, R) maps to exactly one row. (P1.2) AR.48d gains a load-bearing first step (a0): `set(composite.component_families) == {"responses-sdk-adapter-cutover"} ∪ set(sibling_parity_passes.family_id)`. P10 cannot silently drop a parity-passing sibling (shrinking the promoted workload after P9's decision) and cannot include a parity-failing one. New halt code `composite_descriptor_component_families_mismatch_p9`. The other AR.48d steps now sit on top of this anchor: every fixture-eligibility/content-hash/trace-assembly check assumes `component_families` reflects P9's eligible set, which (a0) proves. (P2.1) Halt-code text for `composite_descriptor_parity_fixture_eligibility_mismatch` extended to describe both outer (family-key mismatch) and nested (per-family mutated-kernel mismatch) cases. Surface message includes `level: outer` or `level: nested` so an operator can tell at-a-glance which level the violation is at.
- **v0.2.9-plan (2026-04-26)** — Eighth reviewer P1/P2 pass. Four fixes: (P1.1) Resolved threshold contradiction. v0.2.6's "advance needs ≥5 when P=1" was inconsistent with the formula `I ≥ ⌈0.75 × S⌉` which gives `⌈5.25⌉ = 6` for S=7. Picked the formula explicitly: §6.6.5's "advance needs ≥6 when P=0, ≥5 when P=1" rewritten to "S=8 needs I≥6, S=7 needs I≥6 (since `⌈5.25⌉ = 6`)". §8.4 row examples corrected: "≥5 of 7" → "≥6 of 7" in advance row, S<7 cases removed entirely (P ≥ 2 halts before §8.4 is consulted, so S ≤ 6 never reaches the table). All other formula examples re-verified consistent. (P1.2) AR.48d step (f) split into outer + nested invariant. Outer: `parity_fixture_refs.keys() == component_families` (existing). Nested: `parity_fixture_refs[family].keys() == set(m.kernel_target for m in kernel_mutations)` per family — every mutated kernel target has a per-family fixture entry; no extras for non-mutated kernels. Closes the dual-mutation loophole where a family could carry only DeltaNet fixture while GatedAttn was also mutated, leaving the GatedAttn fixture content unbound from bundle identity. §6.6.2 identity-binding paragraph extended to call out the nested invariant. (P2.1) §1 cross-family pool paragraph updated: "16-probe parity gate per sibling" → "per (sibling × mutated kernel target)" with explicit dual-mutation vs single-mutation framing. Now matches AR.48c4 and the P9 phase row. (P2.2) Stale placeholder `<sha256_of_yaml_plus_npz_companion>` in the §6.6.1 schema example replaced with `<sha256_fixture_manifest_per_6_6_6>` everywhere — pointers an implementer at the canonical manifest procedure rather than the old incomplete hash rule.
- **v0.2.8-plan (2026-04-26)** — Seventh reviewer P1/P2 pass. Five fixes: (P1.1) Closed the dual-mutation P9 parity gap. P9 step (b) and AR.48c4 reframed: parity check is per (sibling × every mutated kernel target). In dual-mutation rounds (P7+P8 both produced winners), each sibling is checked against BOTH the DeltaNet and GatedAttn mutated kernels. A sibling enters `sibling_parity_passes.tsv` ONLY if it passes for ALL mutated kernel targets; failure on any one routes it to `sibling_parity_failures.tsv` with `failing_kernel_target` recorded. Per-(sibling, kernel) artifact: `parity_check_sibling_<fid>_<kernel>.json`. AR.48c4 verifier adds a completeness audit: every (sibling, mutated_kernel) pair in the composite has a `pass: true` artifact. (P1.2) Fixture content_hash redefined as a canonical manifest hash. New §6.6.6 specifies the procedure: yaml file bytes + every referenced blob (`probe_input_ref`, `reference_logits_ref`, optionally `reference_state_snapshots_ref`) in sorted-key delimiter-framed order. Closes the previous loophole where changing `probes_input.jsonl` or one referenced npz could leave content_hash unchanged. AR.48d step (d) updated to call §6.6.6 explicitly. New halt code `composite_descriptor_parity_fixture_blob_missing`. (P2.1) §0 scope bullet stripped of "per-sibling holdout capture at decision time" — replaced with "staleness check on P1-pre-captured sibling holdouts" + "per (sibling × mutated kernel target)" framing. The bullet now matches v0.2.7's P1-captures / P9-verifies-staleness contract. (P2.2) Composite-size YAML comment changed from "1..9 entries" to explicit `{8, 9}` set: 9 if zero sibling parity failures, 8 if exactly one failed, ≥2 failures halts before P10. The "1..9" wording was technically possible to read as "tiny composite after broad failure" which contradicts the halt rule. (P2.3) §9 verification group intro updated from "four groups" to "seven groups" — matches the actual list after the v0.2.7 split (L0a, L0b, L0c, P1, P2b, P9, P10). Editorial but the count was wrong as a scan target.
- **v0.2.7-plan (2026-04-26)** — Sixth reviewer P1/P2 pass. Five fixes: (P1.1) §6.6.1 stale paragraph still claimed "P9 captures sibling holdouts and commits to BOOTSTRAP" — directly contradicting v0.2.6's P1-pre-capture / P9-staleness-check design. Rewrote that paragraph to say sibling holdouts are P1-captured-and-committed-to-main; P9 only verifies staleness; no commit-to-BOOTSTRAP-after-bootstrap pattern is used. (P1.2) P1 phase-table exit gate strengthened. Was: heavy `workload.yaml` + heavy seed + heavy holdout + reproducible workload_distribution_id. Now also requires: all 8 sibling holdout files exist on main, each captured against `vllm-default`, each passes thinking-probe row-3, `git log --diff-filter=A` on main shows the 8 new files added by P1. Without these, P1 PASS gate is not satisfied and downstream phases cannot precondition on it. New halt code path tied to existing `sibling_holdout_capture_failed`. (P1.3) Composite identity now binds to per-family parity fixture contents. `parity_fixture_refs` is no longer just heavy `{deltanet, gatedattn}` paths — it's a per-family map with both `path` and `content_hash` fields, populated for every family in `component_families` (heavy + parity-passing siblings). content_hash = sha256(yaml_bytes + npz_companion_bytes). Body field, falls under yaml_hash, propagates into `workload_distribution_id`. AR.48d gains step (d) (fixture content hash recompute) and step (f) (eligibility consistency: fixture_refs.keys() == component_families). New halt codes `composite_descriptor_parity_fixture_content_drift` and `composite_descriptor_parity_fixture_eligibility_mismatch`. Closes the previous loophole where a sibling fixture could change without the bundle's id reflecting it. (P2.1) Composite descriptor's `seed_trace_ref` / `holdout_trace_ref` comments stripped of "9 family" wording — replaced with "for the families in component_families (1..9 entries depending on P9 outcome)". §6.6.1 component_families YAML comment expanded to call out the dynamic-size case explicitly. Dependency rationale §7.3 line about P10 hashing "over all 9 family traces" updated to "over the trace files for component_families (size dynamic per P9 outcome) + every mutation patch hash + every per-family parity fixture content hash". (P2.2) §9 verification group split: AR.48c3 in its own P1-precapture group; AR.48c4b in its own P2b group (was missing entirely); AR.48c2/c4 retain the P9 group; AR.48d retains the P10 group with explicit "incl. fixture content hash" note.
- **v0.2.6-plan (2026-04-26)** — Fifth reviewer P1/P2 pass. Five fixes: (P1.1) Closed the "single sibling parity failure ships in composite" hole. New §6.6.5 specifies P10's dynamic composite sizing: P10 reads `sibling_parity_passes.tsv`, sets `component_families = ["responses-sdk-adapter-cutover"] + parity_passing_siblings`, builds composite trace files over only the included families, runs parent §5.9 hash over the reduced descriptor. Composite NEVER includes parity-failing siblings. New AR.48c4 invariant audit (`set(component_families) ∩ set(parity_failures.family_id) == ∅`). New halt code `composite_descriptor_includes_parity_failed_sibling`. §8.4 thresholds rescaled to use `len(parity_passing_siblings)` denominator (≥75% improved to advance, ≥37.5% regressed to reframe). (P1.2) Sibling parity probe now has actual reproducible reference artifacts. New §2.4-pre defines per-sibling parity fixture schema (16 probes per sibling, captured against §2.2.0 reference, same structure as heavy fixtures but smaller). P2b's scope expanded to capture all 9 family fixtures (heavy at 64 probes, 8 siblings at 16 probes each). New AR.48c4b verifies all 16 sibling fixture files exist with matching `weight_version_id`. New halt code `sibling_fixture_capture_failed`. (P1.3) Sibling holdout capture moved from P9 to P1 — closes the "cannot commit to BOOTSTRAP after bootstrap" ledger conflict. P1 captures all 8 sibling holdouts up front and commits them to main before any L0 round bootstraps. P9 step (a) becomes a staleness check (file present, thinking-probe still passes, weight_version_id current), not a fresh capture. AR.48c3 rewritten. New halt code `sibling_holdout_stale_or_invalid`. (P2.4) P10 row's "existing v0.1.5 multi-family capture" wording corrected to "P1-captured holdouts + existing v0.1.5 seed_trace_v5" — distinguishes seeds (existing artifacts) from holdouts (P1-produced artifacts). (P2.5) §1 cross-family pool paragraph updated to spell out P9's three sub-steps (holdout staleness check + sibling parity + paired throughput) — replaces the stale "n=5 measurements on each sibling, 40 total" one-arm summary.
- **v0.2.5-plan (2026-04-26)** — Fourth reviewer P1/P2 pass. Five fixes: (P1.1) Factual blocker — claimed sibling `holdout_trace_v5.jsonl` files don't actually exist for the 8 sibling families (only seed traces do). Plan no longer pretends they pre-exist. P9 is extended to capture them as the first sub-step: for each sibling, fresh holdout capture against `vllm-default` (same semantic as heavy family's §2.1 holdout — second independent capture, same prompts, different seed), written to `benchmark_blueprints/families/<sibling_fid>/holdout_trace_v5.jsonl` and committed to the round's BOOTSTRAP commit before any measurement. New AR.48c3. New halt code `sibling_holdout_capture_failed`. (P1.2) Composite promotion now has a per-sibling correctness gate. P9 step (b): 16-probe parity-vs-§2.2.0-reference check against the kernel-mutated stack for each sibling family. Siblings that fail parity are flagged `sibling_parity_fail` and excluded from §8.4's 6-of-8 outcome (counted in neither numerator nor denominator). ≥ 2 sibling parity failures triggers `cross_family_correctness_broad_fail` halt — kernel mutation is workload-overfit on correctness and not safely promotable. New AR.48c4. New halt codes `sibling_parity_fail`, `cross_family_correctness_broad_fail`. (P1.3) Composite descriptor's GatedAttn `kernel_mutations` entry must have a Triton `base_kernel` (regex `^triton-`) — vendor kernels (FA-style, FlashInfer) are not source-mutable and P8 only runs when L0a winner is Triton, so a vendor base is internally inconsistent. Example fixed from `flash-attn-4` to `triton-gatedattn-<variant>`. New AR.48c5. New halt code `composite_descriptor_gatedattn_base_invalid`. (P2.4) §8.4 decision table now covers the 4–5 sibling middle band — partial generalization is reframe-scope (heavy-family-only ship + v0.3 per-cluster scope), distinct from both the strong-generalization advance and the weak-generalization or regression cases. (P2.5) §0 scope bullet updated to reflect P9's actual shape: holdout capture + parity probe + paired n=5 baseline + n=5 winner per parity-passing sibling.
- **v0.2.4-plan (2026-04-26)** — Third reviewer P1/P2 pass. Five fixes: (P1.1) P9 cross-family round now requires **paired n=5 baseline + n=5 winner per sibling family**, all 80 measurements committed in the same round under the same vLLM lifecycle, with `Measurement-Role` trailers `sibling_baseline` / `sibling_winner`. Per-sibling Welch-t consumes only contemporaneous pairs — never historical `objective_mean`. Closes the "no defensible cross-family comparison" hole. New AR.48c2. New halt code `sibling_baseline_inconsistent_repeated`. (P1.2) Sibling holdout traces sourced from existing v0.1.5-plan multi-family-v5 capture (`benchmark_blueprints/families/<fid>/holdout_trace_v5.jsonl`) — every included family already has both seed and holdout from the v0.1 stratified-split work. §6.6.1 schema fully spells out the 9 sibling holdout paths. P10 finalize verifies every per-family path resolves before assembly. New halt code `composite_descriptor_missing_per_family_trace`. (P1.3) Composite descriptor's `kernel_mutation` (singular) replaced with `kernel_mutations` (list, sorted by `kernel_target` for hash stability). P7 (DeltaNet) and P8 (GatedAttn) winners both bind into the composite identity if both produced winners; single-winner rounds have a 1-element list. AR.48d step (c) iterates over all entries. (P2.4) `assemble_composite_trace` rewritten as **newline-safe**: enforces every per-family input ends with `b'\n'`, inserts explicit single-LF separator between component files, refuses to fuse records across file boundaries. P10 runs a pre-flight check before assembly. New halt code `composite_descriptor_input_not_newline_terminated`. (P2.5) Stale references fixed: `§10.6` → `§6.6` in the P10 phase row; halt-code phase identifiers `P4` updated to `P2b` (and to the broader list of consuming phases for `fixture_weight_mismatch`).
- **v0.2.3-plan (2026-04-26)** — Second reviewer P1/P2 pass. Five fixes: (P1.1) Composite bundle hash now goes through **parent §5.9 canonical procedure unchanged** — no second hash algorithm. New §6.6 defines composite descriptor schema (concatenated trace files, mutation patch hash inside descriptor body, alphabetical-by-family_id assembly). Mutation patch identity flows through `yaml_hash` automatically because `kernel_mutation.patch_hash` is a body field. AR.48d rewritten to a 4-step recomputation check (assemble traces, recompute patch hash, recompute workload_distribution_id via parent procedure, verify evidence-only treatment of heavy-family bundle). Added §6.6.4 capture-mints/bootstrap-verifies contract — `/admin/load_tuned_config` will now correctly recompute and verify. (P1.2) `tune-kernel-select` Effects-step-2 rewritten: smoke phase explicitly runs both the determinism probe AND the parity-vs-reference probe before any combo proceeds to screen measurement. Added explicit fixture-existence precondition check at bootstrap (refuses to start if fixtures missing). (P1.3) AR.43 now admits structured pre-parity rejection paths: every `candidates/<NNN>/` has `parity_check.json` with `reason` ∈ {`ran_passed`, `parity_logit_diverged`, `parity_state_diverged`, `intermittent_parity`, `patch_apply_failed`, `compile_timeout`, `compile_nvcc_error`, `sandbox_setup_failed`}. The "parity didn't run" branch is now a first-class outcome with structured reason, not a missing-file ambiguity. (P2.4) AR.44 rewritten to match the actual schema: `parity_fixture_refs.{deltanet,gatedattn}` are path strings; verifier resolves the path, loads the fixture yaml, reads top-level `fixture_id`, compares. (P2.5) §8.1 row 1 advance instruction corrected from "→ P4 (parity fixture capture) → P6" to "→ P6 directly; fixture already in place from P2b".
- **v0.2.2-plan (2026-04-26)** — Reviewer P1/P2 pass. Seven fixes: (P1.1) parity fixture is now built **before** L0a, against an externally-trusted §2.2.0 reference baseline (forced FA4 + Triton DeltaNet defaults + cuBLAS FP8 + default torch_compile + cuda_graph off), not against the L0a winner. L0a's smoke phase becomes determinism + parity-vs-reference, catching the FlashInfer #35138 class of deterministic-but-wrong kernels before they're measured. Phase P4 (post-L0a fixture capture) replaced with P2b (pre-L0a). New AR.38b records parity-vs-reference culls. (P1.2) `tune-kernel-autotune` and `mutate-kernel` both gain `--base-measurements 5` flag — paired-A/B baseline is re-measured in the same round, in the same vLLM lifecycle, before autotune/mutations run; Welch-t uses contemporaneous baseline rows tagged `Measurement-Role: l0{a,b}_baseline_remeasured`, not the prior bundle's `objective_mean`. New AR.41b + AR.48c. (P1.3) P10 mints a **new composite bundle** with `family_id: multi-family-v5-l0-kernel-tuned` and `workload_distribution_id` hashed over (heavy seed+holdout, all 8 sibling seed traces, mutation patch). The heavy-family pseudo-bundle is preserved as evidence but does NOT itself ship for sibling families. New AR.48d. (P1.4) L0c gains three independent caps: `accepted_iteration_cap=12` (parity-passing measured candidates), `total_attempt_cap=36` (total spawn count regardless of accept/reject), `round_timeout_hours=12` (round-level wall-clock). Closes the alternating fail/pass infinite-loop hole. New AR.48b. (P2.5) `workload.yaml` schema gains explicit `parity_fixture_refs: {deltanet: ..., gatedattn: ...}`; AR.44 now references it. (P2.6) P9 precondition fixed: advances if AT LEAST ONE of P7/P8 (whichever ran) produced a winning mutation, matching §8.3's decision rule. (P2.7) AR.47 split into two-part check: `sha256(mutation.patch)` matches `Mutation-Hash` trailer, AND `git apply --check` passes against declared base. New halt-reason codes: `l0a_precondition_missing_fixture`, `l0a_parity_fail_winner`, `flashinfer_passes_parity`, `total_attempt_cap_reached`, `round_timeout`, `composite_bundle_hash_mismatch`. §6.3 clarified: fixture is round-independent, rebuilds only on weight rotation.
- **v0.2.1-plan (2026-04-26)** — Reframed the entire plan around dependency-ordered phases (DAG, not calendar) per operator directive: "do not estimate how long for work item, focus on clear verification — if a step is good, or should fail and let human diagnose." Removed every "X weeks / X days / X hours" estimate. Replaced §7 sequencing table with phase DAG (P1–P10), each with explicit precondition + exit gate + halt-and-surface conditions. Replaced §3.6, §4.5, §5.7 effort sections with **step-level verification tables** (PASS / FAIL-retry-by-agent / FAIL-escalate-to-human). Replaced §8 "Week N" framings with classification-and-action tables (advance / halt-and-diagnose / ship-as-is / reframe-scope). Added §9.1 — named `HALT_REASON` catalogue (19 codes) so a human looking at a halted round has a stable greppable code + diagnosis entrypoint. Cycle-time targets in §5.4 reframed from time budgets to hard timeout + watchdog rails (per parent §5.7). The principle: coding agents drive each phase to completion at whatever pace; the plan defines what success looks like, what recoverable failure looks like, and what failure-modes warrant human diagnosis.
- **v0.2.0-plan (2026-04-26)** — Initial L0 kernel action plan. Pivots from v0.1's L1/L2 config-search direction (which both hardened rounds confirmed as headroom-empty within noise) to L0 kernel work. Single heavy family (`responses-sdk-adapter-cutover`, decode-heavy 4-turn trajectory). Three workstreams: L0a kernel selection (deterministic grid, ~180 combos pruned by determinism smoke), L0b kernel autotune (`@triton.autotune` + CUTLASS Python tuner), L0c kernel mutation (Karpathy-style LLM-in-the-loop with parity gate as load-bearing primitive). Parity-fixture infrastructure (per-family probe set, logit + state-snapshot compare for DeltaNet, logit-only for GatedAttn) is the v0.2 net-new correctness substrate. Cross-family generalization across 8 sibling families. Bundle promotion gated on live family gate per parent §11.5. 11 new verification items (AR.38–48). Inherits the entire v0.1.5-plan substrate (replay-round, statistical machinery, worktree, BOOTSTRAP, trailers, watchdog, thinking probe). Open questions track multi-instance compile, Triton cache, LLM proposer pin, FA4-pinning-as-L0-win semantic, and per-mutation sandbox isolation.

---

*End of v0.3 plan v0.3.3-plan.*

---

## 12. References (v0.3.0)

External sources informing §0.5 hardware grounding and §0.6 priority order:

- **NVIDIA DGX Spark / GB10 product page** — https://www.nvidia.com/en-us/products/workstations/dgx-spark/ (128 GB LPDDR5x, 273 GB/s, 1 PFLOP FP4 sparse / 500 TFLOPS dense)
- **NVIDIA Blackwell architecture overview** — https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/ (5th-gen Tensor Cores, FP8/FP6/FP4 native, MXFP block scaling, `tcgen05` PTX)
- **CUTLASS Blackwell examples** — https://github.com/NVIDIA/cutlass/tree/main/examples (sm_100 kernel examples 62–67 series)
- **FlashAttention repo (FA4 status)** — https://github.com/Dao-AILab/flash-attention (FA4 beta on Blackwell as of late 2025)
- **vLLM source tree** — https://github.com/vllm-project/vllm/tree/main/csrc and `/vllm/model_executor/layers/` (rms_norm kernels, paged_attention, quantization, sampler, fused_moe)
- **FlashInfer** — https://github.com/flashinfer-ai/flashinfer (paged batch attention, sampling kernels)
- **flash-linear-attention (DeltaNet kernels)** — https://github.com/sustcsonglin/flash-linear-attention
- **Qwen3 / Qwen3-Next** — https://github.com/QwenLM/Qwen3 + HuggingFace `Qwen/Qwen3-Next-*` model cards (hybrid attention reference)
- **@TheAhmadOsman kernel framing** (x.com / r/LocalLLaMA admin post, 2026-04-27) — enumerates kernel categories operators care about: MatMul, Attention, RMSNorm, KV cache, Quantized linear, Sampling, Fused.

**Caveats explicitly flagged** (informed §10.6/§10.7/§10.8 open questions):
- GB10 SM count, register file per SM, shared memory per SM, L1/L2 cache sizes are NOT in public NVIDIA docs as of late 2025.
- GB10-specific FA4 perf table and CUTLASS tile-scheduler perf table not published.
- vLLM PR numbers cited in research are best-effort; verify against current tracker before quoting in implementation handoffs. The gist for reviewers: kernel headroom is what's left after L1 was exhausted by v0.1 hardening. Karpathy's autoresearch loop shape is the right reference for L0c — short cycles, LLM-driven, git-as-ledger, never-stop-until-terminal — but the parity gate is what kernel work needs that training-loop work doesn't. Coding agents drive every phase, so the plan does not estimate calendar time; it defines what each phase produces, when it advances, and which named failure modes a human needs to look at.*
