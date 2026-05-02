# L0 FFN GEMM Pivot — Backend Selection + Conditional Mutation

Generated: 2026-05-02

## Scope

This document specifies the next L0 auto-research target after the P3a roofline data (2026-04-30) invalidated `chunk_delta_h` (DeltaNet) as the primary L0c kernel-mutation target **for warm-cache decode workloads**. It covers:

- L0-style auto-research on FP8 FFN GEMM **backend selection** (Phase A).
- Conditional L0c-style auto-research on **kernel mutation** of the selected backend (Phase B), runs only if the winner is Triton-source-mutable.

Out of scope:
- Attention backend selection (FA3/FA4/FlashInfer) — covered in HLD §3.1.1, deferred for now since prefill (where attention dominates) is a smaller fraction of typical agent decode workflows.
- Multi-family workload composite — keep heavy-family-only per v0.3.3.
- Continuing L0c-DeltaNet as primary investment — that work continues at low priority for pipeline-validation signal but is no longer the throughput-headroom bet.

## Measurement Target: Warm-Cache Decode (Agent Workflow Reality)

**This pivot is calibrated to warm-cache decode performance, not cold-start prefill performance.** Agent workflows have a specific characteristic shape that drives this choice:

- **Multi-turn trajectories.** A typical agent run is N turns of (prompt → reasoning → tool call → observation → next reasoning). After turn 1, every subsequent turn enters with a populated KV cache containing all prior turns' tokens. The fraction of total wall-time spent on warm-cache decode dwarfs the fraction spent on cold-start prefill.
- **Long context, autoregressive generation.** Each decode token reads the full KV cache (warm), runs the full FFN/attention forward, and emits one new token. Decode is bandwidth-bound by weight-loading on GB10 (273 GB/s LPDDR5x) — the dominant cost is the FFN GEMM weight-load, not the attention path that dominates cold-start prefill.
- **Cold-start prefill is amortized.** First-turn prefill cost is paid once per agent run; subsequent turns benefit from KV cache. Over a 4-turn trajectory with 4096+512+512+4096 thinking tokens, the warm-cache decode portion is ~85-90% of total wall-time. Optimizing prefill saves first-token latency; optimizing warm-cache decode saves overall throughput.

**P3a measured both phases on the heavy workload and the data shows them as nearly-disjoint problems:**

| Phase | Top bottleneck | Why it's there |
|---|---|---|
| **Prefill (cold-start, empty cache, first response)** | `gatedattn_attention_with_kv_read` at 77.4% | Dominated by attention's compute over long-context QKV with no cached KVs to short-circuit |
| **Warm-cache decode (subsequent responses, populated cache)** | `ffn_linear` at 59.4% + `deltanet_projection_linear` at 20.6% | Dominated by FP8 GEMM weight-loading from LPDDR5x; attention's KV-read share collapses to 6% because only one new token's QK attends to cached KVs |

**Cold-start prefill optimizations (attention backend choice, FA4 vs FA3) target a different bottleneck than warm-cache decode optimizations (FFN GEMM backend choice).** This pivot targets the warm-cache decode bottleneck specifically because agent workflows live there. A separate future investment could target prefill if first-token latency matters more than overall agent throughput; that's not this round.

The parity fixtures, probe sets, and Phase A/B measurement methodology specified below are all calibrated to **warm-cache decode call patterns** — small-M FFN GEMM shapes (M=1 to M=16 corresponding to single-token-at-a-time decode in batches), with KV cache populated to thousands of tokens.

## HLD Stage Context

This work sits inside `docs/HLD-Serving-Backend-AutoResearch-v0_2-L0KernelPlan.md` and pivots two priority decisions:

- **§0.6 priority-order ranking #1** (FP8 Quantized Linear / GEMM): was "L0a backend selection only in v0.3.1; Triton-mutation deferred to v0.3.2 if shipping kernels are mutable." The P3a data now promotes this to the active L0 target.
- **§0.6 priority-order ranking #2** (DeltaNet Triton kernels): was the v0.3.3+ executable target. Demoted to parallel-low-investment track per the P3a result.

Concrete HLD stage mapping for this work:

- **Phase A** is a re-run of HLD §3 L0a (kernel selection), narrowed to `fp8_gemm_kernel` as the only varying knob, against the heavy decode workload that P3a profiled.
- **Phase B** is a conditional HLD §5 L0c (kernel mutation) round on the Phase A winner's Triton source, applicable only if the winner is Triton-mutable. The L0c evaluation ladder from `l0c-evaluation-ladder-and-memory-prior-art-20260430.md` carries forward unchanged.

The intended HLD transition:

```text
v0.3.4 current:
  P3a roofline → P7a L0c-DeltaNet (chunk_delta_h)

v0.3.6 pivot:
  P3a in-process timing (already supports pivot; full external roofline still gates AR.54) →
  Phase A: L0a-FFN-GEMM backend selection (auto-research) →
  Phase B (conditional): L0c-FFN-GEMM mutation if winner is Triton-mutable
```

## P3a Evidence Driving The Pivot

**Evidence type — read carefully:** the data below is from a **scaled in-process timing pass** on 2026-04-30 (`docs/reports/auto_research/l0c-evaluation-ladder-and-memory-prior-art-20260430.md` and the in-process P3a notes recorded against the heavy decode workload `responses-sdk-adapter-cutover-heavy`, 11760 cached-token follow-ups). It is **subsystem-level wall-time evidence**, not a fully-instrumented external roofline.

What this measurement IS:
- Wall-time decomposition of the forward pass into named subsystems (`gatedattn_attention_with_kv_read`, `ffn_linear`, `deltanet_projection_linear`, `deltanet_core`, etc.).
- Captured in-process during real serving against the heavy workload.
- Sufficient to establish relative cost ordering across subsystems — which IS what the pivot decision turns on.

What this measurement is NOT:
- **Not a fully-validated external roofline.** DRAM throughput percentages, SM occupancy percentages, tensor-core utilization, and per-kernel launch counts are not yet captured at the precision needed to activate AR.54's priority enforcement (HLD §0.6).
- **Not a substitute for the future `ncu`-based roofline pass** that v0.3.6 still needs to run before the bandwidth-bound thesis is treated as quantitatively validated.

This pivot is justified by the relative cost ordering the in-process timing makes clear (FFN GEMM dominates warm-cache decode by a wide margin), not by a quantitative roofline claim. The bandwidth-bound thesis (HLD §0.5.2) is **plausible and consistent with this evidence**, not validated. A separate NCU diagnostic profile pass (per §"NCU Diagnostic Profile" below, and HLD §10.8 open question) is required before AR.54 priority enforcement activates.

**Weights are assumed already loaded** — this is a steady-state serving measurement, not a cold-launch measurement.

### Cold-start prefill (first response, empty KV cache)

Wall time: **56.7 seconds**. Component self-time breakdown:

| Component | Self-time | Share |
|---|---|---|
| `gatedattn_attention_with_kv_read` | 43,221 ms | **77.4%** |
| `ffn_linear` | 7,047 ms | 12.6% |
| `deltanet_projection_linear` | 2,346 ms | 4.2% |
| `deltanet_core` | 1,667 ms | **3.0%** |

Cold-start prefill is **GatedAttn-KV-read dominated** — attention computes QKV over the full sequence with no cached KVs to short-circuit. This is the *first-token-latency* problem, not the *throughput-while-the-agent-is-running* problem.

### Warm-cache decode (follow-up responses 2-10, populated KV cache)

This is the **load-bearing measurement for agent workflow throughput**. Each follow-up:
- Operates against an **11,760-cached-token** KV cache (populated from prior turns).
- Runs in ~1.35-1.41 seconds wall-time per response.
- Aggregate over 9 follow-ups: ~10.7 seconds.

Component self-time breakdown (aggregate over follow-ups 2-10):

| Component | Self-time (aggregate) | Share | Why it matters for agents |
|---|---|---|---|
| **`ffn_linear`** | 6,381 ms | **59.4%** | FFN GEMM weight-load from LPDDR5x; bandwidth-bound on GB10's 273 GB/s memory |
| `deltanet_projection_linear` | 2,215 ms | **20.6%** | Same FP8 GEMM family as `ffn_linear`; backend choice generalizes |
| `gatedattn_attention_with_kv_read` | 648 ms | 6.0% | KV-read share collapses from 77% (cold) to 6% (warm) — only one new token's QK attends to cached KVs |
| `gatedattn_projection_linear` | 645 ms | 6.0% | Same FP8 GEMM family |
| `deltanet_core` | 564 ms | **5.3%** | The L0c-DeltaNet target — bounded above at 5.3% even with perfect optimization |

**Combined FP8 GEMM share in warm-cache decode: 59.4% + 20.6% + 6.0% = 86.0%**. This is the dominant cost in agent workflow steady-state.

### Implications for L0 target selection

- **`deltanet_core` (the v0.3.4 L0c-DeltaNet target) is 5.3% of warm-cache decode time.** Best-case ceiling is ~5% even with perfect optimization; FLA's `chunk_delta_h` is already tuned by domain experts so realistic best-case is well below the ceiling. **For agent workflows where warm-cache decode is the binding constraint, this target's headroom is too low to justify the budget.**
- **`ffn_linear` is 59.4% of warm-cache decode time.** Even a 5% improvement here is **3.0% e2e decode**, equal to a 60% perfect optimization of `deltanet_core`. A 10% improvement is 5.9% e2e — already past the entire ceiling of optimizing `deltanet_core`.
- **FP8 GEMM family (`ffn_linear` + `deltanet_projection_linear` + `gatedattn_projection_linear`) is 86% of warm-cache decode.** Backend selection on this family is the load-bearing choice.
- **The bandwidth-bound thesis (HLD §0.5.2) is plausible and consistent with this in-process timing evidence**, but NOT yet quantitatively validated. The dominant decode subsystem is `ffn_linear` (FP8 GEMM weight-loading from LPDDR5x), and that's the pattern bandwidth-bound on 273 GB/s would predict — but turning the in-process timing share into "this kernel is at X% of DRAM peak" requires the NCU diagnostic pass (§"NCU Diagnostic Profile" below). The thesis is **directionally supported**; full validation gates AR.54 activation per HLD §10.8.

### Why prefill data does NOT change the conclusion

A reader looking at the prefill numbers might propose targeting `gatedattn_attention_with_kv_read` (77% of prefill). For agent workflows, this is the wrong optimization for two reasons:

1. **Prefill cost is amortized over decode.** Over a 4-turn trajectory with 4096+512+512+4096 thinking tokens, prefill is the first ~10-15% of wall time; warm-cache decode is the remaining 85-90%. A 20% prefill speedup is ~2-3% e2e wall time; a 10% warm-cache decode speedup is 8-9% e2e wall time.
2. **The `gatedattn_attention_with_kv_read` path is vendor (FA3/FlashInfer) — not L0c-mutable.** Optimizing it requires backend selection (L0a-style) on the attention backend, not source mutation. That's a separate workstream and not the FFN GEMM pivot's scope.

For agent workflows specifically — where the user-perceived "throughput" is tokens-per-second across many turns, and first-token latency matters less than steady-state generation rate — **warm-cache decode is the binding constraint**. This pivot optimizes for that.

## Decision

**Pivot the primary L0 auto-research target from `chunk_delta_h` (DeltaNet) to `ffn_linear` (FP8 FFN GEMM).**

Concretely:

1. The currently-running L0c-DeltaNet round (`l0c_deltanet_long_20260430T183323Z`) finishes its budget. Don't kill it — the round's pipeline-validation signal is useful even if the throughput win is small. **Don't extend or relaunch L0c-DeltaNet** as a primary investment after it concludes.
2. Author Phase A (FFN GEMM backend selection, this document) and execute it as the next L0 round.
3. If Phase A's winner is Triton-mutable, author and execute Phase B (FFN GEMM kernel mutation).
4. If Phase A's winner is cuBLAS/CUTLASS C++ (vendor, not L0c-mutable), the round ships the Phase A bundle as the production artifact for FFN GEMM backend choice, no Phase B.
5. The L0c-DeltaNet pipeline (controller, parity fixtures, evaluator ladder) carries forward unchanged — Phase B reuses it, with the kernel target swapped from `chunk_delta_h.py` to whichever Triton FP8 GEMM kernel Phase A selects (if any).

This document is the design target for both phases. It is not a request to implement source code in this pass.

## Stack Under Auto-Research

The active surface for the pivoted work:

- model family: `qwen3.5-27b`,
- workload family: `responses-sdk-adapter-cutover-heavy` (decode-heavy thinking trajectory),
- hardware/runtime focus: GB10 / local vLLM serving stack,
- kernel target: FP8 FFN GEMM (the GEMM behind `ffn_linear` and `deltanet_projection_linear` and `gatedattn_projection_linear`),
- backend dispatch surface: vLLM `Fp8LinearMethod` and adjacent quantization paths (`vllm/model_executor/layers/quantization/fp8.py`, `csrc/quantization/cutlass_w8a8/`, etc.),
- correctness fixture family: new — `responses-sdk-adapter-cutover-fp8-gemm-v1` (specified in §"Parity Fixture For FFN GEMM" below),
- baseline comparison: contemporaneous paired measurements against the current Phase A baseline (initially: vllm-default routing, which on this stack resolves to cuBLAS per L0b-empirical-winner bundle 4866bc3f).

## Phase A — FFN GEMM Backend Selection (L0a-Style)

### Action Space

Phase A is a finite, enumerable, deterministic grid sweep over FP8 FFN GEMM backends. It uses HLD §3 L0a infrastructure (`tune-kernel-select` CLI, smoke phase, screen, rescreen, top-K winner selection).

| Knob | Candidates | Notes |
|---|---|---|
| `fp8_gemm_kernel` | `cublas` (current default), `cutlass_blackwell_scaled_mm`, `triton_fp8_scaled_mm`, possibly `marlin` / `machete` / `tensorrt_llm_fp8` if exposed in current vLLM | Phase A's only varying knob. **Prerequisite: enumerate which of these are actually exposed in the current vLLM build on the current weight format.** This is the first implementation task — `vllm/model_executor/layers/quantization/fp8.py` and the registered `LinearMethod` subclasses are authoritative. |
| All other knobs | Pinned to L0b-empirical-winner bundle 4866bc3f resolved values | Same pinning discipline as v0.3.3. `actually_resolved_kernel_selection` recorded at fixture-capture time. |

**Open question (Phase A prerequisite):** the action space cannot be finalized until someone audits which FP8 GEMM backends vLLM currently dispatches on this stack. This is a 30-minute audit, not a multi-day project, but it must happen before the Phase A round bootstraps.

### Search Strategy

Standard HLD §3.2 three-phase grid sweep:

1. **Smoke phase**: determinism probe + parity-vs-reference probe per backend. Eliminates non-deterministic backends and deterministic-but-wrong backends (the FlashInfer-style #35138 class).
2. **Screen phase**: n=5 baseline replays at Screen profile + n=2 measurements per surviving backend at Screen profile.
3. **Top-K rescreen**: n=4 measurements at Screen profile per top-K (K=2 — Phase A is small enough that a wide rescreen is unnecessary).

Each phase uses the existing HLD §3 measurement infrastructure, paired-A/B baseline, Welch-t with `confidence: defensible`.

### CLI Subcommand

Use existing `lumoserve auto-research tune-kernel-select` with action-space file. **Schema constraint (P1 fix):** the existing `tune-kernel-select` loader expects an `axes` mapping with all 5 axes specified — `attention_backend`, `deltanet_kernel`, `fp8_gemm_kernel`, `torch_compile_mode`, `cuda_graph_capture`. To narrow to "vary only `fp8_gemm_kernel`", pin the other four axes to single-value lists. **Valid `fp8_gemm_kernel` values today are `cublas` and `cutlass` only** — the runtime activation hook does not yet recognize names like `cutlass_blackwell_scaled_mm` or `triton_fp8_scaled_mm`. Adding new backend names is a CLI/runtime extension that gates Phase A's action space.

```yaml
# kernel_search/phase_a_action_space.yaml
# All 5 axes specified per existing tune-kernel-select loader contract.
# Phase A varies only fp8_gemm_kernel; the other 4 axes are pinned to single-value lists.
axes:
  attention_backend:    [vllm-default]                # pinned per actually_resolved_kernel_selection
  deltanet_kernel:      [triton-chunked-delta-v2]     # pinned
  fp8_gemm_kernel:      [cublas, cutlass]             # the varying axis (current valid values only)
  torch_compile_mode:   [default]                     # pinned
  cuda_graph_capture:   [off]                         # pinned
```

**If Phase A should consider backends beyond cuBLAS/CUTLASS** (e.g., a Triton FP8 scaled-MM if one exists in vLLM, or Marlin/Machete/TRT-LLM-FP8), the prerequisite audit (Open Q1) must produce both:
1. The list of backends to add to the runtime activation hook (in `vllm/model_executor/layers/quantization/fp8.py` adjacent code).
2. The actual code to recognize each new value as a `fp8_gemm_kernel` choice.

These are coding tasks, not just doc tasks. Phase A's executable scope in v0.3.6 is whichever subset of {cuBLAS, CUTLASS, ...post-audit additions} is wired in time.

Round invocation. **Flags marked NEW are CLI extensions required by v0.3.6**; flags unmarked are part of the existing `tune-kernel-select` parser:

```
lumoserve auto-research tune-kernel-select \
  --workload-file benchmark_blueprints/workloads/responses-sdk-adapter-cutover-heavy/workload.yaml \
  --action-space-file kernel_search/phase_a_action_space.yaml \
  --baselines 5 \
  --screen-measurements-per-combo 2 \
  --rescreen-top-k 2 \
  --rescreen-measurements-per-candidate 4 \
  --parallel-instances <empirical from P2> \
  --round-root output/auto_research \
  --harness real \
  # NEW v0.3.6 CLI extensions (must land before Phase A bootstraps):
  --base-stack-resolution vllm_default \                  # NEW: pin baseline stack source per HLD §"Sequencing inversion"
  --round-prefix qwen3.5-27b-fp8-gemm-phase-a \           # NEW: round-directory prefix to avoid colliding with prior L0c-DeltaNet rounds
  --phase-a-screen-method replay                           # NEW: replay screen → uniform full-vLLM rescreen, see §"Architecture R1"
```

**CLI extension prerequisites (must implement before Phase A round can run):**

1. `--base-stack-resolution {vllm_default, reference_baseline, bundle}` — the loader must accept this and use it to determine which baseline-stack to pin pinned-axes against. Existing `tune-kernel-select` does NOT have this flag today.
2. `--round-prefix <string>` — appended to the auto-generated round directory name. Existing CLI uses a fixed naming scheme; needs the prefix slot.
3. `--phase-a-screen-method {replay, full_vllm}` — selects whether the screen tier uses isolated kernel replay (Architecture R1) or full vLLM end-to-end (Architecture R2). NEW in v0.3.6 per the architecture choice.

These three CLI extensions are part of the v0.3.6 implementation prerequisites. The doc-level invocation example above is forward-looking; do NOT copy-paste it into a shell before the extensions land. The audit (Open Q1) plus these three CLI extensions plus the harness work (§"Does This Apply To Phase A Backend Selection?") are the gating items before Phase A can bootstrap.

### Phase A Exit Conditions

- **Winner is cuBLAS (current default).** Round emits `ROUND_NULL_RESULT` for backend-selection — current default is empirically optimal. Phase B does not run; instead, the operator should reconsider whether L0c-FFN-GEMM mutation makes sense on a vendor backend (it doesn't directly; would require pivoting the question to Triton fallback or sub-component fusion).
- **Winner is CUTLASS or another vendor backend.** Phase A bundle ships as the new FP8 GEMM baseline. **Phase B does NOT run** because vendor C++ is outside the L0c-mutable surface. The operator decides whether to invest in CUTLASS-side mutation work (out of v0.3.6 scope; deferred to v0.4+).
- **Winner is `triton_fp8_scaled_mm` or another Triton-source backend.** Phase A bundle ships. Phase B (L0c kernel mutation on the Triton source) is unblocked.
- **All backends are within noise of each other.** ROUND_NULL_RESULT. Operator reconsiders whether the parity fixture or workload is exercising the GEMM in throughput-relevant ways.

### Phase A Verification (extends HLD §9 AR list)

Reuses HLD §9 AR.38–40 (L0a determinism + parity-vs-reference + intermediate-bundle marking) without modification. The `round_type: l0a_select_only` semantics carry forward. New AR items:

- **AR.58 Phase A action space exhaustively enumerated.** The action-space file lists every FP8 GEMM backend that vLLM actually exposes on this stack, derived from a documented audit of `Fp8LinearMethod` and adjacent registered methods. No backend silently absent. Audit log committed at `output/phase_a_action_space_audit.md`.

- **AR.59 Phase A backend identity preserved across rounds.** Each backend's identity is pinned to a concrete vLLM symbol path (e.g., `cublas` → `Fp8LinearMethod.apply_cublas_path`) AND a content hash of the dispatch site. If vLLM is upgraded between fixture capture and round execution, the hash mismatch surfaces as `phase_a_backend_dispatch_drift` halt code rather than silent re-routing.

## Phase B — Conditional FFN GEMM Kernel Mutation (L0c-Style)

### Precondition

Phase B runs **iff** Phase A's winner is a Triton-source-mutable backend (`triton_fp8_scaled_mm` or equivalent). If the winner is cuBLAS or CUTLASS C++, Phase B is skipped and the round terminates with the Phase A bundle.

### Stage Mapping To v0.3.5 L0c Ladder

Phase B reuses the v0.3.5 L0c evaluation ladder from `l0c-evaluation-ladder-and-memory-prior-art-20260430.md` without modification. Only the kernel target swaps:

- **kernel target**: from `deltanet` (`chunk_delta_h.py`) to `fp8_gemm_triton` (whichever Triton source Phase A's winner resolves to — likely something like `vllm/model_executor/layers/quantization/utils/fp8_utils.py::triton_fp8_gemm` or adjacent; final path determined by Phase A).
- **parity fixture**: §"Parity Fixture For FFN GEMM" below replaces the DeltaNet fixture for this round.
- **Tier 3 isolated kernel replay**: same architecture, different entry point. Replay harness imports the patched FP8 GEMM module from the candidate kernel workdir, calls it on captured (M, N, K, A, B, scale_a, scale_b) tuples, compares output and timing against fixture-recorded baseline.
- **Top-K survivor selection**: same defaults (`confirmation_top_k=5`, `minimum_replay_speedup=1.01`, `max_allowed_case_slowdown=0.99`).

### Mutation Contract

Same shape as v0.3.5 L0c, kernel-source path varies:

- Allowed mutation target: the Triton FP8 GEMM kernel source identified by Phase A.
- Allowed mutation classes: tile-shape choices, swizzle patterns, `BLOCK_M/N/K`/`num_warps`/`num_stages` defaults (when not via `@triton.autotune`), per-channel vs per-tensor scale handling within the kernel, async-copy / TMA patterns, epilogue choice (none vs activation-fused vs quant-fused), masking/boundary-handling rewrites.
- Required candidate metadata: `mutation_hash`, `speed_thesis`, `expected_affected_path`, `prior_failure_relation` — same as v0.3.5.
- Forbidden mutation targets: parity fixture builders, parity-check implementation, controller code, measurement-recording code, tests, ledgers — same as v0.3.5.

### Cross-Round Memory

Inherits v0.3.5 L0c cross-round memory architecture without modification:

- `prior_mutations_rejected.tsv`: cross-round failure history (separate file from DeltaNet's, since the kernel surface is different).
- `mutations_rejected.tsv`: current-round failures.
- `winning_diffs.md`: positive memory of accepted mutations (separate file scoped to this kernel target).
- candidate-local artifacts: `mutation.patch`, `parity_check.json`, `static_preflight.json`, `small_replay_check.json`, `kernel_replay_check.json`, `BLOCKED.md`.

The DeltaNet-specific memory files DO NOT bleed into the FFN GEMM round and vice versa. Each kernel target maintains its own memory — if you re-run DeltaNet later, its prior history is intact; the FFN GEMM round starts clean.

## Parity Fixture For FFN GEMM

Phase A and Phase B both require a parity fixture for FFN GEMM correctness gating. The DeltaNet fixture (`responses-sdk-adapter-cutover-deltanet-v1`) does not cover GEMM correctness — it captures DeltaNet kernel inputs/outputs, not FP8 linear-projection outputs.

### What "Correctness" Means In This Spec

Correctness is **never byte-identical**. FP8 has ~3-bit mantissa (native rounding ~10⁻²); different MMA orderings, different `BLOCK_M/N/K` choices, different reduction orders all produce different output bytes for mathematically-equivalent code. Byte-identity is only achievable for no-op patches (comment-only, identifier rename) and is NOT the goal — we WANT tile-shape and reduction-order changes to pass, because that's the entire optimization surface.

Correctness is a **stack of progressively-tighter checks**. A mutation passes the gate iff it passes every check applicable at its tier:

| Check level | Metric | Tolerance | Where the check runs | Catches | Lets through (intentionally) |
|---|---|---|---|---|---|
| **Tight numerical (kernel boundary)** | element-wise rel-error on FP8 GEMM output | `rtol=2e-3, atol=2e-3` (FP8 has stronger native rounding than FP16) | **Tier 3 isolated kernel replay** (achievable from a backend-agnostic harness with just `(A, B, scale_a, scale_b)` inputs) | algorithm bugs, indexing errors, scale-handling mistakes, NaNs/infs | tile-shape / MMA-ordering / reduction-order rewrites that are mathematically equivalent |
| **Compounding-error guard (downstream)** | element-wise rel-error on **logits 32 layers downstream**, NOT just the GEMM output | `rtol=1e-3, atol=1e-3` (model logits are FP16/BF16 reductions; tighter) | **Tier 4 vLLM parity probe** (NOT Tier 3 — see §"Why downstream-logit moved to Tier 4" below) | mutations that pass at the kernel boundary but cause small drift that compounds across remaining layers (the "slow drift" problem) | drift small enough to survive 32 layers of attenuation |
| **Distributional / behavioral (optional)** | KL divergence on post-top-k-top-p sampling distribution; perplexity on held-out probe set | KL ≤ 1e-4 | **Tier 4 add-on**, when vLLM is already warm | subtle semantic drift that doesn't change individual sampled tokens but shifts probability mass | truly behavior-equivalent mutations |

**Rule for FFN GEMM mutations:** pass Tight numerical at Tier 3 (kernel boundary, isolated replay) → pass Compounding-error guard at Tier 4 (vLLM parity, downstream logits) → optionally pass Distributional check at Tier 4 add-on for high-stakes promotions = correctness.

#### Why downstream-logit moved to Tier 4 (P2 fix)

A v0.3.5-style isolated kernel replay harness can compute the GEMM output for a probe `(A, B, scale_a, scale_b)` — that's just a math operation on captured tensors. But computing **logits 32 layers downstream of that GEMM call** requires resuming the model from the GEMM call site through the remaining 32 layers, which needs:
- Layer index and call-site role (which of the 64 layers; gate-up-proj vs down-proj vs attn-output-proj).
- Residual stream state at the call site.
- Attention KV cache state at the call site.
- A replay wrapper that can run the rest of the model from this checkpoint.

That's not "isolated kernel replay" — it's **partial-model replay**, which is much closer to vLLM parity in structure. v0.3.6 keeps Tier 3 as truly isolated (just GEMM-output check) and routes the downstream-logit compounding-error guard into Tier 4 (vLLM parity), where the model state at any point in the forward pass is naturally accessible.

**Implication for the parity fixture (§"Parity Fixture For FFN GEMM"):** the fixture stores per-probe `(A, B, scale_a, scale_b, gemm_output)` for Tier 3 use, and SEPARATELY stores Tier 4 vLLM parity check inputs (the existing `responses-sdk-adapter-cutover-deltanet-v1`-style logit + state snapshots, adapted for FFN GEMM call sites). The two fixture artifacts are decoupled — Tier 3 doesn't need model context; Tier 4 does. A future v0.4+ could add a "model-resume" replay wrapper to restore the compounding-error guard to Tier 3 cost regime; out of v0.3.6 scope.

**Examples of what passes vs. fails this gate:**

- **Tile shape 128×64 → 64×128** (mathematically equivalent, different reduction order): bit-different output, passes Tight (Tier 3) and Compounding-error guard (Tier 4). ✅ accepted.
- **Per-channel → per-tensor scale layout** with correct recompute: bit-different, may pass Tight at Tier 3, **fails Compounding-error guard at Tier 4** because precision loss compounds across 64 layers. ❌ rejected (at Tier 4, after Tier 3 already passed — operator sees a Tier-3-pass-Tier-4-fail signal which is itself diagnostic).
- **Async-copy / TMA bulk pattern change** that produces the same numerical output: bit-different, passes both checks. ✅ accepted.
- **Indexing bug producing NaN at one tensor position:** fails Tight at Tier 3 immediately (NaN beats any tolerance). ❌ rejected at Tier 3 — never reaches Tier 4.
- **Mutation that produces small numerical drift on cold-cache probes:** **fails Tight at Tier 3** OR fails the per-probe non-regression-guard tolerance check. ❌ rejected at Tier 3 (this is a correctness failure, NOT a timing-instability flag — see §"What `kernel_replay_timing_unstable` actually means" below).

This is the same correctness model as v0.3.5's L0c ladder (§2.2.6 four-checkpoint compare for fused_epilogue, §2.2.4 token-id + KL divergence for sampling). The new contribution here is making it explicit so operators understand why bit-different ≠ wrong.

#### What `kernel_replay_timing_unstable` actually means (P3 fix)

`kernel_replay_timing_unstable` is **strictly a runtime-noise flag**, NOT a numerical-drift flag. Two distinct phenomena, two distinct outcomes:

| Symptom | Diagnosis | Outcome |
|---|---|---|
| Per-probe `duration_ms` shows high dispersion (e.g., p80/p20 > 1.5) across repeated invocations of the SAME mutation on the SAME input | Timing measurement is noisy — possibly thermal effects, autotune cache state, GPU concurrency, or driver-level scheduling jitter | `kernel_replay_timing_unstable` flag set; the mutation's speedup score is unreliable; operator reviews. **NOT a correctness failure.** |
| Mutated kernel produces numerically-different output on cold-cache vs warm-cache invocation of the SAME mutation | Numerical drift — the kernel's behavior depends on cache state in a way the reference doesn't | **Tight numerical check FAILS at Tier 3** (the cold-cache probes are part of the probe set; one of them shows tolerance overshoot). Mutation rejected as a correctness failure with reason `kernel_replay_output_diverged`. **NOT a timing flag.** |

The previous wording "passes Tight on the warm-cache probe set but flagged as `kernel_replay_timing_unstable`" mixed these — a mutation that drifts numerically on cold-cache probes should fail correctness on those probes, not be flagged for timing instability. v0.3.6 treats them as independent: timing instability is about runtime measurement noise; numerical drift is a correctness failure routed through the Tight numerical check on whichever probes catch it.

### Reference Baseline

Same §2.2.0 reference baseline pattern: capture against the externally-trusted reference stack (FA3 or vllm-default + cuBLAS FP8 + Triton DeltaNet defaults + cuda_graph off). The reference is what Phase A's smoke phase compares candidates against. Reproducibility-3-times verified before fixture is sealed.

### Fixture Schema (split: Tier-3 isolated replay + Tier-4 vLLM parity)

Per the P2 fix in §"What 'Correctness' Means", **Tier 3 isolated kernel replay** and **Tier 4 vLLM parity** consume different fixture artifacts. The single `fp8_gemm_v1` fixture file is structured as TWO decoupled artifact sets so the implementation does not accidentally treat downstream-logit data as a Tier-3 input.

Path: `benchmark_blueprints/families/responses-sdk-adapter-cutover/parity_fixture/fp8_gemm_v1.yaml` plus per-tier `.npz` companions.

#### Tier-3 artifacts (isolated GEMM replay — no model context required)

```yaml
fixture_id: responses-sdk-adapter-cutover-fp8-gemm-v1
kernel_target: fp8_gemm
generated_at: <iso8601>
generated_against:
  vllm_version: <version>
  weight_version_id: <sha>
  reference_baseline:
    fp8_gemm_kernel: cublas        # or whichever current default; record actually_resolved
    attention_backend: vllm-default
    deltanet_kernel: triton-chunked-delta-v2
    kv_cache_dtype: fp8_e5m2
    torch_compile_mode: default
    cuda_graph_capture: off
  reference_reproducibility_runs: 3

# ====== Tier 3 (isolated kernel replay) artifacts ======
# Per-probe inputs are sufficient to invoke the GEMM kernel from a backend-agnostic
# harness with NO model context. NO downstream logits here — those are Tier 4.
tier_3_probe_count: 512                 # default per §"The 512-probe default"
tier_3_smoke_probe_count: 64            # smaller subset used for the L0a-style smoke phase (determinism + parity-vs-reference) before the full screen
tier_3_probe_shapes:                    # representative (M, N, K) coverage; see §"Probe-Set Construction"
  - {M: 1,    N: 11008, K: 4096}
  - {M: 1,    N: 4096,  K: 11008}
  - {M: 16,   N: 11008, K: 4096}
  - {M: 256,  N: 11008, K: 4096}
  - {M: 4096, N: 11008, K: 4096}
  - {M: 1,    N: 4096,  K: 4096}
  # ... rest derived per §"Probe-Set Construction" trajectory-derived sampling
tier_3_probe_input_a_ref:           tier_3_inputs/gemm_input_a.npz
tier_3_probe_input_b_ref:           tier_3_inputs/gemm_input_b.npz
tier_3_probe_input_scale_a_ref:     tier_3_inputs/gemm_input_scale_a.npz
tier_3_probe_input_scale_b_ref:     tier_3_inputs/gemm_input_scale_b.npz
tier_3_reference_gemm_output_ref:   tier_3_inputs/gemm_reference_output.npz
tier_3_tolerances:
  rtol_gemm_output: 2.0e-3              # FP8 has stronger native rounding than FP16
  atol_gemm_output: 2.0e-3
tier_3_parity_check_method: gemm_output_compare_only

# ====== Tier 4 (vLLM parity) artifacts ======
# Tier 4 reuses the existing vLLM parity infrastructure (HLD §2.2.1/§2.2.2 style
# logit + state snapshots), captured at FFN GEMM call sites of the heavy workload.
# Tier 4 inputs are model-state checkpoints (residual stream, KV cache, layer
# context), NOT bare GEMM tensors — so they're stored separately.
tier_4_call_site_count: 16              # FFN GEMM call sites tagged for downstream-logit capture
tier_4_call_site_layer_indices: [0, 4, 8, 16, 24, 32, 40, 48, 56, 60, 62, 63]   # cover early/mid/late layers
tier_4_probe_input_state_ref:       tier_4_vllm_parity/probe_state_snapshots.npz   # residual + KV cache + layer context per call site
tier_4_reference_downstream_logits_ref: tier_4_vllm_parity/reference_downstream_logits.npz
tier_4_tolerances:
  rtol_downstream_logit: 1.0e-3         # FP16/BF16 reductions, tighter than Tier 3 GEMM-output tolerance
  atol_downstream_logit: 1.0e-3
tier_4_parity_check_method: vllm_parity_with_downstream_logit_compounding_guard

content_hash: <canonical manifest hash per HLD §6.6.6 over all tier_3_* + tier_4_* artifacts>
```

**Implementation note:** the Tier 3 and Tier 4 artifact sets share the same `fixture_id` and the same `content_hash` (so bundle identity binds to both), but consumers MUST select the correct tier's artifacts for their respective check. A Tier 3 isolated replay harness reads only `tier_3_*` fields. A Tier 4 vLLM parity probe reads only `tier_4_*` fields. **Do not implement a "downstream logits in Tier 3" path** — that's the partial-model-replay design the P2 review flagged as out-of-scope for v0.3.6.

### Probe-Set Construction (trajectory-derived, warm-only timing, cold-only cache-warm-up)

#### Terminology — what "probe" means in this doc

A **probe** is one captured kernel invocation: an `(M, N, K, A, B, scale_a, scale_b)` tuple representing a single specific GEMM call that occurred during trajectory replay. **NOT a token, NOT a forward pass.**

Concrete relationship:
- **Token:** one decoded output token. Triggers one forward pass through all 64 layers.
- **Forward pass per token:** invokes ~3 GEMM calls per layer × 64 layers ≈ **~192 GEMM invocations per decoded token**.
- **Probe:** one of those individual GEMM invocations.

During the 3 warm-cache turns of the heavy trajectory (turns 2-4 generating ~5,000 thinking tokens of decode), the model invokes roughly `5,000 × 192 = ~1 million` GEMM calls. The fixture stores a deterministically-subsampled subset of these (~256-1024 probes total) bucket-stratified by `(M, N, K)` shape — most of those million calls cluster in 50-200 distinct shape buckets, so subsampling preserves shape coverage at manageable fixture size.

When this doc says "512 probes per candidate", it means 512 individual GEMM-call inputs are replayed through the candidate kernel and timed. Not 512 tokens, not 512 forward passes.

#### Workload scope: ONE family, ONE variant, ONE trajectory

v0.3.6 commits to a single specific agent flow rather than a multi-trajectory composite:

- **Family:** `responses-sdk-adapter-cutover` (the heavy family; P3a in-process timing identifies it as decode-FFN-dominated).
- **Variant:** `v5` (per existing convention).
- **Trajectory:** `benchmark_blueprints/families/responses-sdk-adapter-cutover/seed_trace_v5.jsonl` (4 turns: 4096+512+512+4096 thinking tokens).

This narrowing is intentional: one specific agent flow is sufficient signal for kernel-level optimization on this hardware. Multi-family generalization is v0.4+ work; the per-family fixture-construction policy below applies unchanged when more families are added.

#### Turn role discipline

| Turn | Role | Captured? | Used in evaluation timing? |
|---|---|---|---|
| **Turn 1** (4096 thinking tokens, empty KV cache → first response) | **Cache warm-up only.** Populates the KV cache so turns 2-4 are realistic warm-cache decode. | Yes (kernel inputs captured for cold-start probes — non-regression guard) | **No** — turn 1's GEMM invocations are NOT the optimization target |
| **Turn 2** (512 thinking tokens, KV cache populated from turn 1) | Warm-cache decode | Yes | **Yes** |
| **Turn 3** (512 thinking tokens, KV cache populated from turns 1-2) | Warm-cache decode | Yes | **Yes** |
| **Turn 4** (4096 thinking tokens, KV cache populated from turns 1-3) | Warm-cache decode | Yes | **Yes** |

**Phase A/B accept-or-reject decisions are made on turns 2-4 timing aggregate ONLY.** Turn 1 is captured because some probes need to come from cold-start shapes (large-M prefill paths) for the non-regression guard, but turn 1's per-probe timing does NOT enter the speedup score. This is the practical realization of "warm-cache decode = agent workflow" from §"Measurement Target".

#### Construction policy

1. **Replay** all 4 turns of `seed_trace_v5.jsonl` against the §2.2.0 reference baseline (forced reference-stack; deterministic-3-times verified).
2. **Tag every kernel invocation by phase:** `prefill_cold_start` (turn 1) or `decode_warm_cache` (turns 2-4).
3. **Capture** every FP8 GEMM invocation's Tier-3 tuple: `(M, N, K, dtype_a, dtype_b, scale_layout, A, B, scale_a, scale_b, output)`. ~5,000 invocations per turn × 4 turns = ~20K total invocations to bucket. **Do NOT capture `downstream_logits_32_layers_later` here** — that's a Tier 4 fixture artifact captured separately at a smaller per-call-site set (16 call sites × 1 logit-snapshot each = 16 Tier-4 artifacts, not 20K).
4. **Bucket by `(M, N, K)` shape.** Distinct shapes typically number 50-200 for this family.
5. **Deterministic subsampling within each bucket** to bound fixture size:
   - For `decode_warm_cache` buckets: keep up to **K=4 invocations** per bucket, chosen by deterministic hash (sort by sha256 prefix of `A.tobytes()`, take first 4).
   - For `prefill_cold_start` buckets: keep up to **K=1 invocation** per bucket, same deterministic-hash selection.
6. **Tag retained probes by `evaluation_role`:**
   - `decode_warm_cache` probes → `evaluation_role: throughput_signal` (timing enters speedup score)
   - `prefill_cold_start` probes → `evaluation_role: non_regression_guard` (correctness must pass; timing must not exceed `max_allowed_case_slowdown=0.99`, but does NOT enter speedup score)
7. **Result:** ~256-1024 probes per family, roughly 75/25 warm/cold by construction (3 warm turns × 4 per bucket vs 1 cold turn × 1 per bucket).
8. **Tier-4 capture (separate sub-pass).** During the same trajectory replay, capture `(layer_index, residual_stream_state, kv_cache_state, downstream_logits_at_end_of_model)` snapshots at 16 chosen FFN GEMM call sites (covering early/mid/late layers per `tier_4_call_site_layer_indices`). These are state-snapshots for vLLM-parity-style use; NOT per-probe partial-model replay. Stored at `parity_fixture/fp8_gemm_v1/tier_4_vllm_parity/`.
9. **Persist** via `scripts/build_parity_fixture.py --fixture-type fp8_gemm --family responses-sdk-adapter-cutover --turns-captured 1,2,3,4 --turns-scored 2,3,4 --decode-per-bucket 4 --prefill-per-bucket 1 --tier-4-call-site-count 16`.

#### Eval cost — per-candidate (the unit that matters)

**Per-candidate replay cost** is the load-bearing number. Round-level cost = (per-candidate cost) × (candidates run); the candidate count is an operator decision, not a fixed v0.3.6 number.

**What "per-probe overhead" includes (and does NOT include):**

Per-probe replay captures a **single number** per probe: `duration_ms` (wall-time of one kernel invocation, measured via `triton.testing.do_bench`-style warmup + repetition + median). It's just timing.

| Per-probe replay captures | Per-probe replay does NOT capture |
|---|---|
| `duration_ms` (wall time, median + dispersion) | DRAM throughput |
| Output tensor (for parity check) | SM occupancy |
| Downstream-logits (for compounding-error guard) | Compute/tensor-core utilization |
| Pass/fail flag | Register spill count |
| | Per-counter Nsight Compute (NCU) data |

**NCU profiling is NOT per-probe.** Capturing rich kernel counters (DRAM throughput, occupancy, SM utilization, top-3 stalls) requires Nsight Compute, which adds 10-100× overhead per kernel call. Capturing NCU on 1 million GEMM calls would take hours-to-days. Capturing NCU on ~8-16 representative shape-bucket exemplars takes a few minutes. NCU is a one-shot diagnostic path, not a routine per-probe path. See §"NCU Diagnostic Profile" below.

Per-candidate cost ≈ `probe_count × per_probe_overhead` where per-probe overhead is just the timing+parity overhead:

| Probe count | Per-probe overhead | **Per-candidate replay cost** | Notes |
|---|---|---|---|
| 256 | ~100ms | **~25 sec** | Minimum viable; OK if shape diversity is low (≤50 distinct shapes) |
| 512 | ~100ms | **~50 sec** | Recommended default for v0.3.6 |
| 1024 | ~100ms | **~2 min** | Use only if fixture coverage requires it (rare for FFN GEMM) |
| 4096 | ~100ms | ~7 min | Approaches the cost regime that motivated the pivot — defeats the purpose |

**The per-candidate cost is what you compare to v0.3.4's ~25 min/candidate.** At 512 probes, evaluating one candidate (correctness + warm-cache decode timing) takes ~50 seconds. That's ~30× faster than v0.3.4 per candidate.

#### NCU Diagnostic Profile (one-shot, NOT per-probe)

NCU's rich counters (DRAM throughput, SM occupancy, compute utilization, register pressure, top-3 stalls) are valuable as **input to the proposer's reasoning** — they let the agent form a grounded `speed_thesis` ("this kernel is bandwidth-bound on M=1 small-N decode shape; the leading optimization is to reduce DRAM read traffic in the gate-up-proj weight load"). But NCU is heavy: 10-100× overhead per kernel call. We use it surgically.

**NCU profiling cadence:**

| When | Scope | Why |
|---|---|---|
| Once per fixture build | ~8-16 representative shapes (one per shape-bucket cluster, sampled to span small-M decode + large-M prefill) | Establishes baseline counters for the unmutated kernel. Cost: ~5-10 min. |
| Once per major weight rotation | Same ~8-16 shapes | NCU counters change when weights change; refresh keeps proposer's profile-context current |
| Once per accepted mutation (optional) | Same ~8-16 shapes, run on the winning kernel | Captures which counter changed (DRAM↓ vs occupancy↑ vs compute↑) so `winning_diffs.md` can record a profile-grounded rationale |

**What gets captured per NCU run:**

```yaml
# parity_fixture/ncu_diagnostic_profile_v1.yaml
fixture_id: responses-sdk-adapter-cutover-fp8-gemm-ncu-v1
generated_at: <iso8601>
generated_against:
  reference_baseline: <§2.2.0>
ncu_probe_count: 12   # 8 small-M decode + 4 large-M prefill, shape-bucket-stratified
profiles:
  - shape: {M: 1, N: 11008, K: 4096}
    role: decode_warm_cache
    counters:
      dram_throughput_pct: 87.3
      sm_throughput_pct: 22.1
      occupancy_pct: 24.0
      register_spill_count: 0
      top_stalls: [global_memory_load_wait, shared_memory_bank_conflict, tensor_core_issue_pipe]
    duration_median_ms: 0.42
  - shape: {M: 16, N: 11008, K: 4096}
    role: decode_warm_cache
    counters: ...
  # ... 10 more
```

**What the proposer sees (distilled, NOT raw NCU dumps):**

The `iteration_brief.md` template gains a new section:

```markdown
# Profile-guided hints (from NCU on baseline, captured <date>)
The unmutated kernel's bottleneck profile (top 3 shapes by frequency in workload):

  Shape M=1 N=11008 K=4096 (small-M decode, most-frequent shape):
    Memory-bound (DRAM 87%, SM 22%, occupancy 24%).
    Top stall: global-memory-load wait.
    Operational interpretation: weight-loading from LPDDR5x is the bottleneck.
    High-EV mutation classes: async-copy / TMA bulk patterns; reduce gate-up-proj
    weight reloads via tile reshape; better weight-prefetch scheduling.

  Shape M=16 N=11008 K=4096 (small-batch decode):
    Still memory-bound but compute share rises (DRAM 78%, SM 41%).
    Top stall: shared-memory-bank-conflict.
    Operational interpretation: the batch-dim is starting to amortize; SMEM access
    patterns matter more.
    High-EV mutation classes: SMEM bank-conflict avoidance; swizzle pattern changes.

  Shape M=4096 N=11008 K=4096 (long-prefill):
    Compute-bound at this scale (DRAM 35%, SM 81%, occupancy 65%).
    Operational interpretation: long-prefill regime is outside the warm-cache-decode
    optimization target. Don't optimize for this shape; just don't regress on it.
```

This is **distilled** ranked operational interpretation, not raw NCU dumps. LLMs misread raw profiler output (per the prior audit's finding); this is the wrapper that makes it useful as proposer input.

**NCU on winners (post-hoc):**

When a Phase B candidate is accepted (passes parity + beats baseline by ≥ minimum_replay_speedup), the controller MAY (operator opt-in) run NCU on the winner against the same ~8-16 shapes. The output is appended to `winning_diffs.md`:

```markdown
## Winner: candidate <NNN>, accepted with +4.2% replay speedup, parity ✅

```diff
<the patch>
```

NCU delta vs baseline (top-3 shapes):
  Shape M=1 N=11008 K=4096:
    DRAM throughput: 87.3% → 79.1% (-8.2pp; less memory pressure)
    SM throughput:   22.1% → 28.4% (+6.3pp)
    Top stall:       global_memory_load_wait → tensor_core_issue_pipe
    Operational interpretation: the patch successfully reduced DRAM traffic;
    the bottleneck shifted from memory-bound to slightly more compute-bound.

  Shape M=16 N=11008 K=4096:
    DRAM throughput: 78.4% → 76.1% (-2.3pp; small change)
    ...
```

This profile-grounded rationale is what makes positive memory (winning_diffs.md) work as proposer guidance. The next round's proposer sees not just "this patch worked" but "this patch worked because it reduced DRAM traffic in the small-M decode shape" — concrete causal mechanism.

**Total NCU budget per round:**
- One-shot baseline profile (already done at fixture build): no per-round cost.
- Optional NCU-on-each-winner: ~5-10 min × number of accepted candidates. For a 5-winner round, ~30-50 min total. Operator opt-in.

**On DGX Spark specifically:** NCU is supported (Nsight Compute runs on Blackwell sm_100). The 10-100× overhead is the hardware-independent cost of profile collection — not a DGX-Spark-specific limitation. The 8-16-probe scope keeps the absolute time bounded.

#### Round-level wall-clock — operator's choice on candidate count

v0.3.4's `accepted_iteration_cap=24` was a search-budget number set when per-candidate cost was ~25 minutes. **With per-candidate cost dropping to ~50 sec, the optimal candidate count is a different decision.** Some examples:

| Candidate count | Phase A or B? | At 512 probes, total replay cost | Plus top-K vLLM confirm (~25 min × K) | Notes |
|---|---|---|---|---|
| **1** | "is this single mutation worth keeping?" check | ~50 sec | ~25 min if it passes parity | Smallest possible; defensible if you want to gate one specific patch |
| **3-5** | Phase A backend selection | ~2.5-4 min | ~25 min for top-1 | Phase A's natural shape — small action space |
| **10-12** | Phase B narrow search | ~10 min | ~2 hours if top-K=5 | Modest Phase B round |
| **24** | v0.3.4-shape Phase B (legacy) | ~20 min | ~2 hours if top-K=5 | Inherited cap; not load-bearing under fast eval |
| **100+** | Phase B aggressive search | ~85 min | ~2-4 hours | Becomes feasible only because per-candidate is cheap |

**The point:** v0.3.6 doesn't commit to a candidate count. The operator chooses based on (proposer cost, search-quality goals, available wallclock). For Phase A this is small by construction (3-5 backends). For Phase B it's a deliberate operator decision — could be 1 if you're verifying a specific mutation, could be 100 if you're doing aggressive search. **Don't carry the v0.3.4 24-cap forward as a default; make it explicit on each round.**

#### The 512-probe default

512 is the recommended default: enough to cover the ~50-200 distinct shapes the workload exercises with multiple per-bucket samples (4 per warm-cache bucket, 1 per cold-start bucket), while keeping per-candidate cost ≤ 1 min. If implementation discovers higher shape diversity, bump to 1024. **Do not exceed 1024 without explicit operator decision** — beyond that, per-candidate cost approaches the regime that motivated the pivot in the first place.

#### Cold-start probes act as non-regression guard, NOT optimization target

The ~25% cold-start probes are NOT the optimization target — Phase A/B's `max_allowed_case_slowdown=0.99` applies per-probe, so a candidate that's 5% faster on small-M decode probes but 50% slower on M=4096 prefill probes is rejected by the slowdown guard. **Warm-cache decode wins are the goal; prefill non-regression is the correctness side-constraint.** This matters because some FP8 GEMM optimizations (e.g., aggressive scale-recompute strategies, narrow-tile-only kernels) win on small-M but tank large-M; the non-regression guard catches that asymmetry without the cold-start probes contaminating the speedup score.

#### Why turns 2-4 specifically (not turns 2-3 or 1-3)

- Turns 2-4 cover **all three** warm-cache decode segments (short-thinking, short-thinking, long-thinking). Skipping turn 4 (the long-thinking response) would miss long-decode shape diversity.
- Turns 2-3 alone would under-sample the long-decode path that's representative of agent workflows.
- Turns 1-3 would include turn 1's cold-start shapes in the timing measurement, contaminating the warm-cache signal — exactly what we're trying to avoid.

#### Generalization note (deferred to v0.4+)

This trajectory-derived probe-capture policy applies unchanged when more families are added in v0.4+. The same `--family` flag plus same `--turns-captured 1,..N --turns-scored 2,..N` discipline produces a per-family fixture. **No fixture-construction work is wasted by the v0.3.6 single-family scope.** Per-family fixtures stand on their own AND aggregate cleanly into a future composite when the multi-family direction is reactivated.

### Does This Apply To Phase A Backend Selection?

**Short answer: yes for correctness, probably for timing, with one prerequisite.**

The trajectory-derived probe set serves both Phase A (backend selection) and Phase B (kernel mutation) — but only if a backend-agnostic isolated replay harness can be authored.

#### Correctness — yes, trivially

For correctness checking, swapping backend implementations is the same operation as swapping mutated kernels. The harness loads the candidate backend, calls it on each probe's `(A, B, scale_a, scale_b)` inputs, compares output to fixture-recorded reference, compares downstream-logits-32-layers-later to fixture-recorded reference. cuBLAS produces output X, CUTLASS produces output Y, Triton-FP8 produces output Z; all three compared to the same reference. No Phase-A-specific machinery needed for correctness.

#### Timing — yes if backend-agnostic harness is achievable

For timing measurement, the harness needs to:

1. Take a backend identifier (e.g., `cublas`, `cutlass_blackwell_scaled_mm`, `triton_fp8_scaled_mm`).
2. Construct the appropriate `LinearMethod` instance with realistic FP8 weight + scale tensors (recreated from probe-recorded layouts).
3. Call its `apply()` (or equivalent dispatch) with captured `(A, scale_a, weight_b, scale_b)`.
4. Time the call with `triton.testing.do_bench`-style warmup + repetition.

This works **if backends can be invoked outside vLLM**. The risk is init-state entanglement: cuBLAS might require vLLM-init-time setup (workspace allocation, handle management) that's hard to replicate in a standalone harness. CUTLASS scaled-MM might require a specific build configuration. Triton-FP8-fallback should be cleanest because Triton kernels are typically Python-importable.

**Prerequisite (added to Open Q1's audit scope):** for each FP8 GEMM backend exposed in vLLM, verify it can be invoked from a standalone Python harness with a synthetic-tensor input. The audit's output `output/phase_a_action_space_audit.md` should include a column "isolated_invocable: yes/no/partial-with-wrapper" per backend.

#### Phase A round shape if isolated-replay-harness works for ALL backends in the action space

Per-candidate cost drops from ~25 min (full vLLM restart) to ~50 sec (replay 512 probes through the candidate backend). Phase A's smoke + screen + rescreen completes in well under an hour for a 5-backend action space, instead of 5+ hours.

#### Phase A round shape if isolated-replay-harness does NOT work for some backend

**Mixing measurement methods within one ranking is invalid.** A backend timed via fast isolated replay and a backend timed via full vLLM end-to-end measure different things — different overheads, different state, different concurrency. Comparing their numbers in the same Welch-t to pick a winner would produce a meaningless ranking. The fix is one of two architectures, NOT mixing per backend:

**Architecture R1 (replay-as-screen-and-ranker, full-vLLM-as-uniform-confirmation-on-top-K):**
- ALL backends in the action space run replay (correctness + timing) as the screen tier.
- Replay produces both a correctness verdict (pass/fail) AND a timing-based ranking score.
- The replay-tier ranking selects **Top-K (default K=2)** highest-throughput-score candidates from among the replay-correct backends.
- Top-K survivors then ALL receive identical full-vLLM end-to-end measurement (uniform methodology, no mixing).
- Phase A winner is selected by the **full-vLLM Welch-t comparison among the Top-K only**, NOT by the replay ranking.
- A backend that fails replay correctness is excluded from Top-K (it can't progress).
- A backend that returns `replay_unavailable` (cannot be isolated-replayed at all) is excluded from the replay ranking but **routed directly to the full-vLLM rescreen pool alongside Top-K survivors**, where uniform-methodology measurement determines its rank vs. them. (Practical effect: an unreplayable backend always reaches full-vLLM; it just bypasses the replay-based Top-K filter.)

**Architecture R2 (uniform-full-vLLM throughout):**
- If even one backend cannot be isolated-replayed AND the operator wants strict measurement uniformity even at the screen tier, ALL backends use full vLLM restart for both screen and rescreen.
- Per-candidate cost reverts to ~25 min for the entire round.
- This is the v0.2.x L0a fallback path — slower, but produces a fully-uniform-methodology ranking from screen onward.

**v0.3.6 commits to Architecture R1.** Replay is both a correctness screen AND a Top-K ranker; full-vLLM is the authoritative winner-selection measurement on the Top-K (plus any replay-unavailable backends that bypassed the rank filter). All candidates reaching the full-vLLM tier are measured with identical methodology — no per-backend cost-model mixing. The CLI knob is `--phase-a-screen-method replay|full_vllm` choosing one method for the entire screen tier; the rescreen tier is always full-vLLM regardless.

**Resolving the prior "all candidates" vs "Top-K only" inconsistency** (P2 fix from external review): the rule is "Top-K only, BUT replay-unavailable backends bypass the replay rank-filter and proceed to full-vLLM directly." This is NOT "all candidates that pass replay" (that would be a Top-N=action-space-size routing, which is too expensive). It IS "Top-K replay-correct survivors ∪ all replay-unavailable backends" — bounded by `K + count(replay_unavailable)`. For typical Phase A action-spaces of 3-5 backends, K=2 means 2-5 candidates reach full-vLLM, not all of them by default.

**Phase A round wall-clock under Architecture R1 (typical):**
- Screen tier (all 3-5 backends): ~50 sec each = ~3-5 min total via replay; OR ~75-125 min via full-vLLM if any backend forces R2 fallback.
- Rescreen tier (top-K=2 backends): 2 × full-vLLM measurement at n=4 each ≈ 1.5-2 hours.
- Total: ~2-3 hours under R1 (replay screen + full-vLLM rescreen), or ~4-5 hours under R2.

This is much closer to v0.2.x's L0a wall-clock than the doc previously claimed. The "well under an hour" framing was optimistic; correcting to ~2-3 hours under Architecture R1.

#### Architectural symmetry: same harness, two phases

Phase A and Phase B both consume the same backend-agnostic isolated replay harness. The only difference is what's varying:

- **Phase A:** backend identifier varies; kernel source is whatever the backend ships.
- **Phase B:** kernel source varies (mutated by L0c proposer); backend identifier is the Phase A winner.

This symmetry simplifies the implementation: one harness, one fixture format, one parity-check function. Phase A is L0a-style auto-research over backend identifiers; Phase B is L0c-style auto-research over patches to a single chosen backend's kernel source. Both reduce to "swap kernel impl, replay probes, compare + time".

### Parity-Check Semantics (split per-tier)

#### Tier-3 (isolated kernel replay) parity check — GEMM output only

```python
def fp8_gemm_tier_3_parity(
    reference_gemm_output: np.ndarray,
    mutated_gemm_output: np.ndarray,
    rtol_gemm: float, atol_gemm: float,
) -> dict:
    # Single checkpoint: GEMM output element-wise tolerance.
    # No downstream logits here — that's Tier 4.
    result = logit_parity(reference_gemm_output, mutated_gemm_output,
                          rtol=rtol_gemm, atol=atol_gemm)
    if not result["pass"]:
        return {"pass": False, "reason": "kernel_replay_output_diverged", **result}
    return {"pass": True, "checkpoint_passed": "gemm_output"}
```

Achievable from a backend-agnostic isolated harness given just `(A, B, scale_a, scale_b, reference_gemm_output)` per probe. No model context required.

#### Tier-4 (vLLM parity) compounding-error guard — downstream logits

```python
def fp8_gemm_tier_4_parity(
    reference_downstream_logits: np.ndarray,
    mutated_downstream_logits: np.ndarray,
    rtol_logit: float, atol_logit: float,
) -> dict:
    # Single checkpoint: downstream logits (32 layers from the GEMM call site,
    # measured under full vLLM forward pass).
    result = logit_parity(reference_downstream_logits, mutated_downstream_logits,
                          rtol=rtol_logit, atol=atol_logit)
    if not result["pass"]:
        return {"pass": False, "reason": "downstream_logit_diverged", **result}
    return {"pass": True, "checkpoint_passed": "downstream_logits"}
```

Requires full-model state at the GEMM call site — runs only inside vLLM where the residual stream, KV cache, and layer context are naturally available.

**A mutated FP8 GEMM kernel must pass BOTH** to be ranked into the Phase A/B winner pool: Tier 3 catches kernel-boundary correctness failures cheaply (~50 sec per candidate), Tier 4 catches downstream compounding-error failures expensively (~25 min per candidate, run only on top-K survivors). The two-tier split is the cost-amortization architecture.

## Workload

Phase A and Phase B both run against `responses-sdk-adapter-cutover-heavy` — the same 4-turn thinking-heavy heavy-family workload from v0.3.3 — **measured in warm-cache decode mode** matching agent workflow reality.

Justification:

- It is the workload P3a profiled (warm-cache decode dominated), so the FFN GEMM bottleneck signal is calibrated to it.
- Its multi-turn decode trajectory (turn 1 = 4096 thinking tokens, turns 2-4 = 512+512+4096 against accumulated KV cache) exercises the warm-cache decode FFN GEMM path heavily — exactly the path agent workflows live in.
- The L0b-empirical-winner bundle 4866bc3f was tuned against this workload, so the paired baseline is contemporaneous.

**Measurement mode discipline (per §"Measurement Target"):**

- Baseline measurement = warm-cache decode aggregated over follow-up responses 2-N (skip turn 1 cold-start prefill from the throughput signal). **Turn 1 is captured for completeness but does NOT enter the Welch-t comparison** for kernel-mutation acceptance/rejection. Acceptance is judged on warm-cache decode speedup specifically, not on total wall-time including cold-start prefill.
- This means the round's `Measurement-Role` schema gains a sub-role: rows are tagged `decode_warm_cache` or `prefill_cold_start`. Only `decode_warm_cache` rows feed acceptance.
- The probe set for the new FP8 GEMM parity fixture (§"Parity Fixture For FFN GEMM") oversamples small-M shapes (M=1 to M=16, single-token-at-a-time decode) over large-M shapes (M=256+, prefill bulk-token compute). This matches the M-distribution that warm-cache decode actually exercises.

The composite multi-family workload from v0.3.2's architectural intent is **not** a Phase A/B prerequisite — single-family is sufficient for FFN GEMM since the GEMM kernels generalize across families more than DeltaNet recurrent-state behavior does, AND the warm-cache decode M-distribution is dominated by single-token-at-a-time shapes (M=1 in batch=1 decode) which generalize trivially across families. Multi-family generalization remains a v0.4+ question; Phase A/B does not need to wait for it.

## Cross-Round Memory Across Targets

Each L0 round maintains its own memory namespace. The DeltaNet round's `prior_mutations_rejected.tsv` and `winning_diffs.md` do not influence the FFN GEMM round and vice versa. Reasoning:

- The kernels are different surfaces with different mutation classes.
- A "forbidden pattern" learned on DeltaNet doesn't transfer to FP8 GEMM.
- A "winning diff" on FP8 GEMM doesn't help a future DeltaNet round.

The strategy_brief.md template includes both target's history at the SECTION level when relevant (e.g., "previous L0 round on DeltaNet found X" as background context), but does NOT treat the cross-target memory as constraint.

## Open Questions And Risks

### Open Q1: Which FP8 GEMM backends does vLLM actually expose on this stack?

The Phase A action space cannot be finalized without auditing `vllm/model_executor/layers/quantization/fp8.py` and the registered `LinearMethod` subclasses. Possibilities:

- cuBLAS via `Fp8LinearMethod.apply` standard path (definitely present).
- CUTLASS scaled-MM via `csrc/quantization/cutlass_w8a8/` (likely present on Blackwell).
- Triton FP8 fallback via `vllm/model_executor/layers/quantization/utils/fp8_utils.py` or adjacent (probably present but as fallback, not default).
- Marlin / Machete / other quant-specific kernels (FP8 may or may not have them; INT4 does).
- TensorRT-LLM FP8 via vLLM's TRT-LLM integration (depends on build).

The 30-minute audit must produce `output/phase_a_action_space_audit.md` listing each registered backend, its dispatch path, its known correctness/hardware constraints. **This is the first concrete implementation task.**

### Open Q2: What if cuBLAS wins decisively?

Phase A may produce `ROUND_NULL_RESULT` if cuBLAS-default is the empirical optimum (which is plausible — cuBLAS is heavily optimized for general matrix shapes). In that case:

- Phase B does not run.
- The throughput-headroom thesis on FFN GEMM is invalidated for L0c-style mutation: there's no Triton-mutable surface to mutate.
- The remaining options are CUTLASS-side mutation (out of v0.3.6 scope), or accepting cuBLAS as the production choice and pivoting to a different bottleneck (e.g., attention-output projection, RoPE, or other Triton-mutable paths in the FFN forward).

This is acceptable as a research outcome — Phase A's job is to find the answer, not to assume one.

### Open Q3: What if the workload's M-distribution is too narrow?

If the heavy workload's FFN GEMM calls are nearly-all (M=1, N=11008, K=4096) decode-shape, then a backend optimized for that single shape may win Phase A but generalize poorly to longer sequences. The probe set must oversample the M-distribution that the workload actually exercises, NOT a uniform sample over (M, N, K). The probe-set-construction step (§"Probe-Set Construction") addresses this — verify in P3a-instrumented run that the captured shapes match the workload's actual call distribution.

### Open Q4: Triton FP8 GEMM may not exist in vLLM's shipping Blackwell path.

The HLD §0.6 deferred FP8 GEMM Triton mutation as "unreachable" because Triton FP8 GEMM is rarely the default in vLLM's Blackwell path. If the audit confirms this, **Phase A's action space may be all-vendor**, in which case Phase B cannot run regardless of Phase A's winner. Acceptable outcome — Phase A still produces useful empirical data on backend choice.

### Open Q5: Parity fixture re-capture cost.

The new `responses-sdk-adapter-cutover-fp8-gemm-v1` fixture has TWO probe tiers, NOT one (P3 fix from external review — disambiguated to avoid the prior "is it 64 or 512?" contradiction):

- **`tier_3_smoke_probe_count: 64`** — used for the L0a-style smoke phase (determinism + parity-vs-reference culls before full screen). 64 deterministically-selected probes from the broader replay set.
- **`tier_3_probe_count: 512`** (default) — used for the full screen and rescreen replay timing. 512 probes per the §"Probe-Set Construction" trajectory-derived sampling.
- **`tier_4_call_site_count: 16`** — Tier-4 vLLM-parity state snapshots at 16 chosen FFN GEMM call sites. NOT comparable to the per-probe Tier-3 counts; these are state-checkpoints, not GEMM tuples.

Capture cost (one-time per weight rotation):
- Tier-3 (512 probes + their 64-probe smoke subset): one trajectory replay of all 4 turns with debug-export at FFN GEMM sites, ~10-15 min of GPU time, plus serialization of ~512 × `(A, B, scale_a, scale_b, output)` tensors.
- Tier-4 (16 call-site state snapshots): one additional trajectory replay sub-pass with debug-export of full-model state at the 16 chosen call sites, ~5-10 min additional.
- Total: ~15-25 min one-shot per weight rotation. Acceptable cost.

### Open Q6: Does the L0c-DeltaNet running round's data tell us anything before we pivot?

The Apr 30 status report shows candidates passing parity but not improving objective. Even one or two of those that "passed parity" are suspect — could the proposer have found something? The pivot doesn't depend on this answer, but if a chunk_delta_h mutation does measurably win, that's signal worth preserving in `winning_diffs.md` even though it's not the primary investment. **Action: when the round terminates, archive its results regardless of whether it "won".**

## Design Actions

| Action | Status | Notes |
|---|---|---|
| Audit vLLM FP8 GEMM backends on current stack | Prerequisite | Output: `output/phase_a_action_space_audit.md`. Blocks Phase A. |
| Build `responses-sdk-adapter-cutover-fp8-gemm-v1` parity fixture | Designed here | Probe set sampled from actual workload's FFN GEMM call distribution. |
| Phase A action-space file (`phase_a_action_space.yaml`) | Designed here | Authored after backend audit completes. |
| Phase A round (L0a-style backend selection) | Designed here | Reuses existing `tune-kernel-select` CLI; minor adapter to handle FP8 GEMM-specific knob. |
| Phase B round (L0c-style mutation, conditional) | Designed here | Reuses v0.3.5 L0c ladder; kernel target swap from `chunk_delta_h.py` to Phase A winner's Triton source. |
| Cross-round memory namespacing | Designed here | DeltaNet and FFN GEMM rounds maintain separate memory artifacts. |
| Parity-check semantics for FP8 GEMM | Designed here | Two-checkpoint compare: GEMM output (looser tol) + downstream logits (tighter tol). |
| Continue running L0c-DeltaNet round to conclusion | Ready | Pipeline-validation signal; archive results regardless of outcome. |
| Stop relaunching L0c-DeltaNet as primary investment | Ready | Operator decision based on P3a evidence. |
| Update HLD §0.6 priority order | Designed here | Promote FP8 GEMM (#1) to active target; demote DeltaNet (#2) to parallel-low-investment. |

## Prior-Art Support

This pivot inherits all relevant prior-art from `l0c-evaluation-ladder-and-memory-prior-art-20260430.md` (KernelBench, Sakana AI CUDA Engineer, Flash Linear Attention, Triton autotune/`do_bench`). One additional reference applies specifically to FP8 GEMM mutation:

- **CUTLASS Blackwell scaled-MM examples** (NVIDIA, 2024-2025): the public CUTLASS 3.6+ examples 62-67 series demonstrate sm_100 FP8 scaled-MM kernels with `tcgen05` MMA instructions. These are the reference designs CUTLASS C++ kernels are derived from. **Relevance:** confirms the structure of the Triton FP8 GEMM problem — block size, swizzle pattern, scale-load epilogue — is well-understood, so a Triton mutation has a clear search surface even if the win is bounded.

- **vLLM `Fp8LinearMethod` source** (https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/fp8.py): authoritative reference for which FP8 GEMM backends are registered.

## Answer: Are We Doing The Right Thing?

The 2026-04-30 P3a audit by an independent online researcher concluded: "single-kernel × single-workload × 25-min cycle time is 50-100x slower than published systems and 1-2 orders of magnitude narrower in scope. The path forward is inverting Tier 3 / Tier 5 usage so most candidates get evaluated at 30-second cycle time." The l0c-evaluation-ladder doc landed that inversion. This pivot landing the **target retargeting** is the second half.

After this pivot, the L0 auto-research direction is:

1. P3a-in-process-timing-supported target (`ffn_linear` is 59.4% of warm-cache decode in the in-process timing pass; full external roofline validation gating AR.54 is still pending).
2. Backend selection first (L0a), kernel mutation conditional on Triton-mutable winner (L0c).
3. Tier 3-as-primary evaluator from the v0.3.5 ladder (~30-sec cycles).
4. v0.3.5 cross-round memory + proposer-quality mechanisms inherited.
5. Existing infrastructure reused; only kernel target, parity fixture, and action-space file are new.

This is the target the bandwidth-bound thesis (§0.5.2) was actually pointing at. The earlier DeltaNet target was a misaligned interpretation of the thesis. The thesis is right; the target is now corrected.
