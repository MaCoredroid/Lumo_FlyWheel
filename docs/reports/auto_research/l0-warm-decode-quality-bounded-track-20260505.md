# L0 Warm-Decode Quality-Bounded Mutation Track (Track B)

Generated: 2026-05-05

Companion to: `docs/reports/auto_research/l0-ffn-gemm-pivot-20260502.md` (Track A — numerical-bounded mutation, currently exhausted at the 1.10× e2e ceiling per the May 4–5 CUTLASS rounds).

## Why this spec exists

The L0c numerical-bounded mutation loop has been measured-exhausted on this hardware × this model. After 70+ candidate attempts across two rounds, no candidate has cleared the 3% e2e threshold. The May 5 CUTLASS round closeout established **the surface is exhausted at the cost regime we can afford** — the workload is at 73% of LPDDR5x bandwidth ceiling (198 GB/s of 273 GB/s theoretical), and tile-shape / schedule / swizzle / async-copy mutations cannot reduce bytes-per-token. The remaining 27% of theoretical bandwidth is the entire envelope tile-level kernel mutation can unlock.

**To get past the bandwidth wall, mutations must change bytes-per-token.** That requires changing the actual numerical computation: weight precision (FP8 → NVFP4 → INT4), KV cache compression, sparsity (2:4, MoD), math approximations (faster softmax / fused approximate activation), speculative decoding tricks. **All of these will fail tight numerical parity by design — the parity gate they need is quality-bounded, not byte-similar.**

This spec defines **Track B**, a parallel auto-research mutation regime with:
- A wider edit surface across multiple files in the kernel + dispatch + caller tree.
- A **3-tier quality gate** (B-1 distributional, B-2 behavioral, B-3 full quality benchmark) replacing tight numerical parity.
- Direction-injection from P3a + NCU profile data so the agent (Codex or Claude Code) optimizes load-bearing surfaces.
- Reward-hacking resistance via held-out probe sets, multi-metric gating, and human review on accepted candidates.

The first concrete Track B targets, **under the operator constraints (FP8 weights immutable + Codex agent workload)**, are **prefix caching tuning + LMCache CPU/disk tier (Round 0)** and **Eagle-3 + PLD hybrid speculative decoding (Round 1)**. Combined ceiling on cache-hit agent turns: 3-5× e2e. These exploit Codex-workload-specific characteristics (heavy prefix reuse, edit/echo decode patterns) AND DGX Spark's unique unified-memory advantage (CPU/disk-tier prefix caching is nearly free on this hardware). Subsequent Track B mutations (XGrammar, Quest-style page attention, fused epilogues, KV INT4, math approximations) reuse the same quality gate stack.

## Scope and non-goals

**In scope:**
- Track B mutation regime: quality-bounded, multi-file, anchored on the FFN/attention GEMM stack that dominates warm decode per P3a.
- 3-tier quality gate construction (probes, thresholds, tools, anti-gaming).
- Workload-derived held-out probe set built from `responses-sdk-adapter-cutover-heavy/seed_trace_v5.jsonl`.
- Edit surface: git-tree-based with explicit allow/deny lists.
- Direction-injection from NCU profile data + P3a self-time breakdown + winning-patches memory.
- First Track B target: NVFP4 weight conversion (concrete bring-up sequence).

**Out of scope (deferred):**
- Track A (numerical-bounded) mutation rounds — covered by `l0-ffn-gemm-pivot-20260502.md`. Track A is paused as a primary investment; Track B replaces it as the primary throughput initiative.
- Cold-start prefill optimization — different bottleneck (`gatedattn_attention_with_kv_read` at 77% of prefill), different regime, different surface. Separate workstream when warranted.
- Multi-machine / distributed inference. Single GB10 only.
- Production deployment / rollout. Track B accepts produces a candidate-tuned-bundle artifact; deployment is a downstream decision.

## Codex agent workload characteristics (v0.7 update)

This inference stack serves a Codex-style agent workload (Codex CLI / Claude-Code-like agent loops). That workload has specific characteristics that differ materially from generic chatbot inference, and they unlock optimizations that wouldn't apply to a chat or single-shot serving stack:

| Characteristic | What it implies for optimization |
|---|---|
| **Heavy prefix reuse across turns.** Each agent turn shares system prompt + tool definitions + accumulated history; only the next observation + reasoning differs at the tail. | **Prefix caching is the largest single win.** vLLM `--enable-prefix-caching` + LMCache CPU/disk tier on DGX Spark's unified LPDDR can eliminate ~30K-token prefill on cache-hit turns. Saves multiple seconds per turn. |
| **Long context (10K-100K+ tokens of accumulated tool output).** | Long-context attention optimization (Quest-style page selection on GatedAttn-only layers) gives 2-5× decode at >32K ctx. |
| **Structured output (tool calls follow JSON schemas, code follows language patterns).** | XGrammar / lm-format-enforcer can short-circuit token sampling for schema-forced tokens. ~1.2-1.6× speedup on tool-call-emitting turns. |
| **Reasoning-heavy generation.** Long thinking blocks before short tool calls or final answers. | Eagle-3 / Medusa speculative decoding wins on novel reasoning content. |
| **High repetition in generation.** Recently-seen identifiers, file paths, function names appear again. | **Prompt Lookup Decoding (PLD)** wins on echo/edit/repeat patterns where Eagle-3 underperforms. **PLD + Eagle-3 hybrid is the right shape.** |
| **Low / zero temperature for deterministic agent behavior.** | Speculative decoding's rejection sampling is even cheaper at low T (high acceptance rate). |
| **Multi-turn within one task.** KV cache accumulates across turns. | Prefix cache hit rates are >90% on the static portion. |
| **Latency-sensitive (agent waits between turns).** | User-perceived latency = first-token + decode rate. Prefix cache + speculative decoding both attack first-token; PLD specifically wins on edit/echo turns where the agent is rewriting recently-read content. |
| **Streaming output.** | Speculative decoding is streaming-compatible (verify in chunks). |

**These characteristics specifically advantage Track B optimizations that are useless or marginal in generic chat workloads.** Agent inference has structure that generic inference doesn't, and Track B should exploit it.

**Unique GB10 / DGX Spark advantage for agent workloads:** the unified-memory architecture (128 GB LPDDR5x shared CPU/GPU at 273 GB/s) makes CPU/disk-tier prefix caching nearly free on this hardware. On a discrete-GPU system, evicting prefix KV to host memory is a PCIe-bandwidth-bottlenecked operation (~64 GB/s); on Spark it's a pointer change with no copy. This makes LMCache + aggressive multi-tier prefix caching dramatically more attractive on DGX Spark than on B200 / H100. **This is one of the few places where DGX Spark architecturally beats discrete-GPU systems for inference.**

## Hard constraint: weights are immutable, model identity preserved (v0.6 update)

**Operator constraint:** Track B mutations MUST preserve the model's weights as shipped. The serving stack must serve the **same model** as the unoptimized baseline — not a different model that happens to benchmark well. Quality may drift slightly from unoptimized due to compute/decoding optimizations, but **the underlying weights stay FP8 as-shipped, untouched**.

**Excluded from Track B scope (all of these change the weights):**

| Excluded mutation class | Why excluded |
|---|---|
| **NVFP4 / FP4 / INT4 weight conversion** | Quantizing weights to a smaller format produces a different model — even if it benchmarks well, it's not the model we shipped |
| **GPTQ / AWQ / SmoothQuant weight-only quantization** | Same logic — different weights |
| **2:4 structured sparsity** (hardware-supported on Blackwell) | Forces 50% of weights to zero; that's pruning, which is model modification |
| **Mixture-of-Depths / layer skipping with weight modification** | If layer skipping requires fine-tuning a router or modifying weights, excluded |

**In scope for Track B (none of these change weights):**

| Allowed mutation class | What it changes | Same weights? | Same output distribution? |
|---|---|---|---|
| **Speculative decoding** (Eagle-3 / Medusa / self-speculation / vanilla draft+verify) | Decoding loop; uses a draft model, but verifies against the original model with rejection sampling | ✅ Same target weights | ✅ Mathematically identical output distribution (rejection sampling guarantees this) |
| **Multi-Token Prediction** (MTP) if Qwen 3.6 supports it | Adds small MTP heads; main transformer weights untouched | ✅ Main weights same; small extras | ✅ Verified output |
| **Fused epilogues** (Triton sidecar fusing residual + RMSNorm + FP8 quant after GEMM) | Memory access pattern; same arithmetic, fewer memory passes | ✅ | ≈ Within FP rounding tolerance |
| **Math approximations** (Newton-Raphson rsqrt for faster softmax; polynomial-approx GELU; faster RoPE) | Compute path; same end-result up to approximation tolerance | ✅ | Within numerical-approximation tolerance — quality gate validates |
| **KV cache compression** (INT4 KV quantization) | KV is computed at runtime, NOT model weights | ✅ Weights untouched | Within KV-quantization tolerance — quality gate validates |
| **Async-copy / TMA bulk-load improvements** | Memory hierarchy / load scheduling | ✅ | ✅ Bit-identical with proper sync |
| **Persistent kernels / cluster launch control** | Kernel launch overhead | ✅ | ✅ Bit-identical |
| **Routing improvements / batching** if it doesn't touch weights | Scheduler-level | ✅ | ✅ |

**Two correctness regimes within Track B (under the constraint):**

1. **Strong-equivalence mutations** (speculative decoding, async-copy improvements): output distribution mathematically identical to baseline; quality gate primarily verifies correctness of the implementation, not quality drift.
2. **Quality-bounded mutations** (fused epilogues, math approximations, KV cache compression): introduce small numerical drift; quality gate validates "drift is small enough to not change downstream task quality".

Both regimes use the same B-1 / B-2 / B-3 gate stack — only the threshold sensitivity changes.

**Updated theoretical headroom on this hardware × this model × Codex agent workload:**

| Mutation | Verified or Plausible Speedup | Phase attacked | Composes? |
|---|---|---|---|
| **Aggressive prefix caching + LMCache CPU/disk tier** | **2-10× on cache-hit turns** (eliminates prefill on cached prefixes) | Prefill | Independent of decode-side wins |
| **Eagle-3 + PLD speculative decoding hybrid** | 1.5-2.5× on average; PLD wins echo/edit-heavy, Eagle-3 wins novel reasoning | Decode | Independent of prefix cache |
| **XGrammar constrained generation** | 1.2-1.6× on tool-call-emitting turns | Decode | Composes with both above |
| **Quest-style page attention on GatedAttn-only layers** | 2-5× decode at >32K context | Decode (long-context) | Composes |
| Fused epilogues (Triton sidecar) | 1.05-1.15× | Decode | Composes |
| KV cache INT4 quantization | 1.05-1.15× | Decode | Composes |
| Math approximations (Newton-Raphson rsqrt etc.) | 1.03-1.08× | Decode | Composes |
| Async-copy / TMA improvements | 1.05-1.10× | Decode | Composes |
| **Combined ceiling on cache-hit turns** | **3-5× e2e** if everything lands | — | — |
| **Combined ceiling on cache-miss / cold turns** | **1.8-2.5× e2e** | — | — |
| **Realistic combined achievable** | **2-3× e2e** on typical agent traffic mix | — | — |

**This is materially higher than my earlier "1.8-2.5×" estimate** because:
1. Prefix caching attacks a different phase (prefill) than speculative decoding (decode), so they compose multiplicatively.
2. Codex agent workloads have >90% prefix cache hit rate on the static portion, making prefix caching a near-constant win.
3. PLD + Eagle-3 hybrid covers complementary decode regimes (echo/edit vs novel reasoning) — together they win on more turns than either alone.
4. DGX Spark's unified memory uniquely advantages CPU/disk-tier prefix caching.

**Excluded from headroom calculation per the v0.6 constraint** (weight modifications):
- ❌ NVFP4 / INT4 weight conversion
- ❌ 2:4 sparsity
- ❌ GPTQ / AWQ weight-only quantization

**Compared to the NVFP4-allowed scenario:** with NVFP4 in scope, the combined ceiling could be 4-6× e2e (NVFP4's 1.29× × everything else). Excluding NVFP4 costs ~30% of theoretical maximum, but preserves model identity. Engineering tradeoff is clean.

## Model selection (v0.5 update)

**Preferred model: Qwen 3.6 27B FP8 hybrid attention (if available).** Qwen 3.6 is the more recent series; if a dense 27B FP8 hybrid-attention checkpoint exists in the 3.6 line, use it. **Fallback: Qwen 3.5 27B FP8 hybrid attention** — the model the existing P3a + L0c rounds were measured against.

**Verification step required as a Track B prerequisite:**

The impl agent (Codex / Claude Code) audits HuggingFace + NVIDIA's published Qwen 3.6 catalog and outputs `output/qwen36_availability_audit.md`:
- Is there a Qwen 3.6 27B FP8 dense hybrid-attention checkpoint published?
- If yes: name + HF path + parameter count + attention layout + tokenizer compatibility with the existing seed_trace_v5.jsonl.
- If no: confirms Qwen 3.5 27B FP8 stays as the target.

**Architecture-level robustness:** Bandwidth physics, P3a self-time breakdown shape, and the FFN-GEMM-dominated warm-cache decode pattern are SAME across Qwen 3.5 and 3.6 at the same parameter count. The Track B infrastructure (quality fixtures, edit surface, NCU diagnostic, agent workflow) does NOT change between the two — only the weight checkpoint and any tokenizer-format detail changes. The verified 1.29× NVFP4 result on Spark sm_120 was measured on a Qwen 3.x 30B-class model and is expected to apply to either 3.5 or 3.6 27B.

**If Qwen 3.6 27B FP8 is available AND tokenizer-compatible with the existing trajectory** (`seed_trace_v5.jsonl` of the heavy family), reuse the trajectory unchanged. **If tokenizer format differs**, the workload's seed trajectory must be re-tokenized OR re-captured against the new model. The latter is preferred for trajectory-derived probe consistency; ~10-15 min of capture work.

**If Qwen 3.6 27B FP8 brings architecture changes** (e.g., layer count change, head dim change, attention layout change beyond hybrid 16 GatedAttn + 48 DeltaNet), the P3a self-time breakdown may shift. Re-run the in-process timing pass against the new model BEFORE Round 1 — this is a ~30-minute investment that prevents optimizing the wrong bottleneck. The bandwidth-bound conclusion almost certainly holds, but specific component shares may differ.

## Anchored measurement target: warm-cache decode

Per the 2026-04-30 P3a in-process timing pass and the 2026-05-05 May 5 CUTLASS round diagnostics:

| Phase | Wall time | Top component | Share | Optimization regime |
|---|---|---|---|---|
| Cold-start prefill (first response, empty cache) | 56.7 s | `gatedattn_attention_with_kv_read` | 77.4% | NOT this spec's target |
| **Warm-cache decode (follow-up responses 2-N)** | ~1.4 s/response | **`ffn_linear`** | **59.4%** | **THIS spec's target** |
| Warm-cache decode | — | `deltanet_projection_linear` | 20.6% | Same FP8 GEMM family — wins generalize |
| Warm-cache decode | — | `gatedattn_projection_linear` | 6.0% | Same FP8 GEMM family |
| Warm-cache decode | — | `deltanet_core` | 5.3% | Track A's exhausted target |

**Combined FP8 GEMM family in warm-cache decode: 86%.** Bandwidth utilization: ~198 GB/s of 273 GB/s = **73%**. Measured throughput: 7.36-7.39 tok/s, 135.5-135.9 ms/token.

**Track B targets the FP8 GEMM family specifically because:**
1. It dominates warm-cache decode (86% combined).
2. It is bandwidth-bound by FP8 weight streaming from LPDDR5x.
3. The most direct way to reduce bytes-per-token IS to reduce weight precision — exactly what NVFP4 does.
4. NVIDIA-published measured speedup on this exact hardware class with this kind of workload: 1.29× ([NVIDIA forum, verified](https://forums.developer.nvidia.com/t/fp4-on-dgx-spark-why-it-doesnt-scale-like-youd-expect/360142)).

Agent-workflow framing carries forward from the pivot doc: agents run multi-turn trajectories with populated KV cache. Warm-cache decode is the binding constraint, NOT cold-start prefill.

## Two-regime correctness model

Track A (existing, exhausted on this hardware) and Track B (this spec) run as parallel mutation regimes with different gates:

| | Track A (existing) | **Track B (this spec)** |
|---|---|---|
| Mutation classes | schedule, tile, swizzle, MMA ordering, async-copy, register pressure, kernel rewrite | **NVFP4 weight conversion, KV cache quant, sparsity, math approximation, speculative decoding** |
| Per-probe gate | Tight numerical: `rtol/atol ~1e-3` | **None** at probe level — drift is expected by design |
| End-to-end gate | Tier 4 vLLM parity (downstream logits within tolerance) | **3-tier quality benchmark** (§"Quality gate stack" below) |
| Cost regime | Per-candidate: ~50 sec Tier-3 + ~25 min Tier-4 for top-K | Per-candidate: ~5 min B-1 + ~30 min B-2 for top-K + ~60 min B-3 for accepted-as-winner |
| Theoretical e2e ceiling on this hardware | **1.35× absolute max** (bandwidth physics) | **2-3×+** if NVFP4/sparsity/quant lands well |
| Risk profile | Numerical bugs (low — gate catches) | **Reward hacking** (high — Sakana CUDA Engineer cautionary tale; multi-metric mitigation required) |
| Mutation surface | One source file per round (`chunk_delta_h.py`, CUTLASS template, etc.) | **Multi-file in kernel + dispatch + caller tree, git-controlled** |
| Direction signal | P3a in-process timing + warm decode diagnostic | **P3a + NCU profile distillation + speed thesis + positive/negative memory** |

**Track B does NOT replace Track A; it operates alongside.** Track A continues for research-signal value (validates the mutation pipeline works at all). Track B is the primary throughput initiative.

## Quality gate stack (the load-bearing piece)

Three tiers, ascending in cost and rigor. A candidate passes Track B iff it passes B-1, B-2 (if selected as top-K), and B-3 (if accepted as round winner). Probe sets are **versioned, content-hashed, and held-out from the proposer** — the agent's filesystem view excludes them.

### Tier B-1: Distributional sanity (per-candidate, ~1-5 min)

**Purpose:** catch gross output-distribution drift; cheap reward-hacking detection.

**Metrics:**
- **Per-token KL(p_ref || p_cand)** under teacher-forcing: mean, p95, max.
- **Top-1 argmax agreement rate** across positions.
- **Output entropy delta** vs reference.

**Probe set construction:**
- 64 prompts × 256 generated tokens.
- Sampled from a frozen slice of `benchmark_blueprints/families/responses-sdk-adapter-cutover/seed_trace_v5.jsonl` — the production workload's trajectory.
- Stratified across (length bucket, task-type bucket).
- Reference logits captured ONCE against the §2.2.0 reference baseline (forced FA3 + CUTLASS FP8 + Triton DeltaNet defaults), stored at `benchmark_blueprints/families/responses-sdk-adapter-cutover/quality_fixture/b1_distributional/`.
- **Held out from the proposer:** the proposer's git checkout DOES NOT contain `quality_fixture/`. The controller injects probes via stdin or a temp directory inaccessible to the proposer's source-reading.

**Thresholds (tunable, default values):**
- `mean_kl ≤ 0.05 nats`
- `p95_kl ≤ 0.25 nats`
- `top1_agreement ≥ 98%`
- `|entropy_delta| ≤ 0.05 nats`

A candidate failing ANY threshold is rejected at B-1; recorded in `mutations_rejected.tsv` with `cost_bucket: b1_distributional_fail`.

**Implementation:**
- vLLM with `logprobs=full` to capture per-token distribution.
- Cached reference logits on disk keyed by probe-set content hash.
- Standalone Python script `scripts/run_b1_distributional.py --candidate-bundle <path> --fixture quality_fixture/b1_distributional/v1.yaml`.

**Reward-hacking resistance:**
- Full-distribution KL is hard to game without actually reproducing the distribution.
- Top-1 agreement check catches "always emit the same token" hacks.
- Output-entropy floor catches "emit low-entropy garbage" hacks.
- Probe set rotates quarterly OR on weight rotation; proposer's exposure is bounded by round duration.

### Tier B-2: Behavioral micro-suite (top-K candidates, ~10-30 min)

**Purpose:** catch behavioral drift that distributional check misses; multi-axis quality validation.

**Metrics (run via `lm-evaluation-harness` with `--limit` flags):**

| Benchmark | Slice | Time | What it catches |
|---|---|---|---|
| **MMLU-mini** | 500 questions, stratified | ~2 min | knowledge regression across 57 subject areas |
| **GSM8K-mini** | 200 problems, CoT | ~5-8 min | math reasoning regression |
| **HumanEval** | full 164 problems, pass@1 | ~5 min | code generation regression |
| **TruthfulQA-mini** | ~250 questions | ~3 min | long-tail belief distribution drift (quantization-sensitive) |
| **Workload-derived behavioral probe** | 100 prompts from production trajectory | ~5-10 min | actual agent task quality |
| **Needle-in-haystack** | 16k and 64k context | ~3-5 min | KV-cache integrity at long context |

Optional: MT-Bench-mini (40 prompts judged by a frozen local Llama-3-70B-Instruct judge), ~10-15 min.

**Probe set:**
- Standard benchmark slices (well-documented, not workload-specific).
- Workload-derived behavioral probe: 100 prompts sampled from `seed_trace_v5.jsonl`, scored by a fixed local judge (NOT GPT-4 — deterministic, sandboxed).
- Probe set is held out from the proposer's source view.

**Thresholds:**
- `avg_quality_delta ≥ -1.0 percentage point` across (MMLU-mini, GSM8K-mini, HumanEval, TruthfulQA-mini).
- `workload_behavioral_judge_score_delta ≥ -0.1` (5-point Likert from judge).
- `needle_recall ≥ 95% of reference recall`.
- Optional MT-Bench delta: ≥ -0.1.

A candidate failing aggregate threshold OR regressing > 1.5pp on any single benchmark is rejected at B-2; recorded with `cost_bucket: b2_behavioral_fail`.

**Implementation:**
- `lm-evaluation-harness` with `--model vllm --model_args base_url=http://localhost:8100 --limit 500` (etc.).
- Custom workload-behavioral-probe scorer: load 100 prompts, generate, run through judge model, score on instruction-following rubric.
- Standalone Python script `scripts/run_b2_behavioral.py --candidate-bundle <path> --fixture quality_fixture/b2_behavioral/v1.yaml`.

**Reward-hacking resistance:**
- Multi-axis (knowledge / math / code / belief / agent / long-context) makes single-axis gaming infeasible.
- Judge model weights are pinned and the proposer cannot see judge prompts.
- Needle test specifically catches KV-cache compression failures that B-1 distributional check misses.

### Tier B-3: Full quality benchmark (top-1, pre-promotion, ~30-60 min)

**Purpose:** final quality validation before declaring a candidate the round winner. Run rarely.

**Metrics:**
- Full **MMLU**, **GSM8K**, **HumanEval+**, **MATH-500**, **IFEval**, **RULER** (long-context).
- **MAUVE** between candidate samples and reference samples on 500 workload prompts at T=0.7 (embedding-distribution divergence; designed exactly for "are these two text distributions equivalent").
- **Held-out perplexity** on a 4 MB workload-derived corpus the B-1 and B-2 probe sets never touched.

**Probe set:**
- Complete benchmark suites.
- Workload corpus is a separate held-out shard rotated quarterly.

**Thresholds:**
- `aggregate_quality_score_delta ≥ -0.5 pp`.
- `mauve ≥ 0.95`.
- `delta_perplexity / reference_perplexity ≤ 0.5%` (much tighter than B-2's casual threshold; this is the production-grade gate).
- No single benchmark regresses > 1.5 pp.

**Implementation:**
- `lm-evaluation-harness` full suite.
- HuggingFace `mauve` package for distribution similarity.
- `RULER` from NVIDIA's repo.
- Standalone script `scripts/run_b3_full.py --candidate-bundle <path>`.

**Reward-hacking resistance:**
- Only tier with the truly held-out workload corpus.
- MAUVE is embedding-distribution-based — reproducing it requires actually matching the reference's generation distribution, not just gaming token statistics.
- Independent re-evaluation on a fresh probe slice before any production promotion.
- Human review of the candidate's diff and B-1/B-2/B-3 metrics before promotion.

### Cross-tier anti-gaming controls

- Probe sets versioned and content-hashed; proposer's filesystem view excludes them via sandboxed checkout.
- Per-evaluation randomized prompt subsetting (sample 64 of 256 each B-1 run, redrawn).
- Output-entropy floor on any tier (reject candidates whose generations are suspiciously low-entropy).
- Log every (candidate, tier, score) tuple to `quality_gate_history.tsv`. Detect proposers that overfit B-1 over generations.
- Quarterly probe-set rotation regardless of round results.
- Independent re-evaluation of B-3-passing candidate on a fresh probe slice before promotion.
- Human review on every accepted Track B candidate (Sakana mitigation; cost is amortized — these are rare high-impact decisions).

## Edit surface architecture

**Track A allowed mutation of one source file per round** (`chunk_delta_h.py`, or one CUTLASS template). **Track B widens this** because mutations like NVFP4 weight conversion necessarily span:
- Weight loading / dtype dispatch (`vllm/model_executor/layers/quantization/fp8.py` → adjacent NVFP4 path).
- Kernel selection (`Fp8LinearMethod` vs `Nvfp4LinearMethod`).
- Caller call sites (whichever layers in the model dispatch through these methods).
- Optional: weight format conversion script (`scripts/convert_weights_to_nvfp4.py`).

The agent gets a **git-tree-based edit surface** with explicit allow/deny lists.

### Allow list (the agent CAN edit)

```yaml
# benchmark_blueprints/auto_research/track_b_edit_surface.yaml
edit_allow:
  - vllm/model_executor/layers/quantization/fp8.py
  - vllm/model_executor/layers/quantization/nvfp4.py        # if exists
  - vllm/model_executor/layers/quantization/utils/**
  - vllm/model_executor/layers/quantization/compressed_tensors/**
  - vllm/model_executor/layers/linear.py                    # call-site dispatch
  - vllm/attention/ops/**                                   # KV cache layout if mutating that
  - csrc/quantization/fp8/**                                # CUDA-side FP8 implementations
  - csrc/quantization/cutlass_w8a8/**                       # CUTLASS scaled-MM
  - kernels/**                                              # any L0c kernel workdir source
  - scripts/convert_weights_to_*.py                         # weight format conversion utilities (NEW agent-creatable)
  - tests/test_quantization_*.py                            # only to UPDATE existing tests for new dtype, not to weaken them
```

### Deny list (the agent CANNOT edit, hard reject in preflight)

```yaml
edit_deny:
  - tests/test_l0c_*.py                                     # L0c controller tests; evaluator-corruption risk
  - tests/test_auto_research.py                             # auto-research controller tests
  - tests/test_quality_gate_*.py                            # quality gate tests; evaluator-corruption risk
  - tests/test_parity_fixture.py                            # parity fixture tests
  - src/lumo_flywheel_serving/auto_research.py              # the controller itself
  - src/lumo_flywheel_serving/quality_gate/**               # quality gate implementation; evaluator-corruption risk
  - benchmark_blueprints/families/**/quality_fixture/**     # held-out probe sets; reward-hacking surface
  - scripts/run_b1_distributional.py                        # quality gate runner
  - scripts/run_b2_behavioral.py
  - scripts/run_b3_full.py
  - scripts/build_parity_fixture.py
  - .git/**                                                 # git internals
  - .github/**                                              # CI configuration
```

### Git-tree-based memory and ratchet

Each candidate is one **git commit** on a per-round branch. The branch is:
```
auto_research/track_b/<round_id>/candidate_NNN
```

Branching architecture:
- `main` is the reference baseline (untouched during round).
- Each round's `auto_research/track_b/<round_id>/baseline` is checked out from main at round start.
- Each candidate `auto_research/track_b/<round_id>/candidate_NNN` is checked out from the previous accepted candidate (or baseline if none accepted yet).
- Accepted candidates are kept on the round branch; rejected candidates' commits are preserved as artifacts but not promoted.
- The round winner branch is merged back to main only after B-3 + human review.

This gives:
- **Persistent memory across attempts** — each candidate sees the prior accepted candidate's state, not a clean slate. A 5-attempt round can incrementally compose 5 winning mutations.
- **Reproducible artifacts** — every candidate is a commit hash; every round is a branch.
- **Easy diff inspection** — `git diff candidate_005..candidate_006` shows what changed.
- **Human review surface** — pre-promotion human review is a normal git PR.

Round artifacts on disk at `output/auto_research/track_b/<round_id>/`:
- `branch_log.json` — per-candidate commit hashes, parent commits, timestamps.
- `winning_diffs.md` — accepted candidates' diffs + B-1/B-2/B-3 scores.
- `mutations_rejected.tsv` — rejected candidates with structured reason.
- `quality_gate_history.tsv` — every (candidate, tier, score) tuple.
- `candidates/<NNN>/` — per-candidate workdir with patch, gate outputs, NCU diagnostic delta if winner.

## Direction architecture

The agent doesn't search blindly. The strategy brief and iteration brief inject load-bearing direction from three sources:

### 1. P3a self-time breakdown (anchored bottleneck)

Embedded in `strategy_brief.md`:
```
P3a warm-cache decode bottleneck (measured 2026-04-30):
  ffn_linear:                    59.4% — primary target
  deltanet_projection_linear:    20.6% — secondary target (same FP8 GEMM family)
  gatedattn_projection_linear:    6.0% — same FP8 GEMM family
  gatedattn_attention_with_kv_read: 6.0%
  deltanet_core:                  5.3% — Track A territory, NOT Track B
  
  Bandwidth utilization: 73% of 273 GB/s LPDDR5x ceiling.
  Bytes-per-token: ~27 GB FP8 weight stream.
  
  Bandwidth-bound conclusion: optimizations that DON'T reduce bytes-per-token
  are bounded above at 1.35x. Optimizations that DO reduce bytes-per-token
  (NVFP4, KV quant, sparsity, math approximation) can plausibly reach 1.5-2x
  for individual mutation classes.
```

### 2. NCU diagnostic profile (distilled, NOT raw)

Per the prior pivot doc's §"NCU Diagnostic Profile" — one-shot capture on ~8-16 representative shapes against the reference baseline. Distilled into the iteration brief:

```
NCU baseline diagnostic (captured 2026-XX-XX, ~12 shapes):

  Shape M=1, N=11008, K=4096 (small-M decode FFN, most-frequent shape):
    DRAM throughput: 87% of 273 GB/s ceiling
    SM throughput: 22%
    Tensor-core utilization: 18%
    Top stalls: (1) global-memory-load wait, (2) shared-memory-bank-conflict
    Operational interpretation: weight-loading dominates; reducing weight bytes
    OR overlapping load with compute is the lever.
  
  [... 11 more representative shapes]
```

### 3. Memory: positive (winning) + negative (rejected)

**Positive memory** at `winning_diffs.md` — the last 5 accepted candidates across all Track B rounds:
- Diff content
- Speed thesis (what the candidate claimed it would do)
- Achieved B-1/B-2/B-3 scores
- Achieved e2e speedup
- NCU delta vs baseline if captured (DRAM↓ vs SM↑ vs occupancy↑)

**Negative memory** at `mutations_rejected.tsv` — every rejected candidate across all Track B rounds:
- Patch hash
- Failure tier (B-1 / B-2 / B-3 / preflight / compile)
- Failure reason (quality drift / behavioral regression / numerical NaN / compile error / etc.)
- First-failing-metric where applicable
- Operator-tagged "do not repeat" flag for systematic failures

Per the AlphaEvolve / FunSearch pattern: the agent sees diverse positive memory (not just best), the negative memory is shown as evidence-and-guidance not enforced filter (per `l0c-evaluation-ladder-and-memory-prior-art-20260430.md` decision).

### Speed thesis requirement

Every candidate's `mutation.patch` MUST be accompanied by a `candidate_analysis.md` containing:
- **Speed thesis:** what the patch is supposed to do (e.g., "convert the FFN gate-up-proj weight layout from FP8 per-channel-scale to NVFP4 micro-block-scale, reducing weight bytes from 0.5B/elem to 0.25B/elem, expected ~1.3x bandwidth reduction on this layer").
- **Expected affected counter:** which NCU counter should change and how (e.g., "DRAM throughput should drop from 87% to ~70%; SM throughput should rise correspondingly").
- **Quality risk:** what's at stake if the candidate is wrong (e.g., "NVFP4 micro-block-scale at granularity 16 may insufficient on outlier-channel layers; if MMLU drops > 1pp this is the cause").
- **Why-not-prior-failure:** if the candidate resembles a prior rejected family, why is this materially different.

Without speed thesis, the candidate is rejected at preflight (`preflight_speed_thesis_missing`).

## Workflow for Codex / Claude Code agents

Per-iteration workflow:

```
1. Controller reads:
   - strategy_brief.md (P3a + NCU + positive/negative memory + edit surface)
   - prior_mutations_rejected.tsv (cross-round failure history)
   - quality gate fixture metadata (NOT the probes themselves; only schema/version)
   - current round branch state (git log + last accepted candidate)

2. Controller spawns ONE agent worker per candidate via:
     codex exec --cwd <round_workdir> --prompt iteration_brief.md
     # or:
     claude-code agent --workdir <round_workdir> --prompt iteration_brief.md

3. Agent worker:
   - Reads strategy_brief.md, iteration_brief.md, recent winning diffs.
   - Inspects current source state (git checkout of round branch).
   - Writes ONE mutation as a git commit OR multiple commits squashed:
       commit subject: "candidate NNN: <short speed-thesis summary>"
       commit body: structured speed thesis (machine-parseable)
       diff: actual changes across allow-listed files
   - Writes candidate_analysis.md with the four required fields.
   - DOES NOT run the quality gate (controller owns that).
   - Exits.

4. Controller validates:
   - All edits are within allow-list (deny-list violation → preflight reject).
   - candidate_analysis.md present and parseable.
   - speed_thesis is non-empty and references at least one NCU counter or P3a breakdown line.
   - Patch builds (compile preflight: ~2 min).

5. Controller runs B-1 distributional gate (~1-5 min).
   - Pass: candidate enters top-K ranking pool.
   - Fail: candidate rejected; mutations_rejected.tsv updated; agent feedback for next iteration.

6. Top-K (default K=3) candidates from B-1 ranking proceed to B-2 behavioral micro-suite (~30 min each).
   - Pass: candidate stays in winner pool.
   - Fail: rejected with structured reason.

7. Top-1 from B-2 winner pool proceeds to:
   - vLLM end-to-end throughput measurement (the actual reason for the round).
   - B-3 full quality benchmark (~60 min) ONLY if e2e improvement > 5%.

8. If B-3 passes: candidate is round winner.
   - Branch merged into round-winner branch.
   - Pre-promotion human review.
   - On approval: merged to main as new baseline.
```

### Codex specifics

- `codex exec --cwd <workdir> --prompt iteration_brief.md`
- Per-iteration spawn (Karpathy fresh-agent pattern); no persistent state across iterations.
- iteration_brief.md is the only context; brief is regenerated per iteration with updated memory.

### Claude Code specifics

- `claude-code agent --workdir <workdir> --prompt iteration_brief.md`
- Slash command `/auto-research-track-b-iteration` available if defined as a project skill.
- Same fresh-agent-per-iteration pattern.
- Tool restrictions enforced via Claude Code's permission system (deny-list paths blocked at file-tool level).

## Round 0 (mandatory baseline): aggressive prefix caching + LMCache tier

**Why Round 0:** prefix caching is the largest single agent-workload win, AND DGX Spark's unified memory uniquely advantages it. This is mandatory baseline work — every subsequent Track B round measures against the prefix-cache-tuned baseline, not the cold baseline.

**Audit + tune (1-2 days):**

1. Verify vLLM is launched with `--enable-prefix-caching` (it's not the default in all versions).
2. Verify `--block-size 32` (vs default 16) — larger blocks reduce radix lookup cost and fragmentation matters less when prefixes are reused.
3. Verify `--num-gpu-blocks-override` is set high — DGX Spark's unified pool means we can be aggressive without OOM risk.
4. Verify `--enable-chunked-prefill` is enabled — partially redundant with prefix caching but helps when only the tail mismatches.
5. Audit DeltaNet state caching: vLLM caches K/V tensors per layer for prefix; DeltaNet's recurrent state needs separate cache infrastructure. **Verify the inference stack handles DeltaNet state checkpoint/restore correctly**. This is a real correctness risk and engineering cost item — the DeltaNet recurrent state is layer-local and per-sequence; mishandling causes silent quality drift on cache-hit turns.
6. **Add LMCache CPU/disk-tier prefix caching.** DGX Spark's LPDDR5x is shared between CPU and GPU — CPU offload is essentially a pointer change, no PCIe copy. This is the unique GB10 advantage.
7. Measure cache hit rate on representative agent traces (`responses-sdk-adapter-cutover-heavy/seed_trace_v5.jsonl`). Target: >85% on the static portion (system + tool defs + early history).
8. Measure latency-per-turn before/after on the first 10 turns of a typical agent task.

**Quality gate:** prefix caching is mathematically identical to no-prefix-caching. B-1 distributional KL should be near zero by construction; if it's NOT zero, that's an implementation bug (probably in DeltaNet state handling). Run B-1 anyway as the bug-detection gate.

**Expected outcome:** 2-10× speedup on cache-hit turns; cumulative agent-task latency reduction 30-60%.

## Round 1: speculative decoding (Eagle-3 + PLD hybrid)

**Why Round 1 (after prefix cache):** speculative decoding is the LARGEST documented decode-phase speedup that preserves the model's output distribution mathematically (via rejection sampling). Same FP8 weights. Same final tokens. Just amortizes the same weight stream over multiple tokens per forward pass.

**Why "Eagle-3 + PLD hybrid":** they dominate complementary regimes for the Codex workload.

| Method | Wins on | Loses on |
|---|---|---|
| **Eagle-3** | Novel reasoning content (CoT, free-form thinking blocks) | Echo/edit content where tokens repeat from prompt |
| **PLD (Prompt Lookup Decoding)** | Echo/edit content (rewriting recently-read function, repeating tool args, file paths, identifiers) | Novel reasoning |
| **Hybrid** | All of the above | — |

For Codex specifically, code-edit turns and tool-arg-emission turns are PLD's home turf — PLD reports 4-7 token acceptance per step on edit-heavy outputs, dominating Eagle-3 on those. Reasoning turns are Eagle-3's home turf. The hybrid runs PLD as cheap first-tier draft, falls back to Eagle-3 when PLD's k-gram match confidence is low.

**Concrete approach: Eagle-3-style draft+verify within vLLM.**

- A small draft model (typically 1B-3B parameters, much smaller than the 27B target) generates N draft tokens autoregressively.
- The 27B target model verifies the N draft tokens in parallel via a single forward pass.
- Tokens are accepted via rejection sampling — output distribution is mathematically identical to running the target model alone.
- Net effect: average ~1.7-2× tokens generated per forward pass (depends on draft acceptance rate; typical 70-80% acceptance gives 1.5-2× e2e).

**Why this matches the operator's constraint:**
- Target model weights: untouched FP8.
- Output distribution: mathematically identical to non-speculative baseline.
- Quality drift: zero by construction (rejection-sampling correctness theorem).
- The "different model" risk is on the DRAFT model, but the draft model is just a speedup mechanism — the final accepted tokens come from the target model.

**Bring-up sequence:**

1. **Audit (1-2 days):** does vLLM's speculative decoding path on Blackwell sm_120 work end-to-end? Read `vllm/spec_decode/` and adjacent. What draft models are compatible with Qwen 3.6 / 3.5 27B target? Eagle-3 specifically requires a trained draft head — does one exist for this target model? Output: `output/spec_decode_audit.md`.

2. **Identify or build a draft model:** options ordered by preference:
   - (a) Use a published Eagle-3 draft head trained for Qwen 3.x 27B (HuggingFace search; verify in audit).
   - (b) Use a smaller Qwen 3.x checkpoint as draft (e.g., Qwen 3.x 1.5B FP8) — works as "self-speculation" with lower acceptance but no draft training needed.
   - (c) Build a draft head via published Eagle-3 training procedure (multi-day effort; avoid in Round 1).

3. **Build B-1 / B-2 / B-3 quality fixtures (~1-2 days):**
   - B-1: 64 prompts × 256 tokens of reference logits captured against §2.2.0 reference baseline (without speculative decoding).
   - B-2: lm-evaluation-harness slices configured + workload behavioral probe + needle-in-haystack probe.
   - B-3: full benchmark suite + held-out workload corpus + MAUVE config.

4. **Build edit-surface enforcement (~1 day).**

5. **Build git-branch round controller (~2 days).**

6. **Capture NCU diagnostic baseline (~30 min one-shot):** representative shapes against non-speculative reference. NCU tells us how much of the bandwidth-wall the speculative path saves.

7. **Run Track B Round 1 — speculative decoding bring-up (~few hours):**
   - Goal: enable Eagle-3-style speculative decoding on Qwen 3.6 27B FP8 (or 3.5 fallback), validate output equivalence, measure e2e speedup.
   - Expected accepted candidates: 1-3 (speculative decoding is a single feature; few candidates needed).
   - Expected outcome: e2e speedup in **1.5-2.0× range** if a usable draft model exists; output distribution mathematically identical to baseline (B-1 distributional KL near zero by construction).

8. **Round closeout report:** measured speedup, B-1/B-2/B-3 deltas, draft acceptance rate, NCU delta, recommendations for Round 2.

**If Round 1 lands:** subsequent Track B rounds compose on top of speculative decoding for additional wins:

- Round 2: **Fused epilogues** (Triton sidecar fusing residual + RMSNorm + FP8 quant after GEMM). Independent of speculative decoding; composes multiplicatively.
- Round 3: **KV cache INT4 quantization**. Independent; composes.
- Round 4: **Math approximations** (Newton-Raphson rsqrt, polynomial-approx activation). Independent; composes.

**Combined target after Rounds 1-4: 1.8-2.5× e2e** with FP8 weights unchanged.

## Per-round implementation details

For each round below: edit surface, prior art the agent should study, tools the agent has, cheap self-verify (CLI calls the agent makes itself before spending controller wall-clock), time breakdown, pass/fail/move-on criteria, and gating verify (expensive validation before declaring a mutation good).

The agent's iteration_brief.md is regenerated per round to include the round-specific edit surface + prior art + cheap-verify CLI list. The controller enforces the gating-verify phase; the agent owns cheap self-verify.

### Round 0 — Prefix caching tuning + LMCache CPU/disk tier

| Field | Value |
|---|---|
| **Edit surface** | (a) vLLM launch flags / config — `vllm/engine/arg_utils.py` if needed for new flags. (b) LMCache integration adapter at `vllm/distributed/kv_transfer/lmcache_connector.py` if not already present. (c) DeltaNet state caching audit: `vllm/model_executor/models/qwen3_*.py` and `vllm/model_executor/layers/fla/*.py` — look for KV cache integration points; verify DeltaNet recurrent state has prefix-cache-compatible serialize/deserialize. **Most edits are config + small integration glue, NOT kernel mutation.** |
| **Prior art** | vLLM automatic prefix caching docs (https://docs.vllm.ai/en/latest/automatic_prefix_caching.html, verified). LMCache repo (https://github.com/LMCache/LMCache, verified). SGLang RadixAttention paper (arXiv 2312.07104, verified). Parrot OSDI 2024 — multi-turn agent workload optimization. The agent reads these to understand the cache invalidation rules + the LMCache integration contract. |
| **Tools** | (a) `vllm benchmark_throughput.py --enable-prefix-caching` for baseline. (b) `curl localhost:8000/metrics | grep prefix_cache` for cache-hit-rate. (c) `nsys profile` (cheap; no full NCU yet) for high-level latency breakdown. (d) Custom `scripts/measure_prefix_cache_hit_rate.py --trace seed_trace_v5.jsonl`. |
| **Cheap self-verify** (agent CLI before declaring patch ready) | (1) `vllm serve --enable-prefix-caching --block-size 32 ... --dry-run` (config valid). (2) `python scripts/measure_prefix_cache_hit_rate.py` against a 5-turn slice of seed trace; expect >85% hit on static portion. (3) `python scripts/run_b1_distributional.py --candidate <patched_serve_config>` — KL must be near zero (mathematically identical; non-zero = DeltaNet state cache bug). (4) Latency check: 1 turn cold + 1 turn warm; ratio should show prefill elimination on warm turn. |
| **Time breakdown** | 0.5 day: vLLM flag tuning + dry-run validation. 1 day: LMCache integration + DeltaNet state-cache audit. 0.5 day: cache-hit-rate measurement + B-1 gate. **Total: 2 days.** |
| **Pass criteria** | Cache hit rate ≥ 85% on representative agent trace. Latency-per-turn drops ≥ 30% on cache-hit turns. B-1 distributional KL near zero (mathematically identical). LMCache CPU/disk tier active and serving evicted prefixes. No DeltaNet state correctness bugs. |
| **Fail criteria** | Cache hit rate < 50% (something is wrong — likely block size mismatch, or DeltaNet state not actually cached). B-1 KL non-zero (DeltaNet state cache correctness bug — halt and fix BEFORE continuing). LMCache crashes or corrupts cache. |
| **Move-on criteria** | If LMCache integration cost balloons beyond 2 days, ship Round 0 with vLLM-only prefix caching (no LMCache tier) and revisit the disk tier in a later cycle. The vLLM-only baseline still gives most of the win. |
| **Gating verify** (controller-owned, expensive) | Full B-1/B-2/B-3 quality gate stack. End-to-end agent-task latency measurement: run a 10-turn representative task once cold, once with the prefix-cached path; cumulative wallclock reduction must be ≥ 25% to ship. |

---

### Round 1 — Eagle-3 + PLD hybrid speculative decoding

| Field | Value |
|---|---|
| **Edit surface** | (a) vLLM launch flags — `--speculative-model "[ngram]" --ngram-prompt-lookup-max 8` for PLD, `--speculative-model <eagle-3-checkpoint>` for Eagle-3. (b) Custom proposer if hybrid PLD-then-Eagle-3 fallback isn't supported natively: `vllm/spec_decode/proposer/hybrid_proposer.py` (new file). (c) DeltaNet-aware verifier: vLLM's spec_decode verifier must handle DeltaNet recurrent state correctly during draft tree verification. **This is the highest-risk integration item — if hybrid-attention spec_decode doesn't work in vLLM today, this round needs vLLM-side code changes.** |
| **Prior art** | Prompt Lookup Decoding (Apoorv Saxena) — https://github.com/apoorvumang/prompt-lookup-decoding (verified). Eagle-3 paper (arXiv 2503.01840, uncertain). Lookahead decoding (LMSYS, https://lmsys.org/blog/2023-11-21-lookahead-decoding/, verified). vLLM spec_decode docs. Sequoia (arXiv 2402.12374) for tree-based draft. The agent reads PLD's repo first because it's the simplest-to-deploy path and dominates exactly the workload pattern (echo/edit) Codex generates. |
| **Tools** | (a) `vllm completions` with spec-decode enabled for sanity. (b) Built-in vLLM acceptance-rate logger (`--spec-decode-log-acceptance-rate`). (c) NCU on draft+verify path vs full forward — see how much weight stream is amortized. (d) `scripts/measure_draft_acceptance.py --trace seed_trace_v5.jsonl --tier1 pld --tier2 eagle3` to characterize PLD vs Eagle-3 wins per turn type. |
| **Cheap self-verify** | (1) `vllm completions --speculative-model "[ngram]" ...` returns same tokens as non-spec-decoded for greedy decoding (token-id equality on a 32-prompt set; mathematically identical by rejection sampling). (2) `scripts/measure_draft_acceptance.py` reports per-turn acceptance rate; for code-edit turns expect ≥ 0.5 acceptance with PLD, for reasoning turns Eagle-3 fallback should fire. (3) B-1 KL near zero (rejection sampling guarantees this). |
| **Time breakdown** | 1 day: vLLM spec_decode + PLD audit (does it work for hybrid attention?). 1 day: Eagle-3 draft head identification (HuggingFace search) OR self-speculation setup with smaller Qwen. 1 day: hybrid proposer (if needed). 0.5 day: measurement + gates. **Total: 3.5 days.** |
| **Pass criteria** | Average decode tok/s improves ≥ 1.5×. PLD acceptance rate ≥ 0.5 on edit/echo turns; Eagle-3 acceptance ≥ 0.5 on reasoning turns. B-1 distributional KL near zero (rejection-sampling correctness). Output token-id equality vs non-spec-decoded baseline at greedy decoding. |
| **Fail criteria** | B-1 KL non-zero at greedy decoding (rejection sampling correctness bug — halt). Acceptance rate < 0.2 on average (draft model is too dissimilar to target — try a different draft). |
| **Move-on criteria** | If vLLM spec_decode doesn't support hybrid attention (DeltaNet + GatedAttn), this round needs deeper vLLM-side code changes (>1 week). Move to Round 2 (XGrammar) and Round 3 (Quest) which are independent of spec_decode; revisit Round 1 after the others ship. |
| **Gating verify** | B-1 distributional (must be near zero by rejection sampling). B-2 behavioral (MMLU/GSM8K/HumanEval — must match baseline within 1pp). B-3 full benchmark + MAUVE (output distribution equivalence). E2E agent-task wall-clock measurement on 10-turn task; should improve 30-50% vs Round 0 baseline. |

---

### Round 2 — XGrammar constrained generation for tool calls

| Field | Value |
|---|---|
| **Edit surface** | (a) vLLM launch flag `--guided-decoding-backend xgrammar` (replaces the older `outlines` backend). (b) Per-tool JSON schema configuration in the agent's tool-call request shape — agent must emit `guided_json` field. (c) Optional: pin XGrammar version in `pyproject.toml` since vLLM's bundled version may lag latest. **Mostly config + dependency pinning, NOT source mutation.** |
| **Prior art** | XGrammar paper (arXiv 2411.15100, uncertain) and repo (https://github.com/mlc-ai/xgrammar, verified). Outlines (https://github.com/outlines-dev/outlines, verified). lm-format-enforcer (https://github.com/noamgat/lm-format-enforcer, verified). vLLM guided generation docs. The agent reads XGrammar's compressed-pushdown-automaton design notes to understand why forced-token skipping is mathematically identical (within-schema, the forced token has prob=1). |
| **Tools** | (a) `vllm completions --guided-json '<schema>' ...` for sanity. (b) `scripts/measure_constrained_decode_speedup.py --schema tool_calls.json --trace seed_trace_v5.jsonl`. (c) JSON schema validator (`jsonschema` package) for output verification. |
| **Cheap self-verify** | (1) Generate a tool call output with XGrammar enabled; verify it's valid JSON via `jsonschema.validate(output, tool_schema)`. (2) Measure forced-token-skip rate — XGrammar reports this; expect 30-50% of emitted tokens are forced on tool-call turns. (3) Token-id equality at forced positions vs non-constrained baseline (forced positions ARE deterministic); free-token positions may differ slightly due to logit masking. |
| **Time breakdown** | 0.5 day: vLLM XGrammar enablement + version pinning. 1 day: per-tool schema definition for our agent's tool calls (JSON schemas for `read_file`, `bash`, etc.). 0.5 day: measurement. **Total: 2 days.** |
| **Pass criteria** | Tool-call decode tok/s improves ≥ 1.2× on tool-call-emitting turns. 100% schema validity on emitted tool calls. B-1 within-schema KL near zero (forced-token skipping is mathematically lossless within schema). |
| **Fail criteria** | Schema violations on emitted tool calls (XGrammar bug or schema misconfig). B-1 free-token KL > threshold (logit masking is too aggressive). |
| **Move-on criteria** | If XGrammar isn't in current vLLM build, upgrade vLLM (typically a 1-day port). If upgrade is hard, fall back to Outlines or lm-format-enforcer (1.0-1.2× win instead of 1.2-1.6×). |
| **Gating verify** | B-1 with within-schema and free-token KL separately reported. B-2 behavioral on tool-call-heavy benchmark slice. E2E measurement on tool-call-heavy turns specifically (not aggregated with reasoning turns since the win is tool-call-localized). |

---

### Round 3 — Quest-style page attention on GatedAttn-only layers

| Field | Value |
|---|---|
| **Edit surface** | (a) New custom Triton kernel for GatedAttn page-selection: `kernels/gatedattn/page_attention.py`. (b) Page-selection scoring logic: `vllm/attention/ops/page_selection.py` (new). (c) GatedAttn dispatch integration: `vllm/model_executor/layers/fla/gated_attn.py` (or wherever the model dispatches attention) to add a `--page-attention-top-k` flag and route through new kernel for layers with `kv_cache_dtype != none`. **DeltaNet layers untouched** — page attention applies only to the 16/64 GatedAttn layers in the hybrid architecture. |
| **Prior art** | Quest (arXiv 2406.10774, uncertain) — page-level KV retrieval with top-k selection. RetrievalAttention (2024) — embedding-based KV page selection. InfiniGen (2024) — long-context decode optimization. The agent reads Quest's page-scoring algorithm specifically because the published version is what informs the top-k threshold defaults. |
| **Tools** | (a) Custom microbench for the new page-selection kernel: `scripts/microbench_page_attention.py --probe-set <tier3_fixture>`. (b) NCU on full attention vs page-selected attention to measure DRAM throughput delta. (c) Needle-in-haystack benchmark (NVIDIA RULER repo) for long-context recall validation. |
| **Cheap self-verify** | (1) Compile check: Triton kernel compiles on Blackwell sm_120. (2) Microbench correctness: page-selected output matches full-attention output within `rtol/atol=5e-3` on 16 captured GatedAttn invocations from seed trace. (3) Top-k recall: at top-k=32 (out of ~256 pages at 32K context), recall the right pages on a needle test ≥ 95%. |
| **Time breakdown** | 2 days: Triton kernel implementation (page selection + attention compute). 1 day: vLLM integration + dispatch routing. 1 day: needle-in-haystack tuning (top-k threshold for GatedAttn page selection). 1 day: measurement + gates. **Total: 5 days.** |
| **Pass criteria** | Long-context (>32K) decode tok/s improves ≥ 2× vs Round 2 baseline. Needle-in-haystack recall ≥ 95% of reference at 16K, 32K, 64K contexts. B-1/B-2 within tolerance (some numerical drift acceptable but downstream behavior must match). |
| **Fail criteria** | Needle recall drops > 5% (page selection too aggressive — increase top-k). B-1 KL > tolerance on long-context probes. Numerical instability under speculative decode verifier. |
| **Move-on criteria** | If kernel integration with hybrid attention dispatch is complex (>5 days), defer to v0.4. Round 4-7 are independent of long-context optimization and continue. |
| **Gating verify** | Full needle-in-haystack at 16K, 32K, 64K, 128K (the most expensive of the gating verifies — runs at every context length). B-1/B-2 with long-context-specific probes added. E2E measurement on long-context turns specifically. |

---

### Round 4 — Fused epilogues (Triton sidecar)

| Field | Value |
|---|---|
| **Edit surface** | (a) New Triton sidecar kernel: `kernels/fused/attn_residual_norm_quant.py` (per pivot doc §"fused_epilogue" target). (b) vLLM dispatch integration in `vllm/model_executor/layers/linear.py` to route through sidecar after applicable GEMM calls. (c) Per pivot doc: Triton-sidecar surface only, NOT CUTLASS-side fusion (deferred to v0.4+). |
| **Prior art** | This repo's `l0-ffn-gemm-pivot-20260502.md` §"fused_epilogue" target spec. CUTLASS Blackwell epilogue examples (https://github.com/NVIDIA/cutlass, examples 70+, verified for repo existence). Triton fused kernel patterns from FLA library (https://github.com/fla-org/flash-linear-attention, verified). |
| **Tools** | (a) Triton's `@triton.testing.do_bench` for cycle-time measurement. (b) Custom microbench harness (Tier 3 isolated kernel replay from pivot doc): `scripts/run_b1_distributional.py` extended for fused-epilogue probes. (c) NCU on fused vs unfused path — should show DRAM throughput drop and SM utilization rise. |
| **Cheap self-verify** | (1) Compile check on Blackwell sm_120. (2) Tier 3 isolated kernel replay correctness: 4-checkpoint compare (post-residual, post-norm, post-quant, downstream-logit) within tolerances. (3) Per-probe runtime improves vs unfused reference; aggregate speedup ≥ 1.05× across 512-probe set. (4) NCU shows ≥ 1 fewer DRAM round-trip per layer. |
| **Time breakdown** | 1.5 days: Triton sidecar implementation (residual + RMSNorm + FP8 quant in one kernel). 0.5 day: vLLM dispatch integration. 0.5 day: parity fixture capture (4-checkpoint per pivot doc). 0.5 day: measurement + gates. **Total: 3 days.** |
| **Pass criteria** | Per-layer memory-pass count drops by ≥ 1 (verified via NCU). E2E decode tok/s improves ≥ 1.05× vs Round 3 baseline. Tier 3 4-checkpoint parity within tolerances. B-1/B-2/B-3 quality gates pass. |
| **Fail criteria** | Numerical drift exceeds Tier 3 tolerance. Sidecar kernel doesn't compile on Blackwell. Integration bugs (wrong input layout, etc.). |
| **Move-on criteria** | If Blackwell sm_120 lacks register file headroom for the 4-fusion sidecar (only 99 KB SMEM), drop to a 3-fusion variant (residual + norm only, leave quant separate) — gives 1.03-1.08× instead of 1.05-1.15×. |
| **Gating verify** | Tier 3 isolated kernel replay (the v0.3.5 ladder applies). B-1/B-2/B-3 quality gate. E2E measurement on full agent task. |

---

### Round 5 — KV cache INT4 quantization

| Field | Value |
|---|---|
| **Edit surface** | (a) vLLM launch flag `--kv-cache-dtype int4`. (b) KV INT4 implementation in `csrc/quantization/kv_cache/` if not present. (c) Per-channel scale handling: `vllm/attention/ops/paged_attn.py` for INT4 dequantization on read. **KV cache is RUNTIME state, NOT model weights — this respects the v0.6 weight-immutability constraint.** |
| **Prior art** | KIVI (Liu et al., arXiv 2402.02750, uncertain). KVQuant (arXiv 2401.18079, uncertain). vLLM KV quantization docs. The agent reads KIVI specifically because it documents per-channel-vs-per-token scale tradeoffs that matter for INT4 KV. |
| **Tools** | (a) `scripts/microbench_kv_attention.py --kv-dtype int4` vs FP8 baseline. (b) Needle-in-haystack benchmark for KV precision validation. (c) NCU on attention with INT4 vs FP8 KV. |
| **Cheap self-verify** | (1) Compile check for INT4 KV path. (2) Attention output correctness: at FP8 KV, attention matches reference within tolerance; at INT4 KV, attention output drifts but stays within `rtol=1e-2`. (3) Needle test sanity at 16K context: recall ≥ 90% (looser than Round 3's 95% because INT4 KV inherently loses some precision). |
| **Time breakdown** | 0.5 day: vLLM INT4 KV enablement (if shipping) OR 2-3 days if implementing from KIVI. 0.5 day: scale calibration tuning. 0.5 day: measurement + needle test. **Total: 1.5-4 days depending on shipping support.** |
| **Pass criteria** | KV cache memory drops 4× vs FP8 KV (INT4 is 4 bits, FP8 is 8 bits → 2×; combined with int4 group-quant headers vs fp8 scales → ~4×). E2E decode tok/s improves ≥ 1.05×. Needle test recall within 5% of FP8 KV reference. B-1/B-2/B-3 within tolerance. |
| **Fail criteria** | Needle test recall drops > 5% vs FP8 KV. B-1 KL exceeds tolerance. INT4 KV implementation has correctness bugs (e.g., scale drift over long context). |
| **Move-on criteria** | If INT4 KV not in shipping vLLM and KIVI integration is too expensive, ship FP8 KV (already deployed, modest 1.0-1.05× win) and defer INT4 to a later cycle. |
| **Gating verify** | Full needle-in-haystack at 16K, 32K, 64K, 128K. B-1/B-2/B-3 quality gate stack. E2E measurement. |

---

### Round 6 — Math approximations (Newton-Raphson rsqrt, polynomial activation, fast RoPE)

| Field | Value |
|---|---|
| **Edit surface** | (a) Triton softmax kernel: `vllm/model_executor/layers/ops/softmax.py` — replace exact `1/sqrt(x)` with Newton-Raphson rsqrt. (b) Activation kernel `vllm/model_executor/layers/activation.py` — polynomial-approx GELU/SiLU. (c) RoPE kernel — fused / approximate cos/sin tables. **Each is a small targeted Triton mutation; the agent picks ONE per candidate.** |
| **Prior art** | FlashAttention's softmax tricks (Dao et al., online-softmax). Polynomial GELU approximations in JAX/PyTorch (e.g., `gelu_approximate='tanh'` is the std reference). Triton kernel optimization patterns. The agent reads FlashAttention-3 source for softmax-specific patterns. |
| **Tools** | (a) Triton `do_bench`. (b) Custom probe set for kernel-level correctness: `scripts/microbench_softmax.py`, `scripts/microbench_activation.py`. (c) NCU per-kernel. |
| **Cheap self-verify** | (1) Compile check. (2) Per-kernel correctness vs exact reference within `rtol=1e-3, atol=1e-3` on captured input tensors. (3) Per-kernel runtime improvement ≥ 5%. |
| **Time breakdown** | 1 day per math-target kernel. **Each round is 1 candidate per kernel; multiple kernel families can run as separate rounds.** |
| **Pass criteria** | Per-kernel speedup ≥ 5% confirmed via microbench. B-1 KL within tolerance. E2E decode tok/s improvement marginal (1.01-1.03× per round) but cumulative across kernels can hit 1.05-1.10× total. |
| **Fail criteria** | Numerical drift exceeds B-1 tolerance. Compile error. Approximation quality worse than expected (e.g., Newton-Raphson rsqrt converges too slowly for FP8 dynamic range). |
| **Move-on criteria** | If a particular math-approximation candidate doesn't move the needle (≥ 5% kernel speedup not achieved), abandon that kernel and try the next. |
| **Gating verify** | Tier 3 isolated kernel replay. B-1/B-2/B-3 quality gate. |

---

### Round 7+ — Async-copy / TMA + persistent kernels

| Field | Value |
|---|---|
| **Edit surface** | (a) Triton kernels with `cp.async` or TMA bulk-load patterns: `kernels/deltanet/*.py`, `kernels/gatedattn/*.py`, `kernels/fused/*.py`. (b) CUTLASS persistent kernel templates if shipping vendor path is mutable. (c) Cluster launch control (CLC) settings. **This is the lowest-leverage round per §0.6 — modest gains, deferred unless other rounds underdeliver.** |
| **Prior art** | CUTLASS persistent kernel examples (NVIDIA, Blackwell sm_100 examples 70+). Triton TMA documentation. FlashAttention-3 + 4 source for async-copy patterns on Blackwell. |
| **Tools** | NCU. Triton testing. CUTLASS profiler if vendor path is exercised. |
| **Cheap self-verify** | Compile + correctness microbench. NCU should show: increased DRAM concurrency, decreased pipeline stalls, increased TC utilization. |
| **Time breakdown** | 2-3 days per kernel target. |
| **Pass criteria** | Per-kernel speedup; sync correctness; B-1/B-2/B-3 within tolerance. |
| **Fail criteria** | Synchronization bugs (race conditions); numerical drift from incorrect cp.async ordering. |
| **Move-on criteria** | If per-kernel speedup < 3%, abandon; the cumulative gain is too small to justify the synchronization-correctness risk. |
| **Gating verify** | Tier 3 isolated kernel replay. B-1/B-2/B-3 quality gate. Long-running stress test for race conditions. |

---

### Common round mechanics (apply to all rounds)

**The agent's iteration_brief.md per-iteration includes:**
1. Round header (which round number, target, and the round's edit surface YAML).
2. P3a self-time breakdown + bandwidth-bound thesis (carries forward).
3. NCU diagnostic baseline distillation (carries forward; refreshed on weight rotation).
4. Last 3 winning_diffs.md entries (positive memory).
5. Last 5 mutations_rejected.tsv entries (negative memory, advisory not blocking).
6. Round-specific prior-art read list (URLs; the agent should read these before proposing).
7. Round-specific cheap-self-verify CLI list (the agent runs these BEFORE writing the patch).
8. Round-specific tools list (NCU, microbench scripts, etc.).
9. Required candidate metadata (speed_thesis, expected_affected_path, prior_failure_relation).

**Cheap self-verify is agent-owned, fast, and short-circuits.** If cheap self-verify fails, the agent writes BLOCKED.md and exits — the controller never spends gating-verify wallclock on a candidate the agent already knows is broken.

**Gating verify is controller-owned and expensive.** Once a candidate passes cheap self-verify, the controller runs the full B-1/B-2/B-3 + e2e measurement stack. Top-K (default K=3) results in winner promotion via human review per §"Quality gate stack" anti-gaming controls.

## Bookkeeping (artifacts and ledgers)

Each Track B round produces under `output/auto_research/track_b/<round_id>/`:

| Artifact | Format | Purpose |
|---|---|---|
| `round_spec.yaml` | YAML | round configuration: target, action surface, gate thresholds |
| `branch_log.json` | JSON | per-candidate git commit hashes + parent links |
| `candidate_analysis.md` (per candidate) | Markdown | speed thesis, expected counter, quality risk |
| `b1_distributional.json` (per candidate) | JSON | B-1 metric outputs |
| `b2_behavioral.json` (per top-K candidate) | JSON | B-2 benchmark deltas |
| `b3_full.json` (per pre-promotion winner) | JSON | B-3 full benchmark deltas + MAUVE |
| `winning_diffs.md` | Markdown | accepted candidates' diffs + scores (carries forward across rounds) |
| `mutations_rejected.tsv` | TSV | rejected candidates with structured reason |
| `quality_gate_history.tsv` | TSV | every (candidate, tier, score) tuple |
| `ncu_baseline_diagnostic.yaml` | YAML | one-shot NCU baseline (re-captured on weight rotation) |
| `ncu_winner_delta.yaml` (per accepted winner) | YAML | NCU counters of winner vs baseline |

## Halt conditions

- `preflight_edit_outside_allow_list` — agent attempted to edit deny-listed file.
- `preflight_speed_thesis_missing` — agent's candidate_analysis.md missing or unparseable.
- `compile_failure` — patch doesn't build.
- `b1_distributional_fail` — quality drift exceeds B-1 thresholds.
- `b2_behavioral_fail` — behavioral regression exceeds B-2 thresholds.
- `b3_full_fail` — full benchmark regression exceeds B-3 thresholds.
- `e2e_speedup_below_threshold` — passes quality but doesn't move throughput; not promoted but kept as null result.
- `quality_gate_runtime_unavailable` — lm-eval-harness or judge model unavailable; round halts pending fix.
- `proposer_stuck_after_canary_round` — three consecutive candidates fail at the same tier with the same family.
- `proposer_reward_hacking_suspected` — candidate's B-1 distributional check passes but its diff resembles a known-hack pattern (e.g., output dropping, constant-folding); flagged for human review.

## Open questions and risks

### Q1: Does vLLM's speculative decoding path on Blackwell sm_120 work end-to-end with Qwen 3.x 27B targets?

**Status:** uncertain. vLLM has speculative decoding support (`vllm/spec_decode/`), but its maturity varies across model architectures and Blackwell support is still landing as of early 2026. Specifically:

1. Does Eagle-3 work as a draft method for Qwen 3.x 27B? Eagle-3 typically requires a published trained draft head — does HuggingFace have one for the chosen target?
2. If no Eagle-3 head exists, does self-speculation work — using a smaller Qwen 3.x checkpoint (e.g., 1.5B FP8) as the draft?
3. Are draft acceptance rates reasonable on the workload's actual prompts (target ≥ 70% acceptance for 1.5×+ effective speedup)?
4. Does vLLM's spec_decode path work correctly on hybrid-attention models (DeltaNet + GatedAttn) — or only on standard transformer architectures?

The audit (Bring-up step 1) answers these. **If the answer to #4 is "no", speculative decoding's bring-up cost is much higher** (vLLM-side code changes needed), and Round 1 may need to fall back to fused epilogues (Round 2 candidate) as the actual first target.

### Q1b: Model selection sub-question

If Qwen 3.6 27B FP8 is the chosen model (per §"Model selection"), does Qwen 3.6's hybrid attention structure differ from 3.5's (16 GatedAttn + 48 DeltaNet layout)? The audit checks this. Hybrid attention vs vanilla transformer matters for both speculative decoding (Q1#4 above) AND for fused epilogue placement (Round 2 target).

### Q2: Where do the B-1 reference logits come from?

The §2.2.0 reference baseline (forced FA3 + CUTLASS FP8 + Triton DeltaNet defaults) is the canonical reference per `l0-ffn-gemm-pivot-20260502.md`. B-1 captures full logits at 64 prompts × 256 positions against this baseline. Re-captured on weight rotation only.

### Q3: How many quality gate runtime hours per round?

Per-candidate: B-1 ~3 min average. Per top-K=3: B-2 ~30 min × 3 = 90 min. Per pre-promotion winner: B-3 ~60 min. For a typical 10-candidate round: ~30 min B-1 + ~90 min B-2 + ~60 min B-3 = ~3 hours of pure gate runtime, plus ~25 min e2e measurement on top-K. Total round: ~4-5 hours. Workable.

### Q4: Reward-hacking risk magnitude?

Real but bounded. Multi-metric gate + held-out probes + human review on accepted candidates is the published mitigation set. Sakana's failures all happened at single-metric gates with proposer-readable test sets. Track B avoids both.

### Q5: What if the agent can't navigate the multi-file edit surface?

This is the biggest open implementation question. Codex and Claude Code handle multi-file edits, but the agent needs to understand the dispatch path, not just edit one file. **Mitigation:** the iteration brief explicitly lists the dispatch path the candidate must respect, and the agent MAY use additional tools (read other files, run profiler) before producing its diff. If the agent struggles, narrow the surface back to fewer files at a time and accept smaller wins per candidate.

### Q6: Track B priority order under both constraints (weight-immutable + Codex agent workload)

In rough order of expected magnitude × engineering cost. **Excludes all weight-modifying mutations per v0.6 constraint** AND **prioritizes agent-workload-specific wins per v0.7 update**:

| Round | Target | Phase attacked | Expected | Agent-workload-specific? |
|---|---|---|---|---|
| **0** | Prefix caching tuning + LMCache CPU/disk tier | Prefill | **2-10×** on cache-hit turns | ✅ Codex's heavy prefix reuse + DGX Spark unified-memory advantage |
| **1** | Eagle-3 + PLD hybrid speculative decoding | Decode | 1.5-2.5× | ✅ PLD dominates on edit/echo; Eagle-3 on reasoning |
| **2** | XGrammar constrained generation for tool calls | Decode (tool-call turns) | 1.2-1.6× | ✅ Codex tool-call workload |
| **3** | Quest-style page attention on GatedAttn-only layers | Decode (long-context only) | 2-5× at >32K ctx | ✅ Codex long-context (10K-100K+ tokens) |
| **4** | Fused epilogues (Triton sidecar) | Decode | 1.05-1.15× | ⚪ Generic |
| **5** | KV cache INT4 quantization | Decode | 1.05-1.15× | ⚪ Generic |
| **6** | Math approximations (Newton-Raphson rsqrt etc.) | Decode | 1.03-1.08× | ⚪ Generic |
| **7+** | Async-copy / TMA + persistent kernels | Decode | 1.03-1.08× | ⚪ Generic |

**Explicitly NOT recommended for agent workloads** (despite published wins on other workloads):
- ❌ **StreamingLLM / Attention Sinks** — agents need to retrieve from arbitrary positions in tool output; sliding-window eviction breaks this. Behavioral drift risk too high.
- ❌ **SnapKV / PyramidKV / H2O score-based KV eviction** — same problem; eviction can drop the file contents the agent needs three turns later. Score-based heuristics underestimate retrieval-heavy agent patterns. Behavioral drift risk too high.
- ❌ **REST retrieval-based speculative decoding** — once context has accumulated (which is most of the agent run), PLD already wins on the same patterns. REST only helps cold-start.

**Excluded from priority order per the v0.6 constraint** (these change weights):
- ❌ NVFP4 / FP4 / INT4 weight conversion (different model)
- ❌ GPTQ / AWQ weight-only quantization (different model)
- ❌ 2:4 structured sparsity (forces 50% weight zeros = pruning = model modification)

**Combined target after Rounds 0-3 (the four agent-specific wins): 3-5× e2e on cache-hit turns, 1.8-2.5× on cache-miss turns. After all rounds: same ceiling, marginal improvement.** The agent-workload-specific targets dominate; the generic targets are diminishing returns.

Each in-scope target re-uses the same Track B infrastructure; only the bring-up steps 1 and 6 are target-specific.

## Implementation sequence summary

| Step | Output | Dependency | Estimated work |
|---|---|---|---|
| **0. Qwen 3.6 27B FP8 availability audit** | `output/qwen36_availability_audit.md`; selected model name + HF path; tokenizer-compatibility verdict; if architecture differs from 3.5, P3a re-run flag | none | half-day |
| 0a. (conditional) Re-capture seed_trace_v5 against the chosen model if tokenizer differs | `benchmark_blueprints/families/responses-sdk-adapter-cutover/seed_trace_v5.jsonl` (re-captured) | Step 0 | ~30 min capture |
| 0b. (conditional) Re-run P3a in-process timing if architecture differs from Qwen 3.5 | `output/p3a_qwen36_27b_fp8.md` with new self-time breakdown | Step 0a if applicable | ~30 min |
| **1. Round 0 — Prefix caching audit + tuning** | `output/prefix_cache_audit.md`; vLLM launch flags tuned (`--enable-prefix-caching`, `--block-size 32`, `--enable-chunked-prefill`); LMCache integration; DeltaNet state-cache verification; cache-hit-rate measurement on representative agent traces | Step 0 (model fixed) | 1-2 days |
| 2. Speculative decoding audit of vLLM | `output/spec_decode_audit.md` (vLLM spec_decode path support, draft-model availability for chosen target, PLD support, Eagle-3 + PLD compose mechanism) | Step 0 (model fixed) | 1-2 days |
| 3. Build quality fixtures (B-1, B-2, B-3) | `quality_fixture/` directory + reference logit captures against prefix-cache-tuned baseline | §2.2.0 reference baseline + chosen model + Round 0 prefix cache tuning | 1-2 days |
| 4. Build edit-surface enforcement | `auto_research/edit_surface_check.py` | none | 1 day |
| 5. Build git-branch round controller | extend `auto_research.py` for Track B branch model | quality fixtures + edit surface | 2 days |
| 6. NCU diagnostic baseline capture | `ncu_baseline_diagnostic.yaml` against prefix-cache-tuned baseline | Nsight tooling on Spark + chosen model + Round 0 baseline | 1 day |
| **7. Track B Round 1 — Eagle-3 + PLD hybrid speculative decoding bring-up** | round artifacts + winning diff if accepted; draft acceptance rate measured for both Eagle-3 and PLD components | all of 0-6 | few hours |
| 8. Round 1 closeout + recommendations for Round 2 | report at `docs/reports/auto_research/track-b-round-1-closeout.md` | Round 1 complete | 1 day |
| 9. Round 2 — XGrammar constrained generation | round artifacts measuring tool-call decode speedup | Round 1 baseline established | 1-2 days |
| 10. Round 3 — Quest-style page attention on GatedAttn-only layers (long-context optimization) | round artifacts measuring long-context decode speedup | Round 2 baseline | 3-5 days (custom kernel work) |
| 11+. Subsequent rounds (fused epilogues, KV INT4, math approximations) | per-round artifacts | prior round baseline | varies |

Total prerequisite work before Round 1: ~5-7 days of engineering (~1 additional day if Qwen 3.6 needs re-capture / re-P3a). Round 1 itself: hours.

**Step 0 specifics (the new prerequisite):** the audit must answer:
- Is `Qwen/Qwen3.6-27B-FP8` (or equivalent naming) published on HuggingFace as a dense hybrid-attention model with FP8 quantization?
- If not, is there an equivalent in the Qwen 3.6 line at 27B parameter count?
- Does its tokenizer match the one used in the existing seed_trace_v5.jsonl?
- Does its layer architecture match Qwen 3.5's (16 GatedAttn + 48 DeltaNet) or differ?
- Does vLLM's current shipping version support it without code changes?

**If Step 0 fails to find Qwen 3.6 27B FP8**: the spec falls back to Qwen 3.5 27B FP8 unchanged. Track B is robust to this — the bandwidth-bound thesis, the FFN GEMM dominance, and the verified 1.29× NVFP4 measurement all hold for either model.

## References

| Source | Status | Relevance |
|---|---|---|
| `l0-ffn-gemm-pivot-20260502.md` | This repo | Track A architecture; Tier 3 isolated replay design; trajectory-derived probe capture |
| `l0c-evaluation-ladder-and-memory-prior-art-20260430.md` | This repo | Hard-vs-soft constraint memory model; positive/negative memory pattern |
| `l0c-fp8-cutlass-round-20260505-closeout.md` | This repo | Track A exhaustion evidence; bandwidth utilization measurement |
| [NVIDIA forum: FP4 on DGX Spark — Why It Doesn't Scale](https://forums.developer.nvidia.com/t/fp4-on-dgx-spark-why-it-doesnt-scale-like-youd-expect/360142) | Verified | NVFP4 1.29× measurement; sm_121 hardware limits explanation |
| [LMSYS DGX Spark Review](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/) | Verified | Bandwidth utilization context; SGLang FP8 numbers |
| [rikkarth Qwen3.6-35B-A3B-FP8 vLLM benchmark](https://rikkarth.com/blog/2026-04-23-benchmark-results-for-qwen-qwen3-6-35b-a3b-fp8-nvidia-dgx-spark-gb10-serving-via-vllm) | Verified | Concurrency curves, bandwidth realization analysis |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Verified | Standard benchmark infrastructure |
| MAUVE (Pillutla et al., NeurIPS 2021) | Verified, arXiv 2102.01454 | Distribution-similarity metric for B-3 |
| GPTQ (Frantar et al.) | Verified, arXiv 2210.17323 | Quantization gate methodology |
| AWQ (Lin et al.) | Verified, arXiv 2306.00978 | Calibration-set construction; per-channel scale rationale |
| Sakana CUDA Engineer retraction | Verified | Reward-hacking cautionary tale; multi-metric mitigation rationale |
| AlphaEvolve / FunSearch | Verified | Evolutionary code search gate patterns; positive-memory architecture |

---

*This is the design target for Track B mutation. Implementation order in §"Implementation sequence summary". Track A continues as research-signal track; Track B is the primary throughput initiative going forward.*
