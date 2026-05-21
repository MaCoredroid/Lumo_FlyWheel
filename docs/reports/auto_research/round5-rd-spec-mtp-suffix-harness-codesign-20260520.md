# Round 5 R&D Spec — MTP + SuffixDecoding Hybrid, Codex Harness Co-Design, and Auto-Completion Substrate

**Generated:** 2026-05-20
**Audience:** Track B team + broader inference R&D
**Status:** Forward-looking spec. None of the implementations described here have shipped. Builds on Round 4b ablation findings (T1 alone is the load-bearing technique; Q36-A is now the leader at 22.46 tps; sqlalchemy 4/4 effect on Q35-D does not replicate on Q36-D).
**Constraint:** stay under FP8 weights, Qwen 3.6 27B as base model, FP8 KV cache.

**Lossless vs lossy taxonomy (load-bearing for what we measure):**

- **Drafter-side changes (Paths 1, 2, 3, and most of Path 4) are lossless by construction.** Speculative decoding with rejection sampling preserves the target model's output distribution exactly — *any* drafter is valid as long as the verifier checks every draft token against the target model's logits. The published rejection sampling theorem (Leviathan et al., 2023; Chen et al., 2023) guarantees this. Changing data structures (DAWG vs suffix tree), changing the proposer (MTP vs SuffixDecoding), changing the ranking function (frequency vs recency), changing the candidate set (typed vs untyped) — none of these alter the output distribution. The verifier ensures distribution fidelity; the drafter only controls *speed*. These changes need only B-1/B-2/B-3 distribution-equivalence gates (KL divergence on a fixed prompt set, byte-exact match under greedy decoding), not full benchmark re-runs.
- **Harness-side changes (a subset of Path 4) are NOT lossless.** If Codex CLI changes turn segmentation, sampling parameters, context truncation, tool-schema rendering, reasoning preservation, or anything else that alters the *input to the model* or the *content sent back to the harness*, the output distribution can shift and pass rates can move. These changes need full benchmark re-runs against external corpora to demonstrate no quality regression.

We need both gates. Our internal v4a_v2 corpus (11 tasks) is too small to give external audiences confidence on the lossy-change side. **External quality gating: SWE-Bench Verified + SWE-Bench Pro before-and-after on every lossy Path 4 sub-change.** See §10.

---

## 1. Why this spec

Round 4b on Qwen 3.5 27B FP8 demonstrated:

- T1 (SuffixDecoding extension) carries the spec-decode win (3.26× over OFF).
- T2/T3/T4 layered on top of T1 contribute small, mixed, and per-task-noisy effects.
- The shipping pick depends on which model: Qwen 3.5 → D (full stack) was best (driven by a single-task sqlalchemy effect); Qwen 3.6 → A (T1 only) is best at +23% throughput, with the sqlalchemy effect absent.

The natural Round 5 questions:

1. **Can MTP + SuffixDecoding compose into something faster than either alone?** Qwen 3.6 ships with a native MTP head we haven't used.
2. **Can we co-design SuffixDecoding around the MTP head and Codex harness state — beyond the current bolt-on architecture?**
3. **Since Codex CLI is open source, can we co-design the harness side too** — not just the inference side?

This spec lays out four parallel paths: a tight τ-threshold hybrid (lowest effort), a per-frame drafter mixture (medium effort, biggest published-art jump), an auto-completion-substrate rework (R&D bet), and a Codex CLI fork with explicit harness-oracle protocol (largest scope, most upside).

---

## 2. State of the published art (as of 2026-05-20)

| Work | Relevance | Lesson |
|---|---|---|
| **SuffixDecoding** (He et al., NeurIPS 2025, arXiv 2411.04975) | What we already run as T1 | Defines τ-threshold hybrid pattern: SuffixDecoding primary, model-based fallback when score < τ. Reports +2.5× on Spec-Bench using SD+EAGLE-3 hybrid. **τ ≈ MAT of fallback** is the published calibration rule. Paper does not combine with MTP. |
| **AgentInfer / AgentSAM** (arXiv 2512.18337, Feb 2026) | Closest agentic co-design paper | Uses **suffix automaton (DAWG)** instead of suffix tree — ~50% memory at same recall. Designed for "reuse token-level predictions across agent sessions." Validated on Deep Research Agents; principle transfers to Codex. |
| **SAM Decoding** (arXiv 2411.10666) | DAWG substrate details | Documents suffix automaton substitution with comparable hit rate. |
| **Gemma 4 MTP drafters** (Google AI, May 2026) | Production MTP at scale | 3× speedup at zero quality loss with small MTP heads. Trained for cross-step coherence. Implies MTP-as-drafter generalizes well. |
| **FastMTP** (arXiv 2509.18362) | Better MTP training | Position-shared MTP head + draft-tree organization beats per-step independent MTP heads. Conceptual sibling to Qwen 3.6's MTP layer. |
| **Speculative Search** (arXiv 2511.20048) | Search-agent co-design | Algorithm+system co-design for LLM search agents. Defines the cross-layer protocol pattern we already use (oracle API). |
| **vLLM Issue #40831** | Compatibility flag | TurboQuant KV × any spec_decode produces degenerate token loops. We're FP8 KV (unaffected) — flag if quant changes. |
| **vLLM Issue #41190** | Compatibility flag | TP=2 + `qwen3_next_mtp` crashes on hybrid-GDN Qwen 3.6. We're TP=1, unaffected. |

### Two specific findings to base the design on

1. **τ-threshold pattern is well-validated.** Plug in *any* model-based fallback; the only parameter to tune is τ. Swapping EAGLE-3 for MTP in that slot is open territory — no published paper has done this yet.
2. **Suffix automaton (DAWG) over suffix tree is the right substrate.** Same recall, ~50% memory, and Codex agents create the kind of repetition where DAWG compresses well.

---

## 3. Path 1 — MTP + SuffixDecoding τ-threshold hybrid

**Loss classification: LOSSLESS by construction.** Both proposers feed into the same rejection-sampling verifier. Output distribution is identical to base decode. Quality gate is B-1/B-2/B-3 distribution-equivalence only; no SWE-Bench re-run required (other than as a sanity check on implementation correctness).

### Goal

Make decode faster and acceptance higher by falling through to Qwen 3.6's native MTP head when SuffixDecoding's pattern match is weak.

### Implementation sketch

```
SuffixDecodingProposer.propose(request):
    suffix_candidates = self._suffix_propose(request)
    score = self._compute_score(suffix_candidates, request)
    if score >= self.tau:
        return suffix_candidates                       # SD primary path
    return self._mtp_fallback.propose(request)          # MTP fallback
```

Concretely:

1. **Patch `arctic_inference.SuffixDecodingProposer.propose`** — already prelaunch-patched by us for T1's per-session router. Adding the fallback branch is a single new call site after score computation.
2. **Reuse vLLM's existing `Qwen3NextMTPProposer`** — already implemented and tested for Qwen 3.6. Inject as `self._mtp_fallback` at proposer init.
3. **τ calibration:** start with τ ≈ MTP's expected mean accepted tokens. From the llama.cpp Qwen 3.5 MTP experiment, acceptance ≈ 0.475 × num_speculative_tokens=1 = ≈ 0.475 MAT. For Qwen 3.6 at num_speculative_tokens=2 (officially supported), expect MAT ≈ 1.0. So τ ≈ 1.0 is the starting calibration. Sweep τ ∈ {0, 1, 2, 3, 5} on a 3-task slice to find corpus-optimal.

### Why this should help on Qwen 3.6 specifically

| Observation | Implication for hybrid |
|---|---|
| Q36-A acceptance = 0.548 (vs Q35-A 0.512) | Q36 output is more SD-friendly; SD picks up more tokens before falling through |
| Q36 MTP head ~47.5% acceptance per the llama.cpp experiment | When SD fails (mid-reasoning chunks, novel content), MTP fills the gap |
| The two are largely orthogonal | SD wins on structural tokens (tool-call envelopes, file paths); MTP wins on freshly-generated reasoning |

### Expected outcome

At the published SD+EAGLE-3 hybrid gain (+0.1× over SD alone on mixed workloads), Q36-A could lift from 22.46 → ~25 tps median. On harder reasoning tasks (transcript-merge, multi-tool), the win could be larger. Worst case: it matches SD alone at τ=0.

### Engineering effort

3–5 days. We already own the SuffixDecodingProposer patch path; this adds one fallback call site + τ configuration env var.

### Risks

- **Acceptance does not compose linearly.** SD's score might be the wrong proxy for "MTP will do better here." Likely refinement: try multiple score functions (suffix-length, leaf-frequency, posterior-prob).
- **Per-call latency overhead.** Running MTP fallback adds ~5-10ms per fallback call. If most calls fall through, the hybrid could be slower than SD alone. Mitigation: warm-start MTP only when SD score is already approaching τ.

---

## 4. Path 2 — Per-frame drafter mixture with harness oracle bias

**Loss classification: LOSSLESS by construction.** Per-position mixture changes *which* candidates the drafter proposes, but every candidate is still rejection-sampled against the target model. Output distribution preserved. Same B-1/B-2/B-3 gate as Path 1.

### Goal

Move from "SD or MTP" to a per-token-position mixture, where the harness oracle steers the mixing weight based on what kind of frame is being generated.

### Three principles

**A. Treat the draft tree as a learned mixture, not a fixed proposer.** Compute both candidate sets and merge by per-position confidence:

```
For each token position i in the draft window:
  s_i  = SD's confidence at position i  (suffix-match count)
  m_i  = MTP's confidence at position i (head logit margin)
  oracle_bias_i = harness_oracle(position_i, recent_tokens)

  P(token | pos_i) = α(oracle_bias_i) · P_SD(token) + (1-α) · P_MTP(token)
```

**B. Harness-oracle-driven per-frame regime.** The Codex harness already emits signals via `lumo_oracle_registry`. Extend with regime-class hints:

| Harness signal | Regime | Recommended α (SD weight) |
|---|---|---:|
| `tool_call_started` | structural | 0.9 (SD-heavy; schema-bound) |
| `tool_arguments_freetext` | hybrid | 0.5 (balanced) |
| `reasoning_block_started` | semantic | 0.2 (MTP-heavy; novel reasoning) |
| `file_path_emission` | structural | 1.0 (SD-only; paths repeat) |
| `plan_emission` | structural | T4 pre-drafter primary, SD secondary |

**C. MTP-aware suffix tree pruning.** The suffix tree should *avoid* indexing tokens MTP can predict well anyway. Inverse of the paper's "fallback to MTP when SD is weak":

- During training of priority weights, track per-token "SD-unique-gain" = SD acceptance − MTP-counterfactual acceptance.
- Tokens with low unique-gain get evicted from the suffix tree first, freeing memory for tokens MTP can't help with.

### Engineering effort

2–4 weeks. New protocol fields in the oracle API, new regime-class detector in the harness, MTP-aware pruning in SuffixDecodingProposer.

### Expected outcome

Larger than Path 1 — per-frame regime steering could yield 15-30% additional acceptance on the frames where one proposer's prediction is much better than the other. The biggest win comes from removing the redundant "both proposers waste compute on the same easy tokens" overhead.

---

## 5. Path 3 — Auto-completion substrate (R&D bet)

**Loss classification: LOSSLESS by construction.** All substrate substitutions (DAWG, Patricia trie, FM-index, etc.) change *how* candidates are looked up but not *which* tokens the verifier accepts. Ranking changes (time-decay, type-aware, multi-source merge) change candidate ordering and inclusion in the draft set; rejection sampling still corrects to the target distribution. Same B-1/B-2/B-3 gate as Paths 1 and 2.

The drafter is effectively an auto-completion engine. Editor auto-completion has 30+ years of engineering tricks that the spec-decode literature hasn't borrowed. Treating it that way unlocks several upgrades.

### Substrate substitutions

| Structure | What it gives us | Tradeoff |
|---|---|---|
| **DAWG / suffix automaton** | Same recall as suffix tree, ~50% memory | More complex updates; build cost |
| **Patricia / radix trie** | Compact prefix matching on identifiers | Less repetitive-pattern friendly than suffix |
| **FM-index (compressed SA)** | Sub-linear search over very large corpora | Read-only; rebuild needed on updates |
| **Learned index over n-gram** | LSP-style "smart completion" via small model | Adds inference cost, but small head amortizes |
| **Bloom filter on negative cache** | Skip lookups that we know won't hit | Saves CPU; doesn't improve hits |
| **Skip list of ranked candidates** | O(log n) ranked retrieval | Replaces our current greedy-leaf approach |

### Ranking ideas from auto-completion

1. **Time-decay scoring.** Editor completion weights recently-typed tokens higher. We currently treat all turns equally. Add per-turn decay factor — patterns from the last 5 turns get weighted higher than patterns from turn 50. Codex editing is *intensely* recency-biased.
2. **Multi-source merge with provenance.** Editor completion merges file-local + project-wide + language built-ins + LSP semantic. Maps to T1 (session) + T2 (file content) + T3 (schema) + T4 (plan). We currently *layer* them as parallel proposers — a unified merged candidate set with provenance tags lets the drafter pick the best source per-token, not per-frame.
3. **Edit-distance / fuzzy matching.** When SD's exact-suffix match fails, fall back to Levenshtein-1 fuzzy matches before going to MTP. Cheap CPU, captures typo-equivalent patterns.
4. **Type-aware filtering.** Tool-call argument values have known JSON-schema types. The suffix tree should index candidates BY type, and lookups should filter by expected type for the current parameter. T3 schema-aware idea extended into the substrate.
5. **Frequency × recency Bayesian prior.** Standard editor ranking: `score = log(frequency) + λ · recency_bonus`. Our current picker is roughly frequency-only. Adding recency improves hit-rate on Codex's "edit recently-touched file" pattern.

### The deepest idea unlocked by this framing

If drafter = auto-completion, then the model is generating into a "search box" with predicted completions. The auto-completion world has invented techniques for:

- **Typo correction** — when the model produces a slightly-off-from-expected token, can we still match a suffix? (Levenshtein-bounded suffix lookup)
- **Natural-language ranking** — BM25-style relevance scoring beyond raw frequency
- **Learn-to-rank** — use the harness oracle as features for a small ranker on top of the candidate set

None of these are in the spec-decode literature. This is genuine R&D territory.

### Engineering effort

DAWG swap alone: ~1 week, +50% memory headroom for free. Time-decay + type-aware filtering: ~2 weeks, +10–20% acceptance on relevant frames. Multi-source provenance merge: ~3–4 weeks, ambitious — could net +30% on top of T1 or could regress if the ranker is wrong.

### Why this is a real publishable contribution

The spec-decode literature has fixated on suffix tree vs neural drafter vs ngram. Borrowing from auto-completion engineering at the data-structure level is open territory. If this works, write it up properly with ablations against published baselines.

---

## 6. Path 4 — Codex CLI fork with explicit harness-oracle protocol

**Loss classification: MIXED. Split per sub-change.** This is the load-bearing distinction for Round 5 quality measurement. Sub-changes are either lossless (only affect the drafter; rejection sampling preserves distribution) or potentially lossy (change what the model sees or how output is rendered back to the harness; can alter pass-rate behavior). Every potentially-lossy sub-change is gated by SWE-Bench Verified + SWE-Bench Pro before-and-after (§10).

| Sub-change (§6.x) | What it changes | Loss class | Quality gate |
|---|---|---|---|
| §6.1 Turn-boundary signals | Drafter lifecycle / session-tree pruning | **Lossless** | B-1/B-2/B-3 only |
| §6.2 Per-request regime hint | Drafter proposer-mix α | **Lossless** | B-1/B-2/B-3 only |
| §6.3 Hot-path identifier registration | Drafter cache priority | **Lossless** | B-1/B-2/B-3 only |
| §6.4 Stream-side priming | Drafter content (suffix tree input) | **Lossless** | B-1/B-2/B-3 only |
| §6.5 Bidirectional oracle (predict_next) | Drafter only IF harness doesn't act on predictions | **Lossless if read-only**; lossy if Codex changes behavior based on predictions | Lossless variant: B-1/B-2/B-3 only. Lossy variant: SWE-Bench gate |
| §6.6 Regime opt-out | Drafter selection per frame | **Lossless** (no draft = base decode, distribution unchanged) | B-1/B-2/B-3 only |
| §6.7 Pre-emit token budget hints | Drafter `num_speculative_tokens` per call | **Lossless** (rejection sampling holds for any draft length) | B-1/B-2/B-3 only |
| §6.8 Sampling parameter overrides (`temperature`, `top_p`, `top_k`) | **Model input distribution** | **Lossy** | SWE-Bench gate required |
| §6.9 Context window management (truncation, summarization) | **Model input content** | **Lossy** | SWE-Bench gate required |
| §6.10 Tool-schema rendering changes | **System prompt content** | **Lossy** | SWE-Bench gate required |
| §6.11 Reasoning preservation (`preserve_thinking`, interleaved-thinking) | **Model input content (prior turn thinking blocks)** | **Lossy** | SWE-Bench gate required |
| §6.12 Per-request `max_tokens` caps | Truncates output | **Lossy if cap binds**; lossless if cap doesn't bind | Conditional SWE-Bench gate |
| §6.13 Tool-execution sandbox / approval-flow changes | Behavior at the agent loop level | **Lossy** | SWE-Bench gate required |

The first 7 sub-changes (§6.1-6.7) are pure drafter inputs — same correctness guarantees as Paths 1, 2, 3. Ship them under the B-1/B-2/B-3 gate alone. The last 6 (§6.8-6.13) require external benchmark re-runs.

### Why this is newly available

Codex CLI is open source. We've been treating it as a closed wire-format dependency, inferring harness state from API events (proxy-side regime detection, session-id heuristics). That's the wrong contract for a Track B that wants to push the harness/inference boundary. **A fork lets us push state explicitly from the harness instead of inferring it from the wire.**

### What Codex CLI currently does that we can't change without forking

- Stream-based SSE response parsing
- Reasoning summary handling (`model_supports_reasoning_summaries` toggle)
- Turn boundary detection — we currently infer this from API events at the proxy
- Context window management — Codex decides when to truncate / summarize
- Tool schema sent per request — we get it via the request envelope, not as a session-bound declaration
- Sampling parameters — Codex chooses temperature/top_p (we recently added a proxy override, but it's a workaround)

### What we could change in a Codex fork

#### 6.1 Explicit harness-oracle signals at turn boundaries

Currently we *infer* turn boundaries from `previous_response_id` and request shape. A fork would emit explicit signals:

```jsonc
// New: sidecar JSON-RPC channel alongside the responses API stream
{ "event": "session_open",     "session_id": "sess_abc", "task_meta": {...} }
{ "event": "turn_open",        "turn_index": 0 }
{ "event": "tool_call_started", "tool_name": "read_file", "expected_args": {...} }
{ "event": "tool_call_finished","tool_name": "read_file" }
{ "event": "tool_result_received", "tool_name": "read_file", "content_hash": "sha256:..." }
{ "event": "reasoning_block_started" }
{ "event": "plan_emission_likely" }   // heuristic from Codex's planning state
{ "event": "turn_close",       "turn_index": 0 }
{ "event": "session_close",    "session_id": "sess_abc" }
```

This is exactly T5 (the lifecycle hooks we deferred in Round 4b). Becomes implementable once we own the harness.

#### 6.2 Per-request regime hint

Codex has client-side knowledge of "what kind of frame I'm about to ask for." Send it:

```jsonc
POST /v1/responses
{
  "model": "qwen3.6-27b",
  "messages": [...],
  "x-lumo-frame-hint": {
    "regime": "tool_call_emission",  // structural | hybrid | semantic | plan
    "expected_schema": "read_file",
    "predicted_completion_length": "short",  // short | medium | long
    "preserve_thinking": false               // override per-request
  }
}
```

The drafter consumes the hint to pick proposer mix (Path 2's α modulation).

#### 6.3 Hot-path identifier registration

Codex knows which strings are *project-locally* repeated — function names, file paths, framework identifiers. Currently we wait for the suffix tree to learn them. A fork can pre-register:

```jsonc
{ "event": "register_hot_paths", "paths": ["src/incident_handoff/adapter.py", "process_responses_event", "ApplyPatchTool"] }
```

vLLM keeps these in the suffix tree even if their frequency drops below the eviction threshold.

#### 6.4 Stream-side priming (T2 done right)

When `read_file` returns content, currently we have to wait for the model to reference it. Codex could push it to the drafter *before the next inference call*:

```jsonc
{ "event": "prime_drafter", "source_tag": "file:src/foo.py", "content": "...", "ttl_turns": 10 }
```

This is T2 (read_file priming) but with explicit harness signaling instead of proxy-side inference. Lower latency, deterministic, no inference-time decision.

#### 6.5 Bidirectional oracle — pull predictions back from vLLM

Currently the oracle protocol is one-way: harness pushes state into vLLM. A fork can pull predictions back:

```jsonc
GET /v1/lumo-oracle/predict_next?session_id=...&recent_tokens=...&n=10
→ { "candidates": ["foo_bar", "process_event", "..."] }
```

Codex can use this for:
- Speculative pre-rendering — start drawing the next likely UI before the model emits it
- Confidence display — show the user a probability bar on the next token
- Early-termination — if the model's next-token distribution is too uncertain, bail out and re-prompt

#### 6.6 Regime opt-out

For frames where spec_decode hurts (e.g., creative writing in docstrings, novel reasoning chunks where SD has no patterns), let Codex opt out of speculation for that frame:

```jsonc
{ "x-lumo-spec-decode": "off" }   // skip drafting entirely for this request
{ "x-lumo-spec-decode": "mtp_only" }   // only MTP, no SD
{ "x-lumo-spec-decode": "sd_only" }    // only SD, no MTP fallback
```

The harness knows when speculation will fail. Tell it instead of guessing.

#### 6.7 Pre-emit token budget hints

Codex can predict roughly how long the next emission will be (short tool call vs long reasoning chunk vs long file edit). Tell vLLM so it can right-size the spec_decode buffer:

```jsonc
{ "x-lumo-budget-hint": { "completion_tokens_estimate": 50, "thinking_tokens_estimate": 200 } }
```

vLLM uses this to allocate spec_decode draft buffer (`num_speculative_tokens` per call could be dynamic).

### What that unlocks

| Without fork (current) | With fork |
|---|---|
| Proxy guesses turn boundaries from API events | Explicit `turn_open` / `turn_close` events |
| Suffix tree learns hot paths organically | Pre-registered hot paths persist across cache eviction |
| T2 priming triggered by inference-time pattern detection | T2 priming triggered by Codex on file-read completion (deterministic, lower latency) |
| Drafter picks proposer mix from no signal | Per-request regime hint chooses α directly |
| Spec_decode either on for all calls or off | Per-frame opt-in/opt-out (saves wasted draft compute) |
| One-way oracle (harness → vLLM) | Bidirectional (Codex can query vLLM for predictions) |

### Engineering effort

3–6 weeks for a clean fork:
- Fork Codex CLI repo, set up rebase pipeline to track upstream
- Implement the sidecar JSON-RPC channel for harness-oracle events (~1 week)
- Add per-request regime hint field + client-side regime detector (~1 week)
- Add hot-path registration API + push-priming + bidirectional oracle (~2 weeks)
- Wire it into our vLLM oracle registry (already exists; new fields only) (~1 week)
- Integration testing on the v4a_v2 corpus (~1 week)

### Risks of forking a maintained tool

- **Upstream divergence.** Codex CLI gets updates from OpenAI. Keep diffs small and contribute upstream where possible (the harness-oracle protocol could go upstream as a vendor-neutral extension).
- **Maintenance burden.** Each upstream rebase needs CI to verify the harness-oracle protocol still works against vLLM.
- **Compatibility.** If OpenAI ships an incompatible Codex CLI update, we may be stuck on an old version. Mitigation: keep the fork small enough to forward-port quickly.

### Why this is the highest-leverage path

The published spec-decode literature is bound by what the inference layer can do *given an opaque agent harness*. Open-sourcing the harness and pushing state explicitly across the boundary is the change that unlocks the next 2-3× of optimization headroom. Every other path here (1, 2, 3) is bottlenecked by what we can infer at the proxy; a harness fork removes that bottleneck.

This is also the only path that turns into a real ecosystem contribution. A vendor-neutral "harness-oracle protocol" specification — pushable upstream to Codex CLI, Claude Code, Aider, etc. — would be a meaningful piece of infrastructure for the entire LLM-agent serving stack.

---

## 7. Recommended sequence

| Order | Path | Effort | Loss class | Quality gate | Expected gain |
|---|---|---|---|---|---|
| 0a | Finish Q36-D measurement (7 tasks pending) | ~3 days | n/a | n/a | Baseline complete |
| 0b | **Baseline SWE-Bench reproduction** — Verified + Pro on Qwen 3.6-27B FP8 D-point | ~150 wall-hrs | n/a | published Qwen baseline ±2 | Establishes our stack's reference number |
| 1 | **Path 1** — τ-threshold hybrid (SD + MTP) | ~1 week | Lossless | B-1/B-2/B-3 only | +10–15% tps over Q36-A |
| 2 | **Path 3** — DAWG substrate swap | ~1 week | Lossless | B-1/B-2/B-3 only | +50% memory headroom |
| 3 | **Path 2** — per-frame regime mixture | ~3 weeks | Lossless | B-1/B-2/B-3 only | +15–30% acceptance on regime-aware frames |
| 4 | **Path 4 lossless bundle** — §6.1-6.7 (turn signals, regime hints, hot paths, priming, predict_next read-only, opt-out, budget hints) | ~3 weeks | Lossless | B-1/B-2/B-3 only | +20–40% acceptance composed |
| 5a | **Path 4 lossy bundle v1** — §6.11 preserve_thinking changes | ~1 week | Lossy | **Subset SWE-Bench gate (Tier 1, ~10 wall-hrs)** | Token efficiency on Q36 reasoning patterns |
| 5b | Tier 2 full SWE-Bench on Path 4 lossy v1 (Verified + Pro) | ~150 wall-hrs | Lossy | **Full benchmark PASS criterion** | Reviewer-credible quality claim |
| 6 | **Path 4 lossy bundle v2** — §6.8-6.10, §6.12 (sampling, context mgmt, schema, max_tokens) | ~4 weeks | Lossy | Tier 1 then Tier 2 SWE-Bench | Additional speed/quality optimization |
| 7 | **Path 3 extended** — time-decay, type-aware, multi-source merge | ~3 weeks | Lossless | B-1/B-2/B-3 only | Publishable; +10–30% acceptance contingent |
| 8 | **Round 5 closeout** — full SWE-Bench Verified + Pro re-run on composed final stack | ~150 wall-hrs | n/a | Headline number for publication | Closes the round |

### Critical sequencing constraints

- **All lossless paths (1, 2, 3, 4-lossless-bundle, Path 3 extended) gate on B-1/B-2/B-3 distribution equivalence only.** No need to re-run SWE-Bench for these — the rejection sampling theorem guarantees correctness if the verifier is implemented correctly. Spot-check with 20-task Tier 0 smoke after every merge for catastrophic breakage.
- **Every lossy Path 4 sub-change requires SWE-Bench gating before merge.** Tier 1 subset for fast iteration; Tier 2 full benchmark before shipping in a release.
- **Baseline reproduction (Order 0b) comes before any change.** If we can't reproduce Qwen team's published numbers ±2 points, our harness is already broken relative to the canonical baseline and any further measurement is on unstable ground.
- **Path 1 must come before Path 2.** Path 2 generalizes Path 1; verify Path 1 composes at all before generalizing.
- **Path 3 DAWG swap can run in parallel with Path 1.** Independent substrate change.
- **Path 4 lossless bundle unblocks Path 2's true potential.** Path 2 with proxy-inferred regime hints is heuristic; with explicit harness signals from Path 4 lossless bundle it's authoritative.
- **Path 4 lossy bundle ships only after the lossless bundle has paid its way in measurable speedups.** Don't take on lossy regression risk until lossless gains are confirmed.

### Total estimated wall time

- Lossless paths + B-1/B-2/B-3 gates: ~8-10 weeks engineering, ~5 wall-hours of benchmark per merge
- Lossy paths + SWE-Bench gates: ~5-6 weeks engineering + ~300 wall-hours of benchmark across the full set
- Closeout SWE-Bench run: ~150 wall-hours
- **Total: ~14-16 weeks engineering + ~450 wall-hours benchmark on dedicated hardware**

This makes Round 5 a quarter-scale program, not a sprint. Pre-register the gates and the publication target up front; the cost is justified by the credibility it buys.

---

## 8. Calibration and measurement plan

Reuse the existing Round 4b harness:

- `scripts/full_data_sweep.py` — per-cell decode_tps, prefill_s, accept_rate, power_w, gpu_util
- `scripts/grade_all_cells.py` — per-cell deterministic grader, P_benchmark + M_aggregate
- `scripts/contamination_sweep.py` — power/tps inversion detector for measurement integrity

New cell namespaces for Round 5 (internal v4a_v2 corpus):

| Namespace | Path | Cells | Quality gate |
|---|---|---:|---|
| `Q36_HYBRID_TAU=1` | Path 1, τ=1.0 | 44 (11 tasks × 4 attempts) | B-1/B-2/B-3 |
| `Q36_HYBRID_TAU=3` | Path 1, τ=3.0 | 44 | B-1/B-2/B-3 |
| `Q36_HYBRID_TAU=5` | Path 1, τ=5.0 | 44 | B-1/B-2/B-3 |
| `Q36_REGIME_MIX` | Path 2 baseline | 44 | B-1/B-2/B-3 |
| `Q36_DAWG` | Path 3 substrate | 44 | B-1/B-2/B-3 |
| `Q36_DAWG_TIMEDECAY` | Path 3 + recency | 44 | B-1/B-2/B-3 |
| `Q36_DAWG_TYPED` | Path 3 + type-aware | 44 | B-1/B-2/B-3 |
| `Q36_HARNESS_FORK_LOSSLESS` | Path 4 §6.1-6.7 | 44 | B-1/B-2/B-3 |
| `Q36_HARNESS_FORK_LOSSY_v1` | Path 4 §6.11 | 44 | + SWE-Bench Tier 1+2 |
| `Q36_HARNESS_FORK_LOSSY_v2` | Path 4 §6.8-6.10, §6.12 | 44 | + SWE-Bench Tier 1+2 |

Internal-corpus total: 10 × 44 = 440 cells. At ~30 min/attempt + 30 min/grading = ~6 hours per task × 110 tasks across configs ≈ 130 wall-hours. Doable in 2-3 weeks of dedicated hardware time.

External-benchmark namespaces (Qwen 3.6-27B FP8):

| Namespace | Benchmark | Tasks | Purpose | Wall time |
|---|---|---:|---|---|
| `Q36_BASELINE_VERIFIED` | SWE-Bench Verified | 500 | Reproduce published 77.2 ± 2 | ~125 wall-hours |
| `Q36_BASELINE_PRO` | SWE-Bench Pro | ~700 | Reproduce published 53.5 ± 2 | ~175 wall-hours |
| `Q36_HARNESS_LOSSY_v1_VERIFIED_T1` | Verified, 100-task subset | 100 | Tier 1 gate for §6.11 | ~25 wall-hours |
| `Q36_HARNESS_LOSSY_v1_PRO_T1` | Pro, 100-task subset | 100 | Tier 1 gate for §6.11 | ~25 wall-hours |
| `Q36_HARNESS_LOSSY_v1_VERIFIED_T2` | Verified full | 500 | Tier 2 gate before ship | ~125 wall-hours |
| `Q36_HARNESS_LOSSY_v1_PRO_T2` | Pro full | ~700 | Tier 2 gate before ship | ~175 wall-hours |
| `Q36_HARNESS_LOSSY_v2_*` | repeat for §6.8-6.10, §6.12 | | | (same costs) |
| `Q36_ROUND5_CLOSEOUT_VERIFIED` | Verified full on composed stack | 500 | Headline number | ~125 wall-hours |
| `Q36_ROUND5_CLOSEOUT_PRO` | Pro full on composed stack | ~700 | Headline number | ~175 wall-hours |

External-benchmark total: ~1,150 wall-hours across all SWE-Bench runs in Round 5.

### Decision criteria per path

- **Path 1 PASS:** Q36-A → Q36-Hybrid median tps ≥ +5%, B-1/B-2/B-3 distribution equivalence holds (KL < 0.01, ≥ 99% byte-exact match on 32-prompt greedy sample). No pass-rate regression on v4a_v2.
- **Path 2 PASS:** Q36-Hybrid → Q36-Regime-Mix shows ≥ +5% on regime-distinguished tasks (transcript-merge, multi-tool, dead-flag), B-1/B-2/B-3 hold.
- **Path 3 PASS:** DAWG substrate ≥ 95% of suffix tree's tps at ≤ 60% memory, B-1/B-2/B-3 hold. Time-decay ≥ +5% accept on recency-biased tasks.
- **Path 4 lossless bundle PASS:** Harness-fork hot-path registration ≥ +10% accept on path-heavy tasks (fanout, multi-tool); bidirectional oracle measurable user-facing latency reduction; B-1/B-2/B-3 hold.
- **Path 4 lossy bundle PASS:** Tier 1 SWE-Bench (100-task subset, Verified + Pro) shows (after − before) ≥ −1.0 absolute on both. Tier 2 full benchmark required before ship: same gate but on full Verified (500) + Pro (~700).
- **Round 5 closeout PASS:** Full composed stack at v0.6 weight-immutability constraint achieves Verified ≥ 77 (within 1 absolute of published Qwen 3.6-27B 77.2), Pro ≥ 53. Aggregate v4a_v2 tps ≥ 28 tps median (vs current Q36-A 22.46). Pre-register these gates before measurement.

---

## 9. External quality gate — SWE-Bench Verified + SWE-Bench Pro

### Why external benchmarks

Our internal v4a_v2 corpus (11 tasks × 4 attempts = 44 cells per point) is calibrated for ablation sensitivity, not for external persuasion. A reviewer outside Track B has no reason to accept claims like "Q36-A pass rate = 10/44" as evidence that a harness-side change preserved quality. External benchmarks with published baselines from the model authors give us a defensible reference.

### Benchmarks to run

| Benchmark | Tasks | Why | Published Qwen 3.6-27B baseline |
|---|---|---|---|
| **SWE-Bench Verified** | 500 human-verified GitHub issues | Industry-standard, 500 problems, gold tests | **77.2** |
| **SWE-Bench Pro** | ~700 harder tasks | Newer, harder, ScaleAI-curated | **53.5** |

Qwen 3.6-27B published baselines come from the Qwen team's [model card](https://huggingface.co/Qwen/Qwen3.6-27B) using their internal agent scaffold (bash + file-edit tools), `temp=1.0`, `top_p=0.95`, 200K context. SWE-bench Multilingual (71.3) and Terminal-Bench 2.0 (59.3) are optional secondary references but not required for the loss gate.

### Measurement protocol per lossy change

For every Path 4 lossy sub-change (§6.8-6.13):

1. **Baseline reproduction.** Run SWE-Bench Verified + Pro against Qwen 3.6-27B FP8 via our current Codex CLI + vLLM stack, *without* the Round 5 change. Compare against Qwen team's published numbers; we should be within ±2 absolute points. If we're not, our harness already diverges from the published baseline — the Path 4 change is being measured against a non-canonical reference, and we need to investigate the harness gap first.
2. **Before-change measurement.** Run both benchmarks against the immediately-prior stack (most recently shipped configuration).
3. **After-change measurement.** Run both benchmarks against the candidate stack (with the lossy change applied).
4. **Gate criterion:** PASS if (after − before) ≥ −1.0 absolute on both Verified and Pro. Pre-register the gate before measuring to avoid p-hacking. A −1 to +1 drift is within run-to-run noise on these benchmarks at our sampling rates; a regression of −2 or worse is a real signal.

### Cost

| Benchmark | Tasks | Time per task | Total wall |
|---|---|---|---|
| SWE-Bench Verified | 500 | ~5-15 min agent budget | 40-125 hours per run |
| SWE-Bench Pro | ~700 | ~5-15 min agent budget | 60-175 hours per run |

So a single before+after pair is ~100-300 wall-hours per lossy change. This is expensive. Mitigations:

- **Subset sampling.** For initial fast iteration, run a stratified 100-task subset of Verified (10 per category × 10 categories) and a 100-task subset of Pro. Cuts cost ~5×. Use the subset gate to filter changes; only re-run the full benchmark for changes that pass the subset gate.
- **Per-change parallelism.** Each task is independent. With multiple DGX Sparks we can run ~4-8 in parallel.
- **Bundle lossy changes.** Don't gate each of §6.8-6.13 separately — bundle them into a "Path 4 lossy bundle vN" and gate the bundle once per release.

### Acceptance ladder

- **Tier 0 (smoke):** 20-task subset of Verified runs in <2 hours. Catches catastrophic breakage. Run before every commit to Path 4 lossy code.
- **Tier 1 (subset gate):** 100-task stratified subset of Verified + Pro. ~10 wall-hours. Run per merge of a lossy change.
- **Tier 2 (full benchmark):** Full Verified + Pro. ~150 wall-hours. Run per release / per major bundle merge.

### What we already know about the published baselines

From Qwen 3.6-27B model card (April 2026):

| Score | Qwen 3.5-27B | Qwen 3.6-27B | Δ |
|---|---:|---:|---:|
| SWE-Bench Verified | 75.0 | **77.2** | +2.2 |
| SWE-Bench Pro | 51.2 | **53.5** | +2.3 |
| SWE-Bench Multilingual | 69.3 | 71.3 | +2.0 |
| Terminal-Bench 2.0 | 41.6 | **59.3** | +17.7 |

The published Qwen 3.6 numbers are meaningfully better than Qwen 3.5 on SWE-Bench. **If our Round 5 reproduction lands at ≥75 on Verified and ≥51 on Pro for the Qwen 3.5 baseline + our shipped D-point config, we know the model improvement transfers through our stack.** If we land significantly below those numbers, our harness is leaving quality on the table and that's a more important finding than any speedup.

### Why this matters for the publishable contribution

Published spec-decode papers (SuffixDecoding, EAGLE-3, FastMTP, etc.) all report SWE-Bench numbers to demonstrate lossless quality. A Round 5 paper claiming "+2-3× speedup with harness co-design" without SWE-Bench Verified + Pro numbers is not going to convince a reviewer. The benchmark cost is real, but the credibility cost of not running them is higher.

---

## 10. Open questions for design review

1. **What's the right MTP draft tree topology?** vLLM's `Qwen3NextMTPProposer` defaults to linear. The FastMTP paper suggests tree topologies help. Worth testing both inside Path 1.
2. **Is τ static or per-frame?** Path 1 starts static. Path 2 makes it dynamic by regime. Verify the gain from dynamic τ separately from the regime mixture.
3. **How aggressive should DAWG eviction be?** Codex sessions can run 200+ turns. Memory cap on the suffix tree / DAWG matters. Currently we hold 50-200 MB per session. With DAWG that's 25-100 MB. With sessions running into the hundreds, total RAM could matter.
4. **Should the harness-oracle protocol be vendor-neutral from day 1?** If yes, spec it up front and propose to OpenAI as an upstream Codex extension. If no, optimize for our setup first, generalize later.
5. **What's the right surface for predict_next bidirectional oracle?** REST? Server-sent events? gRPC? Codex CLI is Node-based; we should pick something that doesn't require new dependencies on the Codex side.
6. **Quality preservation taxonomy.** v0.7 constraint says output distribution mathematically identical (rejection sampling). All lossless paths satisfy this by construction. **Lossy Path 4 sub-changes do not** — they need full SWE-Bench gating before ship. See §9.
7. **Can we reproduce Qwen team's published SWE-Bench baselines?** Order 0b in §7 is the first measurement we must do. If our stack lands more than ±2 points below published Qwen 3.6-27B Verified=77.2 / Pro=53.5, we have a harness gap to debug before anything else. **This is the most important single open question** — answer it before committing engineering time to Round 5.
8. **Tier 1 subset stratification.** The 100-task SWE-Bench Verified subset for fast iteration needs to be stratified to be representative. Naive random sampling could bias toward easy or hard problems. Worth picking the stratification scheme (by repo, by difficulty tag, by category) before starting.
9. **What's the credibility threshold for an external publication?** SWE-Bench Verified + Pro alone, or do we also need Terminal-Bench 2.0 (Qwen 3.6 score: 59.3) and SkillsBench (48.2)? More benchmarks = more credibility but more wall-time. Decide before kickoff.
10. **Can we contribute the harness-oracle protocol upstream?** Pre-register intent with the OpenAI Codex maintainer team. If they're receptive, design the protocol for upstream-ability from day 1. If not, optimize for our fork.

---

## 11. Files referenced

- Round 4b formal report (final): `docs/reports/auto_research/track-b-round4b-ablation-formal-report-20260516.md`
- Round 4b D-point remeasure: `docs/reports/auto_research/track-b-round4b-dpoint-remeasure-results-20260517.md`
- Power-w contamination methodology: `docs/reports/auto_research/track-b-round4b-power-w-remeasure-list-20260516.md`
- Qwen 3.6 temp=0.6 experiment: `docs/reports/auto_research/qwen36-temp06-experiment-results-20260518.md`
- Parent engineering spec: `docs/reports/auto_research/codex-harness-spec-decode-engineering-20260507.md`
- Sweep + grading scripts: `scripts/full_data_sweep.py`, `scripts/grade_all_cells.py`, `scripts/contamination_sweep.py`

## 12. External sources

- SuffixDecoding paper — [arXiv 2411.04975](https://arxiv.org/pdf/2411.04975)
- SuffixDecoding NeurIPS 2025 spotlight site — [suffix-decoding.github.io](https://suffix-decoding.github.io/)
- Snowflake Arctic Inference + vLLM SuffixDecoding blog — [snowflake.com](https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/)
- AgentInfer paper — [arXiv 2512.18337](https://arxiv.org/pdf/2512.18337)
- SAM Decoding paper — [arXiv 2411.10666](https://arxiv.org/pdf/2411.10666)
- FastMTP paper — [arXiv 2509.18362](https://arxiv.org/pdf/2509.18362)
- Speculative Search agent co-design — [arXiv 2511.20048](https://arxiv.org/pdf/2511.20048)
- Gemma 4 MTP drafters — [MarkTechPost article, May 2026](https://www.marktechpost.com/2026/05/06/google-ai-releases-multi-token-prediction-mtp-drafters-for-gemma-4-delivering-up-to-3x-faster-inference-without-quality-loss/)
- vLLM Qwen 3.6 recipe — [vLLM recipes](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)
- Codex CLI repo — [github.com/openai/codex](https://github.com/openai/codex)
- vLLM Issue #40831 (TurboQuant KV × spec_decode bug) — [github.com/vllm-project/vllm/issues/40831](https://github.com/vllm-project/vllm/issues/40831)
- vLLM Issue #41190 (TP=2 MTP crash on Qwen 3.6) — [github.com/vllm-project/vllm/issues/41190](https://github.com/vllm-project/vllm/issues/41190)
- SWE-Bench Verified leaderboard — [swebench.com/SWE-bench/swe-bench_verified.html](https://www.swebench.com/SWE-bench/swe-bench_verified.html)
- SWE-Bench Pro dataset — [ScaleAI/SWE-bench_Pro on Hugging Face](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro)
- Qwen 3.6-27B model card (published SWE-Bench baselines) — [huggingface.co/Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
- Leviathan et al. "Fast Inference from Transformers via Speculative Decoding" (ICML 2023) — rejection sampling theorem
- Chen et al. "Accelerating Large Language Model Decoding with Speculative Sampling" (2023) — independent derivation of the same losslessness proof

---

**End of spec.**
