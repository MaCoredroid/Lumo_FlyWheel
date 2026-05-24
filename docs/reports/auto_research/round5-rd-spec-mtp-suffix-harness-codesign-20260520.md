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

## 13. Addendum (2026-05-24) — Probe16d empirical updates

The original spec (sections 1–12, dated 2026-05-20) was written before the conc-probe16d SWE-Bench Verified data landed. Probe16d's clean per-stream measurements (with the fixed `upstream_compute_s` counter and the c1off arm with `speculative_config` cleared) shift three things in the spec: the workload acceptance number we should anchor Path 1 to, our ability to measure GPU compute saturation directly, and what an additional empirical-priors path looks like alongside Paths 1–4. This addendum amends those three places without rewriting the original sections; the dated provenance of §§1–12 is preserved.

### 13.1 Measurement constraint confirmed — DCGM profiling not officially supported on DGX Spark / GB10

The DCGM sampler in `tests/test_track_b_dcgm_sampler.py` is wired for the rich profile fields (`dram_active_pct`, `sm_active_pct`, `sm_occupancy_pct`, `pipe_tensor_active_pct`, `pipe_fp16_active_pct`), but every row in `output/swe_bench_q36_a_temp06/.../dcgm_samples.jsonl` returns `null` for them with `profile_fields_unavailable_reason: "nvml_fallback_only"`. Initial suspicion was a setup misconfiguration; confirmed via the NVIDIA Developer Forums (thread "Spark failed to retrieve SM Activity and the profiler module failed to load") that this is **NVIDIA-declared product policy**, not a fixable configuration issue.

Quoting the NVIDIA staff response on that thread (DGX Spark / GB10 forum, late 2025):

> *"DCGM is not officially supported on DGX Spark."*
>
> *"Since the Spark is not a datacenter, there are no plans to support DCGM on Spark."*

Confirmed by the same thread's reproducer log: `dcgmi modules -l` shows **Module 8 (Profiling): Failed to load**, **Module 9 (SysMon): Failed to load**. `dcgmi profile -l` returns `"Unable to Get supported metric groups: This request is serviced by a module of DCGM that is not currently loaded."` This is the same failure shape our sampler hits, and it is by design — DGX Spark is classified as a workstation/edge product and not eligible for the DCGM Profiling module that's reserved for datacenter SKUs.

**The hardware can produce these counters; NVIDIA's product positioning withholds them via the DCGM module.** The same underlying CUPTI counters that DCGM-Profiling exposes are still accessible via Nsight Systems on Spark — they're just not exposed through the streaming-telemetry API that production observability stacks expect.

#### Implications for Round 5 measurement

For continuous production telemetry (1 Hz - 100 Hz streaming during inference), our reliable signals on GB10 are:

| Signal | What it actually measures | Reliable? |
|---|---|---|
| `gpu_util_pct` (NVML) | SM-issue activity, includes memory-stall cycles | **No** — over-reports utilization on bandwidth-bound workloads |
| `mem_copy_util_pct` (NVML) | DMA copy engine activity | Yes but uninteresting (decode is not copy-bound) |
| `power_w` (NVML) | Board power draw | **Yes — best proxy for true compute utilization on GB10** |
| `iter_cnt` delta over time (step trace) | Per-step latency as a function of B | Yes — direct measure of forward-pass time |
| Step latency vs B slope | Bandwidth-bound iff flat; compute-bound iff linear | Yes — derived signal, not noisy |

Empirically, the probe16d step trace shows power_w at 48 W (B=1) → 49 W (B=2) → 53 W (B=3) against a GB10 GPU TDP of ~120-140 W. **The GPU's compute side is running at roughly 35-40% of capacity during decode.** This is the load-bearing fact that justifies Path 1, Path 2, and the new Path 5 below — there's substantial unused compute headroom that aggressive drafting (more candidates, learned drafter, mined priors) can spend without contention.

#### One-off characterization workaround — Nsight Systems

For offline characterization (not continuous telemetry), **Nsight Systems works on DGX Spark** and CAN expose the per-kernel metrics DCGM-Profiling would have given us, via the `--gpu-metrics-devices` option (which uses CUPTI under the hood — and CUPTI is supported on Spark for developer profiling even though DCGM-Profiling is not).

Concretely:

```bash
nsys profile --gpu-metrics-devices=0 --gpu-metrics-frequency=10 \
  --gpu-metrics-set=tu10x-gfxt \
  python -m vllm.entrypoints.openai.api_server [...]
```

This produces an `.nsys-rep` file with per-kernel DRAM throughput, SM occupancy, tensor-core utilization, etc. — the same data we'd see from DCGM Profiling on a Hopper host. The data isn't streamed into our metrics JSONL pipeline (Nsight is post-hoc, not realtime), and the profiling overhead is non-trivial (~5-10% slowdown), so it's not a continuous-telemetry replacement. But for a one-off "is the GPU memory-bound or compute-bound during decode" question — which is the question Round 5 most cares about — a single Nsight run on a representative workload would give a definitive answer.

There are known Nsight-on-Spark issues to plan around:
- `[nsys profile] gpu-metrics-devices fails with "Already under profiling"` — a documented forum issue when multiple profilers contend for CUPTI. Mitigation: ensure no other CUPTI consumer (PyTorch profiler, nvprof, etc.) is running concurrently.
- Nsight Systems on GB10 (SM 12.1) needs a recent-enough version to handle the unified-memory traces correctly; check the Nsight Systems release notes for SM121 support.

#### Recommendation

1. **For Round 5 production measurement (all paths):** Use `power_w` as the primary compute-saturation proxy. Document in any external write-up that DCGM Profiling is not available on Spark per NVIDIA policy and that power-draw is the substitute.
2. **For Round 5 closeout characterization (optional but valuable):** Run one Nsight Systems profile on the composed final stack (SD + MTP fallback + Codex prior + Path 4-lossless) against a 30-min SWE-Bench astropy run. Extract the per-kernel DRAM throughput and tensor-core utilization, and report them alongside `power_w` to substantiate the bandwidth-bound claim. This is the publishable evidence path.
3. **For Round 5 paper credibility:** If a reviewer challenges the "bandwidth-bound on GB10" claim, having both `power_w` (continuous, the operational metric) and one Nsight `.nsys-rep` (point-in-time, the gold-standard verification) is the responsible position. It's not a fundamental hardware limitation — it's a product-policy limitation — and the workaround is well-known.

Note that this is also a useful general lesson for any team running on Spark: **Spark gives you NVML+Nsight; it does not give you DCGM Profiling.** Production observability on Spark must accept this constraint; offline characterization can route around it.

### 13.2 Updated workload reality: SWE-Bench Verified astropy acceptance ≈ 0.22

Section §3 calibrates Path 1's τ from CNB-55 numbers (Q36-A acceptance 0.548). The probe16d data shows SWE-Bench Verified astropy acceptance is **far lower than the CNB-55 calibration point**, with concrete implications for Path 1.

Probe16d c=1 measurements (n=16 instances, fixed `upstream_compute_s` counter):

| Metric | Value | Range across 16 instances |
|---|---:|---|
| Pooled acceptance | 0.217 | per-instance: [0.127, 0.307] |
| Pooled decode-only tps | 9.22 | per-instance: [6.62, 13.82] |
| Pooled wall tps | 8.60 | per-instance: [6.19, 12.23] |
| Spec-on vs c1off (no spec) speedup | 1.95× | (8.60 / 4.40) |

c1off baseline (no speculative decoding, n=5 instances): pooled 4.40 tps median 4.14 tps. **The end-to-end speedup attributable to T1 SuffixDecoding on SWE-Bench astropy is 1.95×, not the 4.02× we measured on CNB-55.** Per-instance Pearson r(acceptance, decode-only tps) = +0.618 — within-workload variance is dominated by suffix-cache hit rate, not by context length (r(prompt_tokens, tps) ≈ +0.15).

What this means for Path 1's τ calibration:

- The original spec's starting point (τ ≈ 1.0, drawn from CNB-55 conditions) is calibrated for an acceptance regime where SuffixDecoding alone yields ≥3 accepted tokens/step. On SWE-Bench astropy, SuffixDecoding alone yields **~1.4 accepted tokens/step.** A τ that triggers "fall through to MTP" any time SD scores below 1.0 will fire on ~70-80% of decode positions for SWE-Bench workloads, vs ~20-30% on CNB-55. Per-call MTP latency (5-10 ms) at that fire rate will *erase* the bandwidth amortization the hybrid is supposed to buy.
- The corrective recalibration: **start τ at 0.5-0.7, not 1.0.** Engage MTP only when SuffixDecoding's score signals near-zero confidence, not when SD is doing its average job on a low-acceptance workload. This is the inverse of the CNB-55 prescription — CNB-55's optimal τ is high because SD is reliable there; SWE-Bench's optimal τ is low because SD is unreliable and you want to avoid over-engaging MTP.
- The Path 1 PASS criterion in §8 (≥+5% on Q36-Hybrid vs Q36-A on the internal v4a_v2 corpus) is calibrated for the high-acceptance regime. **The Round 5 v4a_v2 PASS criterion will under-stress Path 1's true value, because v4a_v2 is CNB-55-shaped.** Add a **SWE-Bench-Verified astropy mini-sweep** (16 instances, ~5 wall-hours per τ point) to the §8 cell namespaces — Path 1 must pass *both* gates before shipping.

Updated §3 expected outcome (replaces "Q36-A could lift from 22.46 → ~25 tps median"):

- On v4a_v2 (CNB-55-shaped): minimal lift, possibly flat. SD-alone already captures most of the available speedup at ~0.55 acceptance.
- **On SWE-Bench Verified astropy (probe16d-shaped): potential lift from 8.6 → 11-13 tps if MTP fallback captures 30-40% of the 78% of positions where SD currently fails.** This is where Path 1's marginal value actually lives. The CNB-55 v4a_v2 corpus was designed for ablation sensitivity, not for stressing the new hybrid.

This is the right framing for the closeout: **Path 1's lift is largest precisely where SuffixDecoding-alone is weakest.** The published τ-threshold pattern says "fall through to fallback when SD is weak." On low-acceptance workloads SD is *always* weak — so the hybrid should win by a larger margin than the CNB-55 calibration suggests, *if* τ is set low enough not to over-engage the fallback.

### 13.3 New Path 5 — Empirical Codex prior via trace mining

The spec's Paths 3 (auto-completion substrate) and 4 (Codex CLI fork) gesture at workload-specific suffix-tree priming (Path 4 §6.3 hot-path registration, §6.4 stream-side priming; Path 3's "multi-source merge with provenance"), but treat them architecturally. There is a complementary **empirical** path the spec doesn't operationalize: mine our own production traces to construct a Codex-deployment-specific prior, pre-load it into the global suffix tree at session start, and let it amortize across all subsequent sessions on this harness.

The motivation is concrete: probe16d has produced ~22 M tokens of real Codex SWE-Bench traces across c=1, c=2, c=4, c1off arms (16 instances × ~10–15 K tokens × multiple arms). These traces contain the *empirical distribution* of what Codex actually emits on SWE-Bench Verified. The current suffix-decode design starts every session with an empty global tree and learns the high-frequency patterns turn-by-turn at acceptance ~0.22. A pre-loaded prior tree could start every session with the Codex-prior already populated, lifting the acceptance floor immediately rather than waiting for organic discovery.

**Loss classification: LOSSLESS by construction.** Pre-loading the suffix tree changes *which candidates* the drafter proposes but not which tokens the verifier accepts. Rejection sampling preserves the target model's distribution exactly. Gate as Paths 1–3: B-1/B-2/B-3 distribution-equivalence only, no SWE-Bench re-run required for correctness.

#### 13.3.1 What the prior would contain

Token sequences whose frequency in production Codex traces is high enough that pre-indexing them lifts the drafter's average match length. From inspection of the existing probe traces, candidates include:

| Pattern class | Examples | Expected frequency |
|---|---|---|
| **Tool-call envelope** | `<function_call><name>...</name><arguments>{`, `"path":"`, `"command":["` | Every tool call; ~16-32 tokens per call |
| **Codex tool names** | `read_file`, `exec_command`, `apply_patch`, `write_file`, `list_directory`, `view` | One per tool call |
| **Common bash invocations** | `pytest -q`, `pytest -x`, `python -c "`, `grep -rn`, `find . -name`, `cd /repo &&` | High frequency in agentic loops |
| **Diff format tokens** | `--- a/`, `+++ b/`, `@@ -`, `@@ +`, `*** Begin Patch`, `*** End Patch` | Every `apply_patch` call |
| **Python control flow** | `def `, `class `, `if `, `else:`, `for `, `return `, `raise `, `import ` | Pervasive in patch hunks |
| **Test runner output stubs** | `PASSED`, `FAILED`, `ERROR`, `=== test session starts ===`, `collected ` | Test-output references |
| **Repository structure (workload-specific)** | For astropy: `astropy/io/fits/`, `astropy/modeling/`, `astropy/units/`. For sympy/django/sphinx: equivalents. | Many turns reference paths |

Note the last row: **the prior is stratifiable by repo or benchmark slice.** A general-purpose Codex prior (the first 6 rows) is workload-agnostic; an astropy-specific prior (or sympy-specific, etc.) adds another tier of repeated content for that benchmark slice. The same architecture supports both.

#### 13.3.2 Construction pipeline (offline, ~3-5 days of analyst time)

1. **Aggregate trace corpus.** Concatenate token streams from all c=1 (RAW) and c=1 (spec-on) probe traces into one corpus. Source: `output/swe_conc_probe16d/*/per_task/*/vllm_request_metrics.jsonl` — extract `completion_tokens` per session, plus prompts.
2. **N-gram frequency analysis.** Build frequency tables for 4-grams, 8-grams, 16-grams over the corpus. Compute (a) raw frequency, (b) document frequency (fraction of distinct sessions containing the n-gram), (c) "stability" score = document frequency × log(raw frequency).
3. **Cross-validation slice.** Hold out 20% of sessions; verify the top-K most-frequent n-grams from the training slice also appear in the held-out slice with similar frequency. Discard n-grams that don't generalize (over-fit to specific tasks).
4. **Prior tree construction.** Use the existing `arctic_inference.suffix_decoding.SuffixDecodingCache` API to insert each stable n-gram as a synthetic sequence at session-start. The cache already handles tree construction; we just feed it pre-mined data alongside live session traffic.
5. **Eviction policy adjustment.** The current `max_cached_requests=1000` FIFO would evict the prior as soon as live sessions exceeded 1000. Two clean fixes:
   - **Tiered cache:** mark prior entries as "immutable" (or assign them a very early seq_id that's pinned). Live sessions evict each other in FIFO order; the prior persists.
   - **Two-tree extension:** add a third tree alongside `_local_trees` and `_global_tree` — `_prior_tree` — that's read-only and pre-populated. `speculate()` queries all three, picks the highest-scoring draft. Same wrapper pattern as T1 session-scoping; adds ~3 days engineering on top of the existing patch.

#### 13.3.3 Offline measurement protocol (do this before the engineering)

The Arctic Inference codebase ships `arctic_inference/suffix_decoding/simulator.py` (referenced in the spec's §10). The simulator runs `SuffixDecodingCache.speculate()` against trace data without invoking vLLM. Use it to predict the lift from pre-loading priors *before* writing any production code:

```
For each held-out probe16d session:
  baseline_path:
    cache = SuffixDecodingCache(max_tree_depth=32, max_cached_requests=1000)
    cache.start_request(req_id, prompt)
    For each emitted token at position i:
      draft = cache.speculate(req_id, context[-32:], ...)
      record accepted_count_baseline_i
      cache.add_active_response(req_id, [token_i])

  prior_path:
    cache = SuffixDecodingCache(max_tree_depth=32, max_cached_requests=2000)
    # Pre-load: insert each mined n-gram as a synthetic completed session
    For each prior_ngram:
      cache.start_request(prior_req_id, [])
      cache.add_active_response(prior_req_id, prior_ngram)
      cache.stop_request(prior_req_id)
    cache.start_request(req_id, prompt)
    [same loop as baseline]
    record accepted_count_prior_i

  Compare: total_accepted_prior / total_accepted_baseline
```

If the simulator reports a **lift of <5%** the empirical-prior path is not worth shipping. If **>15%**, fund the engineering side immediately. The 5-15% middle ground is where the cost-benefit calls for sensitivity analysis on prior-tree size vs lift.

This experiment is *cheap*: no GPU time required, runs in Python on a single CPU host, completes in hours. **Adding this offline simulator pass as a pre-engineering measurement is the single highest-information-per-cost step in Round 5.**

#### 13.3.4 Why this is a publishable contribution

Path 1 (MTP+SD hybrid) and Path 2 (per-frame regime mixture) are architectural compositions of techniques already published independently. They're worth doing but they're incremental on the literature.

Path 5 (empirical workload priors) makes a stronger claim:

> *"Production deployments of suffix decoding on a fixed agent harness (e.g., Codex CLI) operate over a non-uniform distribution of token sequences. Mining the empirical frequency distribution from production traces yields a workload-specific prior that, when pre-loaded into the suffix-decode cache, lifts acceptance from a cold-start baseline to a steady-state baseline at turn 0 instead of turn N. We measure a ΔAcceptance of [X]% on a [Y]-token trace corpus of [N] distinct sessions, equivalent to [Z]× per-stream throughput improvement at production-relevant batch sizes."*

This frames the contribution as **empirical infrastructure**, not algorithmic. Reviewers respond well to that — the methodology is clear (mine your traces, freeze the top-K, pre-load), the ablation is clean (prior vs no-prior, holding everything else fixed), and the result is reproducible by anyone running a fixed agent on a fixed benchmark. It also generalizes: anyone running any agent harness against any benchmark can apply the same methodology.

The literature has nothing in this slot. Snowflake's SuffixDecoding paper assumes cold-start. AgentInfer's "reuse across agent sessions" hand-waves at this idea but doesn't operationalize the priors-from-trace-mining angle. Gemma's MTP drafters are model-side, not deployment-side. Empirical priors are open territory and we are uniquely positioned to do this work because we have the trace corpus.

#### 13.3.5 Engineering effort

- **Offline simulator measurement:** 2-3 days (n-gram extraction, simulator harness, lift measurement on held-out traces).
- **Prior tree construction + integration:** 3-5 days (third-tree wrapper, eviction policy fix, serialization of mined priors to disk).
- **Production deployment:** 1-2 days (load prior at vLLM container start, regenerate priors on a weekly cadence from accumulated production traces).
- **B-1/B-2/B-3 distribution-equivalence gate:** standard, ~1 day.
- **Total:** ~2 weeks engineering after the offline measurement validates the approach.

#### 13.3.6 Composition with Path 1

Path 1 and Path 5 are **complementary, not redundant**:

- **Path 1 (MTP fallback)** picks up the *"novel content"* arm — tokens the model is generating freshly, where pattern memorization can't help but a learned distribution can.
- **Path 5 (empirical Codex prior)** picks up the *"structural repetition"* arm — tool envelopes, command shells, file paths, diff format tokens — that recur across the workload distribution but aren't yet in this specific session's cache.

Stacked:

```
                            Acceptance on SWE-Bench Verified astropy   Per-stream tps (est)
SD-alone (current)              0.22                                    8.6
SD + MTP fallback (Path 1)      0.28-0.32                              10.5-11.5
SD + Codex prior (Path 5)       0.28-0.35                              10.5-12.5
SD + Path 1 + Path 5            0.35-0.45                              12-15
```

The composition arithmetic is approximate; the simulator measurement in §13.3.3 will give the real numbers for Path 5, and the τ-sweep on SWE-Bench (§13.2) will give the real numbers for Path 1. Both should be measured before committing to a particular composition.

### 13.4 Updated recommended sequence

The §7 sequence assumed the v4a_v2 corpus was sufficient for calibration. Probe16d shows it's not — the SWE-Bench Verified slice has a fundamentally different acceptance regime and a different per-stream optimum. The amended sequence inserts the empirical measurements before any architectural commitment:

| Order | Step | Effort | Cost | Decision rule |
|---|---|---|---|---|
| **0a** | Finish Q36-D measurement (per original §7) | ~3 days | — | (unchanged) |
| **0b** | **Baseline SWE-Bench reproduction** (per original §7) | ~150 wall-hrs | — | (unchanged) |
| **0c** | **NEW: Offline simulator measurement of empirical Codex prior lift** (§13.3.3) | 2-3 days | CPU-only | If simulated lift on probe16d traces ≥ 10%, fund Path 5 engineering. If 5-10%, sensitivity analysis. If <5%, defer Path 5. |
| **0d** | **NEW: τ-sweep on SWE-Bench astropy mini-corpus (16 instances)** for Path 1 | ~5 wall-hrs per τ point × 5 points | ~25 wall-hrs total | Identify optimal τ for low-acceptance regime; carry forward to Path 1 production calibration. |
| **0e** | **NEW (PREREQUISITE): GPU compute-saturation metric enablement via Nsight Systems on Spark** (§13.7) | ~1-2 days setup + 1-2 hours profile | one ~30-min run (~10% profiling overhead) | Establish whether the operational `power_w` proxy is consistent with per-kernel DRAM_ACTIVE / SM_ACTIVE / PIPE_TENSOR_ACTIVE from CUPTI. If Nsight confirms bandwidth-bound (DRAM_ACTIVE > 75%, PIPE_TENSOR_ACTIVE < 50%), proceed with Paths 1/5 confidently. If Nsight reveals already-compute-bound regime (PIPE_TENSOR_ACTIVE > 75%), Paths 1/5 lose their headroom argument and the spec needs revision before continuing. |
| 1 | **Path 1** — τ-threshold hybrid (SD + MTP) with SWE-Bench-anchored τ (from 0d) | ~1 week | B-1/B-2/B-3 + 5h SWE-Bench mini | +10-30% tps on SWE-Bench, possibly flat on v4a_v2 |
| 1.5 | **NEW: Path 5 — empirical Codex prior via trace mining** (conditional on 0c) | ~2 weeks | B-1/B-2/B-3 + 5h SWE-Bench mini | +5-25% acceptance on SWE-Bench |
| 2 | **Path 3** — DAWG substrate swap | ~1 week | (unchanged) | (unchanged) |
| 3 | **Path 2** — per-frame regime mixture | ~3 weeks | (unchanged) | (unchanged) |
| 4 | **Path 4 lossless bundle** | ~3 weeks | (unchanged) | (unchanged) |
| 5+ | **Path 4 lossy bundles** | (unchanged) | (unchanged) | (unchanged) |
| 8 | **Round 5 closeout** | ~150 wall-hrs | (unchanged) + 1 Nsight profile on composed stack | Composed SD + Path 1 + Path 5 + Path 4-lossless target: 14-18 tps median on SWE-Bench astropy. Include closeout Nsight profile as supplementary evidence for the publication. |

The key insertions are 0c (free, decisive), 0d (cheap, recalibrates Path 1 for the actual workload), and 0e (cheap, validates the bandwidth-bound assumption that motivates the whole engineering direction). All three can run in parallel and complete in the same week. They unblock Paths 1 and 5 with empirically-calibrated parameters instead of CNB-55-extrapolated ones and with verified compute-saturation evidence.

**Why 0e is a prerequisite, not an afterthought:** Paths 1, 2, 5, and "draft more aggressively" all assume the GPU is bandwidth-bound and that adding compute (more drafts, MTP fallback, mined-prior lookups) is essentially free. That assumption is currently supported by `power_w ≈ 48 W` on a ~120-140 W TDP GPU — a strong but indirect signal. A single Nsight Systems run resolves the assumption directly. The downside risk we want to eliminate before spending 8-10 weeks of engineering: if the GPU is in fact already compute-saturated (e.g., tensor cores running near peak on attention compute for our 24K context), then adding more drafted positions, MTP fallback, or aggressive priors would *steal* compute we currently rely on, slowing down decode rather than speeding it up. The probability is low — the power_w signal makes it unlikely — but the cost of running Nsight is hours, and the cost of being wrong is multiple weeks of engineering on a flawed premise. Burn the prerequisite, get the certainty, then commit the engineering. This is the highest-information-per-cost step in the entire Round 5 plan.

### 13.5 Open questions added to §10

11. **What is the actual lift from empirical Codex priors on probe16d traces?** Answered by §13.3.3 offline simulation. Must run before §13.3 engineering.
12. **What is the optimal τ for SWE-Bench Verified astropy specifically?** Answered by §13.4 step 0d τ-sweep. Likely materially different from the v4a_v2-optimal τ.
13. **Should empirical priors be benchmark-specific or workload-general?** Mine traces per-benchmark (astropy / sympy / django / sphinx) and measure prior overlap. If overlap is >70%, ship a general prior. If <30%, ship per-benchmark priors. If 30-70%, ship a tiered prior with a general base + per-benchmark deltas.
14. **What's the GB10 alternative for DCGM-quality compute-saturation measurement?** Either accept `power_w` as proxy (current default), or run one-off characterization on a Hopper/Blackwell discrete-GPU host with DCGM enabled. The latter is required if we want to publish DRAM_ACTIVE / PIPE_TENSOR_ACTIVE numbers in a Round 5 paper.
15. **Trace corpus refresh cadence for empirical priors.** Production deployments accumulate new trace data continuously. How often should the prior be re-mined and re-deployed — weekly? Monthly? Per-release? Stability of top-K n-grams across time windows is the answer; measurable from existing data.

### 13.6 Provenance of this addendum

This addendum is based on data captured in:

- `output/swe_conc_probe16d/c1/per_task/` — 16 instances of c=1 with spec-on, fixed `upstream_compute_s`
- `output/swe_conc_probe16d/c1off/per_task/` — 5 instances of c=1 with `speculative_config` cleared
- `output/swe_conc_probe16d/c2/per_task/` — 16 instances of c=2
- `output/swe_conc_probe16d/c4/per_task/` — 16 instances of c=4
- `output/swe_conc_probe8c/dgx_steptrace.jsonl` — 1946 samples of vLLM iter_cnt / running / power_w at ~1.5 s sampling
- `docs/reports/auto_research/swe-bench-telemetry-definitions-20260523.md` — authoritative counter definitions

Author commits relevant to this addendum: `e26a12d1` (fixed `upstream_compute_s`), `ea6e3035` and follow-ons (probe16d raw data), `ed432190` (richer telemetry sampler), `5d6eee0d` (proxy upstream timestamps).

### 13.7 Prerequisite — GPU compute-saturation metric enablement (Nsight Systems on Spark)

Referenced as step 0e in §13.4. This section gives the detailed methodology for the prerequisite Nsight Systems characterization run.

#### 13.7.1 Why this is a prerequisite

Three engineering directions in this spec (Paths 1, 2, 5, and the broader "draft more aggressively" reasoning in §13.1) all rest on the claim that **the GPU is bandwidth-bound during decode and has substantial compute headroom available**. The current evidence for that claim is indirect:

- `power_w ≈ 48 W` at B=1 vs ~120-140 W TDP → ~37% of compute capacity used
- Step latency essentially flat from B=1 to B=2 (184 → 169 ms) → no per-pass time penalty for adding work, the signature of bandwidth-bound batching
- vLLM `gpu_util_pct` reads 95% but this is the SM-issue-activity metric which counts memory-stall cycles as "utilized" — known-unreliable on bandwidth-bound workloads

The claim is well-supported by the indirect signals, but **directly measuring DRAM_ACTIVE and PIPE_TENSOR_ACTIVE via CUPTI through Nsight Systems** gives us the ground-truth answer in a single profiling run. Cost: a few hours to set up plus one 30-minute run with ~10% profiling overhead. Benefit: definitively unblocks (or definitively redirects) 8-10 weeks of engineering.

This is also the publishable evidence path. If Round 5 produces a paper, reviewers will expect either DCGM-Profiling numbers or Nsight-equivalent numbers as backing for any "memory-bound" claim — the operational `power_w` proxy alone is not sufficient for external credibility, even though it is sufficient for internal operational decisions.

#### 13.7.2 Why this is doable on Spark despite the DCGM limitation

Per §13.1: DCGM Profiling is not available on Spark per NVIDIA product policy. However:

- **CUPTI is available on Spark** for developer profiling. The same underlying GPU performance counters that DCGM-Profiling exposes via DCGM_FI_PROF_* fields are accessible to CUPTI-based tools.
- **Nsight Systems uses CUPTI under the hood.** `nsys profile --gpu-metrics-devices=0` will report the same metrics that DCGM-Profiling would have shown, just packaged as a post-hoc `.nsys-rep` file rather than streamed telemetry.
- **vLLM has documented Nsight integration** for production profiling, so we don't need to invent the profiling pattern from scratch.

The practical translation: DCGM is closed off; Nsight is open. We can get the data; we just route through a different API surface.

#### 13.7.3 Setup steps

```bash
# 1. Verify Nsight Systems is installed and recent enough for SM 12.1 / GB10.
nsys --version  # Need 2025.4+ for full GB10 unified-memory trace support.

# 2. Stop any other CUPTI consumers (PyTorch profiler, nvprof, etc.).
#    Concurrent profilers will cause "Already under profiling" errors per
#    the known forum issue.

# 3. Confirm we have the right GPU metrics set for sm_121.
nsys profile --list-gpu-metrics

# Expected output should include DRAM throughput, SM active, PIPE tensor active.
# If those are missing, the Nsight version is too old for GB10.
```

#### 13.7.4 Profile run methodology

Use a representative SWE-Bench Verified astropy instance — one of the c=1 probe16d instances with median tps (e.g., `astropy__astropy-13033` which sat at 8.95 tps decode-only, close to the pooled 9.22 median). Run it under Nsight:

```bash
nsys profile \
  --gpu-metrics-devices=0 \
  --gpu-metrics-frequency=10 \
  --gpu-metrics-set=tu10x-gfxt \
  --trace=cuda,nvtx,osrt \
  --output=output/swe_conc_probe16d_nsight/q36a_astropy_13033 \
  --duration=1800 \
  --delay=30 \
  bash scripts/swe_x86_helpers/run_one_codex_instance.sh astropy__astropy-13033
```

Settings rationale:
- `--gpu-metrics-frequency=10`: 10 Hz sampling — fine enough to see B=1 vs B=2 transitions, coarse enough to avoid drowning in data on a 30-min run.
- `--gpu-metrics-set=tu10x-gfxt`: the metric set including DRAM_ACTIVE, SM_ACTIVE, PIPE_TENSOR_ACTIVE, occupancy. Use the SM121-specific set if available; otherwise tu10x-gfxt is the closest fallback.
- `--trace=cuda,nvtx,osrt`: capture CUDA API calls, NVTX markers (vLLM emits these), and OS runtime traces (thread scheduling).
- `--delay=30`: skip the first 30 seconds (prefill of the initial prompt) so the captured window is steady-state decode.
- `--duration=1800`: 30-min capture, matching the agent budget.

The `.nsys-rep` file will be ~2-5 GB. Extract key metrics:

```bash
# Generate a CSV of GPU metrics over time.
nsys stats --report gpu_metric_usage --format csv \
  --output q36a_astropy_13033_metrics.csv \
  output/swe_conc_probe16d_nsight/q36a_astropy_13033.nsys-rep

# Aggregate by 1-second buckets to compare with vLLM's iter_cnt timeline.
```

#### 13.7.5 Decision criteria

Run a single Nsight profile and inspect:

| Metric | Bandwidth-bound prediction | If true | If false |
|---|---|---|---|
| `DRAM_ACTIVE` median | 75-90% (DRAM mostly busy) | Confirms memory-bound; Paths 1/5 are well-motivated | If DRAM_ACTIVE < 60%, neither memory nor compute is the bottleneck — something else (kernel launch, scheduling) dominates; rethink optimization direction |
| `PIPE_TENSOR_ACTIVE` median | 25-45% (tensor cores partial) | Confirms compute headroom; aggressive drafting is safe | If PIPE_TENSOR_ACTIVE > 70%, drafting more competes with target attention compute; Path 1's MTP fallback becomes risky, Path 5's prior trees less impactful |
| `SM_ACTIVE` vs `SM_OCCUPANCY` | Active ~95%, occupancy ~30-50% | Active SMs spending most cycles on memory stalls; high active but moderate occupancy means lots of warps waiting | If occupancy > 70%, SMs are well-fed and compute is the limit |
| Per-kernel breakdown | Attention + matmul kernels dominate wall time | Expected | If non-matmul kernels (sampling, softmax, layernorm) dominate, focus is on kernel fusion not on drafting |

The PASS condition for proceeding with Paths 1, 5, and the spec's broader compute-headroom assumption:

- `DRAM_ACTIVE` median > 70% AND
- `PIPE_TENSOR_ACTIVE` median < 50% AND
- power_w / TDP ratio consistent with the Nsight-derived utilization (sanity check)

If all three hold, proceed with the spec as-written. If any fail, file a spec amendment before continuing engineering.

#### 13.7.6 What to do with the result

1. **If the bandwidth-bound hypothesis is confirmed (expected):** include the Nsight `.nsys-rep` snapshot in the Round 5 baseline measurement bundle. Reference it in the closeout report as the authoritative compute-saturation evidence. Continue with Paths 1 and 5 engineering.
2. **If the hypothesis is partially confirmed (e.g., DRAM_ACTIVE high but PIPE_TENSOR_ACTIVE also high):** narrow the engineering focus. Path 1's MTP fallback becomes higher-risk because MTP's draft-side compute is non-trivial. Path 5's prior is still safe because it's all CPU-side. Re-prioritize toward Path 5 and DAWG substrate (Path 3) over Path 1.
3. **If the hypothesis fails:** halt §13 engineering. The original spec (§3-§7, calibrated against CNB-55 conditions) may still apply if CNB-55 workloads are differently shaped, but Paths 1/5 specifically for SWE-Bench Verified need re-justification.

#### 13.7.7 Closeout profile (separate from prerequisite)

A second Nsight profile against the composed final stack (SD + MTP fallback + Codex prior + Path 4-lossless bundle) provides the publication evidence for the Round 5 paper. Same methodology as §13.7.4, run on 2-3 representative SWE-Bench instances spanning the per-stream tps range. Cost: ~3-6 wall-hours including profile overhead and analysis.

#### 13.7.8 Engineering effort

- Initial setup + first profile run: **1-2 days** (verify Nsight version, identify representative task, set up the run script, execute, parse).
- Analysis + decision: **half a day** (extract metrics from `.nsys-rep`, compare against PASS criteria, document findings in a short memo).
- Total prerequisite cost: **~2 days wall, well under one engineer-week.**

This is the cheapest measurable insurance against committing 8-10 weeks of engineering effort to a wrong assumption. Run it before Path 1 implementation starts.

---

**End of spec.**
