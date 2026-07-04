# FR13 Context-Compression Design — qwen-code give-up at ~49k (fixv2 era)

Author: red-team subagent, 2026-07-04. Read-only design. No GPU used.
Verified first-hand against: `qwen-code-runner:v1` (= @qwen-code/qwen-code **v0.19.4**, id 40ecf4f50923) on alienware
— including the **exact minified source** of `computeThresholds`, the `reservedOutputTokens`/`hasUserMaxTokensOverride`
branch, and the threshold constants (chunks `chunk-BFG6OZN7.js` + `chunk-YTFBURQD.js`); the served model config
`/models/qwen3.6-27b-fp8/config.json`; `scripts/run_swe_bench_q36_a.py` (template lines 95-102);
`scripts/swe_x86_helpers/relaunch_proxy_remote.sh` (:44, :57-63); `src/lumo_flywheel_serving/inference_proxy.py`
(auto-continue is `/v1/responses`-only via `nonstream_bypass_active`, `normalize_chat_completions` at :173);
`output/fr13_baseline_main/m_tree_cache_fixv1/swe_out/verified/per_task/astropy__astropy-13453/codex_trace.jsonl`
(2×`COMPRESSION_FAILED_EMPTY_SUMMARY`, 2×`hard limit: 48875.2`, `Estimated prompt tokens: 48897`, 15×`cache_read_input_tokens:0`);
`FR13_TREECACHE_CAMPAIGN_20260704.md` (§2 fence, §17 fixv1 wall, §18 route flip).
**Adversarial-review pass (this session) reconciled the arithmetic to the in-image formula, corrected the KV math
against the real config, decoupled the fence fix from R1, added the mandatory re-baseline, and tightened R2a; see
inline notes.** (The trace shows 15 usage records; §17 counts 16 turns — one turn carried no usage block.)

## 0. The failure, restated with verified numbers

- fixv1 (§17) eliminated the drift give-up (16 turns, coherent) but hit a NEW terminal wall:
  qwen-code auto-compression fired **twice at ~49k**, both returned `COMPRESSION_FAILED_EMPTY_SUMMARY`,
  hard-stopped the session with a 0-byte patch.
- Trace (verified this session): `2× COMPRESSION_FAILED_EMPTY_SUMMARY`, `2× "Context is too large"`,
  `Estimated prompt tokens: 48897`, `hard limit: 48875.2`, **15/15 turns `cache_read_input_tokens:0`**.
- Source of the number (Track A, and now re-verified byte-for-byte in the v0.19.4 image — see the exact
  `computeThresholds` body in chunk-BFG6OZN7.js and constants in chunk-YTFBURQD.js):
  `contextLimit = window2` is passed to `computeThresholds(window2, pct=0.7)`, whose real body is
  `effectiveWindow = window2 − SUMMARY_RESERVE(20000)`; `auto = max(0.7·window2, effectiveWindow − AUTOCOMPACT_BUFFER(13000))`;
  `rawHard = effectiveWindow − HARD_BUFFER(3000)`; **`hard = min(window2, max(rawHard, auto + HARD_BUFFER))`**.
  At `window2=65536`: `auto = max(45875.2, 32536) = 45875.2`, `rawHard = 42536`, so
  **`hard = max(42536, 48875.2) = 48875.2 exactly`** (the `auto+3000` branch dominates; rawHard is smaller here).
  (`SUMMARY_RESERVE = COMPACT_MAX_OUTPUT_TOKENS = 20000`, all values confirmed in-image: DEFAULT_PCT=0.7,
  AUTOCOMPACT_BUFFER=13e3, HARD_BUFFER=3e3, WARN_BUFFER=2e4, ESCALATED_MAX_TOKENS=64e3, DEFAULT_TOKEN_LIMIT=131072.)
  The exact match proves *no* settings override was active (all defaults; `hasUserMaxTokensOverride=false`).
- **Why the budget is only 65536 (half the served 131072):** `rawContextLimit = contextWindowSize ?? DEFAULT_TOKEN_LIMIT = 131072`;
  `contextLimit = rawContextLimit − reservedOutputTokens`. With no override,
  `reservedOutputTokens = max(ESCALATED_MAX_TOKENS=64000, tokenLimit(model,"output")) = max(64000, 65536) = 65536`
  (the model-specific `knownTokenLimit(model,"output")=65536`; note the generic `DEFAULT_OUTPUT_TOKEN_LIMIT` is only 32000,
  so 65536 is a known-model entry, not the fallback). So the agent throws away **65536 of its 131072 window on an
  output reserve it never uses** — the proxy already caps real output to `LUMO_PROXY_MAX_OUTPUT_TOKENS=32768`
  (verified `relaunch_proxy_remote.sh:44`).
- **The knob is real and verified in-image** (exact code read this session, chunk-BFG6OZN7.js):
  `parsedEnvMaxTokens = parsePositiveIntegerEnvValue(process.env["QWEN_CODE_MAX_OUTPUT_TOKENS"])`; then
  `hasUserMaxTokensOverride = (samplingParams?.max_tokens != null) || parsedEnvMaxTokens !== undefined`; and
  `reservedOutputTokens = hasUserMaxTokensOverride ? (samplingParams?.max_tokens ?? parsedEnvMaxTokens ?? 0)
  : max(ESCALATED_MAX_TOKENS, tokenLimit(model,"output"))`. Setting the env var therefore **short-circuits the
  `max(64000, …)` floor entirely** and pins `reservedOutputTokens` to the env value → directly moves `contextLimit`
  and the hard limit. **Precedence caveat:** `samplingParams?.max_tokens` wins over the env var; the env var only
  takes effect while `samplingParams.max_tokens` is unset — which it is today (the failing run had
  `hasUserMaxTokensOverride=false`, so nothing sets it). If a `.qwen/settings.json` `generationConfig.max_tokens`
  is ever added, it would override `QWEN_CODE_MAX_OUTPUT_TOKENS`.

### The four binding constraints (every rung is checked against these)
- **(a) NUDGE-FREE** — no mid-run task guidance injected. Deterministic elision of the agent's *own prior tool
  outputs*, and pure budget/threshold config, are NOT nudges. The proxy's `LUMO_PROXY_AUTO_CONTINUE_MESSAGE`
  IS a forceful task-guidance nudge (`relaunch_proxy_remote.sh:35`, default ON with `LUMO_PROXY_AUTO_CONTINUE=1`) —
  but it is inert here: verified in source (`inference_proxy.py:2222/2436/2532`), the auto-continue loop is gated by
  `nonstream_bypass_active`, which is set **only** inside the `/v1/responses` branch, and its retry logic reads the
  Responses-schema `output[].function_call` — so it cannot fire on qwen-code's `/v1/chat/completions`; and the
  compression hard-stop is client-side, so the proxy never sees the over-limit turn. **Precondition:** this
  inertness (and the fairness it buys) holds only while **every** arm runs through qwen-code (`/v1/chat/completions`).
  If any comparison arm (e.g. a codex "native") runs through `/v1/responses`, the nudge fires for *that arm only* →
  fairness break. The tree+cache campaign is all-qwen-code today, so it holds — but confirm before mixing agents.
- **(b) EVAL FAIRNESS** — one identical context policy across native / tree / cache-on / cache-off (same trigger,
  same reserve, same elision text, same seed). Everything below is a container-env/image change applied uniformly.
- **(c) SAME-STACK COMPRESSION** — the summarizer is served by the arm under test at forced temp 0.6 +
  presence_penalty 1.0 (proxy `normalize_chat_completions_request_payload`). Compression quality is therefore
  *coupled to serving health* and differs by arm. This is a confound, addressed head-on in R3/R4 and side-stepped
  by R1 (make compression rarely fire) + R2 (deterministic fallback that needs no model call).
- **(d) GPU/READ-ONLY** — no GB10 use; alienware SSH read-only; no long-lived containers. All GPU-gated steps
  below are specified as operator runnables, not run here.

---

## RANKED LADDER (cheapest first)

| Rung | What | Impl effort | Fires | Removes run-kill? | Fairness |
|---|---|---|---|---|---|
| **R1** | Raise agent budget via output-reserve knob | trivial (1 env var) | before compression triggers | for this task class | uniform env |
| **R2** | Compression-failure → deterministic fallback, never hard-stop | small (image patch or orchestrator classify) | when compression still fires | yes, always | same image/harness |
| **R3** | Fix the summarize call + diagnose EMPTY_SUMMARY (serving-dependency probe) | medium (probe + config) | when a summary is truly needed | reduces empty-rate | off-arm/deterministic |
| **R4** | Structured SWE compaction + subagent quarantine | large (engineering cycle) | steady-state design | yes, by construction | structural, uniform |

---

## R1 — BUDGET: reclaim the wasted output reserve (DO NOW)

**Root cause is not "context too big"; it is "budget artificially halved."** The served window is 131072 and
qwen-code already defaults `contextWindowSize`→`DEFAULT_TOKEN_LIMIT=131072` (matches `max_model_len`). The only
reason the agent compresses at 49k is a 65536-token output reserve that the proxy caps to 32768 anyway.

**The knob (verified in v0.19.4):** container env `QWEN_CODE_MAX_OUTPUT_TOKENS`.
Thread it through `QWEN_CODE_TEMPLATE` in `scripts/run_swe_bench_q36_a.py:95-102`
(add `-e QWEN_CODE_MAX_OUTPUT_TOKENS={value}` to the `docker run`).

**Proposed value: `32768`** — set the reserve equal to the real proxy output cap (`LUMO_PROXY_MAX_OUTPUT_TOKENS`).
Not lower: a single long `apply_patch`/edit turn can legitimately generate up to the cap, and under-reserving
risks truncated generations. Not higher: anything above 32768 is pure wasted input budget. Do **not** raise
`contextWindowSize` above 131072 — that exceeds `max_model_len` and vLLM will reject the request.

**Resulting arithmetic (computed with the in-image `computeThresholds`, constants verified in the v0.19.4 image:
SUMMARY_RESERVE=20000, AUTOCOMPACT_BUFFER=13000, HARD_BUFFER=3000, pct=0.7):**

| reservedOutput | contextLimit (=131072−reserve) | soft/auto trigger | **hard limit** | headroom over 48897 |
|---|---|---|---|---|
| 65536 (today, default) | 65536 | 45875.2 | **48875.2** | −22 (FAILS) |
| **32768 (proposed)** | **98304** | 68812.8 | **75304** | **+26407 (safe)** |
| 16384 (aggressive) | 114688 | 81688 | 91688 | +42791 (risks truncated gen) |

At reserve=32768 the hard limit rises **+26429 tokens (+54%)** and astropy-13453's 48897 clears with ~26k to spare.
Here the `rawHard = effectiveWindow−3000 = 75304` branch dominates (75304 > auto+3000 = 71812.8). Both baseline
(48875.2) and proposed (75304) reproduce **exactly** from the in-image formula. **Note on the 16384 row:** above
`window2 ≈ 110000` the `auto` term flips from the `0.7·window2` branch to the `effectiveWindow−13000` branch, so at
reserve=16384 `auto = max(80281.6, 81688) = 81688` and `hard = min(114688, max(91688, 84688)) = 91688` — an earlier
draft mis-carried these as 80281.6/88304. (Recommendation is the 32768 row, unaffected.)

**KV / memory math at B=1 on GB10 (the key point: R1 is footprint-neutral). Served config now read directly from
`/models/qwen3.6-27b-fp8/config.json` (no GPU used) — resolves the prior open question:**
- vLLM boots with `max_model_len=131072` and **pre-allocates the KV block pool for that window at boot**
  (governed by `GPU_UTIL`). Raising the agent's *usable* budget 49k→75k just lets the agent fill KV blocks vLLM
  **already reserved**; it does not grow the process footprint beyond the boot reservation. **This is the whole
  memory argument — R1 is net-zero on peak VRAM regardless of the per-request KV delta below.**
- Exact hybrid config: `num_hidden_layers=64`, `layer_types = 48 linear_attention (GDN) + 16 full_attention`
  (`full_attention_interval=4`), `num_key_value_heads=4`, `head_dim=256`, `kv_cache_dtype=fp8_e5m2` (1 byte).
  Only the 16 full-attn layers grow KV with sequence length; the 48 GDN layers hold a fixed-size recurrent state.
  Per-token full-attn KV = `2(K+V)·4·256·1B·16 layers = 32 KiB/token`. Going 48897→75304 tokens (Δ26407) ≈
  **+0.81 GiB KV at B=1** (full 75304-token full-attn KV ≈ 2.30 GiB) — *within the already-reserved pool*, not
  net-new. (An earlier draft's "+1.3–1.7 GiB" over-counted; at the true 4 KV-heads / fp8 it is ~0.8 GiB.)
- The campaign's real memory actor (`§2`) is the **bounded ES recurrent-checkpoint store** (plateau ~9.2 GB ≈
  64×144MiB, LRU-capped, **sequence-length-independent**). R1 does not touch it.

**Fence / GPU_UTIL interplay (campaign §2) — apply the §2 fix, but for its OWN reason, not R1's:** §2/§6 already
show `gpu_oom_guard` (floor 9000MiB) was **killing healthy runs** at the ES-store plateau (~8.86–9.2 GB), a
pre-existing tax independent of R1. Because the KV pool is fixed at boot (above), a longer per-request context does
**not** raise the process footprint the guard measures, so R1 adds no fence pressure — the earlier "longer context
raises occupancy → re-grazes the fence" framing was wrong. Still **apply GPU_UTIL 0.82→0.78 (+4.7 GB clearance)**
because §2 independently requires it; it is orthogonal to and complementary with R1. (Diagnostic
`GPU_GUARD_FLOOR_MIB=6500` in use gives ample margin during screening.)

- **Impl effort:** trivial — one `-e QWEN_CODE_MAX_OUTPUT_TOKENS=32768` in the template; no image rebuild.
- **Risk:** very low. Fairness: identical env for every arm (b). Nudge-free (a): pure budget config. Only residual
  risk = a genuinely huge task still exceeds 75304 → compression fires → needs R2 as the safety net.
- **Verify:** (1) re-run astropy-13453 tree+cache; confirm trace `hard limit: 75304` and **zero
  `COMPRESSION_FAILED_*`**; (2) confirm the fingerprint arms (native / tree / cache-off) show the identical
  `hard limit` line (fairness); (3) nvidia-smi/guard log shows no fence graze at the §2 GPU_UTIL setting.

### R1 forces a full re-baseline of ALL arms (fairness, constraint b) — do NOT compare across the budget change
R1 changes the context budget **identically for every arm**, which keeps *future* runs fair — but it makes every
**already-banked qwen-code number invalid for comparison**. All prior baselines (the §1 clean matrix, §5/§10 fix-arm
verdicts, the §17 fixv1 phenotype, the §18 route distribution, and any solve-rate / turns / give-up counts) were
collected at `reservedOutput=65536` (hard=48875.2) and cannot be compared to post-R1 runs at hard=75304.
**Re-baseline all four arms — native, tree, cache-on, cache-off — under R1 before any cross-arm claim.** This is not
optional bookkeeping: R1 changes the *operating regime* of the phenomenon under study. Today the tree+cache arms
hard-stop at ~49k; under R1 they run to ~75k, giving the tree×cache degradation (§11/§15: long single-context drift
under cache) **more context to accumulate**. The give-up/resolve comparison and the lossless/superset gates may move
— possibly the give-up reappears at higher context, or the tree-vs-native gap widens — and that is a *result to
measure*, not an artifact to suppress. Same caution applies to R2b, R3a, and R4 (each changes behavior → re-baseline).

---

## R2 — FAILURE HANDLING: compression failure must never kill the run (DO NOW)

Today the failure path is a **hard stop**: the compression call is `maxAttempts:1`; on
`COMPRESSION_FAILED_EMPTY_SUMMARY` at the hard tier, `shouldStopAfterHardRescue()` aborts the turn and
`getHardRescueFailureMessage()` becomes the terminal agent message → 0-byte patch → `failed`. R1 makes this rare;
R2 makes it non-fatal when it still fires (long tasks > 75k). Two sub-rungs, apply both:

**R2a — orchestrator classify (cheapest, no image rebuild) — but DO NOT blanket-exclude (fairness hazard).**
In `run_swe_bench_q36_a.py`, detect a terminal `COMPRESSION_FAILED_*` / "Context is too large" in
`codex_trace.jsonl` and record it as a distinct terminal reason (`compression_abort`) rather than a generic
`failed`. **The hazard:** auto-excluding every `compression_abort` from the denominator *selectively rescues the
tree+cache arm*, because §18 shows all six tree+cache variants take the monolithic `read_file` route and are the
ones that reach ~49k, while native / tree+nocache take the subagent route and never compress. Reaching compression
is therefore **partly the arm's own behavior** (§18 calls the route flip a tree×cache logit-flip carrier), not a
neutral infra hiccup — excluding it would launder a genuine tree+cache defect. Constraint (c) has two *different*
arm-couplings that must not be conflated: (i) the **summarizer call's quality** (EMPTY_SUMMARY) genuinely depends on
serving health and is fair to neutralize; (ii) **whether the arm reached compression at all** is a route-economics
consequence that may be the defect under study. So:
- **Default: exclude nothing.** Prefer R2b (continue past the failure), which removes the summarizer-quality confound
  (i) *without* removing the run — the arm still has to produce a patch under its own route economics.
- **Conditional exclusion only:** a `compression_abort` may be dropped from the lossless/superset denominator **only
  after R3b proves the EMPTY_SUMMARY is a serving-health artifact for that arm** (the summarizer call itself is
  defective given equivalent context), and the exclusion rule must be identical for all arms (b). Absent that proof,
  count it as a legitimate (arm-caused) non-solve.
- Effort: small Python. Injects nothing into the run (a). Verify: `compression_abort` is a labeled terminal reason in
  the report for every arm; nothing is silently removed from the denominator without the R3b verdict on record.

**R2b — make qwen-code continue instead of aborting (smallest image patch).** Patch the hard-rescue path in
`chunk-BFG6OZN7.js` (rebuild `qwen-code-runner:v2`) so that on `EMPTY_SUMMARY`/`INFLATED_TOKEN_COUNT`:
1. **retry once** (the compression call is currently `maxAttempts:1`);
2. on second failure, **fall back to deterministic oldest-tool-output truncation** — drop/placeholder the *oldest*
   large tool outputs (`read_file`/`bash` bodies > 500 chars), **skip the most recent N (=5)**, and always keep the
   task statement + recent tail — then **continue the turn**. This is the Claude-Code "microcompact" / SWE-agent
   `last_n_observations` shape; it injects zero model-generated and zero human text (a).
3. set a **`hasFailedCompressionAttempt` latch** (gemini-cli #16213) so compression doesn't re-fire every turn.
4. **Never** emit `getHardRescueFailureMessage()` as a terminal message; never fall through to the nudge.
- Effort: medium — patch minified chunk (or rebuild qwen-code from source with the patch) + rebuild image; operator
  action (I don't build images). Risk: low; the fallback is deterministic and identical across arms (b). It removes
  the *class* of run-kill (works even for tasks that exceed any budget).
- **Note:** R2b's deterministic truncation is the same mechanism as R4's steady-state design — R2b is the minimal
  emergency version; R4 is the principled always-on version. Building R2b well is a down payment on R4.
- Verify: force a small `contextWindowSize` (or a low `QWEN_CODE_MAX_OUTPUT_TOKENS`) to trigger compression early on
  a scratch task, confirm the run **continues** past the failure with truncated-but-present history, latch set,
  no terminal error message, and identical behavior native-vs-tree.

---

## R3 — COMPRESSION-CALL HYGIENE + serving-dependency diagnosis (operator probe, GPU-gated)

Two parts: fix the summarize request itself, and settle *why* it comes back empty on this stack.

**R3a — request hygiene (config-only where possible).**
- The compression call is a normal `/v1/chat/completions` side-query, so it inherits the proxy's **full forced
  sampling set** (verified `relaunch_proxy_remote.sh:57-63` + `normalize_chat_completions_request_payload`
  `inference_proxy.py:173-214`, active because `LUMO_PROXY_QWEN_SAMPLING=1` by default):
  **temp 0.6, top_p 0.95, top_k 20, presence_penalty 1.0, min_p 0** — not just temp 0.6 + pp 1.0. It runs with
  `maxOutputTokens=COMPACT_MAX_OUTPUT_TOKENS=20000` over ~49k tokens of slimmed history asking for a 9-section
  `<state_snapshot>` XML. **presence_penalty 1.0 together with the tight top_k 20** on a long *structured* emit is a
  plausible driver of degenerate/empty output. **Route the compression sub-request to deterministic sampling**
  (temp 0, presence_penalty 0) — either via a proxy carve-out that detects the compression system prompt
  (`getCompressionPrompt`/`state_snapshot` marker) and neutralizes sampling, or by preferring R4's model-free path
  so the call never happens. Deterministic + identical across arms satisfies (b)+(c).
- Bound the summarize **input** (don't summarize the whole 49k in one shot at temp 0.6): prefer hierarchical/chunked
  summarization or, better, the deterministic pre-pass (R4) so the LLM call — if any — sees a small, bounded input.
- Confirm `maxOutputTokens` (20000) < served output cap (32768) — it is, so no #7578-class "maxOutputTokens exceeds
  supported range" paradox here; keep it that way if the reserve changes.

**R3b — serving-dependency probe (the decisive experiment; GPU-gated, operator runs).**
Question: is EMPTY_SUMMARY a **serving-health artifact** (tree+cache garbles/empties the 49k structured generation)
or a **parse artifact** (model emits only `<analysis>`, no `<state_snapshot>`, `stripAnalysisBlock`→empty)?
Runnable spec for the operator (no code I can run here — needs the GPU + a live boot):
1. **Capture the exact side-query.** Enable `LUMO_PROXY_PAIR_DUMP_DIR` / `LUMO_PROXY_REQUEST_DUMP_DIR` on the
   **alienware** proxy (GB10-side dumps are empty for offloaded arms — §17b), reproduce astropy-13453 tree+cache to
   the compression turn, and save the `chatreq_*.json` compression request (system=`getCompressionPrompt`,
   contents=slimmed history + the "First, reason in your `<analysis>` block… then `<state_snapshot>`" user turn).
   Fallback if dumps stay empty: reconstruct the request deterministically from `codex_trace.jsonl` history.
2. **Replay as a 2×2** = {(A) healthy native-MTP-5 boot, (B) tree+cache boot} × {production sampling
   (temp 0.6/top_p 0.95/top_k 20/pp 1.0/min_p 0), temp 0}, identical request bytes. The **production-sampling row is
   the decisive one** — that is the setting under which the real EMPTY_SUMMARY occurred, so it directly attributes
   the production failure; the temp-0 row isolates boot-health from sampling. Diff outputs:
   - tree+cache empty/garbled while native yields a valid `<state_snapshot>` (at the SAME sampling) → **serving-health
     artifact** → the summary confound is real → compression must be moved off-arm/deterministic, AND this is the
     evidence that licenses R2a's *conditional* exclusion of that arm's `compression_abort` runs (never a blanket
     exclude — see R2a).
   - both empty → **parse/prompt artifact** (model emits only `<analysis>`, `stripAnalysisBlock`→empty) → fix is
     prompt/parse-side (strip-analysis leniency, or a deterministic path that never depends on the XML).
   - production-sampling both-empty but temp-0 both-fine → **sampling artifact** (top_k 20 + pp 1.0) → R3a's
     deterministic re-routing alone fixes it, arm-independently.
- Effort: medium (capture + 2 replays). Risk: GPU-gated, serialized (respect MAX-2-workflows; this is 1 GPU arm).
- Verify: the A/B diff itself is the verdict; it also tells R4 whether any LLM summary is trustworthy on this stack.

---

## R4 — REAL COMPACTION for SWE sessions (later engineering cycle)

Prior art scanned (Track B): Anthropic API compaction (150k trigger far below the window + `pause_after_compaction`
verbatim re-inject + cached-separately system prompt), Claude-Code **microcompact** / qwen-code FR #2817
(deterministic no-LLM pre-pass over stale tool outputs, skip recent N), **OpenHands condensers** (pinned head +
LLM-summarized middle + verbatim tail, condense only above `max_size` so the prefix stays cache-stable — ~2× cost
cut at equal SWE score), **SWE-agent history processors** (fully deterministic `last_n_observations`, always keep
problem statement, `tag_tool_call_observations`), MemGPT/Letta (recursive summary pinned at index 0), and
**subagent quarantine** (the Explore route that already works on this stack — §18 shows native/tree+nocache pick the
subagent route; all six tree+cache arms picked the monolithic `read_file` route that pumps raw bodies into the main
loop until 49k).

**Design for qwen-code (0.19.4 architecture), three layers:**
1. **Deterministic pre-pass first (primary; model-free).** Before any LLM summary, replace **stale** tool outputs
   (`read_file`/`bash` > 500 chars) older than the most-recent N (=5) with a short fixed placeholder
   (`"[old read_file output elided: <path>, <n> lines]"`). This is the qwen-code partial already shipping
   (`maxRecentFilesToRetain=5`, env `QWEN_COMPACT_MAX_RECENT_FILES`) generalized to bash. Zero model text (a),
   identical transform across arms (b), **removes the served-model dependency entirely** (c). Highest-leverage change.
2. **Structured pin/summarize/drop split** (OpenHands/SWE-agent/Anthropic-common):
   - **PIN verbatim (never summarize):** the task/problem statement (`AGENTS.md`), the current plan, the live
     patch/diff state, and the last failing-test output. These are the SWE working set.
   - **SUMMARIZE (only if the pre-pass is insufficient):** older *exploration* turns — and only via a
     deterministic-sampling / off-arm endpoint (R3a), never the arm under test at temp 0.6.
   - **DROP:** stale tool dumps (handled by layer 1).
   - Keep the **recent tail** verbatim (last few turns) and a **pinned head**, so condensation only touches the
     middle → the prefix stays cache-stable (see prompt-caching note).
3. **Subagent quarantine for exploration** (structural, nudge-free, already proven on this stack). Route
   `read_file`/`grep`/large tool outputs through an Explore subagent with its own fresh context that returns only a
   condensed finding; the main loop never accumulates raw file bodies and never approaches 49k. Because §18 shows the
   route choice is itself the tree×cache logit-flip carrier, forcing the subagent route harness-side *also* fixes the
   context economics regardless of the route flip — but forcing a route must be done **uniformly across all arms**
   (b) and **without task guidance** (a) (a structural harness setting, not a prompt nudge).

**Prompt-caching sub-item (relevant to all rungs).** Trace shows `cache_read_input_tokens:0` on **all 15 turns** —
the offload proxy re-sends full history every turn, so context grows maximally fast and the compression crossing
arrives sooner. Stabilizing the prefix (cache the system prompt/head separately; condense-only-above-threshold to
keep the prefix byte-stable) slows growth and pushes the crossing out. **Tension:** raw `last_n_observations`
elision changes history each turn and *breaks* caching — so prefer **threshold-triggered condensation with a stable
pinned head** (OpenHands shape) over per-turn elision if caching is enabled. Enabling upstream prompt caching also
interacts with the tree+cache defect under study, so treat it as its own gated experiment, not a free win.

- Effort: large (multi-file harness/image work + the caching interaction study). Risk: medium; must preserve (a)/(b)
  and be validated against the lossless/superset gates.
- Verify: on a task that today hits 49k, confirm (1) it completes without compression, (2) the pinned SWE state
  (task/plan/patch/failing-test) is present verbatim at every turn, (3) identical policy fingerprint across arms,
  (4) subagent route never accumulates raw file bodies in the main loop.

---

## What to implement NOW vs later

**NOW (fixv2 pair — minimum):**
- **R1** (`QWEN_CODE_MAX_OUTPUT_TOKENS=32768` in the template) — trivial, moves hard limit 48875→75304, clears the
  astropy-13453 class, memory-neutral at B=1. **Pair with §2 GPU_UTIL 0.82→0.78** so the slightly higher KV
  occupancy never re-grazes the 9000MiB fence.
- **R2a** (orchestrator *labels* a `compression_abort` terminal reason — but does NOT auto-exclude it; exclusion is
  conditional on the R3b serving-health verdict, else it selectively rescues the tree+cache arm) — a few lines, uniform.
- **R2b** if a v2 image build is already in scope (it makes the run-kill class impossible for tasks that still exceed
  75k). If not, R2a covers fairness until the next image cut.

This is the R1+R2 minimum the campaign asked for: R1 stops compression firing for the common case; R2 removes the
run-killing behavior for the tail. Both are nudge-free (a), uniform (b), and side-step the same-stack summarizer
confound (c) — R1 by avoiding the call, R2 by a model-free fallback.

**LATER (engineering cycle):**
- **R3** — run the serving-dependency probe (R3b) to settle whether EMPTY_SUMMARY is a tree+cache serving artifact;
  fix compression-call sampling (R3a). This is the input to trusting *any* LLM summary on this stack.
- **R4** — build the structured deterministic compaction + subagent quarantine as the steady-state design. R2b's
  deterministic fallback is the seed; R4 generalizes it and adds the pin/tail/cache-stable prefix. This also
  neutralizes the §18 monolithic-route context bloat harness-side.

**Do NOT do (recorded so nobody re-proposes it):** using `LUMO_PROXY_AUTO_CONTINUE_MESSAGE` or any injected retry to
"rescue" the stop — it is a nudge (banned, a) and is inert on the qwen-code path anyway. Do not raise
`contextWindowSize` above 131072 (exceeds `max_model_len`). Do not leave LLM summarization served by the arm under
test at temp 0.6 as the primary path (breaks c).

## Open items to confirm before shipping R1
- ~~Exact served attention config~~ **RESOLVED this session** from `/models/qwen3.6-27b-fp8/config.json`:
  64 layers = 48 GDN + 16 full-attn, `num_key_value_heads=4`, `head_dim=256`, `kv_cache_dtype=fp8_e5m2` →
  **32 KiB/token**, Δ48897→75304 ≈ **+0.81 GiB** full-attn KV at B=1 (within the pre-reserved pool). No longer open.
- Confirm the launch env for the fixv2 arms did not already set `QWEN_CODE_MAX_OUTPUT_TOKENS` (or a
  `.qwen/settings.json` `generationConfig.max_tokens`, which would win over the env var) elsewhere — the exact
  48875.2 match plus `hasUserMaxTokensOverride=false` says both were unset for the failing run, but verify the
  launch env directly.
- **Re-baseline gate:** all four arms (native/tree/cache-on/cache-off) must be re-run under R1 before any cross-arm
  comparison; treat the pre-R1 banked numbers as a different regime (see the R1 re-baseline note).
- R3b verdict (serving-health vs sampling vs parse) decides (i) whether R4 may keep any LLM summary on this stack or
  must be fully model-free, and (ii) whether R2a is *ever* allowed to exclude a `compression_abort` run.
- Prompt caching: all 15 usage records show `cache_read_input_tokens:0` (offload proxy re-sends full history each
  turn). Whether enabling upstream prompt caching is safe under the tree+cache defect — and how much a stable prefix
  would push out the compression crossing — is a separate gated experiment (see R4 caching sub-item).
