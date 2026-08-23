# RUNAWAY BRAKE SPEC — thinking-budget first line

Proposal for Mark. Not a landing. Arming is his decision.

Redesigned around the `thinking_budget` finding: the model already supports a
hard cap on thinking tokens, our serve sets none, and the degeneration is a
thinking-block runaway. A native cap beats anything I would build.

---

## 1. HEADLINE

**Our serve sets no thinking budget, and the data says the official default of
4000 would be badly wrong for this workload — it would cut 39% of healthy
task-arms. The right number from our own corpus is ~24,000.**

The separation is not marginal:

| | tokens |
|---|---:|
| healthy per-arm max thinking block, n=100 | median **3,518**, p90 10,316, p95 12,658, p99 **18,653**, max **22,398** |
| degenerate arms (all 5) | 30,731, 32,768, 32,768, 32,768, 32,768 |
| **separation gap** | **8,333 tokens, zero overlap** |

Four of five degenerations sit exactly at **32,768** — they ran until they hit an
existing hard ceiling. A ceiling exists; it is simply too high to prevent the
damage it was meant to bound.

### Cut-rate by candidate budget (healthy arms only, 100 arms / 2,776 blocks)

| budget | healthy arms cut | healthy blocks cut |
|---:|---:|---:|
| **4,000 (official default)** | **39 (39.0%)** | 96 (3.46%) |
| 8,000 | 18 (18.0%) | 25 (0.90%) |
| 12,000 | 6 (6.0%) | 8 (0.29%) |
| 16,000 | 2 (2.0%) | 2 (0.07%) |
| 20,000 | 1 (1.0%) | 1 (0.04%) |
| **24,000 (recommended)** | **0 (0.0%)** | **0 (0.00%)** |
| 30,000 | 0 | 0 |

Finer curve below the recommendation, since Mark may want tighter:

| budget | healthy arms clipped | healthy blocks clipped |
|---:|---:|---:|
| 2,000 | 74 (74.0%) | 246 (8.86%) |
| 4,000 *(official)* | 39 (39.0%) | 96 (3.46%) |
| 6,000 | 26 (26.0%) | 43 (1.55%) |
| 8,000 | 18 (18.0%) | 25 (0.90%) |
| 10,000 | 12 (12.0%) | 16 (0.58%) |
| 12,000 | 6 (6.0%) | 8 (0.29%) |
| 14,000 | 4 (4.0%) | 5 (0.18%) |
| 16,000 | 2 (2.0%) | 2 (0.07%) |
| 20,000 | 1 (1.0%) | 1 (0.04%) |
| 22,398 *(healthy max)* | **0** | **0** |
| 24,000 *(recommended)* | **0** | **0** |

The knee is sharp: everything above ~12,000 costs at most a handful of arms,
and 22,398 is where the cost reaches zero. Anything below ~10,000 is a
behaviour change on a tenth or more of the corpus.

**Recommendation: 24,000.** It is 1,602 above the healthy maximum (+7.2% headroom)
and 6,731 below the smallest degenerate block (−21.9% margin). It is the smallest
round number that cuts nothing healthy in the entire banked corpus.

Method: every `type:"thinking"` block in all 105 banked `qwen_trace.jsonl`
task-arms, tokenized with the served model's own `tokenizer.json` (not a
chars/token estimate — the enumeration class has an atypically low ratio and an
estimate would have skewed the budget low, which is the dangerous direction).

**If Mark prefers the official 4000, that is a behaviour change on ~2 in 5
tasks, not a safety net.** It should be measured, not assumed.

---

## 2. THE MECHANISM ALREADY EXISTS — I MISSED IT

**Correction, and it is the substance of this revision.** `LUMO_PROXY_THINK_BUDGET`
has existed since FR13 `439c43567`, built for the char-8 degeneration era. It
implements the two-phase design I proposed, and implements it better than I
specified. My §2 survey searched the *pinned vLLM tree* for a budget and the
proxy for `"seed"`, and never grepped our own proxy for `think`. That is the
error: I surveyed the vendor for a mechanism the house had already built.

What exists, at `inference_proxy.py:3951` (`_parse_think_budget`), `:4030`
(`_think_build_cutoff_prefill`), `:5074-5130` (the call site):

- cap call A at N tokens, remembering the original budget for call B's answer;
- on a thinking dead-end, force the close via a `continue_final_message`
  re-issue;
- default OFF, byte-identical legacy path when unset;
- `LUMO_PROXY_THINK_CUTOFF` overrides the cutoff text.

Its docstring also answers my layer survey directly and supersedes it: **native
`thinking_token_budget` is voided by MTP**, and `enable_thinking` /
`reasoning_effort` **are not forwarded by the Responses API**. The proxy cap is
the house mechanism by prior design, not a fallback.

### (a) COVERAGE — RESOLVED, and it moved which knob does the job

**The SWE client rides `/v1/chat/completions`.** The proxy ingress ledger
(`logs/fr13_fixed32_proxy_ingress.jsonl`) records a `route` per request:
**11,918 `chat` vs 110 `responses` across 55 banked runroots** — and the
`responses` count is exactly 2 per runroot, both `request_rejected` /
`preflight` auth probes (`missing_bearer`, `malformed_bearer`). Zero real
generation traffic on `/v1/responses`. The proxy's own comment at the chat
branch calls it "the qwen-code path".

**So `LUMO_PROXY_THINK_BUDGET` cannot reach our serves. Arming it would have
been a placebo.**

Extending it to chat is not the cheap build it looked like either: a captured
client body shows `stream: true` with `max_tokens: 32768`. The two-phase cap
needs the complete call-A response, so covering chat means switching the
agent's live streaming path to non-stream bypass and synthesising SSE back —
a change to **every healthy request's** time-to-first-token. That violates
zero-perturbation-when-not-firing, which is the property the whole brake rests
on.

**What was armed instead — and it is the same ceiling, on the right path.**
`LUMO_PROXY_MAX_OUTPUT_TOKENS` is applied in
`normalize_chat_completions_request_payload` (the client's path), only ever
lowers `max_tokens`, and was **already defaulted to 32768** — exactly the
ceiling all five degenerations ran into. The landing lowers it to **24,000**
and pins it in the launcher.

The semantic difference, stated plainly: this caps thinking+answer **total**,
where `THINK_BUDGET` would cap thinking alone and force an answer. It costs
nothing on healthy turns because **healthy answers are small** — the healthy max
total per turn (22,398) is the *same number* as the healthy max thinking block,
so a total cap clips exactly what a thinking cap would (0/105 arms). A
degenerate turn is truncated rather than force-closed. If force-close semantics
are specifically wanted, that needs the chat-path build and its streaming cost.

### (a-old) THE ORIGINAL COVERAGE QUESTION

The cap is inside `if self.path == "/v1/responses":` (`:5078`). The proxy also
serves `/v1/chat/completions` (`:5142`), and **nothing in that branch arms the
cap**. Both facts are pinned by
`tests/test_fr14_think_budget_cap.py::test_the_cap_is_wired_only_into_the_responses_path`.

**I could not determine from banked artifacts which route the SWE client uses,
and I am not going to guess.** What I checked and what it gave:

- `fixed32_proxy_ingress_preflight.json` records both routes — but it is an
  *auth* preflight that probes both by design, not live traffic.
- The 80 `/v1/chat/completions` hits in `docker_*.log` are vLLM's **startup
  route-registration banner**, not requests.
- `fr13_bigdenom_swe_serve_variant.sh:2608` points at `/v1/chat/completions`,
  but that is the `PROBE_ONLY` de-confounder path, not the agent; the agent gets
  `--agent-endpoint .../v1` (`:2663`) and picks its own route.
- Circumstantial for Responses: `LUMO_PROXY_NONSTREAM_BYPASS=1` **is armed** in
  the banked exact16 arm, and it only has effect inside the `/v1/responses`
  branch — arming a no-op would be odd. The capture path and
  `_think_extract_reasoning` are Responses-shaped (`parsed["output"]`).

**Cheap verifications, either settles it in minutes:** (i) one line of proxy
access logging, or (ii) arm `LUMO_PROXY_THINK_BUDGET` at a deliberately tiny
value on a throwaway arm and see whether the cap fires. **If the client is on
chat/completions, extending the cap to that branch is the only build this brake
needs.** If it is on Responses, there is nothing to build at all — only a value
to choose.

### (b) THE INJECTION, RE-VERIFIED ON CURRENT CODE — and I had it wrong

Exercised on CPU against the present proxy (15 tests, `test_fr14_think_budget_cap.py`):
budget parsing, over-budget detection, prefill shape, under-budget no-op.

**The correction matters.** My first spec proposed appending `</think>`. The
existing implementation deliberately does the opposite: it prefills an **open**
`<think>` + call-A's reasoning + a terse cutoff, and lets the **model generate
the close**. The docstring records why, verified on qwen3.6-27b: the qwen3
reasoning parser only watches *generated* tokens, so a **prefilled** `</think>`
mislabels the entire continuation as reasoning and **the tool call is lost**.
My proposal would have silently broken tool calling — the exact failure mode the
brake exists to prevent. The existing design is correct and mine was not.

It also records that the cutoff must be **terse**: the verbose Qwen official
framing lets the model keep thinking.

The interaction claim stands unchanged and is now doubly safe: the close
arrives via a re-issue, so the engine sees a normal prefill, and the fixed32
route has no token-id special-casing either way.

### (c) ONE HAZARD TO FIX BEFORE ARMING

`_parse_think_budget` returns `None` for anything it cannot parse. So
`LUMO_PROXY_THINK_BUDGET="24,000"` — a comma, a stray space, a typo — **silently
disarms the brake** rather than refusing. That is the vacuous-gate shape this
campaign keeps paying for: the safety net turns itself off and nothing says so.
Harmless while unused; it should be strict-parse-or-refuse before the cap is
ever armed in anger. Pinned by
`test_malformed_budget_values_currently_disarm` so a fix reads as a deliberate
change.

### (d) PRECEDENT THAT IT RUNS

The cap has been armed once, at **500 tokens**, in
`fr13_replica_selfnoise_run.sh`. That is evidence the mechanism *executes* —
not a value recommendation. 500 would clip the overwhelming majority of healthy
thinking in this corpus (74% of arms are clipped even at 2,000).

---

## 3. LINES OF DEFENCE

**First line — thinking budget (this spec).** Native model capability, bounded by
data, zero perturbation below budget. Catches **both** classes: it is keyed on
*length*, not on repetition, so the enumeration class (whose word-TTR 0.729 is
the highest in the corpus and whose 12-gram count is *lower* than healthy) is
caught exactly as well as the n-gram loop. That is the property C6 says a
repetition-keyed brake would miss.

**Second line — detectors, already landed, flags only.** The c5 seam gate
(`scripts/fr14_c5_seam_gate.py`, corridor [0.40, 0.70], 4/5 corpus detections at
0/95 false positives, plus the free windowed variant) and the eyeball's n-gram
panel. These stay diagnostics. They flag; the eyeball adjudicates.

**Third line — custom in-serve brake. DEMOTED to optional.** Only if budget
enforcement cannot land cleanly. Sketch retained for completeness: a device-side
counter of consecutive steps with accepted-length at ceiling (round 5's
`MAX_RUN` precedent), fired as a proposer-arbitration switch rather than a
refusal. I do **not** recommend building it if (e) lands — it would add a step-path
mechanism to solve a problem the model already solves natively, and every
step-path mechanism in this campaign has cost a re-attestation.

---

## 4. philox-B — THE RE-QUALIFICATION BUNDLE

Per the ruling, folded in here rather than asked separately.

Per-request seeding is refused by the fixed32 route
(`_fr13_fixed32_fill_uniforms` raises on a non-empty generator map; the route
requires one bulk device RNG call). **Option B** replaces the sequential bulk
stream with **counter-based (philox) addressing keyed on (derived seed, step)**:

- one kernel, as today — the "one bulk device RNG call" invariant survives;
- capture-safe and in-place, no per-request launches;
- each request's uniforms become a pure function of its own key, so a task
  replays exactly regardless of engine history — which is what makes every
  controlled experiment in the forensics plan possible.

It changes the uniforms, therefore the numerics, therefore requires
re-qualification. **Bundle it with whatever brake change is approved so the
campaign pays for one re-qualification window, not two.**

Interim, already approved: `FR13_SG_PIN_UNIFORMS=1` for experiment arms only,
self-labelling `rng_route`, never QC or promotion evidence.

---

## 5. OVERHEAD BOUNDS

| line | when idle | when firing |
|---|---|---|
| think cap (existing, `/v1/responses`) | **exactly zero** — `_parse_think_budget()` returns None and the whole cap block is skipped; byte-identical legacy path | one extra upstream call per capped turn |
| think cap extended to chat/completions (only if §2a says so) | same — one env read per request | same |
| c5 gate | zero — reads banked artifacts post hoc | zero |
| ladder windowed c5 | zero — reuses sidecars already drained at flush | zero |
| third-line in-serve brake (**not recommended**) | ~3 device ops/step on a ≤4-element tensor, ≈15 µs vs a 196.4 ms step = 0.008% | proposer switch |

The first line costs nothing on a healthy turn by construction: the cap only
does work after call A reports `incomplete_details.reason == "max_output_tokens"`,
which a healthy turn never does.

---

## 6. WHAT I NEED RULED

1. **Whether to arm at all.** The cap exists and is OFF. Arming is a behaviour
   change on the serving path and is Mark's call.
2. **Budget value.** I recommend **24,000** (0/100 healthy arms clipped). The
   official 4,000 clips 39%. The curve in §1 is there if a tighter number is
   wanted with eyes open.
3. **Coverage.** Settle which route the SWE client uses (§2a). If
   chat/completions, extending the cap to that branch is the only build.
4. **Strict parsing before arming** (§2c) — so a typo cannot disarm the brake.
5. **philox-B bundling** with whatever is approved, one re-qualification window.
6. Whether to build the third line at all. My recommendation: **no**.

## 7. HONEST LIMITS

- **n=5 degenerations.** The separation gap is large and clean, but five is five.
  24,000 is chosen to cut nothing healthy rather than to sit at a midpoint,
  precisely because the healthy side is the side with n=100.
- The two-phase close **changes the trajectory** — a stop-and-resume is not
  identical to an in-engine forced close. It bounds the damage; it does not
  reproduce what an in-engine budget would have produced. If exact equivalence
  to the official `thinking_budget` semantics matters, that needs the vLLM patch
  in §2(a) and its re-attestation.
- All five degenerate blocks sit at the existing 32,768 ceiling, so I cannot say
  from this corpus how long they *would* have run. The budget bounds them; it
  does not tell us the untruncated distribution.
