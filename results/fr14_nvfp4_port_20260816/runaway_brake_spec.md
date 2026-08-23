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

## 2. IMPLEMENTABILITY — WHERE IT CAN LAND

Assessed lowest-layer-first, as asked. Evidence from the pinned vLLM source and
our own tree.

### (a) vLLM native budget — NOT AVAILABLE

Our pinned build has **no** thinking-budget mechanism. Searched for
`thinking_budget`, `max_thinking`, `reasoning_budget`, `thinking_tokens`:
the only hits are `reasoning_effort`, and it exists **only** for Mistral
tokenizers (`tokenizers/mistral.py`) and the Harmony/gpt-oss responses path
(`entrypoints/openai/parser/harmony_utils.py`). Neither is reachable from our
Qwen3 chat-completions route.

`SamplingParams` has no budget field. Adding one is a vLLM patch — i.e. a new
injected-blob surface and a re-attestation, which is precisely what we avoid
unless it buys something the higher layers cannot.

### (b) Reasoning-parser hook — PRESENT BUT WRONG LAYER

`reasoning/qwen3_reasoning_parser.py` exists and exposes `is_reasoning_end(input_ids)`
and `is_reasoning_end_streaming(...)`. But it is consulted from
`v1/structured_output/__init__.py` — it exists so guided decoding knows when the
grammar starts applying. It is **not** a decode-time budget hook, and wiring one
there would mean running structured output we do not otherwise run.

Useful fact from that parser: for Qwen3.5 the chat template puts `<think>` in the
**prompt**, so only `</think>` is ever generated. Forcing a close is therefore a
**single-token** intervention.

### (c) Per-request stop — AVAILABLE, and the basis of the recommendation

`SamplingParams.stop` and `SamplingParams.stop_token_ids` are supported per
request. This gives a **two-phase close** that needs no engine change:

1. Issue the turn with `max_tokens` unchanged and a budget counter.
2. When thinking reaches the budget, stop the generation.
3. Re-issue with the assistant prefix continued and `</think>` appended, so the
   model resumes in answer mode.

**Zero perturbation below budget**: phase 2 never happens on a healthy turn, and
phase 1 is byte-identical to today's request — the counter is host-side
bookkeeping over tokens we already receive.

**Never fail-open**: if the budget cannot be derived, or the close cannot be
issued, the correct behaviour is to fail the turn loudly, not to continue
unbounded. That is a hard requirement, and it is what distinguishes this from
the vacuous-gate class.

### (d) Proxy-side streaming intervention — POSSIBLE, with one correction

I initially read `inference_proxy.py:1032-1040` as forbidding stream
intervention. It does not: it forbids **dumping** raw requests/streams to disk
(`LUMO_PROXY_PAIR_DUMP_DIR` / `REQUEST_DUMP_DIR` / `SSE_DUMP_DIR`), an evidence
and privacy control. Counting thinking tokens in flight is not capture.

The proxy is nevertheless the **more invasive** of the two host-side options: it
would have to become stateful per request and would sit on the measured path.
The harness/client layer is preferable.

### (e) Harness-side (qwen-code client) — PREFERRED

The client already owns turn construction and already sees the thinking blocks
(that is where I measured them). It is off the measured serving path entirely,
so it perturbs `step_wall` by construction — nothing added to the step loop.

**Recommendation: (e) as the first line, using (c)'s `stop` support, with (d) as
the fallback if the client cannot be changed.**

---

## 3. FAILURE-INTERACTION — VERIFIED

The ask: a forced think-close must not confuse the committer or the suffix
machinery.

**Verified clean.** I searched the fixed32 device path for any token-id
special-casing (`eos`, special tokens, `token_id ==` comparisons) outside the
ordinary draft/bonus/output tensors: **there is none**. The committer commits
accepted draft ids; the Arctic suffix cache is keyed on committed token ids and
is content-agnostic (`fr14_suffix_pass_gate.py` operates on committed-token
n-grams, not on token identity). A `</think>` is an ordinary id to both.

Under the two-phase design the point is moot anyway: the close arrives as part
of the **next request's prompt**, not as an injection into a live decode. The
engine sees a normal prefill. This is the main reason to prefer two-phase over
an in-engine forced token.

---

## 4. LINES OF DEFENCE

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

## 5. philox-B — THE RE-QUALIFICATION BUNDLE

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

## 6. OVERHEAD BOUNDS

| line | when idle | when firing |
|---|---|---|
| thinking budget (harness) | **exactly zero** — off the serving path; a token counter over blocks the client already parses | one extra request per capped turn |
| thinking budget (proxy fallback) | one integer increment per streamed chunk, host-side, off the device path | one extra request per capped turn |
| c5 gate | zero — reads banked artifacts post hoc | zero |
| ladder windowed c5 | zero — reuses sidecars already drained at flush | zero |
| third-line in-serve brake (if ever built) | ~3 device ops/step on a ≤4-element tensor, ≈15 µs against a 196.4 ms step = 0.008% | proposer switch |

---

## 7. WHAT I NEED RULED

1. **Budget value.** I recommend **24,000** (0/100 healthy arms cut). The official
   4,000 cuts 39% of healthy arms — if that is wanted anyway, it is a behaviour
   change to be measured, not a safety net to be assumed.
2. **Layer.** Harness/client (preferred) vs proxy (fallback).
3. **Fail-closed confirmation.** On inability to derive or enforce the budget,
   fail the turn loudly. I want that stated, because the tempting alternative —
   continue unbounded — is the vacuous gate.
4. **philox-B bundling** with the approved change, one re-qualification window.
5. Whether to build the third line at all. My recommendation: **no**, if (1)-(2)
   land.

---

## 8. HONEST LIMITS

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
