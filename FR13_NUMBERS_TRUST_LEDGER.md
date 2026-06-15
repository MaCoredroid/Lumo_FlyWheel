# FR13 NUMBERS TRUST LEDGER — SLICE: SPEED NUMBERS

Audit (CPU read-only, no boot; GPU re-aim wf w8q5jg1k0 in flight). Staples every SPEED
number against the TWO deployment contaminations and assigns a TIER. Sources cited inline.

## The two contaminations (recap) and what they mean FOR SPEED
1. **OOD prompt** — `prompts_swe4.json` is sent as a RAW string to **`/v1/completions`**
   (CONFIRMED in code: `scripts/fr10_quick_decode_tps_probe.py:179` posts to `/v1/completions`;
   no chat template) = off-distribution for this chat/thinking-trained model. This tanks
   ACCEPT/LOSSLESS (native served_token_ids degenerate into the empty-`<think>` loop). The
   deployment regime = real SWE-Verified + codex agent loop, chat-templated `/v1/responses`.
2. **temp 0 vs 0.6** — every speed gate ran GREEDY (temp 0.0, top_p 1.0); deployment = temp 0.6
   top_p 0.95 sampling.

**KEY FINDING FOR THIS SLICE — why s/fwd survives BOTH contaminations (TIER C):**
The speed BASIS is `s/forward = delta(vllm:request_decode_time_seconds_sum) /
delta(vllm:spec_decode_num_drafts_total)` — a per-EVENT (per-forward) decode time scraped from
RAW `/metrics` counters, NEVER TPS, NEVER accept, NEVER wall (banned hand-rolls; reference
fr10_speed_measurement_pitfalls + feedback_dont_handroll_speed). It DIVIDES BY FORWARDS, not
tokens. The forward is **weight-bandwidth-bound**: ~27 GB fp8 / 273 GB/s = ~98.6-98.9 ms floor
on GB10 (FR13_WHY_SLOWER_VERDICT L18; FR13_FUNDAMENTAL_SPEED_FLOOR_BIND L9). Therefore:
- **Content-independent:** the per-forward cost is the weight DMA + fixed launch shape, the SAME
  whether the prompt is in- or off-distribution. DIRECT EVIDENCE: across legacy cat9 arms whose
  ACCEPT spread is 1.82–2.22 (trajectory-confounded across boots), s/fwd moves only 0.3936–0.3976
  (**±0.5%**) — "per-forward is robust to accept differences BY CONSTRUCTION (divides by forwards,
  not tokens)" (FR13_SPEED_TAX_BASELINE.md L44-46). Different served streams (= different prompt
  realizations) do NOT move s/fwd.
- **Temp-independent:** s/fwd is a stable substrate property, rep1==rep2 byte-identical within
  boot (FR13_SPEED_TAX_SCALING_BIND L17; FR13_B1_FIX*_GATE_BIND within-boot identity). Greedy is
  CHOSEN as the basis precisely because **t0.6 carries wall jitter between identical-stream reps**
  — t0.6 is noisier, not different in central value (FR13_B1_FIX2_GATE_BIND L249-250;
  FR13_B1_FIX3_GATE_BIND L269). So greedy s/fwd is the cleaner estimator of the SAME quantity
  temp-0.6 would measure, not a different regime.
- **The ONE genuine regime caveat = CONTEXT LENGTH (not prompt-distribution).** s/fwd has a mild
  KV-length dependence: ~1.045x @64-tok vs 1.056-1.063x @11k (FR13_SPEED_TAX_SCALING_BIND L16;
  the residue is constant/event but the GAP grows with context as the long-KV unified attention
  weighs in — diag_residue_audit_wf_18daa008). Deployment is the long-context agentic end
  (~11k), so the deployment cat9 ratio is the ~1.05-1.06x end, NOT the tighter 1.030x @64-tok.
  This is a context-regime shift WITHIN TIER C — the s/fwd METHOD stays valid; the headline number
  should be quoted as the @64-tok B=1-gate figure (1.030x) WITH the @11k (~1.05-1.06x) end.

> NET: all s/fwd numbers below are TIER C (regime-robust, trustworthy regardless of prompt
> distribution / temp), with the single caveat that the cat9-vs-native RATIO should be read at the
> deployment context length (~1.05-1.06x @11k) not only the @64-tok gate figure (1.030x). The
> 1.030x cat9-vs-native s/fwd verdict STANDS despite accept being contaminated. Any number that is
> secretly TPS or accept-derived is flagged NOT-regime-robust below.

---

## SPEED LEDGER (one row per number)

| # | number | what it is | prompt-regime | temp | B | basis | TIER | conclusion STANDS / RE-MEASURE | source |
|---|---|---|---|---|---|---|---|---|---|
| S1 | **native E5 = 0.2182 s/fwd** | native MTP-5 B=1 per-forward decode reference | raw /v1/completions (swe4) | 0 (greedy) | 1 | decode_seconds/spec_drafts, MEASURED | **C** | STANDS — the canonical B=1 native ref; content/temp-robust | FR13_B1_FIX3_GATE_BIND L246; FR13_B1_SPEED_ATTRIBUTION_BIND |
| S2 | **cat9 ON = 0.2247-0.2249 s/fwd = 1.030x native** | cat9 tree B=1 per-forward, post-FIX-3, clean (metrics OFF) | raw /v1/completions | 0 | 1 | decode_seconds/spec_drafts, MEASURED | **C** | STANDS @64-tok; deployment-context end ~1.05-1.06x @11k. The ratio is regime-robust; accept contamination does NOT touch it | FR13_B1_FIX3_GATE_BIND L256,259-261,286-287; FR13_SPEED_HISTORY_RECONCILE |
| S3 | **chain5 ON_b = 0.2223-0.2226 s/fwd = 1.019x native** (best tree) | chain5 (depth-5 linear) B=1 per-forward, post-FIX-3 | raw /v1/completions | 0 | 1 | decode_seconds/spec_drafts, MEASURED | **C** | STANDS as a per-forward ratio. NOTE chain5 is depth-5 LINEAR; its accept/TPS comparison is a SEPARATE (contaminated) axis | FR13_B1_FIX3_GATE_BIND L253 |
| S4 | **Phase-0 native E5 = 0.2159 s/fwd; cat9 = 0.2241 s/fwd** | Phase-0 re-baseline; MATCHED the banked 0.2182/0.2248 within boot noise | raw /v1/completions (swe4, `fr13_speed_phase0.sh`) | 0 | 1 | decode_seconds/spec_drafts, MEASURED | **C** | STANDS — it is the independent confirmation that s/fwd REPRODUCES (the Phase-0 ACCEPT did NOT reproduce — native 1.70/cat9 3.02 = the raw-regime bug — which is exactly the point that s/fwd is regime-robust while accept is not) | FR13_SPEED_MEASURE_INFRA.md L203; scripts/fr13_speed_phase0.sh; scripts/fr13_build_speed_measure_infra_workflow.js L76-77 |
| S5 | **FIX-1 progression: cat9 0.3118->0.2373 (1.429x->1.088x); chain5 0.3070->0.2294 (1.407x->1.051x)** | drafter single-logits removes the double lm-head GEMV | raw /v1/completions | 0 | 1 | decode_seconds/spec_drafts, MEASURED | **C** | STANDS — the +81.9 ms/draft drafter-head tax (and its removal) is a bandwidth fact (lm-head weight 2.543 GB read; gemvx 11.1->5.9/draft, MEASURED engagement). Content/temp-robust | FR13_B1_FIX1_GATE_BIND L160-163,200; FR13_SPEED_HISTORY_RECONCILE L14,21-32 |
| S6 | **FIX-2 progression: chain5 ->0.2254-0.2265 (1.033x); cat9 ->0.2347-0.2349 (1.076x)** | eager-pack: 102->1 DtoH, 96->2 replay launches | raw /v1/completions | 0 | 1 | decode_seconds/spec_drafts, MEASURED | **C** | STANDS — chain5 saving 3.0-3.9 ms separated from boot noise (cat9 saving NOT separable, conv-dominated). Engagement MEASURED (DtoH 109.6->8.0/draft). Regime-robust | FR13_B1_FIX2_GATE_BIND L226-244; FR13_SPEED_HISTORY_RECONCILE L15,34-44 |
| S7 | **FIX-3 progression: cat9 ->0.2247-0.2249 (1.030x); chain5 ->0.2223-0.2247 (1.019-1.030x)** | tree-conv fusion (~3.4k/5.1k captured nodes saved/fwd) | raw /v1/completions | 0 | 1 | decode_seconds/spec_drafts, MEASURED | **C** | STANDS — cat9 ~7.7 ms saved, beats design window, fully separated from OFF spread. byte-A/B 283/283 lossless-by-construction. Regime-robust | FR13_B1_FIX3_GATE_BIND L244-296; FR13_SPEED_HISTORY_RECONCILE L16 |
| S8 | **FIX-1/2/3 NET: 1.40x -> 1.03x** (cat9), 1.41x->1.019x (chain5) | the full lm-head/eager/conv lineage = the bulk of the original gap | raw /v1/completions | 0 | 1 | decode_seconds/spec_drafts, MEASURED | **C** | STANDS as the headline speed-reduction lineage. All three fixes are MEASURED s/fwd deltas, content/temp-robust. (deployment-context end ~1.05-1.06x) | FR13_SPEED_HISTORY_RECONCILE L9-17,138-143 |
| S9 | **in_proj_ba baked = SPEED-NEUTRAL (0.2248 vs 0.2249 OFF)** | in_proj batch-aware pad, hidden behind bandwidth-bound weight read | raw /v1/completions | 0 | 1 | decode_seconds/spec_drafts, MEASURED | **C** | STANDS — neutral by construction (pad hidden behind weight DMA). Regime-robust | FR13_SPEED_HISTORY_RECONCILE L17,90; commit 4d0452df |
| S10 | **stale "2.336x slower"** | OLD cat9-vs-native at B=4 SWE | raw (B=4 SWE swe4 probe) | 0 (greedy) | 4 | decode_seconds (RAW), but DECOMPOSED into accept x per-fwd | **PARTLY NOT-REGIME-ROBUST -> STALE** | DO NOT USE. (a) The headline 2.336x = 1.432x (MORE forwards = spec_drafts ratio = an ACCEPT-driven term, contaminated) x 1.632x (per-forward tax). The 1.432x FORWARD-COUNT factor IS accept-coupled (fewer accepts -> more forwards) so it is NOT regime-robust; it inflated under the OOD-prompt accept collapse. (b) PRE-FIX-1/2/3 (drafter double lm-head regime). The per-forward 1.632x slice is itself a B=4 sum-basis ratio (concurrency-overlapped, NOT single-stream latency, self-flagged UNKNOWN). Superseded by 1.030x (S2). | FR13_WHY_SLOWER_VERDICT L6-15; FR13_SPEED_HISTORY_RECONCILE L80-82,114 |
| S11 | **OPT-1 GPU-resident committer: reclaims ~4-6 ms of the ~6.6 ms cat9 tax** (-> s/fwd parity / just-below native) | design projection; kills the committer DtoH+sync (census: chain5 91.9% main-thread block vs native 0.8%) | n/a (projection) | n/a | 1 | **INFERRED** (design arithmetic; 35-60 ms was the pre-FIX-2 census; impl=10ebccac, NEVER GPU-verified) | **C-method but INFERRED-not-measured** | RE-MEASURE on GPU. The MECHANISM (91.9% main-thread block in memcpyAsync) is a MEASURED census fact (regime-robust). The 4-6 ms REACH is a projection, never measured. When measured it will be a TIER-C number, but today it is unproven | FR13_BEAT_NATIVE_SPEED_DESIGN_BIND L23,26,29; FR13_SPEED_HISTORY_RECONCILE Appendix-3 L304-340; commit 10ebccac |
| S12 | **OPT-A GB10-tuned fp8 GEMV: ~140-150 ms/fwd = ~1.45-1.55x faster s/fwd vs native 218 ms** | design projection; no GB10/sm_121 fp8 JSON exists (default BLOCK_SIZE_M=64, num_stages=2 = ~45% peak) | n/a (projection) | n/a | 1 | **INFERRED** (design; impl=e90de7ef, NEVER GPU-verified; lossless-by-construction K=128 pinned) | **C-method but INFERRED-not-measured** | RE-MEASURE on GPU. The ROOT (native at 45% of peak = 2.21x the 98.6 ms bandwidth floor; no GB10 config) is source-verified vs LIVE fp8_utils.py (regime-robust). The 1.45-1.55x REACH is a projection. SHARED-kernel TENSION (native's spine runs the same GEMM) NEEDS a user ruling — flagged, not in this slice's verdict | FR13_FUNDAMENTAL_SPEED_FLOOR_BIND L1,9,34-36; FR13_SPEED_HISTORY_RECONCILE Appendix-3 L342-380; commit e90de7ef |
| S13 | **bandwidth floor = 98.6-98.9 ms/fwd (27 GB fp8 / 273 GB/s, GB10)** | the hard per-forward floor both arms share | n/a (hardware) | n/a | n/a | derived from HW spec | **C** | STANDS — pure hardware (GB10 mem bandwidth); the reason s/fwd is regime-robust at all | FR13_WHY_SLOWER_VERDICT L18; FR13_FUNDAMENTAL_SPEED_FLOOR_BIND L9 |
| S14 | **per-forward tax GROWS with N: +0.0108 s/fwd per draft node (OLS, n=2 distinct N, DIRECTION-ONLY)** | legacy-route chain5(N=5)->cat9(N=9) per-forward step +42-46 ms ≈ 7x the row-traffic floor | raw /v1/completions, BI=1 | 0 | 1 | decode_seconds/spec_drafts, MEASURED step / FIT label | **C (the step) / INFERRED (the slope shape)** | The STEP is real (3 N=9 boots agree within 1%, regime-robust). The SLOPE is a fit over 2 distinct N = direction not shape; n_pad 8->16 is a built-in confound. Do not treat the slope as measured | FR13_SPEED_TAX_BASELINE L52-63; FR13_SPEED_TAX_SCALING_BIND L28-37 |
| S15 | **replay@N=9 = 0.3270 s/fwd (1.537x), BI=1** | replay-kernel cat9 per-forward (cheaper than legacy N=5 0.3517) | raw /v1/completions, BI=1 | 0 | 1 | decode_seconds/spec_drafts, MEASURED but ACCEPT-BUG CONFOUNDED boot | **C-basis but boot-confounded** | The per-forward RATIO is informative (s/fwd robust to accept), but it is ONE point at ONE N from an accept-bug-confounded boot (accept 2.02->1.58); no N-invariance test possible. BI=1 = the slow deterministic-GEMM regime (NOT the deployed speed regime, which is BI=0). Treat as direction only | FR13_SPEED_TAX_BASELINE L36-37,64-67; speed_tax_gate_w5pzs35uz |

### Cross-cutting flag — numbers that are NOT regime-robust (secretly accept/TPS)
- **warm TPS** columns (e.g. cat9 13.9-14.4, chain5 17.7-17.9) reported alongside s/fwd in the
  FIX gates: TPS = (accept+1)/s_fwd, so TPS embeds ACCEPT in the numerator -> contaminated by both
  the OOD-prompt accept collapse AND temp. NOT TIER C. These are reported in the docs as a sanity
  sidebar but the BINDING basis is always s/fwd. Do NOT quote TPS as a regime-robust speed number.
- **The 2.336x "MORE forwards" factor (1.432x, S10):** spec_drafts ratio is accept-coupled
  (more rejects -> more forwards) = NOT regime-robust. Only the per-forward tax slice of 2.336x
  is method-valid, and even that is stale (pre-FIX) + B=4-sum-basis.
- **break-even accept thresholds** (e.g. "accept ~3.43 at 1.05x to break even", FR13_WHY_SLOWER
  table): these COMBINE s/fwd (TIER C) with an accept target (contaminated) -> the THRESHOLD math
  is regime-robust but the accept side it is compared against must be re-measured on chat+temp0.6.

---

## TIER SUMMARY (speed slice)

**STAND (deployment-faithful, regime-robust = TIER C):**
- The s/fwd MEASUREMENT METHOD (decode_seconds/spec_drafts, per-forward, bandwidth-bound) is
  content- and temp-independent by construction — DIRECTLY evidenced (accept spread 1.82-2.22 moves
  s/fwd only ±0.5%; rep1==rep2 within boot). This is the load-bearing audit conclusion.
- native E5 0.2182 s/fwd (S1), cat9 1.030x @64-tok (S2), chain5 1.019x (S3), the Phase-0
  reproduction 0.2159/0.2241 (S4), and the FIX-1/2/3 progression 1.40x->1.03x (S5/S6/S7/S8) ALL
  STAND as regime-robust per-forward numbers. **The cat9-vs-native 1.030x s/fwd verdict STANDS even
  though accept was contaminated.** in_proj_ba speed-neutral (S9) stands. Bandwidth floor (S13)
  stands.
- ONE caveat WITHIN TIER C: read the cat9-vs-native ratio at the DEPLOYMENT context length
  (~1.05-1.06x @11k) not only the @64-tok gate figure (1.030x). The METHOD does not change; only
  the context regime shifts the number slightly up. (Context length is a real deployment-regime
  axis; prompt-distribution and temp are not, for s/fwd.)

**RE-MEASURE before binding:**
- OPT-1 (S11) and OPT-A (S12) REACHES are INFERRED design projections, implemented (10ebccac /
  e90de7ef) but NEVER GPU-verified. Their MECHANISMS are measured/source-verified (TIER C), but the
  ~4-6 ms and ~1.45-1.55x numbers are unproven and must be GPU-measured (and OPT-A needs the
  shared-kernel user ruling). When measured they become TIER-C s/fwd numbers.

**STALE / CONTAMINATED — do NOT use:**
- "2.336x slower" (S10): stale (pre-FIX-1/2/3) AND partly accept-coupled (the 1.432x forward-count
  factor) AND B=4-sum-basis. Superseded by 1.030x (S2).
- All warm-TPS sidebars and any break-even ACCEPT target: TPS/accept side is contaminated (OOD
  prompt + temp 0) and must be re-measured on the deployment regime (real SWE-Verified + codex,
  chat-templated /v1/responses, temp 0.6). s/fwd is the only clean break-even input today
  (FR13_SPEED_TAX_SCALING_BIND L66-67).

**NOTE on cross-slice numbers seen but OUT of this slice:** the 7.39x is a confound-free FLIP RATE
(lossless axis, FR13_MATH_HISTORY_RECONCILE L54,176), NOT a speed number — flagged so a reader does
not mistake it for a TPS/s-fwd ratio. accept 3.161/3.18, the +15 per-event superset, the 23-flip
decomposition, the big-denom 13.55%/13.99%, the reshape accept/flip numbers, and the bag-TV floors
belong to the accept/lossless slices and are TIER B/D/E there — out of scope here.

---
---

# FR13 NUMBERS TRUST LEDGER — SLICE: ACCEPT NUMBERS + BREAK-EVEN + PER-EVENT SUPERSET GATE

Audit (CPU read-only, no boot; GPU re-aim wf w8q5jg1k0 in flight). Staples every ACCEPT/event
number, the break-even argument, and the per-event SUPERSET gate against the TWO contaminations and
assigns a TIER. Sources cited inline. (Speed slice above; this slice = accept/lossless axis.)

## The decisive regime fact for THIS slice
Essentially EVERY B=1 accept/event number below came from `prompts_swe4.json` sent as a **RAW text
string** to **`/v1/completions` with NO chat template**, greedy (temp 0.0), seed 1313 — CONFIRMED in
the probe code:
- `scripts/fr13_gold_margin_probe.py:6` — literally "`/v1/completions, RAW text prompt (no chat template)`".
- `scripts/fr13_verify_bisect_probe.py:85-89` — `"prompt": prompt … /v1/completions`, `temperature 0.0`.
- `scripts/fr13_measure.py:19,45` — canonical fr13_measure "sent prompts_swe4.json as a RAW string to
  /v1/completions with NO chat template".
- `prompts_swe4.json` = the astropy-12907 SWE-bench task text as ONE raw string (verified head).

This is **OFF-DISTRIBUTION** for the chat/thinking model. CONFIRMED degenerate (re-aim commits
`be10f299`, `cab6c157`): native E5 served stream loops `[271,248068,271,248069,271,40]` =
`\n<think>\n</think>\nI` (empty-`<think></think>` repetition) → **native accept tanks to ~1.589**, forks
cross-boot; the no-spec recurrent oracle ranks the coherent continuation correct by ~11 nats → it is
the **raw regime, not a kernel bug**. ⇒ **accept/lossless on this regime is TIER D (CONTAMINATED).**

The ONLY chat-templated real-SWE accept-axis number is the **big-denom** (codex agent loop on
astropy-12907 via `/v1/responses`, no degenerate loop) — that one is TIER B.

All accept numbers are temp-0 (greedy) → at best the greedy-superset axis; the temp-0.6 distributional
accept/superset gate is **UNMEASURED on the deployment prompt** ⇒ **no Tier A accept number exists yet.**

## ACCEPT/EVENT LEDGER (one row per number)

| # | number | what | prompt-regime | temp | B | TIER | conclusion STANDS / RE-MEASURE | source |
|---|---|---|---|---|---|---|---|---|
| A1 | **native E5 = 3.1613** | native MTP-5 accept/event, B=1 current gate (the "3.1613 bar") | raw /v1/completions (off-dist) | 0 | 1 | **D** | RE-MEASURE — contaminated native baseline; SAME boot family degenerates to ~1.589. | FR13_B1_CURRENT_GATE_BIND (3.161290, 124ev/620tok); FR13_B1_SUPERSET_PRECONDITION_BIND |
| A2 | **native E5 = 3.076 / 3.08 / 3.154** | the depth-matched SUPERSET BAR (5-spine MTP-5); cross-boot spread | raw /v1/completions | 0 | 1 | **D** | RE-MEASURE — the superset target bar itself is contaminated; the 3.076↔3.154↔3.1613 spread is the native self-floor ON THE RAW REGIME. | FR13_DIRECTION_AND_NUMBERS (3.076); FR13_CHAIN3_DEPTH_LEVER_DEAD_BIND (3.076); FR13_RESHAPE_AB_RECURRENT_BIND (3.08); FR13_ACCEPTANCE_LADDER_BIND R2 (3.154 sum) |
| A3 | **cat9 = 3.18 / 3.1789 / 3.198** | cat9 accept/event (post-FIX-A); "first crossing ABOVE native" | raw /v1/completions | 0 | 1 | **D** | RE-MEASURE — the crossing-above-native claim is raw-regime. (3.1789 controlled probe; 3.198 ladder/cat10.) | FR13_SPEED_HISTORY_RECONCILE L410 (3.1789); FR13_CAT10_BIND (3.1983); FR13_RESHAPE_AB_RECURRENT_BIND (3.198) |
| A4 | **cat9 +0.0176 EDGE over native E5** | controlled pinned-probe accept delta (3.1789 − 3.1613) | raw /v1/completions | 0 | 1 | **D** | RE-MEASURE — a delta of two TIER-D numbers; does NOT survive to deployment as-measured. Aggregate accept draw also UNRESOLVED (−0.43→~0, trajectory-confounded). | FR13_SPEED_HISTORY_RECONCILE L410-466 (MEASURED, ac1d3039 FIX-A) |
| A5 | **cat9 = 2.1515** | pre-FIX-A cat9 accept (B=1 current gate); drove "theorem-precondition failure" | raw /v1/completions | 0 | 1 | **D** | RE-MEASURE — historical; superseded by A3 but same contamination. | FR13_B1_CURRENT_GATE_BIND (2.151515, 165ev/1485tok) |
| A6 | **chain5 = 2.66 / 2.664 / 2.6596 / 3.0391 / 3.2562** | leaf-free 5-spine accept (cross-boot spread) | raw /v1/completions | 0 | 1 | **D** | RE-MEASURE — wide cross-boot spread is itself a raw-regime symptom; "chain5 exceeds native" is contaminated. | FR13_CHAIN3_DEPTH_LEVER_DEAD_BIND (2.664); FR13_B1_FIX3_GATE_BIND gate(c) (ON 3.0391 / on_b 2.6596); FR13_B1_CHAIN_SPEED_DISCRIMINATOR (3.2562) |
| A7 | **chain3 = 2.266 / 2.295** | leaf-free depth-3 spine accept | raw /v1/completions | 0 | 1 | **D** | RE-MEASURE — accept is the binding arbiter for the "reshape lossless but SPEED-NEGATIVE" verdict; it is TIER D. | FR13_RESHAPE_AB_RECURRENT_BIND (2.266); FR13_CHAIN3_DEPTH_LEVER_DEAD_BIND (2.295) |
| A8 | **cat3w = 2.282** | depth-3 spine + root-sib + d1 width accept | raw /v1/completions | 0 | 1 | **D** | RE-MEASURE — "width re-introduces co-residency, accept < native" verdict on TIER-D accept. | FR13_RESHAPE_AB_RECURRENT_BIND (2.282) |
| A9 | **cat10 = 2.9316** (d0 rescue +0.035, net −0.27 vs cat9 3.1983) | root-sibling cat10 accept | raw /v1/completions | 0 | 1 | **D** | RE-MEASURE — whole-window accept self-flagged trajectory-confounded IN the doc AND TIER D. | FR13_CAT10_BIND AXIS 2 |
| A10 | **per-depth ladder: tree 2.082 vs native 3.154 sum** (d0 deficit −0.227, root-concentrated) | R2 marginal/conditional per-depth | raw /v1/completions | 0 | 1 | **D** | RE-MEASURE — the root-concentrated d0-deficit diagnosis is on the regime where native degenerates; BI-asymmetry (tree BI=1 / native BI=0) ALSO open. | FR13_ACCEPTANCE_LADDER_BIND R2 |
| **A11** | **big-denom 13.548% (cat9) / 13.985% (native)** clear-margin flip rate (CIs OVERLAP, cat9 LOWER) | per-token argmax-vs-own-no-spec-recurrent-oracle flip rate at scale | **CHAT /v1/responses, real SWE astropy-12907 + codex loop** | 0 | 1 | **B** | **STANDS** as a lossless-vs-native PASS at the deployment PROMPT (temp-0). The only non-contaminated accept-axis number. RE-MEASURE ONLY to add the temp-0.6 distributional axis (→ A). | FR13_CONFIRM_SPEC_VS_NONSPEC (VALID, 4 CODE-READ links); output/fr13_bigdenom_rescore/consolidated.json; FR13_TEMP06_DRIFT_GATE §1b |

## BREAK-EVEN / ACCEPT-PARITY

| # | claim | regime | temp | B | TIER | STANDS / RE-MEASURE | source |
|---|---|---|---|---|---|---|---|
| A12 | **"cat9 accept ≥ native at parity" (the break-even for the accept-edge TPS win)** | raw /v1/completions | 0 | 1 | **D** | RE-MEASURE — rests on A3 vs A1/A2 (all TIER D). The token-weighted AGGREGATE accept draw is admitted UNRESOLVED (−0.43→~0, trajectory-confounded). The break-even THRESHOLD math is sound but its accept inputs are contaminated. | FR13_SPEED_HISTORY_RECONCILE L180,317,415-466 |
| A13 | **"no break-even available" (pre-FIX-A current gate)** | raw /v1/completions | 0 | 1 | **D** | RE-MEASURE — "tree slower per-fwd AND fewer accepts (2.15<3.16)" is TIER D; superseded by FIX-A. | FR13_B1_CURRENT_GATE_BIND |

## PER-EVENT SUPERSET GATE

| # | number | what | regime | temp | B | TIER | STANDS / RE-MEASURE | source |
|---|---|---|---|---|---|---|---|---|
| A14 | **net +15** (21 lossless leaf-saves − 6 lossy − 0 spine_reg) | per-event superset gate: cat9 IS a lossless superset of E5 | raw /v1/completions (fork-margin boot, 118ev/466pos/4 prompts) | 0 | 1 | **D** | RE-MEASURE — VERDICT computed on raw-regime dump; greedy-only; SMALL-SAMPLE 1 boot. The STRUCTURAL part (0 spine_regressions, strict `>best_lcp` tie-break over 250 recs) is design-true; the +15 magnitude + 21/6 split are TIER D. | FR13_PEREVENT_SUPERSET_GATE_RESULT (w02jpqib2, verify HOLDS) |
| A15 | **23-flip decomp = 6 lossy-leaf-saves + 17 spine-realization** | reframes "cat9 23 flips vs native 3" | raw /v1/completions | 0 | 1 | **D** | RE-MEASURE — the KIND-split logic (leaves net-positive; gap = tree-verify SPINE drift) is informative, but COUNTS (23/6/17, native ~3) are TIER D. | FR13_PEREVENT_SUPERSET_GATE_RESULT; FR13_CARRIER_REOPEN (de-cascade 23→18→~14-15) |
| A16 | **net_lossless/event +0.127, lossless fraction 21/27 = 78%** | per-event superset gain magnitudes | raw /v1/completions | 0 | 1 | **D** | RE-MEASURE — same boot as A14; TIER D. | FR13_PEREVENT_SUPERSET_GATE_RESULT |
| A17 | **confound-free 7.39x flip RATE** (cat9 43.3/1000 vs native 5.9/1000; de-cascaded 19 vs 3) | the "load-bearing defect number" replacing raw 23-vs-3 | raw /v1/completions (recurrent-rescore, wgb0yegin) | 0 | 1 | **D** | RE-MEASURE — confound-free vs LENGTH/CASCADE, NOT vs PROMPT-REGIME; both arms on the off-dist streams. The "M-invariance vs topology BOTH refuted" verdict (every single-op lever dead; floor = spine ~+2) is a per-forward KERNEL/co-residency conclusion that likely STANDS structurally (content-independent channel), but its NUMBERS are TIER D. | FR13_MATH_HISTORY_RECONCILE L52-207 (verify HOLDS); FR13_E5_VS_CAT9_SPINE_DRIFT |
| A18 | **bag-TV floor 0.0593 → superseded by 0.1133** | native-self bag-TV floor (lossless threshold) | raw /v1/completions | 0.6 (served-bag) | 1/4 | **D** | RE-MEASURE — 0.0593 = ONE native-a/native-b draw (`fr13_corruption_gate.py:123`); 0.1133 = one CUDA-captured draw; BOTH single-draw (class #12) AND raw-regime. Must become an N=6-8 p95 floor on the deployment regime. (Rare temp-0.6 number, but native-self on off-dist.) | FR13_TEMP06_DRIFT_GATE §2; FR13_DRIFT_TRACKER_DESIGN L5 |
| A19 | **cat9 22 / 23 raw flips (per-prompt [6,6,4,6] / [5,4,5,9]) vs native ~3** | the headline absolute flip count | raw /v1/completions | 0 | 1 | **D/E** | RE-MEASURE for deployment; valid as TIER E (argmax-localization diagnostic) for kernel work. "22 reproduces the banked oracle → reference validated" is a legitimate E use. | FR13_CAT10_BIND (22); FR13_CARRIER_REOPEN (23) |

## TIER SUMMARY (accept slice)

**STANDS (deployment-faithful):**
- **A11 big-denom 13.548% (cat9) / 13.985% (native) — TIER B.** The ONLY accept-axis number measured
  on the deployment PROMPT (chat `/v1/responses`, real SWE-Verified astropy-12907 + codex loop, no
  degenerate loop). Each arm scored vs its OWN no-spec recurrent decode oracle; CIs overlap, cat9 LOWER.
  Conclusion "cat9 ≈ native within floor ⇒ lossless-vs-native PASS at scale" is deployment-prompt-faithful.
  **B not A** because it is temp-0 argmax (necessary-not-sufficient); the temp-0.6 distributional gate is owed.
- The DESIGN-TRUE / kernel-channel parts survive regardless of prompt: 0 spine_regressions (strict
  `>best_lcp` tie-break, 250 recs), the spec-vs-non-spec framing validity (4 CODE-READ links), and the
  qualitative co-residency-is-the-carrier / single-op-levers-refuted structure (A17). Their MAGNITUDES are TIER D.

**RE-MEASURE on the deployment regime (chat real-SWE + temp 0.6):**
- **EVERY B=1 accept number A1–A10, A12–A19 is TIER D** — all from `prompts_swe4.json` sent RAW to
  `/v1/completions` (no chat template) at temp-0, the regime where native degenerates to ~1.589. So:
  native 3.1613 / 3.076 / 3.08 / 3.154 (the bar); cat9 3.18 / 3.1789 / 3.198 and the **+0.0176 edge**;
  chain5 (2.66–3.2562) / chain3 (2.27/2.30) / cat3w (2.28) / cat10 (2.93); the **per-event SUPERSET gate
  +15** and the **23 = 6+17 decomposition**; the **break-even / accept-parity**; the **7.39x** rate; the
  **bag-TV floor**.
- These do not necessarily REVERSE on the deployment regime (A11 suggests cat9 ≈ native HOLDS there), but
  **the specific accept/event values and the +15 / +0.0176 magnitudes are not deployment-faithful as
  measured** — re-take on chat real-SWE, then run a temp-0.6 distributional accept/superset gate for Tier A.
- **No Tier A accept number exists yet.** The deployment-binding lossless gate is the temp-0.6
  distributional one (FR13_TEMP06_DRIFT_GATE), UNMEASURED on the deployment prompt. Re-aim (`cab6c157`,
  `w8q5jg1k0`) re-points fr13_measure at exactly this.
- **UNKNOWN-REGIME: none in this slice** — every accept number's prompt-path is determinable (raw probe
  scripts) or documented (big-denom = chat). Only the bag-TV floor (A18) is temp-AMBIGUOUS (temp-0.6
  served-bag, but raw regime).

## CITATIONS (accept slice)
- Commits: `cab6c157`, `be10f299` (re-aim / contamination ruling: raw /v1/completions off-dist → native
  ~1.589, `\n<think>\n</think>\nI` loop); `ac1d3039` (FIX-A, cat9 2.03→3.1789); `8add39e6` (+35 greedy
  branch accepts); `77e2a0e8` (per-event superset reducer).
- Probe source (regime proof): `scripts/fr13_gold_margin_probe.py:6,94`; `scripts/fr13_verify_bisect_probe.py:85-89`;
  `scripts/fr13_measure.py:7,16,19,45`; `output/fr13_acceptance_ladder/prompts_swe4.json`.
- Binds: FR13_B1_CURRENT_GATE_BIND, FR13_B1_SUPERSET_PRECONDITION_BIND, FR13_B1_FIX3_GATE_BIND,
  FR13_DIRECTION_AND_NUMBERS, FR13_ACCEPTANCE_LADDER_BIND, FR13_CHAIN3_DEPTH_LEVER_DEAD_BIND,
  FR13_RESHAPE_AB_RECURRENT_BIND, FR13_CAT10_BIND, FR13_PEREVENT_SUPERSET_GATE_RESULT, FR13_CARRIER_REOPEN,
  FR13_MATH_HISTORY_RECONCILE (7.39x; M-invariance both refuted), FR13_E5_VS_CAT9_SPINE_DRIFT,
  FR13_CONFIRM_SPEC_VS_NONSPEC (big-denom VALID), FR13_TEMP06_DRIFT_GATE (deployment temp-0.6 gate;
  bag-TV 0.0593→0.1133), FR13_SPEED_HISTORY_RECONCILE (+0.0176 edge, accept-parity).

---
---

# FR13 NUMBERS TRUST LEDGER — SLICE: DRIVING VERDICTS (disposition — STANDS vs NEEDS-REMEASURE)

Audit (CPU read-only, 2026-06-15; GPU re-aim wf w8q5jg1k0 in flight — no boot). The SPEED and ACCEPT
slices above staple individual NUMBERS. This slice dispositions the five VERDICTS that drove FR13
decisions: the regime of each verdict's EVIDENCE, and whether the verdict STANDS on deployment-faithful
evidence or NEEDS-REMEASURE on the deployment regime (real-SWE+codex chat + temp 0.6). It does not
re-table numbers already stapled above; it cites their rows (A*, S*).

## Audit-confirmed contamination smoking gun (this slice's load-bearing finding)
The wgb0yegin "confound-free" oracle source and the reshape A/B sources are the **raw off-distribution
regime, AUDIT-CONFIRMED IN THE BINDING ARTIFACT**: `output/fr13_recurrent_oracle/rescore_native.json`
points at src `output/fr13_verify_decisive/q3_native_capture.json`, whose native
`served_token_ids[:14] = [271, 248068, 271, 248069, 271, 40, ...]` = `"\n<think>\n</think>\nI"` — the
exact degenerate empty-`<think></think>` loop the re-aim condemned. The capture prompt carries NO chat
markers (raw `## Codex CLI invocation prompt` string; `<|im_start|>`/harmony absent), `temperature=0.0`,
seed 1313. So 7.39× (A17), the 20-flip 2/13/5 KIND split, the held-trajectory-zero control, and the
reshape frontier (A6–A8) are ALL measured against a degenerate native baseline ⇒ TIER D.

## THE FIVE DRIVING VERDICTS

**V1 — "lossless-vs-native met at scale" (big-denom 13.55%/13.99%, A11).**
Evidence regime: **chat / deployment-prompt (real SWE astropy-12907, codex, /v1/responses) but TEMP-0**,
single task, B=1. Spec-vs-non-spec framing CODE-READ-confirmed (4 links, FR13_CONFIRM_SPEC_VS_NONSPEC).
**STANDS at temp-0 (TIER B) — the only deployment-faithful lossless evidence in FR13; necessary-not-
sufficient.** NEEDS-REMEASURE for the binding answer at temp 0.6 (the sub-argmax tail sampling reaches at
0.6 is untested; temp-0.6 TV(q,p) = A11's missing axis, not yet computable — q not banked,
FR13_TEMP06_DRIFT_GATE §3) and to widen past one SWE task.

**V2 — "M-invariance + topology-reshape BOTH refuted" (wgb0yegin / A17).**
Evidence regime: **RAW prompts_swe4 / temp-0**, native source = the empty-think loop; reshape arms also
scored vs a chunked (wrong-frame) oracle. SPLIT VERDICT:
- M-invariance lever refutation (conv-state-feed seam = max_abs 0.0 — a 0.0 seam can't carry 14 flips) is
  a CODE/SEAM fact → **STANDS regime-robust.**
- depth-dead (chain3==chain5) and width-adds-GDN-co-residency are STRUCTURAL topology facts →
  **STAND DIRECTIONALLY.**
- the headline NUMBERS (7.39×, +17 co-residency, 2/13/5 KIND split, "native-3 is the wrong bar /
  spine ~1.67× floor") are **TIER D** → **NEEDS-REMEASURE on deployment + temp 0.6.** The qualitative
  "no cheap deployable single-op route to native losslessness" is plausibly robust; its quantitative
  floor is not.

**V3 — per-event SUPERSET gate PASS (+15, A14).**
Evidence regime: **RAW / temp-0 / 1 boot / 118 events.** `spine_regressions=0` is a structural committer
property (strict `>best_lcp` tie-break, 250 recs) → **STANDS regime-robust.** The quantitative gate
(+15 net, 78% lossless, 21/6 split, A14–A16) is **TIER D** → **NEEDS-REMEASURE on deployment + temp 0.6**
(the binding superset question at 0.6 is distributional, not a greedy leaf-save count). Correctly NOT
baked/shipped.

**V4 — reshape LEADS (R4 cat6root + chain3/cat3w frontier).**
The SPEED rationale (pad8→pad16 = +42–46 ms/fwd, keep N≤7; S14) is **TIER C → STANDS** as a speed lever.
The LOSSLESS/accept rationale and the "reshape EXHAUSTED — no deployable lossless+fast shape" verdict are
**TIER D** (raw + temp-0 + chunked-oracle frame; A6–A8) → **NEEDS-REMEASURE**: decided on a degenerate
native baseline. R4 cat6root is worth trying as a SPEED lever; as a lossless claim it is unproven.
Depth-matched compares also UNMET (a d3 arm must compare to E3, currently UNMEASURED;
feedback_depth_matched_accept_compare).

**V5 — OPT-1 / OPT-A speed path (S11/S12).**
NOT YET A VERDICT — both UNBUILT/un-GPU-verified (OPT-1 sync-kill G2 unbuilt; OPT-A built CPU-byte-A/B,
never GPU-verified; reaches INFERRED). The s/fwd baseline they attack (cat9 1.030×, S2) is TIER C and
STANDS. Lossless-by-construction arguments are strong but unverified; OPT-A has a shared-kernel scope
tension needing a user ruling. When GPU-verified, run the LOSSLESS side on deployment regime + temp 0.6.

## DRIVING-VERDICTS DISPOSITION SUMMARY
- **STANDS (deployment-faithful / structural):** V1 at temp-0 (TIER B, the lone deployment-prompt lossless
  point); the structural/code facts inside V2/V3 (spine_regressions=0; conv-state-feed 0.0; depth-dead;
  width-adds-co-residency); the V4 SPEED rationale (TIER C).
- **NEEDS-REMEASURE on real-SWE+codex chat + temp 0.6 [+ B=4/CUDA]:** V1's temp-0.6 axis (does not yet
  exist); V2/V3 quantitative numbers (7.39×, +15, 2/13/5, reshape frontier); V4 lossless/accept claims;
  V5 GPU-verify.
- **Bottom line:** the only verdict resting on BOTH a deployment-prompt AND a lossless measure is V1
  (big-denom, TIER B, temp-0). The confound-free 7.39×, reshape-exhausted, and superset-PASS verdicts that
  drove recent strategy were all on the raw off-distribution regime where native itself degenerates → TIER
  D → must be re-run on real-SWE+codex chat at temp 0.6 before they bind a lever or ship decision. The
  speed wins (1.030× etc.) stand.

## CITATIONS (driving-verdicts slice)
Commits: be10f299, cab6c157 (re-aim + native-degeneration proof); eabb07f9/wgb0yegin (7.39×);
e720b0be/77e2a0e8 (per-event superset); ac1d3039 (FIX-A 3.1789); 5146e574 (reshape exhausted);
10ebccac (OPT-1), e90de7ef (OPT-A). Docs: FR13_CONFIRM_SPEC_VS_NONSPEC, FR13_TEMP06_DRIFT_GATE,
FR13_MATH_HISTORY_RECONCILE, FR13_PEREVENT_SUPERSET_GATE_RESULT, FR13_RESHAPE_EXHAUSTED_BIND,
FR13_B1_CURRENT_GATE_BIND, FR13_MEASURE_DEPLOYMENT_REGIME, FR13_SPEED_HISTORY_RECONCILE.
Artifacts: output/fr13_bigdenom_rescore/consolidated.json; output/fr13_recurrent_oracle/rescore_*.json
(src output/fr13_verify_decisive/q3_native_capture.json = native served_token_ids[:14]
[271,248068,271,248069,271,40,...] = empty-think loop, AUDIT-CONFIRMED THIS SLICE);
output/fr13_shape_sweep/*_capture.json (temperature=0.0, prompts_swe4).
