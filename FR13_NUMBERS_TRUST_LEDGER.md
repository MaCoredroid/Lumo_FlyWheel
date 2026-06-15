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
