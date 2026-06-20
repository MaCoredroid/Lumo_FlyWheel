# APC arm A (cat9+APC) — POISONING CONFIRMED on our long-prompt config — 2026-06-19

Arm A = cat9 + APC (cache-ON, align+chunked+fp32, mamba_block=1024), 4 astropy SWE tasks, B=1
temp-0.6, OFFLOAD_CODEX alienware. Serve done (driver 2451795, 5616s, 3603 pair-dumps).
**VERDICT: APC reproduces #43559 / #45477 cache-poisoning on our config. NO-GO as-is.** Branch
fr13-prefix-cache. Binding per-token rescore still pending (but agent-level + text-level evidence
is already conclusive).

## The evidence chain (decisive)
1. **Accept collapse (live, 21:50):** Mean acceptance length 1.00, position-0=0.000 sustained ~2-3
   min, gen 3.8 tok/s degenerate. Recovered at the turn boundary (intermittent, not permanent).
2. **Text-level garble (the smoking gun):** scanning all 3577 response dumps, the degenerate turns
   produced runaway/garbled output — the #43559/#45477 signature:
   - seq 54: `</parameter> </parameter> …` ×15+ (runaway malformed tool-call markers)
   - seq 73: `0000…` ×100+ (16357 chars, runaway zeros)
   - seq 47: `用户说 "a file, a file path". 用户说 …` (repetitive loop)
   - seq 56: `Theuserwantsmetobeabletoeditthefilesdirectly.Theuser…` (no-space runaway)
   - seq 30, 75: symbol soup. ~6/3577 SEVERELY garbled (>0.18 repeated-trigram); a 0.17% rate but
     each garbled turn derails a whole agentic task.
3. **Agent-level degradation (vs cache-OFF, SAME 4 tasks):**
   | task | cache-OFF (cat9_b1) | APC-ON (cat9_apc) |
   |---|---|---|
   | 12907 | resolved, 729s, clean | resolved but **1622s TIMED OUT** (2.2x slower) |
   | 13033 | failed, full-wall 1921s, patch **998B** | failed, **gave up 375s, patch 0B** |
   | 13236 | failed, full-wall 1921s, patch **719B** | failed, **gave up 186s, patch 0B** |
   | 13398 | failed, full-wall 1921s, patch **587B** | failed, **gave up 475s, patch 0B** |
   Cache-OFF worked the full wall + produced real patches; APC-ON gave up early with EMPTY patches
   on 3/4. health_rc=1 (FULL-RUN HEALTH RULE flagged the early exits).
4. **Cache-OFF NEVER did this:** 886 accept-windows, min accept 1.75 (never <1.5), position-0 mean
   0.93 (never ~0), zero collapse windows. The asymmetry is clean → APC-attributable.

## What it means
- **APC poisons the GDN recurrent state intermittently during multi-turn generation** → garbled
  runaway output → agent gives up / empty patches → task failures. This is the #43559 "~20%
  accuracy drop" / #45477 "malformed tool calls, runaway gens" REPRODUCED on our config.
- **On LONG prompts (~11-14k, single-stream)** — NOT the #45477 short-prompt+concurrent (a)
  trigger. So this points at **carrier (b): the conv-window reconstruction** (get_conv_copy_spec
  offset-shift at num_accepted>1), the "latent CUDA risk" the red-team said was unconfirmed. We
  appear to have CONFIRMED carrier (b) on long-prompt CUDA tree-spec. (Caveat: the binding rescore
  + the native-E5+APC / CONTROL diagnostic arms are needed to pin the carrier definitively vs a
  trajectory artifact — though the cache-OFF asymmetry makes APC the strong cause.)
- **Gate-0's healthy-acceptance scalar MISSED this** (acceptance 5.0, cold≡warm at gate-0). The
  poisoning only surfaced over the full multi-turn agent loop. Validates
  [[reference_scalar_metric_per_token_blindspot]] AND why the red-team was right to HOLD the "we
  don't see the 20% drop" claim — the Gate-0 feasibility signal was not a losslessness verdict.

## Ship implication + fix
- **NO-GO as-is for deployment** — APC degrades the agent (the ~13x prefill saving is real but
  worthless if generation gets poisoned). Default flag stays OFF (it already is).
- **Fix direction = conv-window full-precision SNAPSHOT-not-reconstruct** (SGLang/NVIDIA #10335 /
  our FR13_CONV_COMMITTED_PATH cheap interim: read the full accepted-node row instead of
  offset-slicing the base row in get_conv_copy_spec). This is the same carrier (b) fix from
  vllm_43559_rootcause.md.
- **#43559 contribution: STRENGTHENED.** We reproduced the poisoning on a long-prompt config with
  concrete garbled-output evidence + the carrier-(b) source pointer. This partly overturns the
  red-team's "long prompts dodge the trigger" caution (we see it on long prompts → carrier b, not a).

## Open / pending (await user steer)
- Binding per-token lossless rescore (LOSSLESS_ARMS=cat9_apc, ~3h GPU) for the rigorous flip-rate
  number vs the recurrent oracle — would quantify the poisoning per-token for the contribution.
- Diagnostic arms: native-E5+APC (does it reproduce on native MTP? = carrier confirmation) +
  CONTROL (cat9 chunked-ON cache-OFF: is chunked-prefill alone enough, or is it the cache?).
- Conv-window snapshot fix (the actual remedy to make APC lossless).

Artifacts: output/fr13_bigdenom_swe/cat9_apc/{health.json, proxy_pair_dumps/ (3603), deploy_speed_b1.json}.

---
## UPDATE 2026-06-20: stock-revert conv fix (c4875fca) is INSUFFICIENT — garble persists
Re-ran arm A as `cat9_apc_fix` (FR13_APC_CONV_FIX=1: get_conv_copy_spec override scoped to preprocess,
postprocess falls through to STOCK offset-slice). DECISIVE dump scan: **8 severely-garbled dumps**
(same </parameter>×N / 用户说 / no-space-runaway signatures), ~same rate as the poison run's ~6.
Agent outcome IDENTICAL to poison: 1/4 resolved (12907 511s), 3 gave_up (13033/13236/13398 124/906/67s)
— NOT cache-OFF's full-wall+real-patches. So:
- **Our override wrong-block misfire was NOT the dominant garble carrier.** Reverting it to stock
  offset-slice did not reduce garble. The fix is still correct (wrong-block read was a real bug) but
  orthogonal to the poison.
- The garble carrier is elsewhere in the align path: the conv-window **reconstruction** (stock
  offset-slice = position-shift, #25587 non-invertibility) OR the SSM-state snapshot OR another align
  seam (block hashing #45477 under chunked-prefill?). NOT yet pinned empirically.
- **METHOD CORRECTION: gate on the DUMP GARBLE SCAN, not the live per-window accept signal** — accept
  looked "milder" (bouncing 1.0-1.2, not pinned-0) yet garble was unchanged. The intermittent garble
  doesn't show reliably in per-window accept.
NEXT: design workflow w84n4mdcz (SGLang snapshot-whole-row) — its committed-conv LAYOUT read is the
diagnostic for whether conv-reconstruction is the carrier; if the offset-slice is correct for our
layout, the carrier is elsewhere and needs empirical instrumentation (tap conv+ssm at a garbled turn
under APC vs APC-OFF). Do NOT assume the snapshot fix works — gate it on the garble scan.
