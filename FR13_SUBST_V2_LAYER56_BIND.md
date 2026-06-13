# FR13 substitution v2 — pos21 flip LOCALIZED to GDN layer 56 (single layer, NOT diffuse)

Workflow `wf_c1bcc077-c1e` (6 agents, positive-control-gated). Raw:
`research/fr13_workflows/subst_v2_layer56_wf_c1bcc077.raw.json`. Verify **holds=True**.
The first VALID, trustworthy carrier localization the chase has produced. HEAD c621c65c+.

## The result: pos21's flip is BORN at one GDN layer, not accumulated
The cat9 pos21 flip (clean oracle `3425 ' files'` → tree-served `1970 ' code'`,
CHANNEL-2 verify-forward) is **generated at LAYER 56 and merely propagated by L57-63.**
Single-layer splice oracle@[N] at the validated row reverts pos21→3425 **iff N≥56** —
razor-sharp: NO-revert at N∈{0,4,8,16,24,32,40,48,49,50,51,52,53,54,55}; REVERT at
N∈{56,57,58,59,60,61,62,63}. Decisive pair: **splice@[55] does NOT revert** (L56 runs
real on a clean L55 output and re-creates the flip) vs **splice@[56] DOES** (L56 clamped
clean → stays clean to lm_head). So L56's GDN transform, *given a correct input*,
introduces the divergence — it's the **computation at L56**, not accumulated upstream drift.

**LAYER 56 = `linear_attention` = GDN** (config full_attention_interval=4 → full-attn at
{3,7,…,55,59,63}; 55 is full-attn, 56 is GDN — the first GDN block right after a full-attn
block). So the carrier is **our own GDN kernel at one layer** — fixable by bit-exact
alignment — NOT the FA2-fork full-attn (accepted floor), NOT diffuse-across-48-GDN-layers.
**This overturns v1's "diffuse" verdict**, which was an instrument artifact (v1's oracle
`pos849` didn't predict pos21; its all-64-layer splice failing to revert was meaningless —
no positive control).

## Why it's trustworthy (the instrument finally works)
- **Positive control reproduced**: splice oracle@[63] at frow=6 → pos21=3425; disarmed
  baseline=1970; 405 `FR13_HSUB spliced` log lines. Real overwrite, not a no-op.
- **Oracle correct**: a no-spec sequential rank-1 FLASH decode of `prompt2 + served[:21]`
  emits `3425` (pos21's clean argmax); oracle_row = the predicting row (not v1's row-0).
- **Same-boot clean**: pos21==1970 this boot, rep1==rep2, served[:21] byte-matches the
  oracle capture context (resolves the GB10 cross-boot fork for pos21<71 by *matching*).
- The `frow` derivation (recon guessed 4, then node-row 7) DISAGREED with reality — the
  positive-control **search** found frow=6 empirically. That disagreement is exactly why
  the gate exists: never trust a derived row, validate it by a splice that actually reverts.

## The FIX target (handed to the fix workflow)
Drill ONE level: per-sub-op GDN capture at **LAYER 56 ONLY** — conv1d_out → scan
(chunked delta-rule recurrent state) → gate (fused_sigmoid_gating, the 1/rms amplifier)
→ o_proj — for pos21's committed flat-row (frow=6) on the cat9 tree forward vs the no-spec
sequential rank-1 oracle. First-nonzero sub-op = the carrier. Then **align OUR kernel
bit-exact to native** (feedback_no_reroute_reward_hacking — fix the kernel, never ship the
splice), remeasure the sub-op→0.0, and re-gate the per-token argmax probe. Seam estimate:
SMALL (1, maybe 2 sub-ops at one layer = a SHORT grind, not the ~48-layer grind); likely
candidates from prior root-causes = conv prior-window bank-row at num_accepted>1, or GDN
scan N-dependent reduction; gate is the 1/rms amplifier not the source.

## Caveats (honest, carried to the fix)
1. **pos21 is ONE flip** — a deep-SPINE flip (committed path `[0,1,3,5,7]`, deep spine).
   Whether the other 21 share L56 (or the same GDN sub-op at *their* tipping layer) is the
   immediate follow-up. A single GDN-op realization fix should generalize to every flip
   through that op, but that must be MEASURED (re-gate the full per-token argmax probe).
2. pos21's verify-forward margin is modest (~0.5 logit, tree 1970 over clean 3425) — a
   near-tie tips easily; the clear-margin flips may need the same fix to bite harder or may
   localize elsewhere. Re-gate decides.

Pairs with [[feedback_top_down_per_layer_lossless_gate]], [[feedback_math_correct_vs_bitexact]],
[[feedback_fr12_subkernel_zero_gate]], [[feedback_no_reroute_reward_hacking]],
[[reference_diffuse_gdn_accumulation_explained]] (this REFINES "diffuse" → single-layer for
this flip), [[feedback_fail_loud_assert_engagement]].
