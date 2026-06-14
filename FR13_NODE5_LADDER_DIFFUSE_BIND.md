# FR13 — node-5 per-layer ladder + deep-accept state-feed code-read: the 22-flip carrier is DIFFUSE (within-floor), NOT a single seam

Date 2026-06-14. **Dual-verified convergence** of two independent methods on the SAME carrier event
(gold-gate p3, step-103, deep-spine node-5, num_accepted=4, the ~7-logit `Let`(9764) vs `\`\`\``(71093) flip):
- **GPU per-layer ladder** `wf_6ba048aa-4aa` / task `wdzd8se3o` (Test+Verify, **holds=TRUE**) —
  `research/fr13_workflows/node5_ladder_diffuse_wdzd8se3o.raw.json`,
  `output/fr13_node5_ladder/{per_layer_maxabs,per_depth_argmax}.json`.
- **CPU deep-accept state-feed code-read** `wd1245bnm` (5 agents) —
  `research/fr13_workflows/deepaccept_statefeed_read_wd1245bnm.raw.json`.

Both land the same verdict; this **resolves the contested diffuse-vs-single-L56 framing** (INDEX row
`tree_vs_seq_deepspine_scan`, "pending ladder") and **overturns FR13_SUBST_V2_LAYER56_BIND's "single GDN
layer 56"** read.

## Verdict: DIFFUSE per-layer realization accumulation — no fixable single op
Same-boot, the live tree-verify node-5 row (flat row 6, num_accepted=4) and the clean teacher-forced
single-forward of the accepted prefix `[0,1,3,5]` (ctx_len 1687, byte-identical banked lp) **enter L0
BYTE-EXACT** (`per_layer_maxabs.json` input_maxabs=0.0). Divergence is **born inside L0's GDN compute
(hid_max_abs 3.9e-3 from identical input)** and grows **monotonically, smoothly** to 12.45 at L63 —
`per_layer_maxabs` re-derived 0/64 mismatch; every L4+ layer-to-layer L2 jump-ratio <1.7× (only L2/L3
exceed 2× and both at trivial absolute magnitude right at signal birth). **No isolated ~0→argmax-flip
spike at any layer.**

The flip mechanism: the `Let`(9764) logit is essentially **matched** in both arms (live 25.38 / clean
24.80); the argmax flips only because the **`\`\`\``(71093) logit collapses live (15.94) vs clean (26.60)**.
That `\`\`\`` deficit is ~0 through L0–L40, then **ratchets across the deep ~22 layers (L41–L63)**, biggest
single steps at the deep full-attention-adjacent layers (L59 +1.14, L62 +1.39, L63 +1.61) with substantial
GDN contributions throughout. The **final-token** argmax crystallizes only at **L60 (clean reaches `\`\`\``)
/ L61 (live locks `Let`)**.

### Correction to my earlier read (recorded so it isn't repeated)
- The "first flip at **L34**" I reported off `per_depth_argmax.json` was a **transient early-exit projection
  artifact** (L34/39/46/48/50/52/53/56 all flicker because those layers don't predict the final token).
  The decisive flip is at L60/L61. Verify issue (2) flags this explicitly.
- My "max_abs 2.30 at L0" used the **wrong file** (`live_node5_layers.pt`); LIMIT=5 produced call0–call7,
  only **`live_node5_layers.call1.pt`** = step 103. Correctly flagged unreliable before banking.

## Why it's diffuse, not the bank/wiring (the CPU code-read half)
The deep-accept GDN **state-feed** (the last unverified candidate) is **correct by construction**, NOT the
carrier:
- FILL (`_tree_gdn_replay_kernel`, `fr10_gdn_tree_kernel.py:624-706`): publishes each accepted token's
  post-state to its own fp32 bank LINEAR column; at acc_len=4, col 3 = state after the 4th accepted token.
- READ (next forward h0, `:439-448`): `h0_column=clamp(num_accepted-1,0)` → col 3. **FILL and READ land on
  the same column.** Same `num_accepted-1` convention across all 4 touchpoints (forward-SSM-read,
  forward-conv-read `:1907-1916`, replay-publish, replay-h0); `_fr10_accepted_lens_tensor` ≡
  `_accepted_lens_buf` (same tensor, best_lcp). No fill-vs-read counting mismatch.
- Reconciled vs **native** roll-slot (`mamba_utils.py:259-275`, `gdn_linear_attn.py:984-1005`): native rolls
  a single bf16 in-place slot at `num_accepted_tokens-1`; ours publishes multi-column **fp32** and reads
  the deepest committed column — same logical state, **intentionally higher precision** (aligning to native
  bf16 would be the reward-hack). The only thing that *could* diverge is which-column wiring — verified
  correct. → "the residual is accumulation, not the bank."

### Reconciles the BV A/B scan=0.0
The GDN scan **op** is bit-exact to native and num_accepted-invariant (BV A/B D16=D32=0.0) — but that was
the **shallow single-forward** regime. Here the carrier is the GDN **recurrent state-feed across the
co-resident accepted chain within ONE tree forward at num_accepted=4**: the live arm builds node-5's state
via **rank-1 tree-scan over [0,1,3,5] seeded from b_h0**; the clean arm builds the same logical state via a
**1687-token chunked-prefill scan**. Different realizations of the same recurrent state (the documented
chunk-vs-recurrent ~ULP gap) born at L0, amplified ~32× by the gate 1/rms and the deep full-attn layers.
The full-attn FA2-fork layers (known 2-ULP floor) **amplify, do not originate** — deep accept introduces
no NEW full-attn seam.

## Disposition: route to the within-floor gate, NOT literal-0.0
This is exactly [[reference_diffuse_gdn_accumulation_explained]] (native same-model fp8 drifts ~7× less =
existence proof it's a realization diff, not a one-op bug), now **dual-verified** at a specific carrier
event. Matches the standing bar [[project_fr13_active_worker_codex_fr15]] (per-depth-argmax + within-floor,
NOT abs-0.0) and [[project_fr13_speed_first_lossless_gate]].

**There is no single-seam fix to chase here.** Next (constructive, no close):
1. (CPU, now) Quantify: how many of the 22 flips share this deep-accept (num_accepted≥3) diffuse signature
   vs any sharp residual; establish/locate the **native-E5 baseline flip count** under the identical
   per-token argmax-vs-clean probe (is "native ~3 flips" banked or assumed?).
2. The lever to reduce the excess flips is **tree-reshape (shallower + root-sibling)**
   [[project_fr13_tree_reshape_unifying_lever]] — fewer co-resident deep nodes = less state-feed
   realization drift — NOT a per-op patch. (Also the speed lever: fewer verify rows.)
3. The verdict instrument is **e2e cat9 vs E5** (FLASH_ATTN native MTP-5): does cat9 stay within E5's
   self-noise floor (spine per-depth-argmax + bag-TV ≤ 0.0593 + accept/event ≥ native)? Bring that
   pass/fail to the user; **do not close on the literal-0.0 carrier.**

Pairs with [[reference_scalar_metric_per_token_blindspot]], [[feedback_math_correct_vs_bitexact]],
[[feedback_top_down_per_layer_lossless_gate]], [[feedback_no_reroute_reward_hacking]],
[[project_fr13_pipeline_lock]].
