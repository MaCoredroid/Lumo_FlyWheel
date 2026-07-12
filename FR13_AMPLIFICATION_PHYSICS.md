# FR13 amplification physics — why a ~1-ULP seed flips a token (workflow wuvlb5ex1, adversarially verified)

User Q: "if we just randomly flip one ULP any direction, why would it flip that much?"

## Answer (skeptic verdict: direction SOLID; two headline numbers CORRECTED)
- **Not a random flip.** Random 1-ULP → random-walk √64 ≈ 8× → ~0.10 L2, negligible. Excluded.
- **The ×14,800 raw ladder growth (0.012→178) is INFLATED ~30×** by ordinary residual-stream norm
  growth (clean root_residual grows 9.61→289.6 = 30.1×) which native shares = NOT instability.
  **Genuine error-vs-signal amplification ≈ 492×; instability exponent ≈ 0.099/layer (not 0.152).**
  Still ~60× above the random-walk ceiling → random walk excluded by 1–2 orders even corrected.
- **Two mechanisms:** (1) SYSTEMATIC same-sign seed (deterministic tree-vs-native kernel geometry:
  reduction order / co-residency / packing; re-injected coherently every layer → adds, doesn't cancel).
  (2) MULTIPLICATIVE amplification: δ→(I+Jₗ)δ = product of Jacobians = exponential in depth; full-attn
  layers ~1.4× vs ~1.1× GDN (softmax over drifted keys sharpens). Coherent seed × exp propagator = blowup.
- **Native immunity ⇒ fix the SEED:** the amplifier (frozen trained Jacobian) is shared bit-for-bit by
  native MTP-5, native B=8, and our tree — same ×492. Native seed ~0 ⇒ 0% garble. Cannot lower a frozen
  Jacobian and it isn't the differentiator. Encouraging: ×492 (not ×14,800) ⇒ seed needs less reduction.
- **Intermittency = margin-gating:** flip only where amplified drift lands on a small top1-vs-near-neighbor
  margin AND projects onto that unembedding direction; high-margin structural tokens absorb it.

## Ladder data check (output/fr13_node5_ladder/per_layer_maxabs.json)
Growth smooth-monotone (only 2/63 steps decrease, <2%), ln-linear R²=0.94 = exponential fingerprint,
NOT random walk (which decreases ~half its steps). Three regimes: L0→L3 fast transient 2.88×/layer
(power-iteration alignment), L3→L31 plateau 1.09×, L31→L63 re-accel 1.13×.

## HONEST caveats (skeptic survives=False on overclaims)
1. Ladder is ONE node, and it's a DIFFUSE-carrier node (```→Let, clean margin ~1.8 nats, drift ~6× margin
   = clean blowout, NOT a knife-edge graze) — NOT one of the 14/16 LOCALIZED near-neighbor garbles.
   Mechanism demonstrated on diffuse node, INFERRED for the localized carrier.
2. Late layers drift ~0.6 of residual norm = outside linearized-Jacobian regime ⇒ "λ" is an effective
   fitted rate, not a measured spectrum; late growth may be nonlinear.
3. "Native seed ~0, shared J" is ASSERTED from the tree trajectory alone — NO native ladder captured.

## Next experiment (the honest gap): capture a NATIVE MTP-5 per-layer ladder + diff vs tree ladder
Empirically confirm the seed IS the differentiator + measure how much smaller native's per-layer seed is
(the reduction target for the fix). This nails the strategy on data instead of assertion.
