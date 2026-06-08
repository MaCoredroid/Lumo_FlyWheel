# FR13 WY Cascade Map — Remaining-Seam Roadmap (cascade-map workflow wbkc915ct, 2026-06-08)

Source: `output/fr13_wy_gateA_20260608T163915Z/` (.pt re-extracted). Live FLA: `/tmp/vllm_live_019/.../fla/ops/`. Kernel: `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` (`_tree_gdn_wy_kernel`:400).

## VERDICT: PARTIAL — the 2 applied taps do NOT cascade-clear
Cascade is **single-root** (layer-0 GDN = 0.0 all spine rows; layer-1 = **exactly 1 bf16 ULP at pos9 only**; then ~125× amplification via RMSNormGated 1/rms). BUT native FLA rounds bf16 at **SIX** boundaries; only #1 applied correctly, #2 mis-located, #3-#6 missing. With ~125× amplification, **any one** un-tapped boundary re-seeds the full failure.
**The gate is FRAGILE (decisive):** pre-fix argmax matches by LUCK — logit drift **exceeds the native top1−top2 margin on 4/6 spine positions** (pos5 2.95>2.50, pos7 1.56>0.75, pos8 1.44>0.125, pos9 1.77>0.25). Must drive drift to the ~6e-5 bf16 floor. Rules out "2 taps, just confirm."

## The 6-boundary minimal set — apply ALL in ONE pass, gated `FLA_BF16_BOUNDARIES`
- **#1 l2norm (504-509): APPLIED ✓ keep.** matches `l2norm.py:92`.
- **#2 solve-T: RELOCATE.** Native runs the substitution fully fp32, rounds ONCE at the final inverse-T store (`solve_tril.py:94-97`). Current tap rounds `coeff_j` per-iter (542-543) = OVER-rounds. **Fix:** remove 542-543; round once at the `solved_v`/`solved_k` writes (548-549): `y_i.to(bf16).to(f32)` / `sk_i.to(bf16).to(f32)`.
- **#3 KKt gram (522-524): NOT subsumed by #1.** Native `chunk_scaled_dot_kkt.py:84-85` = bf16-input dot, beta pre-folded. **Fix:** `b_kb=(b_k*b_beta[:,None]).to(tl.bfloat16); kk=tl.dot(b_kb, tl.trans(b_k).to(tl.bfloat16))` (drop `input_precision="ieee"`), drop the duplicate `*b_beta` at 524.
- **#4 w/u operands (537-538):** native `wy_fast.py:92-93,114-116` rounds bf16 before the T-inverse apply. **Fix:** `y_i=(beta_i*v_i).to(bf16).to(f32)` at 537; `sk_i=(beta_i*k_i*tl.exp(cumg_i)).to(bf16).to(f32)` at 538.
- **#5 state v_new/h0 (551):** native `chunk_delta_h.py:235`. **Fix:** `tv_i=(y_i-incoming_i).to(bf16).to(f32)` at 551. (smaller — decayed delta.)
- **#6 output readout (558-570): bf16 round of the intra outer-product + q_i** to emulate native `chunk_o.py:137` two-term bf16 split. **CAVEAT — reduction order differs (structural).** Round at 567/558; if a residual remains, do NOT restructure speculatively (see user-decision).

Order: #1-#4 carry the dominant injection (undecayed operands); #5-#6 are decayed/sub-floor. **One pass — NOT one-at-a-time** (125× amplification = a one-at-a-time loop wastes a ~5-min ladder per seam).

## USER-DECISION FLAGS (do NOT proceed past these without asking)
1. **#6 readout reduction-order.** The bf16 *round* is tappable, but native splits the readout into `q@h(bf16) + (intra→bf16)@v(bf16)` while WY does one fused fp32 contraction — a **different reduction order**. If after #1-#5 the GDN *output* still drifts above ~6e-5, matching it requires a kernel **rewrite** (algorithmic, not a tap) → **bring to user.** Expected sub-ULP (rides decayed terms); let the post-#1-#5 ladder decide first.
2. **Branch losslessness is NOT certified by the spine ladder.** The gateA ladder compares only spine rows [0,1,2,4,6,8]; branch rows 3,5,7,9 are excluded. The #1-#6 taps clear the branch *numerical* seam (same kernel), but a **mask-correctness** seam (`visible_mask` 516-521 / output-visibility gate 559-569) is invisible to a spine-only ladder. Per `reference_gdn_tree_branch_oracle_losslessness`, branch correctness = **per-depth argmax vs native-run-on-branch-path oracle** (SpecInfer Def 4.1 / STree Eq.4-6). **The e2e lossless verdict cannot be declared from the spine ladder alone — the branch oracle must be added to the same run.**

## Next test (codex)
Apply #1-#6 in ONE pass, then ONE live ladder. PASS = layer-1 GDN ≤ ~6e-5, all GDN within floor, **final-logit drift < the native top1−top2 margin on ALL 6 positions** (the real gate, currently 4/6 fail), per-depth spine argmax. ADD the branch-path oracle (per-depth argmax for rows 3,5,7,9) to the same run. Keep the fp32-oracle path (`FLA_BF16_BOUNDARIES` OFF → ~4.19e-9) as the ℝ-correctness check. NO copy/splice. Then Gate-2 + clean B=4 e2e.
