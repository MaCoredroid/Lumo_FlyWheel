# FR13 WY Tree-Verify GDN Kernel — Op-Order / Precision Seam Fixes (seam-finder workflow wsomqvehb, 2026-06-08)

**Root cause:** WY is byte-exact to the **fp32 CPU oracle** (4.19e-9) but diverges from the **live FLA Triton chunk kernel** by 1 bf16 ULP (1.22e-4) because WY keeps full fp32 where native FLA **deliberately rounds to bf16** at 4 boundaries. WY is *more accurate than the incumbent*; to zero the LIVE ladder it must reproduce native's bf16 rounding (same class as the conv bf16-tap win). Bar = bit-exact to the INCUMBENT FLA, not max accuracy (`feedback_math_correct_vs_bitexact`).

## Two native references (the key framing)
| reference | what | WY gap | source |
|---|---|---|---|
| fp32 CPU oracle | `recurrent_gated_delta_rule.py:166-216` all-fp32 | **4.19e-9, 0 flips** (already exact) | FR13_LOSSLESS_FAST_DERIVATION.md:104 |
| **live FLA Triton** | GB10 server: bf16 l2norm + bf16 solve-T + bf16 KKt-input + bf16 v | **1.22e-4, 2 flips** | chunk.py:35-66, l2norm.py:92, solve_tril.py:96 |

**TEST AGAINST THE LIVE FLA** (the native-server capture `native_layer_hidden.pt`), NOT the fp32 serial oracle (WY already matches that at 9.3e-10).

## The floor (do NOT overshoot)
Native's OWN bf16-vs-fp32 self-noise = **9.5e-5, 3/9 flips** (FR13:110). WY already beats it (6.1e-5, 2/9). Target = **within native bf16 self-noise (~1e-4) per layer + final logits within E5 floor**, NOT literal 0.0. Gate ALL bf16 taps behind one `FLA_BF16_BOUNDARIES` constexpr so the fp32-oracle path (4.19e-9) stays available for the math check.

## Ranked fixes (test in order; #3/#4 subsumed by #1)
1. **#1 l2norm bf16 store** `fr10_gdn_tree_kernel.py:503-505` (HIGH, ROOT, test 1st): after the rsqrt, `b_q = (b_q*tl.rsqrt(...)).to(tl.bfloat16).to(tl.float32)`; same for `b_k`. Native `l2norm_fwd` stores normed q/k to bf16 (`l2norm.py:92,102`) before every downstream dot. 2-line, gated. Expect this alone to drop L1 toward the ~1e-4 floor and kill most of the 128× amplification.
2. **#2 solve-T bf16 round** `:528/537`: `coeff_j = coeff_j.to(tl.bfloat16).to(tl.float32)`. Native stores the WY inverse T to bf16 rtne (`solve_tril.py:96`, `wy_fast.py:71-115`). Most ill-conditioned (1/rms) step → dominates residual after #1.
3. **#3+#4 KKt bf16-input + beta-before-dot** `:518-520`: `b_kb=(b_k*b_beta[:,None]).to(tl.bfloat16); kk=tl.dot(b_kb, tl.trans(b_k).to(tl.bfloat16))` (drop `input_precision="ieee"` + the post-dot `*b_beta`). Matches native bf16-input gram + fold-then-dot (`chunk_scaled_dot_kkt.py:84-85`). **Largely subsumed once q/k arrive bf16 from #1.**
4. **#5 state-update v/k bf16** `:546,557` (med-low, only if residual): round `tv_i`/`k_j` to bf16 (`chunk_delta_h.py:235,241`). Operates on decayed delta (<1) → smaller.

## NO CHANGE (closed)
- Basis (gate-folding) `:519-520,534`: WY already uses native bounded basis `exp(G_i−G_j)`, NOT the rescaled (46573× blow-up). Do not touch.
- Final output store `:564-569`: single bf16 truncation, already matches `chunk_o.py:137-138`.
- chunk/N_PAD grouping: n≤16<64 = one chunk, #42960 ruled out.
- cum_g order `:517`: differences `exp(G_i−G_j)`, order-noise far below the bf16 ULP floor. Defer.

## Gate after #1-#3
Re-measure the LIVE ladder (vs live FLA). PASS = L1 within native bf16 self-noise + the cascade brings final logits within the E5 floor. Then Gate-2 + clean B=4 e2e. Sources: OUR kernel `fr10_gdn_tree_kernel.py:401,503-573`; live FLA `/tmp/vllm_live_019/.../fla/ops/` (chunk.py, l2norm.py:92, chunk_scaled_dot_kkt.py:84, solve_tril.py:96, wy_fast.py, chunk_delta_h.py).
