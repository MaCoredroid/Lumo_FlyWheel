# FR13 WY floor-reducibility verdict (workflow wzkwgcpvi, 2026-06-08, grounded)

## VERDICT: WY is a 1-bf16-ULP-floor kernel — NOT byte-exact-to-native as built. Output = alignable TAP; state = irreducible floor.

### OUTPUT residual = 1.22e-4 = exactly 1 bf16 ULP → ALIGNABLE TAP (not a wall, not a reward-hack)
Native `chunk_o.py:137` builds the readout as **two separately-rounded terms**: inter `b_o = dot(b_q, trans(b_h))` (fp32) **+** intra `dot(b_A.to(b_v.dtype=bf16), b_v)` — `b_A` cast to **bf16 first** — then `*scale` per-term. Ours (`fr10_gdn_tree_kernel.py:683`) does ONE fused fp32 contraction `tl.sum(state_i*q_i)`, never performs native's intra-A bf16 cast. Our readout is *more accurate* than native (offline our-vs-serial = 9.3e-10) → the 1 ULP is native+us rounding opposite ways on boundary-straddling elements. **Replicating native's two-term split + bf16(A) cast inside OUR kernel is a clean tap → spine 0.0, branches sparse single-ULP (MMA-grouping floor, same accepted class as FA2 no-copy 2-in-1M). NOT a reroute/splice (builds native's *structure*, not native's *kernel*).**

### STATE residual = 1.33e-12 pre-round / 58-of-4.7M bf16-flips → IRREDUCIBLE FLOOR (do NOT grind)
Our `_state` store (`fr10_gdn_tree_kernel.py:667-682`) is **op-for-op identical** to native's per-token decode recurrence (`fused_sigmoid_gating.py:158-165`, the fp32 sequential ref). The 1.33e-12 is **invariant across all 4 traversals** (original/reverse-DFS/spine-first/spine-only) **and all 6 op-order variants** (7.45e-9 max_abs, 8-flip floor) = signature of intrinsic fp32 non-associative reduction noise. No edit reaches 0.0. ~6 orders below the E5 floor per-step. Grinding it = the FR11 "diffuse/tolerance" trap inverted: no seam left.

## CRUX 2 — within-floor PASS vs super-floor FAIL: ONLY THE LIVE LADDER DECIDES (no self-declared PASS)
PROVEN offline: output 1.22e-4 = 0.46% relative, **non-compounding** (oscillates ±1 ULP depth0→5, does not grow); state ~6 orders below floor/step. NOT proven offline (live-only): cross-layer coherence of 48 bf16 roundings, o_proj/RMSNormGated Jacobians (no weights in CPU artifacts), **branch losslessness** (spine ladder does NOT certify branches — needs per-path native-on-branch oracle).
- **Decisive single number = rows 3/4 (pos8/pos9) post-fix tree-vs-native final-logit drift** (live run `output/fr13_wy_poststate_live_ladder_20260608T205204Z`, native ref `output/fr13_wy_gateA_20260608T163915Z/native/logs/native_final_logits.pt`).
- **PASS-within-floor** if drift ~0.12–0.26 (incoherent/physical) + spine argmaxes match (only candidate flip = pos8, margin 0.125, but flip-cost 0.0526 bag-TV < 0.0593 E5 floor → absorbed by construction). Margins: pos5=2.50, pos6=3.94, pos7=0.75, pos9=0.25, pos10=12.88.
- **FAIL super-floor** if drift ~0.6–1.0 (coherent accumulation) → pos9 flips (cost 0.089 > floor), possibly pos7 (0.217).
- Offline depth-1 (1.53e-5) under-predicts live L1 (1.22e-4) by ~8× → offline UNDER-counts. This is a calibrated likelihood, NOT a pass.
- Pre-fix 3.32 / 56%-reject was the PRE-state-fix build (state 1.66e-3); `8a975837` → 2.98e-8, so pre-fix cascade numbers do NOT carry forward.

## CRUX 3 — RANKED MOVES
| # | Move | Class | When |
|---|------|-------|------|
| **(a)** | Accept floor + verify LIVE (no kernel change) | clean — measurement in flight | **DEFAULT / DO FIRST.** Verdict is e2e-vs-E5, not per-layer. If live ladder + e2e bag-TV ≤ 0.0593 + accept/event ≥ E5 3.076 → ship. |
| **(b)** | #6 output two-term + bf16(A) tap at `:683` mirroring `chunk_o.py:137` | clean TAP, NOT algorithmic rewrite, NOT reward-hack | **ONLY IF (a) fails super-floor.** Drives spine→0.0, branches→single-ULP MMA floor. Don't double-fix a floor that may already pass. |
| **(c)** | State chunk-accum-order match | irreducible floor, NOT a tap | **NOT WORTH IT — do not grind.** 6 op-orders + 4 traversals all stall at 1.33e-12; ~6 orders under floor. |

No move is a banned reward-hack. Banned forms (route spine through native causal_conv1d_update/FLA, dense/copy gather, splice native kernel) are NOT among (a)/(b)/(c).

## MONITOR ACTION
WAIT for the live ladder → read tree-vs-native final-logit drift at **rows 3/4 (pos8/pos9)** + per-row argmax. Incoherent (≤1 sub-floor flip) → e2e bag-TV vs E5 + add branch-path argmax oracle (rows 3/5/7/9), bring to user (move a, no kernel change). Coherent super-floor → move (b) the #6 tap, re-ladder. Do NOT grind state (c).
