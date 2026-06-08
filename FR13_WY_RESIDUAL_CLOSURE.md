# FR13 WY residual closure — 1.221e-4 is the irreducible 1-bf16-ULP floor; the strict per-layer gate is over-conservative; MEASURE the e2e (read-only verify, 2026-06-08)

Read-only adversarial synthesis. No GPU/docker/server. Every claim ties to file:line or a captured number. Kernel under test: `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` `_tree_gdn_wy_kernel` (readout 567-589), POST-redteam-fix (commits `1f11b9d7` + `978c81e2`, the over-rounds are already gone — line 570 `q_i = ... * OUTPUT_SCALE` with NO bf16 round, line 578 `state_update_ij` fp32 no per-j round). Live native: `/tmp/vllm_live_019/.../fla/ops/chunk_o.py`.

---

## VERDICT: **(B)** — 1.221e-4 is the irreducible 1-bf16-ULP floor, it is WITHIN the user-accepted e2e gate (strict per-layer margin is over-conservative), so STOP tapping and MEASURE the e2e. With ONE hard honesty flag (read §5 before declaring pass).

### Single next action for codex
**Do NOT chase the residual tap. Run the WY-kernel B=4 CUDA-graph e2e (TREE_ATTN, forked FA2, FR13_FA2_PREFILL_NATIVE=1, splice OFF) per FR13_FLOOR_WORKFLOW_VERDICT.md:39-47, with the branch-path oracle added (per-depth argmax for rows 3,5,7,9). Report the E5-vs-TREE table (bag-TV + accept/event) to the user; do NOT self-declare PASS.** Precondition before the timed run: confirm the WY kernel CUDA-graph FULL-captures + serves at B=4 and re-confirm Gate-2 hooks-off, AND confirm the full-attn prefill path is native-aligned (the prior e2e fail in §5 was a pre-prefill-fix build, not the WY kernel).

---

## 1. Q1 — is there a CORRECT bf16 boundary that closes 1.221e-4? **NO. It is irreducible.**

**1.221e-4 is exactly 1 bf16 ULP** at the mismatch magnitude. Persistent first-mismatch element across `bf16`/`bf16_kkt`/`bf16_fullbound`/`redteam_fix`: out[3,13,77] native `0.0264892578125` vs tree `0.0263671875`, abs `0.0001220703125` (`wy_l1_spine_scan_live_fla_redteam_fix.json` out_first_mismatch). Magnitude ~0.0265 → exponent −6 → bf16 ULP = 2^(−6−7) = 2^−13 = **0.0001220703125**; **diff/ULP = 1.0 exactly** (computed).

**No correct tap lowers it.** Cumulative isolation (`output/fr13_wy_l1_payload_20260608T170530Z/`):

| variant | taps | WY-vs-live-FLA out_max_abs |
| --- | --- | ---: |
| bf16 | #1 l2norm | 1.221e-4 |
| bf16_kkt | +#3 | 1.221e-4 (unchanged) |
| bf16_fullbound | +#4 #5 | 1.221e-4 (unchanged) |
| redteam_fix | #1-#5, over-rounds removed | 1.221e-4 |
| 6tap (WRONG q/per-j rounds) | +#6 | **2.441e-4** (DOUBLED) |

#1 (l2norm, kernel 504-509, matches `l2norm.py:92`) sets the floor; #3/#4/#5 are subsumed; the only thing that ever moved it was the WRONG #6 which made it WORSE (`wy_l1_spine_scan_live_fla_6tap.json` out_max_abs `0.000244140625`).

**The hypothesized "correct #6 b_A round" CANNOT be added — WY never forms the object native rounds.** Native `chunk_o.py:137` is a TWO-TERM bf16-SPLIT readout: `b_o = b_o*scale + dot(b_A.to(b_v.dtype), b_v)*scale`, where `b_o` is the inter-chunk q@h term (`:111` `b_o += dot(b_q, trans(b_h))`) and `b_A` is the intra-chunk attention gram q@k (`:113` `b_A += dot(b_q, b_k)`), gated by `exp(g_i−g_j)` (`:120`), causal-masked (`:125`), and **rounded to bf16 ONCE at :137** before `@b_v`. WY's readout (kernel 567-584) is a MATERIALIZED-STATE formulation: it assembles `state_i = b_h0*exp(cumg_i) + Σ_j trans_v_j·k_j·exp(cumg_i−cumg_j)` entirely in fp32 (571-583), then does ONE fused fp32 contraction `out_i = Σ(state_i·q_i)` (584). **WY never builds a q@k gram `b_A`**, so native's `:137 b_A.to(bf16)` rounds an intermediate with NO tensor counterpart in WY. Adding it requires rewriting WY's readout into native's `q@h + (q@k→bf16)@v` split — an algorithmic change, not a tap, and the explicit USER-DECISION-FLAG #1 rewrite (`FR13_WY_CASCADE_MAP.md:20`), reward-hack-adjacent and banned.

Every other native chunk_o bf16 boundary is already covered or correct: q/k/h/v arrive bf16 because the LOADED tensors are bf16 (`chunk_o.py:104-108`) — WY supplies q/k bf16 via #1 (508-509) and v bf16 via #4 (544-546); scale hits the fp32 OUTPUT at `:137` (not the loaded q), which WY matches by scaling q with no post-round (kernel 570). **The 1.221e-4 is the inter-op-order ULP between WY's fused-fp32 readout and native's bf16-split readout under fp non-associativity** — both correctly bf16-bounded, mathematically equivalent reduction orders. WY is in fact MORE accurate than native vs the fp32 CPU oracle: `wy_vs_serial_out 9.31e-10`, `wy_vs_serving 0.0`, 0 flips (`wy_l1_offline_probe.json`). **No correct boundary #1-#6 closes it; this is the irreducible storage-rounding floor.** (Caveat: the spine scan only covers rows [0,1,2,4,6,8]; branch rows 3,5,7,9 are NOT in this scan — §5.)

---

## 2. Q2 — what is the ACTUAL bar? **The user-accepted e2e gate, NOT per-layer bit-exactness.**

`FR13_FLOOR_WORKFLOW_VERDICT.md:36-37` "USER DECISION (2026-06-07): ACCEPT THE FLOOR" sets the verify-path gate to **within-E5-floor / argmax-lossless (NOT literal-0.0)**; the verdict belongs to the e2e: **bag-TV vs E5 ≤ E5 self-noise floor (~0.059) AND accept/event ≥ E5** at B=4 + CUDA-graph-captured + SWE-Verified-4. E5 reference: accept/event **3.076171875** (`FR13_LADDER_LOG.md:135`, `output/fr10_native_mtp5_same8_20260604T210257Z`); fresh aligned native arm measured 3.2133 (`FR13_LADDER_LOG.md:136` — the prompt's "3.21"). MEMORY confirms: "Per-layer 0.0 = DEV check only; verdict is e2e vs E5."

The strict per-layer "final-logit drift < native top1−top2 margin" gate appears ONLY in `FR13_WY_CASCADE_MAP.md:7,24` and `FR13_WY_TAP_REDTEAM.md:24` as a **conservative dev-side proxy**. It is NOT the accepted bar.

---

## 3. Gate reconciliation — is per-layer drift<margin NECESSARY, or a conservative proxy? **CONSERVATIVE/SUFFICIENT, NOT necessary — proven over-conservative.**

I loaded the captured gateA logits (`output/fr13_wy_gateA_20260608T163915Z/{tree,native}/logs/*_final_logits.pt`, both at the current redteam-fixed 1.221e-4 layer-1 state) and computed per-spine-position argmax + native top1−top2 margin + drift coordinate. Spine row map [0,1,2,4,6,8]→native[0..5] (verified self-consistent: argmax 6/6).

| depth | tree_row | tree argmax | nat argmax | match | nat margin | drift_max | drift@nat_argmax | drift_max_idx | drift>margin? |
|---:|---:|---:|---:|:--:|---:|---:|---:|---:|:--:|
| 0 | 0 | 248068 | 248068 | ✅ | 2.500 | 2.951 | 1.875 | 23583 | **YES** |
| 1 | 1 | 3299 | 3299 | ✅ | 3.938 | 1.656 | 0.125 | 13 | no |
| 2 | 2 | 369 | 369 | ✅ | 0.750 | 1.563 | 0.750 | 218224 | **YES** |
| 3 | 4 | 13 | 13 | ✅ | 0.125 | 1.438 | 0.125 | 2085 | **YES** |
| 4 | 6 | 248044 | 248044 | ✅ | 0.250 | 1.766 | 0.375 | 50951 | **YES** |
| 5 | 8 | 198 | 198 | ✅ | 12.875 | 3.320 | 1.625 | 161536 | no |

**ALL 6 spine argmax MATCH native.** The 4 positions the cascade-map calls "fails" (depths 0,2,3,4) all produce the **identical argmax token** as native. The strict gate compares max-over-248320-vocab drift against the argmax margin, but **the max-drift coordinate is never at the argmax index on any row** (drift_max_idx 23583/13/218224/2085/50951/161536, none == tree or native argmax). The margin is the FLIP THRESHOLD; the drift lands on irrelevant vocab entries, so no argmax flips. **The strict per-layer margin gate reads "FAIL 4/6" while the argmax — hence the rejection-sampler/committer accept-reject decision and the emitted token — is bit-identical to native on all 6 spine positions.** It is strictly over-conservative.

Mechanism that ties this to the e2e: `FR13_FLOOR_WORKFLOW_VERDICT.md:32-34` SUPERSET-BY-MATH — if spine verify argmax == native, the rejection sampler makes identical spine accept/reject decisions and accept/event ≥ E5 by construction (SpecInfer Thm 4.2 MSS / Multi-Draft 2410.18234 Thm 1, temp 0.6 multi-candidate). Drift>margin is NECESSARY-not-sufficient for a flip; here no flip occurs. So **per-layer 1.221e-4 does NOT necessarily produce bag-TV>0.059** and WY can pass the accepted e2e despite failing the strict per-layer margin.

---

## 4. Reconcile "WY 6.1e-5 within native 9.5e-5" vs "1.221e-4 fails 4/6" — THREE different quantities, all consistent.

- **6.1e-5 / 9.5e-5** (`FR13_LOSSLESS_FAST_DERIVATION.md:110`, `scripts/fr13_lossless_fast_derivation_validate.py`): WY vs the **CPU SERIAL recurrence oracle** (recurrent_gated_delta_rule, SpecInfer native-on-path) at a bf16 boundary = 6.10e-5 (2/9 flips) — vs native's OWN bf16-input self-noise (same serial recurrence, bf16 vs fp32 inputs) = 9.5e-5 (3/9 flips). WY-vs-serial < native-serial-self-noise ⟹ lossless WITHIN native's bf16 floor. This is the GDN-scan attn_out self-noise measure.
- **1.221e-4** (`wy_l1_spine_scan_live_fla_*.json`, native_reference `vllm.fused_sigmoid_gating_delta_rule_update`): WY vs the **LIVE CHUNKED FLA Triton kernel** (kkt→solve_tril→wy_fast→chunk_delta_h→chunk_o), a DIFFERENT op-order than the serial recurrence; per-element 1-bf16-ULP gap at the single worst spine depth (depth3/row4). Most depths in the SAME run are AT the seam-finder level: depth0=1.53e-5, depth1=7.63e-6, depth2=3.05e-5, **depth3=1.221e-4**, depth4=6.10e-5, depth5=1.221e-4 (`redteam_fix.json` by_depth). Only two depths hit the 1-ULP ceiling.
- **"fails 4/6"** is a THIRD quantity: it is NOT a per-layer-drift fact — it is the cascade-map's PROJECTION of the 1.221e-4 layer-1 GDN seam through ~27000× live amplification (L1 1.22e-4 → final-logit max_abs 3.32, `gateA_spine_ladder.json` logits.stats) onto final-logit max-over-vocab drift, compared against the argmax margin (the over-conservative dev-only gate of §3).

All three are ~1-bf16-ULP-class and ride within native's bf16 self-noise. **No contradiction:** "WY 6.1e-5 within 9.5e-5" (GDN-output, serial oracle, lossless) and "1.221e-4 fails 4/6" (amplified final-logit max-drift vs flip-threshold, live-FLA oracle) measure entirely different things.

---

## 5. HONESTY FLAGS — why this is (B) and not a self-declared pass

The argmax-lossless spine + irreducible-floor finding is **strong evidence the e2e passes, NOT a pass.** Three caveats, one of them hard:

1. **HARD: an FA2-fork tree e2e has ALREADY FAILED catastrophically, and no clean e2e has ever passed.** `FR13_LADDER_LOG.md:135`, `output/fr13_argmax_e2e_20260608T055851Z/e2e_compare_tree_vs_native_fresh.json`: a real B=4 CUDA-graph e2e of the forked-FA2 TREE_ATTN arm got accept/event **1.1134** (vs E5 3.076), accept/token **0.124** (vs 0.64), bag-TV **0.5017** (threshold 0.059), TPS 2.67 — FAIL on both gates. **BUT this is NOT evidence against the WY kernel at 1.221e-4:** (a) it predates the FR13_FA2_PREFILL_NATIVE fix (`FR13_LADDER_LOG.md` takeover @ 07:35 added it AFTER this run; before it, prefill full-attn layers 0-11 were NOT byte-exact); (b) the failure magnitude (4× accept collapse, 0.50 bag-TV) is a STRUCTURAL/prefill/mask breakage signature, not a 1-ULP softmax-mass shift (a per-layer 1-ULP cannot produce a 4× accept collapse without flipping argmax en masse, which contradicts the argmax-lossless single-event finding). The lesson is **the e2e must be MEASURED with the prefill-native-aligned WY build**, and the prior fail is a precedent that argmax-lossless-spine alone has NOT yet yielded a passing e2e. The WY kernel's e2e has **never been run**.
2. **Spine-only / single-event / B=1-eager evidence.** The §3 argmax table is one decode event (call0), B=1 eager, 6 spine rows. The real gate is B=4 + CUDA-graph + SWE-4; argmax-lossless may shift under B=4 co-residency.
3. **Branch rows 3,5,7,9 are UNCERTIFIED.** The spine scan covers only [0,1,2,4,6,8] (`redteam_fix.json` spine_rows). A mask-correctness seam (visible_mask 516-521 / output-visibility gate 573-583) is invisible to a spine-only ladder. Per `reference_gdn_tree_branch_oracle_losslessness`, branch correctness = per-depth argmax vs native-run-on-branch-path oracle (SpecInfer Def 4.1 / STree Eq.4-6) — **the e2e run MUST add the branch oracle** (`FR13_WY_CASCADE_MAP.md:21`).
4. **argmax-lossless ≠ temp-0.6 sample-distribution-lossless.** The e2e uses rejection sampling at temp 0.6 / top_p 0.95; a per-layer drift shifts softmax mass on the selected token, so accept/event could differ even with argmax preserved. Must be measured, not declared.
5. **Offline-smoke necessary-not-sufficient.** Offline depth-1 (1.53e-5) under-predicts live L1 (1.22e-4) (`FR13_WY_TAP_REDTEAM.md:25`); the live ladder, not the offline smoke, is the per-layer reference.

---

## 6. Why NOT (A) and why NOT (C)

- **NOT (A) "a correct closing tap exists":** §1 proves no correct tap remains — #1-#5 leave it at exactly 1 ULP, the only mover was the WRONG #6 (doubled it), and the hypothesized correct #6 (b_A bf16 round) has no tensor counterpart in WY's materialized-state readout and would require a banned readout rewrite. The readout is already ℝ-exact (9.31e-10). Chasing it is exactly the over-engineering FR13_FLOOR_WORKFLOW_VERDICT warns against.
- **NOT (C) "irreducible AND likely fails e2e → wall":** §3 shows the strict gate is over-conservative (argmax-lossless 6/6 on the spine), so 1.221e-4 does NOT obviously fail the accepted e2e. (C) would require evidence the argmax flips or bag-TV exceeds floor at the WY 1.221e-4 state — we have the opposite on the spine. The prior 1.1134 fail (§5.1) is from a pre-prefill-fix non-WY build and does not establish a WY wall. Declaring (C) now would be premature; the honest move is to MEASURE.

**(B) is the decisive, honest call: the residual is the irreducible 1-bf16-ULP floor, it is within the user-accepted argmax/within-floor gate on the spine, so stop tapping and run the e2e — and bring the numbers (with the branch oracle) to the user rather than self-declaring.**
