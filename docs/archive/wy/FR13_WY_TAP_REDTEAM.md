# FR13 WY tap red-team — the 2.44e-4 is a MIS-APPLIED #6 tap, not the rewrite (workflow ww48n3yht, 2026-06-08)

## VERDICT (A): MIS-APPLIED TAP — fix in-kernel, re-smoke, NO ladder, NO rewrite, do NOT escalate
The 2.44e-4 is NOT the #6 readout reduction-order — that is **ℝ-exact** (fp32 oracle = 9.31e-10). It is **two extra bf16 rounds tap #6 added that native FLA never performs.** Cumulative-tap isolation (`output/fr13_wy_l1_payload_20260608T170530Z/wy_l1_spine_scan_live_fla_*.json`):

| variant | taps | WY-vs-live-FLA out_max_abs |
| --- | --- | ---: |
| bf16 | #1 l2norm | 1.221e-4 |
| bf16_kkt | +#3 | 1.221e-4 (unchanged) |
| bf16_fullbound | +#4 #5 | 1.221e-4 (unchanged) |
| **6tap** | +#6 | **2.441e-4** (doubled) |

The doubling happens ONLY when #6 is added → the two #6 rounds are the regression.

## The two over-rounds to DELETE (both contradict live native source)
1. **`fr10_gdn_tree_kernel.py:571-572`** — `q_i = q_i.to(bf16).to(f32)` AFTER `q_i = b_q*OUTPUT_SCALE` (570). Native `chunk_o.py:104` loads q bf16 **unscaled**; `:137` applies `scale` to the **fp32 output**. `b_q(bf16)*0.0884` is not bf16-representable → an extra round native lacks. q's bf16 boundary is already supplied by **#1** (508).
2. **`fr10_gdn_tree_kernel.py:581-582`** — `state_update_ij = (trans_j⊗k_j*decay).to(bf16).to(f32)` PER-j. Native `chunk_delta_h.py:235` rounds only the operand v_new to bf16, accumulates outer products in **fp32** (`:241`), rounds the state **once** at the chunk store (`:134`). Per-j rounding injects up to N ULPs (tree-size-scaling).

## Taps confirmed CORRECT (keep): #1 l2norm (508-509), #2 solve-T relocated (single round at solved stores), #3 KKt bf16-input+pre-beta (trans(b_k).to(bf16) is idempotent post-#1 = benign), #4 w/u (544-546), #5 tv_i (562-564).

## Fix: delete 571-572 + 581-582, keep #1-#5, re-run the OFFLINE smoke → expect return to 1.221e-4.

## HONEST open question (NOT this fix): 1.221e-4 still exceeds the gate floor
The gate-predict agent: at the measured live amplification (L1 1.22e-4 → final logits 3.32, ~27000×), even **1.221e-4 fails 4/6 spine margins** (pos5 2.50/pos7 0.75/pos8 0.125/pos9 0.25). So removing the over-rounds restores 1.221e-4 but does NOT pass the gate — there is a remaining ~1-ULP seam (the reduction-order is exact, so it is a bf16-BOUNDARY mismatch, likely a CORRECT #6 round of the intra-attention matrix `b_A` per `chunk_o.py:137`, distinct from the wrong q/per-j rounds). **Next stage = a CPU workflow to localize the residual 1.221e-4 boundary** before any live ladder.
**Caveats:** offline smoke is necessary-not-sufficient (offline depth-1 1.53e-5 vs live L1 1.22e-4); branch rows 3,5,7,9 uncertified by the spine smoke. Both downstream of restoring the floor.
