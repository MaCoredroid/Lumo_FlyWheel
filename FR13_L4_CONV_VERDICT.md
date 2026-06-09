# FR13 L4 conv divergence — ROOT + fix (workflow wm2k077fr, 2026-06-09, source-verified). WIRING fix, NOT kernel.

## ROOT (PROVEN): the tree conv prior-window READ uses a HEAD-vs-TAIL column convention + off-by-one vs native's rolled tail
- conv INPUT (pre_conv) bit-exact 0.0 → the divergence enters only through the CARRIED history taps, not the live input.
- **Staircase fingerprint (proves it's the 3 history columns, NOT a numeric seam):** conv1d_out spine by depth = depth0 (3 history taps) 0.0556 → depth1 (2) 0.0176 → depth2 (1) 0.0039 → depth3-5 (0 taps, history slides out of the width-4 receptive field) **EXACTLY 0.0**. A bf16/numeric seam would scatter ~1-ULP independent of tap position; the clean cutoff at depth-3 = only the 3 prior columns differ. NUMERIC RULED OUT (the bf16 tap path _fr11_conv_tap_product is already native-aligned).
- SOURCE-INDEX is a red herring for THIS first divergence: spine source indices MATCH native at depths 0-2 (where the divergence is); the caterpillar mislabel only appears at depth>=3 where output is already 0.0. (Real for branches, must still be fixed, but masked here.)

## THE BUG (file:line)
- **Native** `/tmp/vllm_live_019/.../mamba/ops/causal_conv1d.py` reads the **rolled TAIL**: `conv_state_token_offset = num_accepted_tokens-1` (~:859-873), and the kernel reads `prior_tokens = conv_states_base + (state_len-1)*stride` (~:156). 
- **Tree** `scripts/fr10_phase4_patch_vllm_tree_gdn.py` reads the **HEAD**: prior-window read `:800-815` via `gather(spec_state_indices, clamp(accepted_lens-1))`, head col base `_fr10_prior_col_base = arange(width-1) = [0,1,2]` `:868`, assembled `:968-977`. Plus off-by-one: tree `accepted_lens=[0]` vs native `num_accepted_tokens=[1]`.
- **Write-back** `:1254-1312` stores `_fr10_new_state` = prior-bank ⊕ node-x in a different column layout vs native's sliding-window causal_conv1d_update → next step's head-read mis-consumes it.
- `launch_tree_state_linear_remap` (`fr10_gdn_tree_kernel.py:84-203`) copies whole rows, does NOT roll conv columns, no-op at accepted_len=0 — EXONERATED as the corruptor (but the structural reason the layout is never reconciled to native's tail).

## WIRING vs KERNEL → WIRING (do NOT build a kernel, NO splice)
The conv arithmetic is bit-exact given a correct history (offline replays L0/L45 = 0.0; tap/SiLU/accumulate correct). The mismatch is a READ/WRITE *convention* (head-vs-tail column + off-by-one) across the tree-verify step. Fix = reconcile OUR conv-state ring-buffer handoff to native's rolled-tail convention. Finite per-GDN-layer wiring fix.

## FIX (codex)
1. **Read-convention fix:** change the tree prior-window READ (`:800-815`/`:868`) to native's rolled-TAIL convention (read the tail column `num_accepted-1`-relative, not head `[0,1,2]`), + reconcile the `accepted_lens` vs `num_accepted_tokens` `-1` source. This corrects OUR column convention; NOT a splice of native's conv output (read-only convention fix on OUR bank).
2. Re-ladder L4 → expect conv1d_out spine **0.0556 → 0.0**.
3. **If residual remains** → the write-back roll (`:1254-1312`) is also involved → store native's exact `[history…, accepted…]` ordering so the next step's read matches.
4. **CAPTURE CAVEAT:** capture native's prior window **PRE-kernel** (before causal_conv1d_update rolls the buffer). The raw `6.05` overstated the gap (post_update_fallback + arms booted different conv geometry state_len 8 vs 12); the LIVE conv1d_out 0.0556/staircase is the real, self-consistent signal. Expect tree-col1 ≈ native-col0 (a 1-column roll), not a value seam.
- WIRING fix, no kernel, no copy/dense/splice. Why L4: first GDN layer whose carried conv-state was last written by the divergent tree-verify handoff (L0-3 history came from the identical prefill). NO self-declare; the tail-convention re-ladder (conv1d_out→0.0) is the verdict.
