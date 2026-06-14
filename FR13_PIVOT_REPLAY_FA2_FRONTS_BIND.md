# FR13 — PIVOT: L0-GDN sub-op A/B PARKED; two new fronts (replay durable-state + FA2-tree decode)

Date 2026-06-14. After the total-drift reanalysis (v1 `473cbda0` + v2 `e69f3444`, both HOLD) + the one-more-fix
verdict (ww3yd22ry). Two prior fronts complete; two new fronts launched.

## CLOSED: L0-GDN conv/scan sub-op M10-vs-M5 A/B — PARKED (5th consecutive infra block)
The one-more-fix (ww3yd22ry) **assert RECURRED**. `fe0af022` made the *reduced* M5/M1 conv arms kernel-valid
(non-spec, `state_len=width-1=3`) but LEFT the **M10 reference arm** on the spec path: width=4, seqlen=10 →
`state_len = width-1 + (seqlen-1) = 3+9 = 12` columns, overrunning the cloned conv-state bank's physical
`width-1=3` window → OOB global store inside `_causal_conv1d_update_kernel` → async device-side assert →
poisons the CUDA context (unrecoverable) → `recordsWritten=0`. The rebuilt gate's infra WORKED (env-in-worker
✓ via sidecar, engaged ✓ `layers.0.linear_attn tree_n=10`, loud markers fired) — purely a kernel-geometry
blocker. ROOT NOTE: the named file `fused_post_conv_prep.py:215` does NOT exist in the live vLLM 0.19 tree
(it's the audit copy); the real fault is `causal_conv1d_update` spec-path `state_len` store span.
**5th consecutive infra-only block** (env-to-worker ×3, eager, kernel-geometry ×2). The depth-intrinsic
prediction stands by THEORY but was NEVER empirically measured. **PARKED** because: (a) infra-cursed, and (b)
the reanalysis shows it's the WRONG front — even a clean ~0 would only *confirm* L0-GDN-conv/scan isn't the
carrier, not close the question. Bug-class: #9 (vacuous instrument — markers loud, still 0 records), #8 (28
CPU tests never modeled the live M10 spec-store overrun), #3 (per-arm clone isolated the engine to arm-fail;
the OOB is the kernel store SPAN not the clone). A further-fix exists (make the M10 ref arm ALSO kernel-valid:
full-M non-spec + post-hoc deep-row select, OR cap M10 spec seqlen, OR resize the cloned bank) but is NOT on
the critical path — the replay A/B supersedes it.

## Reanalysis v2 (e69f3444) — the verify pass STRENGTHENED the case
- **Back-loading signature**: flip-position norm mean **0.696** = flips concentrate LATE in the stream = the
  fingerprint of **cross-event accumulation** (not a per-forward defect). Corrected v1's cascade-inflation
  from ~1.5× → ~1.17× → **~18 independent events** (slightly MORE residual than v1 said).
- **External corroboration**: sglang #25587 (conv-state corruption after partial accept, diverges after ~100
  tokens = back-loading) + SpecMamba/dflash (recurrent-replay + projections both need stabilizing).
- Every per-forward kernel ruling HOLDS on fresh read; 2 FLAGS (FA2=prefill-only not decode; the 21-flip
  reference is the no-spec DECODE oracle = a different dispatch than tree-verify).

## OPEN FRONT 1 (GPU, w2vaqcsmx) — replay-durable-state-vs-native-MTP A/B  [21-flip axis]
The PRIME lever. Observe-only cross-event A/B: replay cat9's accepted chain through BOTH our
`_tree_gdn_replay_kernel` AND native `fused_sigmoid_gating_delta_rule_update` from the same cloned h0/conv-
state; record per-layer per-event `max_abs(H_ours - H_native_seq)` + first-nonzero + back-loading growth. The
"passed" byte-A/B only compared replay-vs-our-own-scan, never vs native. nonzero+growing ⇒ THE back-loaded
carrier (align OUR replay kernel bit-exact to the sequential reference = build-our-kernel, NOT reroute; the
durable state's spec IS the sequential recurrent state after the accepted tokens). ~0 ⇒ faithful, pivot to
TREE_ATTN. Reuses the PROVEN SUBOP_MAB sidecar-env + stage markers; operates on the LINEAR accepted chain to
DODGE the reduced-row assert that killed conv/scan 5×.

## ~~OPEN~~ CLOSED FRONT 2 (CPU, wkaexrv30 DONE) — FA2-tree-for-decode  [0.00195 DELIVERABLE axis]
**ANSWERED + closed (FR13_FA2_FORK_IS_DECODE_KERNEL_CORRECTION.md):** PREMISE WAS FALSE — `FR13_FA2_TREE_BIAS=1`
in the locked launcher means the **FA2-fork is ALREADY the deployed decode kernel** (cat9 max_query_len=9>1 →
`flash_attn_varlen_func(tree_bias=...)`). The 0.00195 is the `unified_attention` EXP2-Triton FALLBACK residual,
shadowed OFF = MOOT; nothing to swap. Live full-attn decode = fork floor 0.0039 (lossless, ~15× below E5);
FULL-captures + serves at B=4. Full-attn is NOT the carrier (node7 first-nonzero L0 GDN 0.0078 upstream of L3
full-attn 0.00409) ⇒ **reinforces the replay (L0-GDN cross-event) pivot.** Legitimate (fork = FLASH+tree-mask,
the sanctioned fallback, byte-verified splice-OFF), not a reroute. No GPU test needed.

## Standing
2/2 (1 GPU + 1 CPU). Per-forward kernels all M-invariant; the residual is now hypothesized as cross-event
(replay durable-state) + deliverable-backend (TREE_ATTN vs FLASH). accept/event 3.0-3.15 ~ native 3.076 ⇒
still sub-deployment-impact; "21" is not the irreducible floor it was presented as.
