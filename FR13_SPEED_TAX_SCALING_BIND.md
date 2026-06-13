# FR13 Speed-Tax + Scaling — break-even, lm-head anatomy, the real lever

Workflow `wf_c3b84aab-9df` (CPU, 4 agents). Raw:
`research/fr13_workflows/speed_tax_scaling_wf_c3b84aab.raw.json`. Adversarial verify
`holds=FALSE` — the lm-head/tax math is SOURCE-VERIFIED and HOLDS; the "accept-gap-closed"
claim is REFUTED. Act on the math, not the accept conclusion.

## Question (user 2026-06-13)
"How did you get the +0.43..0.65 break-even number? CPU-analyze it. And find ways to let the
tree SCALE without sacrificing lossless. Is the lm-head lever the same as the WY kernel — would
it break lossless?"

## DURABLE (source-verified, verify confirms)
### The speed tax is ~1.05x and it is NOT the lm-head
- TPS = (accept+1)/s_fwd (the +1 = bonus token/event). Break-even: `acc_tree >= (acc_native+1)*tax - 1`.
- **Speed tax = 1.045x @64-tok, 1.056-1.063x @11k = ~+10 ms/forward** (NOT +96 ms — that was the
  PRE-FIX-1 drafter regime). Clean input (s/fwd is a stable substrate property, rep1==rep2).
- **The verify lm-head is ALREADY ONE BATCHED GEMM** — `gpu_model_runner.py:4090-4091` gathers
  all M rows (native 6 / cat9 10 / cat10 11) into one [M,5120] tensor → single `compute_logits`
  → `logits_processor.py:96` → `vocab_parallel_embedding.py:69` → `utils.py:98` F.linear over
  [M,5120]×[5120,248320], bf16 weight 2.5428 GB **read ONCE**. Patch `:7109` is a gather, not a
  matmul. Per added verify row = **+0.0019 ms** (M-invariant; 539 rows per +1 ms).
- **The +81.9 ms "per-row" tax was the DRAFTER head** (M=1 `gemvx` GEMV, full-vocab logits
  computed TWICE per spine step) — **already collapsed by FIX-1** (d407e545: gemvx 11.1→5.9/draft,
  chain5 1.41x→1.05x). So **L1 (batch the verify head) = NO-OP, nothing to collapse.**

### The real scaling lever = WY/replay GDN kernel (answers the user's WY question)
- The residual 1.05x tax is **GDN state-row traffic + the N_PAD≈16 register-spill wall**
  (FR13_SPEED_TAX_BASELINE: +42-46 ms/fwd per node ≈ 7× the row-traffic floor), NOT lm-head.
- So the lm-head can scale to **arbitrary tree width for free** — the binding cost of adding a
  NODE is the GDN per-node state traffic. **The lever that makes width cheap IS the WY-class
  kernel:** the GDN replay / accept-only-state kernel (FR13_GDN_KERNEL_LINEAGE: 36→6
  row-touches/layer = 0.86× native HBM, **spill-free at ANY tree size**, removes the N_PAD wall).
  Ranked levers: **L3 (wide tree) + replay-kernel >> L1 (no-op) > L2 (skip-dead-rows ~0.006 ms,
  do not implement).**
- **cat10 implication:** the +1 root-sibling ROW is free on the lm-head axis (+0.0019 ms), but the
  +1 NODE costs +42-46 ms/fwd of GDN traffic unless paired with the replay kernel.

### Would the lever break lossless?
- **L1/lm-head: no risk** (it's already batched, no change). The only caveat is the **M-bucket
  cuBLAS kernel swap** (bf16 GEMM at M=6 vs 10 vs 11 may pick a different tile/split-K → ULP
  reduction-order change → could flip a verify argmax near-tie, `rejection_sampler.py:394`
  exact argmax). This is NOT introduced by any change here (native already runs M=6, tree M=10);
  judged tree-vs-E5 within the self-noise floor, not bit-vs-a-different-M.
- **WY/replay: parked for failing abs-0.0 (byte-exact), but the current bar is within-floor**
  ([[project_fr13_active_worker_codex_fr15]] "WY PARKED-NOT-DEAD"), so it may pass now. And an
  accept-only-state replay (spine computed from ONLY the accepted path, no co-resident branch in
  the batch) could **FIX the spine contamination** (directive #2 / the cat10 finding) — possibly
  IMPROVING lossless. CAUTION: FR13_ACCEPT_ONLY_GATE4_FAIL_BIND (offline bit-identical ≠ live
  multi-step) — needs a GPU re-gate, not a paper claim.

## REFUTED (do NOT assert) — the accept gap is UNRESOLVED, not closed
The model claimed FIX-A closed the −0.43/−0.65 and the current gap is ~0 (a draw). The verify
REFUTES this on the SAME post-FIX-A raw (wf_d8a86320): **token-weighted AGGREGATE accept tree
3.1493 vs native 3.5765 = −0.427 — the "−0.43" IS reproduced in current data, NOT stale.** The
~0 gap appears ONLY on the per-request-MEDIAN basis (tree 3.5872 vs native 3.5955). Both bases are
trajectory-confounded DRAWS (forked at char-80; tree's long low-accept requests deflate the
aggregate on different token paths). **Accept gap spans −0.43 (aggregate) to ~0 (median) =
UNRESOLVED.** A PAIRED teacher-forced same-prefix accept capture is required before claiming the
tree ties/beats native. So even at tax→1.0 the tree's accept lead is NOT established.

## NET
- The speed tax (~1.05x, ~+10 ms/fwd) is clean and is **GDN state-row traffic, not lm-head**.
- The "easy-scale without sacrificing lossless" lever = **wide tree + the WY/replay GDN kernel**
  (the user's WY instinct was right — at the GDN layer, not the lm-head).
- The accept gap is a confounded draw (−0.43 to ~0); needs a paired teacher-forced capture to
  resolve — the only clean break-even input today is the s/fwd tax.
