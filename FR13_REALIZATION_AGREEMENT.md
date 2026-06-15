# FR13 — REALIZATION AGREEMENT: do our three kernel realizations almost agree, and is there a rounding/op-order kernel mitigation?

Date 2026-06-15. CPU-only, READ-ONLY (a GPU reshape A/B runs concurrently — no code/boot touched; this doc is the
only write). vLLM source read DIRECTLY from the pinned image `vllm/vllm-openai@sha256:3dbe092e`
(= 0.19.2rc1.dev134+gfe9c3d6c5) via `scripts/vllm_src.sh`, NEVER a /tmp cache. Banked captures artifact-checked:
`output/fr13_node5_ladder/` (2026-06-14), `output/fr13_node7_ladder/` (2026-06-13),
`output/fr13_verify_decisive/q1_recur_vs_chunked.json` (2026-06-13). All recent, 0.19.2-keyed.

The user framing (the SPINE): "a flip means our kernel realizations do not agree; we as kernel designers should
make sure PREFILL and our COMMITTER and TEACHER-FORCING almost agree." This doc engages that LITERALLY and
quantitatively, reading all three realizations op-by-op, and answers the relax-vs-fix question the user owns.

**Posture (skeptic):** the incumbent verdict is "genuinely diffuse, lever = topology not kernel-align"
(FR13_DIFFUSION_DEEP_DIVE 5164c454). I only overturn it with op-by-op kernel evidence. The hard counter-evidence
(recompute STATE-aligned the scan to bit-exact, yet e2e flips ROSE 23→32, FR13_SCAN_NOT_E2E_CARRIER_BIND) is
confronted head-on, not hand-waved.

---

## THE THREE REALIZATIONS (named precisely, each kernel read from the pinned image)

| # | name | kernel (source, line-grounded) | l2norm op | beta op | b_h carry | store boundary |
|---|---|---|---|---|---|---|
| **(1)** | **CHUNKED-PREFILL** | `chunk_gated_delta_rule` (fla `chunk.py` / `chunk_delta_h.py`, WY/UT block-parallel), dispatched at gdn_linear_attn.py L990 `num_prefills>0`, `use_qk_l2norm_in_kernel=False` (l2norm applied OUTSIDE in `fused_post_conv_prep`, `apply_l2norm=True`) | torch `rsqrt` pre-kernel | torch sigmoid pre-kernel | fp32 chunk accumulator, **block-parallel WY** (different summation tree) | `last_recurrent_state.to(ssm_state.dtype)` = bf16 once per chunk |
| **(2)** | **DEPLOYMENT RECURRENT-DECODE** = the binding ORACLE | `fused_recurrent_gated_delta_rule_packed_decode` (`fused_recurrent.py` L313-336), dispatched gdn_linear_attn.py L807-810 `enable_packed_recurrent_decode and spec_masks is None and num_prefills==0 and num_decodes>0`. This is `scripts/fr13_recurrent_decode_oracle.py`'s frame. | `b_q / tl.sqrt(sum(b_q*b_q)+1e-6)` (**div**, L314) | `tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)` = **bf16 round-trip** (L325) | fp32 within one token; **ONE token per program** → state RELOADED from bf16 cache each step | `b_h.to(p_ht.dtype.element_ty)` = bf16 EVERY token (L335) |
| **(2′)** | native SPEC-decode update (the kernel cat9 DISPLACES) | `fused_sigmoid_gating_delta_rule_update` (`fused_sigmoid_gating.py` L60-200), dispatched gdn_linear_attn.py L959 `spec_sequence_masks is not None`, `use_qk_l2norm_in_kernel=True` | `tl.rsqrt(sum(b_q*b_q)+1e-6)` (**rsqrt**, L155-156) | `tl.sigmoid(b_b.to(tl.float32))` = **NO round-trip** (L153) | **fp32 b_h carried across the T-token loop** (L137 `tl.zeros(...,fp32)`); store bf16 to `ht` is EXPORT only | `b_h.to(...)` to `ht` each token, but `b_h` register stays fp32 between tokens |
| **(3)** | **OUR TREE-VERIFY COMMITTER** | `_gdn_node_step` + `_tree_gdn_kernel` (`fr10_gdn_tree_kernel.py` L423-650); LCP committer `_lumo_tree_path_lcp_max_greedy_sample` (`fr10_phase4_patch_vllm_tree_gdn.py` L6610). SCAN_ALIGN **default OFF**. | OFF: `tl.rsqrt(...)` (L480) ; SCAN_ALIGN: `b_q/tl.sqrt(...)` (L477) | OFF: `tl.sigmoid(b_raw_b.to(fp32))` NO round-trip (L468) ; SCAN_ALIGN: `.to(bf16).to(fp32)` round-trip (L466) | **fp32 `state_i`/`h_cache` carried in registers across the WHOLE spine** (L564) — NEVER round-trips bf16 between tree nodes | export-only `tl.store(state,...)` (L646), not read in-kernel |

**The structural fact the whole analysis turns on (read, not assumed):** our default tree-scan (3, SCAN_ALIGN OFF)
op-order MATCHES the native SPEC-decode update **(2′)** — `tl.rsqrt`, no-beta-round-trip, fp32 b_h carried in
registers. But the binding ORACLE is **(2)** packed-decode, which uses **div + beta-bf16-round-trip + state
reloaded from bf16 cache every token**. So "align our committer to the deployment oracle" is NOT "match the kernel
we replaced" — it is a real op-order DELTA (rsqrt→div, +beta round-trip, +per-token bf16 state round-trip). The
SCAN_ALIGN seams in our kernel were already built to do seams d (div) and e (beta round-trip) — but NOT the
per-token bf16 b_h store-reload, which is the one with depth growth.

---

## 1. THE FLIP MATH (exact inequality, measured numbers)

A flip at served position *i* is `argmax(verify_forward_logits_i) != argmax(oracle_forward_logits_i)` at clear
margin (`deviation_nat > 1.0` or served-id outside oracle top-k; the gold-margin instrument's definition,
`fr13_recurrent_decode_oracle.py`). It happens **iff the accumulated realization-residual at `final_norm`, pushed
through the lm-head GEMV, exceeds the token's CLEAN margin between top-1 and top-2.**

### The inequality
Let `r = h_live(final_norm) − h_clean(final_norm)` be the post-RMSNorm residual vector (this is what enters the
lm-head). The logit gap induced on the top-2 pair `(a=top1, b=top2)` is `Δ_ab = (W_a − W_b)·r`. The token flips iff
```
        Δ_ab  =  (W_a − W_b)·r   ≥   margin_clean(a,b)        [the flip condition]
```
where `margin_clean = logit_clean(a) − logit_clean(b)` (nats). Bounding by Cauchy-Schwarz,
`|Δ_ab| ≤ ‖W_a − W_b‖ · ‖r‖`, so a token with small clean margin and a residual aligned with a high-norm lm-head
row-difference flips; a format-fixed token (huge margin) does not.

### Measured numbers (node5, the carrier event; per_layer_maxabs.json + drive_result.json, MEASURED, input 0.0)
- Clean margin: ` ``` `(71093) −0.158 vs `Let`(9764) −2.033 = **margin_clean = 1.875 nat** (drive_result clean_reps).
- `final_norm` residual: **max_abs 7.59, L2 103.09** (post-RMSNorm). Pre-norm residual stream L2 = 178.5 (L63).
- The realized logit swing: ` ``` ` collapses live 15.94 vs clean 26.60 = **−10.7 nat** on ONE token (`Let` is
  essentially matched, live 25.38 / clean 24.80). So `Δ_ab ≈ 10.7 nat ≥ 1.875` → **flips**. The 10.7-nat swing from
  a 7.59-max_abs / 103-L2 norm perturbation is exactly the `‖W_a − W_b‖·‖r‖` order for a high-entropy boundary
  row.

### How the 1.166x/layer growth REACHES the crossing (MEASURED, node5 resid_L2 series)
resid_L2: **0.012 (L0) → 178.5 (L63)** = **14,800×** over 64 layers. Geometric-mean per-layer ratio
`= (178.5/0.012)^(1/63) = 1.166×/layer`. Per-layer ratios: median ≈1.10, ALL in [1.0, 1.34] except signal-birth
L1-L3 (trivial magnitude) and the deep full-attn jumps L35 (1.61), L47 (1.32), L51 (1.34), L62 (1.29). The flip
crystallizes L60 (clean reaches `71093`) and locks L61 (`9764`), holding L61→L63 (per_depth_argmax.json, MEASURED).

### Why structural boundaries flip, format-fixed do not (margin × residual interaction, MEASURED across nodes)
- node5 (` ``` ` vs `Let`): margin 1.875 → flips at final_norm max_abs 7.59.
- node7-p2 (` code` vs ` files`): margin **0.5** → flips at final_norm **max_abs 2.5** (cos 0.987) — SMALLER
  residual needed because SMALLER margin.
- node7-p3: final_norm max_abs **14.66**, cos **0.570** (huge accumulated divergence) → flips.
- Format-fixed positions: clean margin is many nat; the SAME ~2.5–7.6 max_abs residual cannot cross it → oracle
  agrees at dev=0. **This is why flips cluster at high-entropy structural boundaries (codefence/prose/JSON/tool):
  margin small enough for the fully-amplified diffuse floor to cross.**

**Flip-math verdict:** the floor is ~1 bf16-ULP at L0 (0.0008–0.0078), geometric-compounds ~1.166×/layer to
max_abs ~2.5–7.6 at final_norm, the lm-head GEMV turns that into a ~2–11 nat swing, and the flip needs only to beat
a 0.5–1.9 nat boundary margin. The math is a margin-vs-amplified-residual race; there is no single dominant layer.

---

## 2. THE THREE PAIRWISE GAPS (the core of the user framing — all banked)

| pair | what it is | banked number | argmax verdict |
|---|---|---|---|
| **(a) PREFILL (1) ↔ DECODE (2)** | does native's OWN chunked-prefill agree with native's OWN recurrent-decode? | L0 GDN: our-tree-vs-RECURRENT **0.000854** (~1 bf16 ULP) vs our-tree-vs-CHUNKED **0.0078** (9.14× larger) — i.e. PREFILL and DECODE realizations differ by ~9× at L0. **BUT at final_norm they CONVERGE: vs-recur (3.125 max_abs, cos 0.986) ≈ vs-chunk (3.375 max_abs, cos 0.988)** (q1_recur_vs_chunked.json, MEASURED) | The prefill↔decode FRAME difference is ~9× at L0 but washes out to ~8% (3.125 vs 3.375) by final_norm. **At the argmax level the two native frames AGREE on these positions** (the residual difference between frames is far below the 1.875/0.5-nat margins). |
| **(b) DECODE (2) ↔ OUR COMMITTER (3)** | the REAL deployment loss | **23 clear-margin flips** (cat9 OFF, per-prompt [5,4,5,9]) vs **native-E5 = 3** ([0,0,2,1]), SAME recurrent oracle (FR13_SCAN_NOT_E2E_CARRIER_BIND, non-vacuous triple-proven) | The binding 7× gap. This is the one the user wants closed. |
| **(c) PREFILL (1) ↔ OUR COMMITTER (3)** | what the node5/node7 LADDERS measured (live tree-scan vs chunked-prefill clean teacher-force) | node5 final_norm max_abs 7.59 / L2 103; node7-p2 2.5; node7-p3 14.66 — first-nonzero L0 GDN, geometric to L63 | The per-layer ladder picture. Its clean reference is CHUNKED-PREFILL (1), not the deployment oracle (2). |

### The frame-mismatch reconciliation (CRITICAL, answers the user's (2a) worry quantitatively)
The node5/node7 ladders score live tree-scan (3) vs **chunked-prefill (1)** teacher-force, but the deployment
oracle is **recurrent-decode (2)**. Is the ladder carrier therefore a measurement-frame artifact? **Quantitative
answer: NO, only ~8% of the final_norm residual magnitude is frame-mismatch.** q1 measures BOTH frames on the SAME
L0 carrier: the prefill frame inflates the L0 floor by 9.14× (0.0078 vs 0.000854), but by final_norm the two frames
give **3.375 vs 3.125 max_abs (within 8%)** and **cos 0.988 vs 0.986** — the frame difference is a SMALL fraction
of the accumulated residual. So the ladder over-states the L0 floor by ~9× but the FINAL flip-relevant residual is
nearly frame-invariant. And decisively: the **binding 23 was scored vs the RECURRENT oracle (2)** (correct frame,
FR13_SCAN_NOT_E2E_CARRIER_BIND `RECURRENT_PATH_ENGAGED=True` all arms), so the 23-vs-3 gap is REAL deployment loss,
not a prefill-frame artifact. The ladder is a valid per-layer GROWTH model; only its L0 seed magnitude is
frame-inflated ~9×.

**The user's "all three almost agree" target, scored:** (a) prefill≈decode at argmax (frame diff <margin) — ALREADY
HOLDS. (b) decode↔committer is 23-vs-3 — the GAP. (c) prefill↔committer is the ladder — same root as (b) modulo the
~8% frame term. So the ONE pair that does not "almost agree" is (b), and (c) is (b) viewed in the prefill frame.

---

## 3. KERNEL-ALIGNMENT CANDIDATES (op-by-op (3) vs (2), each tagged + ranked)

Read op-by-op: our `_gdn_node_step` (L450-488) vs native packed-decode (`fused_recurrent.py` L313-336, the oracle
(2)) AND the spec-update kernel (`fused_sigmoid_gating.py` L137-170, the displaced (2′)).

| # | op | OUR default (3) | ORACLE (2) packed-decode | seam type | reward-hack | plausible flip-reduction |
|---|---|---|---|---|---|---|
| **K1** | **b_h per-token bf16 store-reload** | fp32 `state_i`/`h_cache` carried in REGISTERS across the whole spine — **NO bf16 round-trip between nodes** | **state RELOADED from bf16 cache EVERY token** (one program per token; L313 `b_h=tl.load(p_h0).to(fp32)`, L335 store bf16) | **alignable-to-(2), the ONLY one with depth growth** | numerics-align AUTHORIZED | **the only candidate that could matter** — but see §4 (recompute already tested a stronger form of this and it ROSE). LOW-MED. |
| **K2** | l2norm | `tl.rsqrt(sum+1e-6)` (L480) | `b_q/tl.sqrt(sum+1e-6)` (div, L314) | alignable-seam-to-(2) (SCAN_ALIGN seam d exists, L477) | AUTHORIZED | last-bit only, **no depth growth** (per-token, M-invariant). ~0. |
| **K3** | beta | `tl.sigmoid(b_raw_b.to(fp32))` no round-trip (L468) | `tl.sigmoid(b_val).to(bf16).to(fp32)` round-trip (L325) | alignable-seam-to-(2) (SCAN_ALIGN seam e exists, L466) | AUTHORIZED | one bf16 round on a scalar per token, **no depth growth**. ~0. |
| **K4** | gate/decay op-order | `state_i *= exp(b_g); b_v -= sum(state_i*b_k); b_v *= b_beta; state_i += b_v*b_k; out = sum(state_i*b_q)` (L482-487) | **IDENTICAL** order (L333-335) | **already byte-aligned** (no diff) | — | 0 (already matches). |
| **K5** | conv anchor-row silu/bf16-tap | bf16-tap + ex2.approx silu (the onset seed, FR13_GATEA) | native causal_conv1d_update bf16 tap | topology-intrinsic at num_accepted>1 (tree anchor vs linear) | — | already conv-fixed (bf16 taps); 1-ULP onset seed only. ~0. |
| **K6** | scan summation tree (rank-1 vs tree-mask replay) | `tl.where(ancestor,...)` ancestor replay over h_cache (L567-573) | single roll-slot, no mask | **topology-intrinsic** (tree vs linear) | — | this is the irreducible tree-vs-linear difference; not a rounding seam. |

### The match-not-maximize-precision subtlety (stated explicitly, per the task)
The agreement target is the DEPLOYMENT oracle (2), which ITSELF bf16-stores b_h and uses div+beta-round-trip. Our
fp32-carry / rsqrt / no-round-trip is **MORE precise but DISAGREES with the bf16-store deployment path.** So a
"fix" here is to make our kernel LESS precise to MATCH (2) — adopt K1+K2+K3 so our committer realizes the SAME
rounding as the oracle it is scored against. K2/K3 are no-depth-growth last-bit seams (≈0 flip impact). **K1 is the
only candidate with the depth-growth property the carrier needs** — and it is the one the recompute test already
probed in a stronger (state-level) form (§4).

**Ranking by plausible flip-reduction: K1 (low-med, the only depth-growth candidate) >> K2 ≈ K3 ≈ K4 ≈ K5 ≈ 0.**
This is the op-by-op evidence the incumbent "diffuse, topology-only" verdict demanded — and it does NOT overturn it:
five of six seams have no depth growth (per the MEASURED M-invariance + per-token-only structure), and the one that
does (K1) is bracketed by the recompute-rose result below.

---

## 4. RECONCILE with recompute-rose + diffuse (do NOT contradict silently)

**Is a per-token bf16-store/op-order alignment (K1) the SAME lever as the recompute STATE-alignment that ROSE
23→32? Partly, and that is bad news for K1 being a fix.**

- **What recompute did (FR13_SCAN_NOT_E2E_CARRIER_BIND, MEASURED):** it re-derived each tree node's STATE from the
  spine and made our scan STATE **bit-exact (int-view 0.0)** to native packed-decode. STATE-bit-exact is a STRONGER
  condition than K1's per-token bf16-store: if the final state is bit-exact, the per-token store boundary is
  necessarily matched. So recompute already gave us the K1 endpoint (and more) at the STATE level.
- **Result:** e2e clear-margin flips ROSE 23→32 (per-prompt [10,9,7,6]), common-prefix-normalized 25, per-token
  rate 0.0529→0.0625 — artifact-checked, NOT a length artifact. It rose because recompute CHANGED THE TRAJECTORY
  (different verify logits at near-ties → different LCP-max path → ~369 token-diff stream), and recompute is NOT
  byte-lossless so it is not a drop-in anyway.
- **Is K1 (in-place body bf16-store, NOT recompute) DIFFERENT + plausibly helpful?** It is different in mechanism:
  K1 keeps the TREE structure and h_cache topology and only inserts a `state_i.to(bf16).to(fp32)` round-trip per
  node (a 1-line body change applied 48× across GDN layers), so it does NOT change the accepted path the way
  recompute did — it would be byte-deterministic and could stay a drop-in. BUT the recompute result is the dominant
  evidence: making the scan state MORE aligned to native (in fact bit-exact) did not move flips toward 3; it moved
  them AWAY. K1 is a WEAKER alignment than recompute's bit-exact state, so the prior strongly predicts K1's e2e
  impact is ≤ recompute's (≈0 or negative), not a collapse to native-3.

**Does "50+ layers each 1.1×" mean NO single rounding change lands native, OR could one op-order applied 48×
collapse it?** Honest + quantitative both ways:
- Argument it CANNOT (incumbent): the floor is ~50 comparable 1.05–1.34× contributions; zeroing any ONE layer's
  step changes L63 residual by its ratio (1.1–1.6×), never enough to recover a 1.875-nat margin. K2/K3 are per-token
  last-bit (no depth growth) → provably ~0. K1 is the only systematic one, but recompute (a superset of K1) already
  showed state-alignment does not reduce e2e flips. So the per-layer math says no single rounding change lands
  native.
- Argument it MIGHT (the one honest opening): native-E5=3 EXISTS at the same model/fp8/frame → a 3-flip realization
  is reachable. K1 is "match ONE recurrent op-order across ALL 48 GDN layers" — exactly the kind of SYSTEMATIC
  (not single-layer) alignment that could in principle collapse a correlated diffuse floor, BECAUSE the per-layer
  diffs are CORRELATED (each layer's diff ∝ the residual it receives; that's why it's geometric not additive). A
  correlated floor can in principle be cancelled by a single systematic root change even though no single LAYER
  dominates. **But** the carrier is NOT a single per-forward seam: recompute removed the strongest per-forward state
  diff and flips rose; the de-cascade analysis (FR13_CARRIER_REOPEN) shows the excess is trajectory-fork +
  tree-amplification, not one un-aligned recurrent op. So the "systematic K1 collapses it" hope is real in theory
  but the bank's direct test of a stronger version refuted it.

**Reconciliation verdict:** K1 is mechanistically distinct from recompute (in-place, byte-deterministic,
drop-in-able) and is the ONLY kernel-align candidate worth a cheap test, but the recompute-rose result is a powered
prior that K1 will NOT collapse the gap to native-3. The diffuse + topology-carrier verdict SURVIVES the op-by-op
read; K1 is the one residual doubt, and it is cheap to settle (§5). I do NOT overturn the incumbent.

---

## 5. ONE CHEAP NON-VACUOUS TEST (settles the top candidate K1)

**Test: in-place bf16 b_h store-reload alignment (K1), re-scored vs the RECURRENT oracle (same frame as the binding
23).** This is DIFFERENT from the recompute test: recompute changed geometry+trajectory (BV32/w1, ancestry replay,
non-lossless 369-tok diff). K1 keeps the EXACT cat9 tree-scan geometry/topology and inserts ONLY a per-node
`state_i = state_i.to(tl.bfloat16).to(tl.float32)` round-trip inside `_gdn_node_step` (the b_h store boundary that
(2) realizes every token), gated by a new `SCAN_ALIGN` sub-flag — byte-deterministic, drop-in-able if it helps.

Procedure (single GPU boot, ~minutes, A/B at fixed seed/prompts):
1. Add the per-node bf16 round-trip to `_gdn_node_step` behind a flag (1-line, applied to all 48 GDN layers via the
   shared body). Combine with the EXISTING SCAN_ALIGN seams d (div) + e (beta round-trip) so all three (2)-matching
   roundings are on together.
2. Run cat9 at temp 0.0 seed 1313 on the SAME 4 prompts, re-score every position vs the SAME
   `scripts/fr13_recurrent_decode_oracle.py` (the binding-23 frame).
3. PREDICT (incumbent): flips stay ~23 or rise (per the recompute prior). FALSIFY-the-incumbent outcome: flips drop
   toward native-3 → K1 IS the systematic lever and "diffuse/topology-only" partially reverts to "one un-aligned
   recurrent store boundary."

**Non-vacuity proof (the 4 vacuous instruments this session burned are avoided — playbook #9):**
- **neg-control flips:** run with the bf16-round-trip flag ON but pointed at the WRONG dtype (fp16) — a powered
  control that MUST change the served stream; if the served stream is byte-identical with the flag on, the flag is
  not applied (fail loud), exactly the `negative_control_powered=True` hardening from FR13_SCAN_NOT_E2E (e428db3a).
- **oracle engaged:** assert `RECURRENT_PATH_ENGAGED=True` + nonzero `_forward_core_decode_non_spec` calls on all
  arms (the oracle's built-in class-9 assert); flips are GOLD-MARGIN (`deviation_nat>1.0` + full oracle_topk), NOT
  streamed top_logprobs.
- **flag actually applied:** bridge-needle the worker `/proc/<pid>/environ` for the K1 sub-flag + confirm the served
  stream DIVERGES from SCAN_ALIGN-off (if K1 changes any rounding it must change SOME token; if byte-identical to
  off, the constexpr threaded dead per bug-class #10 → vacuous, fail loud).
- **discriminator powered:** the recompute arm already moved flips 23→32 on this exact harness → the instrument
  resolves a real ±9-flip signal; K1's predicted band (≈23 vs a collapse to ~3) is well inside resolution.

This is the single highest-value GPU minute for the kernel-align question. It is ORTHOGONAL to the reshape A/B
running concurrently (that tests the TOPOLOGY lever, §4's incumbent fix; K1 tests the last kernel-align doubt). If
the user wants ONE test for the relax-vs-fix decision, this is it: it is the only experiment that distinguishes
"in-place store-boundary alignment (drop-in)" from the already-refuted "recompute state-alignment (trajectory
change, non-lossless)."

---

## VERDICT (for the user's relax-vs-fix decision — NOT a close/pass-fail, that is the user's call)

**There is NO strong kernel rounding/op-order mitigation that makes our committer agree with the deployment oracle.**
The op-by-op read found six candidate seams; five (K2 l2norm-div, K3 beta-round-trip, K4 gate-order [already
identical], K5 conv-tap [already fixed], K6 scan-summation-tree [topology-intrinsic]) have NO depth-growth property
and are provably ~0 flip impact (per-token, M-invariant, MEASURED). The ONE with depth growth — K1, the per-token
bf16 b_h store-reload that (2) realizes — is bracketed by the recompute-rose result (a STRONGER state-bit-exact
form already ROSE 23→32), so the powered prior is that K1 does not collapse the gap either. The disagreement is
**predominantly topology-intrinsic** (tree-vs-linear amplification of a diffuse correlated per-layer floor +
trajectory fork), consistent with the incumbent "diffuse, lever=topology" verdict, which I do NOT overturn.

**Honest residual doubt (the only thing keeping this from a hard no):** K1 is mechanistically distinct from
recompute (in-place, byte-deterministic, drop-in-able, keeps the tree topology) and is a SYSTEMATIC alignment
applied 48× to a CORRELATED floor — the one class of change that could in principle cancel a diffuse floor that no
single layer dominates, and native-E5=3 proves a clean realization is reachable. That doubt is CHEAP to settle (§5,
one GPU boot, non-vacuous). **Recommendation to the user: the kernel-align lever is weak (≈ already-refuted via
recompute); the strong lever is TOPOLOGY (the reshape A/B). Run the §5 K1 test ONLY to close the last kernel doubt
before relaxing to the topology/accept-event arbiter — do not invest in K2–K5 (provably ~0).** Relax-vs-fix is yours.

---

## Playbook rows quoted (FR13_BUG_CLASS_PLAYBOOK)
- **#9 Silent fallback / vacuous instrument** — the §5 test's non-vacuity proof (neg-control flips, oracle engaged
  via `RECURRENT_PATH_ENGAGED`, flag confirmed in worker environ + served-stream divergence) is built to avoid the
  4 vacuous instruments this session burned; "a run passes while measuring nothing" is the failure mode being
  guarded.
- **#10 Shared-source ≠ shared-SASS (codegen identity)** — K1/K2/K3 are realization-identity questions: our
  `_gdn_node_step` and native's kernels inline near-identical bodies but compile/round differently (rsqrt vs div,
  fp32-carry vs bf16-reload, tree-mask replay vs roll-slot). The alignment is a byte-A/B / int-view-0.0 question
  (NEVER atol). The recompute int-view 0.0 was the byte-A/B that PROVED the scan state alignable — and per #10 it
  did not transfer to e2e (different SASS path → different trajectory). K1 must be gated by the same byte-A/B.
- **#12 Measurement traps / depth-accumulation-trajectory** — the ~9× prefill-vs-decode L0 frame inflation is a
  frame trap (the ladder over-states the L0 seed; corrected to ~8% at final_norm via q1). The raw 23/32 flip counts
  are length/cascade-inflated; the arbiter is accept/event + per-token gold-margin, not the raw count. The
  per-depth early-exit argmax flickers (L34/39/45…) are the per-pos-counter projection trap, excluded. "Non-like-
  for-like trajectories after fixes" is exactly why recompute's 23→32 is not a clean refutation of K1 and why §5
  holds geometry/topology fixed.

## Reward-hack / hygiene
CLEAN: pure read of banked artifacts (timestamps checked, all 2026-06-13/14) + committed vLLM source via
`scripts/vllm_src.sh` (pinned 3dbe092e, no /tmp cache); no GPU boot; no code/launcher edit; no served-path splice;
this doc is the only write. K1/K2/K3 are NUMERICS ALIGNMENT (op-order/bf16-boundary/l2norm — AUTHORIZED per
reference_no_reroute_reward_hacking), our kernel COMPUTING, matched to native's realization — NOT splice (native =
A/B oracle only), NOT copy-recurrent multi-spine (CLOSED), NOT dense, NOT forced-spine, NOT recompute-bake (369-tok
non-lossless). No self-declared pass/fail/relax; the arbiter remains e2e cat9-vs-E5 (per-depth-argmax + within-floor
+ accept/event ≥ native), brought to the user.

Pairs with FR13_DIFFUSION_DEEP_DIVE.md, FR13_SCAN_NOT_E2E_CARRIER_BIND.md, FR13_CARRIER_REOPEN.md,
FR13_NODE5_LADDER_DIFFUSE_BIND.md, FR13_NODE7_LADDER_BIND.md,
[[reference_diffuse_gdn_accumulation_explained]], [[feedback_math_correct_vs_bitexact]],
[[reference_gdn_verify_sequential_dispatch]], [[project_fr13_tree_reshape_unifying_lever]],
[[feedback_fr13_lossless_compare_target]].
