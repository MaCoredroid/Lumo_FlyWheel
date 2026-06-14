# FR13 — the +2 spine floor (our pure spine 5 vs native 3): seam decomposition, lever ranking, non-WY closability

Date 2026-06-14. **READ-ONLY research** (no kernel/patcher change; pathspec commit only). Concurrent GPU
workflow = cat3w boot — untouched. Informs the decision; does **NOT** conclude the wall
([[feedback_research_before_deadend]]). Pairs with FR13_CHAIN3_DEPTH_LEVER_DEAD_BIND.md,
FR13_CONV_NOT_CARRIER_SCAN_STATEFEED_BIND.md, FR13_FA2_CARRIER_OVERTURNED_BIND.md,
FR13_NODE5_LADDER_DIFFUSE_BIND.md, FR13_GATEA_DEEP_DIVERGENCE.md.

## 0. The question, restated precisely
chain3 (depth-3 pure spine) = chain5 (depth-5 pure spine) = **5** clear-margin flips, each vs its OWN
no-spec teacher-force oracle, thr 1.0 nat, prompts_swe4. native E5 (FLASH MTP-5) = **3**. Depth lever DEAD
(5→5 across D=3/5). leaf co-residency = +17 (cat9 22 vs spine 5), removable by reshape. THIS doc =
**the +2 (spine 5 vs native 3)**: the our-kernel-vs-native per-forward realization gap on a **branchless**
spine. Is it closable WITHOUT reviving WY (PARKED on remote, not revived without the user)?

**Critical comparison caveat ([[reference_multispine_not_lossless_closed_nonship]], class 12):** native E5
and the spine arms run DIFFERENT streams; +2 is a COUNT (5 vs 3 each-vs-own-oracle), NOT position-by-position.
We reason about the MECHANISM (why our realization gap > native's), and do NOT claim 2 identifiable extra
positions. Both arms' oracles flip at the SAME high-entropy boundary set (native p1/p3 = `Let`↔```` ``` ````,
p2 = `"`↔`'`; chain5 p2 = `tool`/`_call`/code-fence cluster) — the COUNT is set by realization-gap MAGNITUDE
over that fixed boundary set (FR13_CHAIN3 corrected model).

---

## 1. THE +2 CARRIER (dominant): GDN recurrent state-feed = chunk-vs-recurrent, NOT attention

**Decisive number (output/fr13_verify_decisive/q1_recur_vs_chunked.json, q1_summary.json), deep-spine carrier
node, byte-exact input (`input_hidden=0.0`):**

| L0 GDN first-nonzero | our tree-row vs ... | max_abs |
|---|---|---|
| vs **chunked-prefill** oracle | the no-spec teacher-force the spine arms are scored against | **0.0078125** |
| vs **TRUE per-path recurrent** (non-MTP, recurrent state build) oracle | the physically-correct reference | **0.0008544921875** (~1 bf16 ULP, at floor) |
| ratio | | **9.14×** |

q1_summary verdict (quoted): *"vs the chunked prefill was ~89% INPUT-DRIFT (chunk-vs-recurrent
state-construction), not op-divergence."* The GDN scan **op** is at the bf16-ULP floor vs the TRUE recurrent
oracle. The ~0.0078 the spine arm shows vs its scored oracle is **dominated (×9) by the realization
difference between two ways of building the SAME recurrent state**:
- **live tree-verify** builds node-N's state via a **rank-1 sequential tree-scan** over the accepted chain
  seeded from `b_h0` ([[reference_gdn_verify_sequential_dispatch]]: native verify =
  `fused_sigmoid_gating_delta_rule_update`, the sequential rank-1 recurrence; ours re-indexes the same loop
  by tree-ancestry → on a pure spine it collapses to native's loop);
- **the no-spec oracle** (`fr13_oracle_stream_teacher_force.py`, `max_tokens=1`, prefix = prompt + served[:i],
  `spec_decode_*` counters delta = 0 → spec OFF) computes the prefix logprobs via a **chunked-prefill scan**
  (`chunk_gated_delta_rule`, [[reference_gdn_verify_sequential_dispatch]]:1142-1148 = prefill-only). Two
  realizations of one recurrent state = the documented **chunk-vs-recurrent ~ULP gap**, fp non-associative
  (Yang 2406.06484: chunked-WY is ℝ-equal but NOT bit-exact to the sequential rank-1 chain).

**Why our COUNT (5) exceeds native's (3):** native E5's scored oracle is ALSO a chunked-prefill teacher-force,
but native's **live verify** is MTP-5's own sequential recurrence over a clean 5-token causal chain — the SAME
kernel family vLLM ships for both prefill and decode, tuned together. Our live verify is the **fr10 tree-scan
re-index** of that loop (different Triton codegen / load-order / register-resident vs HBM-state path), so
our (recurrent-realization − chunked-oracle-realization) gap is larger than native's, by enough to push 2
more boundaries across the margin. This is a **realization** difference, not a math error — and not the bank
(FR13_NODE5_LADDER: FILL/READ land on the same `num_accepted-1` column across all 4 touchpoints; our fp32
multi-column state is INTENTIONALLY more precise than native's bf16 roll-slot — aligning down to bf16 would
be the reward-hack).

**Bug-class framing (FR13_BUG_CLASS_PLAYBOOK.md):**
- class **10** (Shared-source ≠ shared-SASS): the fr10 tree-scan and native's `fused_sigmoid_gating` inline
  the same delta-rule body but compile to a different fp32 instruction realization (the kernel comment at
  `fr10_gdn_tree_kernel.py:357` *intends* "the identical fp32 instruction sequence on bit-identical inputs" —
  the residual is whether codegen actually delivers that). Discriminator = byte A/B / int-view equality.
- class **12** (measurement trap): the 0.0078 was ~89% input-drift, not op-divergence — the chunked oracle is
  NOT the right reference for the op; the recurrent oracle is. Already corrected in q1.

### Non-WY sub-levers for candidate 1, and their in-our-kernel feasibility
| sub-lever | state today | feasibility WITHOUT WY |
|---|---|---|
| **fp32 recurrent state accumulation** | ALREADY DONE. `fr10_gdn_tree_kernel.py`: `h_cache` fp32 (:458), state bank fp32-enforced (:742-744), all loads `.to(tl.float32)`. | No headroom — we are already more precise than native. Cannot move the count by adding precision. |
| **op-order / l2norm / raw-g alignment in the rank-1 scan** | `use_qk_l2norm_in_kernel=True` (:724); raw-g path (`b_g=-exp(A_log)*softplus`, :369-372); beta via `sigmoid(raw_b.to(f32))` (:373). Matches the [[reference_gb10_gdn_backend_fla]] fla_chunk semantics. | OPEN + grindable: drive our **recurrent** scan to byte-exact vs native's **recurrent** verify (NOT vs the chunked oracle). This is the in-kernel, non-WY path. It is class-10 codegen alignment (num_warps / static_range-unroll / load-order / cast-boundary), the SAME family that cracked the conv (bf16-tap) and the prior scan (static_range→tl.range). |
| **match native's MTP-5 single-step recurrence exactly** | our tree-scan re-indexes native's loop; on a pure spine it SHOULD collapse to native's loop identically. | The decisive non-WY test: a **recurrent-vs-recurrent** A/B (ours-recurrent vs native-`fused_sigmoid_gating`-recurrent on the SAME spine prefix + h0), NOT recurrent-vs-chunked. If they byte-match, the +2 is the oracle's chunked-vs-our-recurrent artifact (then native suffers it too and the gap is a measurement frame, not a kernel gap). If they don't, the residual is a localizable codegen seam → align bit-exact. |

**Verdict on candidate 1:** the +2 carrier is HERE, and the closable-without-WY path is the
**recurrent-vs-recurrent op alignment** — NOT the chunked-WY kernel. WY was about aligning a batched
WY/UT-solve to a (non-existent) chunked verify readout (falsified, [[reference_gdn_verify_sequential_dispatch]]);
that is NOT what closes this. The fp32-state sub-lever is exhausted (already done); op-order/codegen alignment
of the **recurrent** scan to native's **recurrent** verify is the open, in-our-kernel, non-WY lever.

---

## 2. FA2-fork TREE_ATTN vs FLASH on a BRANCHLESS spine — THE CRUX, and it is NOT a reward-hack to use a FLASH-identical path

**Finding (proven by direct construction): for a branchless chain the tree-bias matrix IS the standard
lower-triangular causal mask, bit-for-bit.** `_prepare_tree_attn_bias(chain5)` (replicated from live
`tree_attn.py:256-292`) yields exactly:
```
[[  0  -inf -inf -inf -inf -inf]      torch.equal(chain5_bias, lower_triangular_causal) == True
 [  0    0  -inf -inf -inf -inf]      (verified this session)
 [  0    0    0  -inf -inf -inf]
 [  0    0    0    0  -inf -inf]      cat9 (with branches) == pure-causal: FALSE
 [  0    0    0    0    0  -inf]        (off-diagonal -inf blocks cross-branch attention)
 [  0    0    0    0    0    0 ]]
```
A spine node's only ancestors are the prior spine nodes = exactly the causal lower triangle. **The tree-bias
adds NOTHING beyond causality on a pure spine.** The tree-specific structure (off-causal -inf to block
cross-branch attention) appears ONLY when branches are present (cat9).

**What the FA2-fork does to a spine row (apply_tree_bias, fr13_patch_fa2_tree_bias.py:26-74):**
- The bias is applied ONLY to the **query-suffix × query-suffix** block (`q_rel`,`k_rel` both relative to the
  tree's own rows, `[tree_len,tree_len]`). The **context KV** (the long prefix, `k_rel<0` after
  `- context_len`, or `>= tree_bias_cols`) gets **NO bias** → pure causal, identical to stock FA2.
- Within the suffix block, where `bias==0` (ancestor/self): `tensor += 0.0f / scale` — IEEE754 `x + 0.0 == x`
  EXACTLY for finite x (only `-0.0` sign, irrelevant to magnitude/argmax). Where `bias==-INFINITY`
  (above-diagonal): `tensor = -INFINITY` — identical to stock causal masking (`exp2(-inf)=0` exact,
  per [[project_fr13_fa2_fork_nocopy_floor]]).

⇒ **For a pure-causal spine, the FA2-fork tree-bias path is functionally a no-op vs stock causal FA2.** The
spine's attention output is NOT perturbed by the tree-bias mechanism. The +2 is therefore **NOT** an
FA2-tree-bias effect.

**Is the tree-bias even "active" on a chain?** It is *invoked* (the kernel runs the bias loop), but it is
**arithmetically inert** on the spine block (`+0.0` / standard `-inf` causal) — so it is not a *source* of
divergence. This is consistent with FR13_FA2_CARRIER_OVERTURNED (the FA2 query-tile is a downstream
amplifier, not the carrier — even when QPAD zeroed L31, e2e flips did not move) and the NODE7-LADDER
(first-nonzero at L0 GDN, upstream of the first full-attn layer L3).

**The residual 2-ULP FA2 floor ([[project_fr13_fa2_fork_nocopy_floor]]): 14/16 calls whole-tree 0.0; 2
single-bf16-ULP in ~983k.** This is the irreducible **no-copy MMA fp32 fragment-grouping** floor (scattered
ancestor KV slots → different accumulation lane than a packed oracle; `flash_fwd_kernel.h:367` gemm_rs). KEY:
this floor is a property of the **scattered/paged KV layout of the tree**, NOT of the bias. On a pure spine
the KV is contiguous-ish (no branch rows interleaved), so the spine sees AT MOST this ~2e-6 probabilistic
tie-break, max 0.0039 = **15× below the E5 self-noise floor ~0.059, with no depth growth** (rounding
signature). It cannot be the +2 carrier: a 2e-6 non-compounding tie-break does not cross 1.1–9.75-nat
decision boundaries; the GDN state-feed (compounding L0→L63) does.

### The crux: could spine rows route through a FLASH-identical (no-tree-bias) path WITHOUT being a reward-hack?
**YES — and it is NOT a reroute-to-pass-a-metric ([[feedback_no_reroute_reward_hacking]], FR12 lesson).** The
distinction the directive flags:
- **REWARD-HACK (banned):** routing the spine through native's kernel *to pass a parity metric* while OUR
  kernel is unchanged / the spine's real computation still happens elsewhere. (FR12: routing spine through
  native `causal_conv1d_update`/FLA to pass parity = hack because our kernel was the deliverable and stayed
  broken.)
- **LEGITIMATE COMPUTATION (allowed):** a branchless chain **genuinely needs no tree-bias** (proven above:
  its bias == causal). Computing the spine attention via the standard causal FA2 path is the *correct*
  computation for that input, not a route-around — because there is no ancestry to mask beyond causality.
  The deliverable (our tree verifier) is *unchanged*; we are observing that on the no-branch sub-case the
  tree op DEGENERATES to causal FA2, which is what it already does (the `+0.0` no-op).

**BUT — the practical lever is empty, because there is nothing to close on candidate 2.** The spine attention
is ALREADY effectively FLASH-identical (the bias is inert; the only residual is the 15×-below-floor
non-compounding 2-ULP). The +2 does NOT live in attention. So "route spine through FLASH" would buy ~0 flips
(the spine attention already matches FLASH to within the do-not-grind floor). The deployable shape (cat9/cat3w)
needs branches for accept, and **branch rows legitimately DO need the tree-bias** (their bias is non-causal);
they cannot route through plain FLASH. **A spine-only-FLASH dispatch is feasible and clean, but moves ~0 of
the +2** — confirming the +2 is the GDN state-feed, not attention.

---

## 3. L60/L61 deep full-attn crystallization — AMPLIFIER confirmed, not origin

Confirmed from `output/fr13_node5_ladder/per_layer_maxabs.json` (deep-spine carrier, byte-exact input):
divergence is born at **L0 GDN (hid_max_abs 0.0039)**, stays small/diffuse and **monotone** (L23 0.053,
L30 0.16, L50 1.09 — all gentle, jump-ratios <1.7×), then the residual stream **explodes through the deep
full-attn band**: L51 (full_attn) resid 3.0, L52 9.0, L53 15.75, L54 18.75, L55 (full_attn) 24.75, L58 25.5,
L60 18.0→L62 18.5, **L63 (full_attn) hid 11.5 / resid 11.0, final_norm 3.125, cos 0.986**. The final-token
argmax crystallizes at **L60 (clean reaches ```` ``` ````) / L61 (live locks `Let`)** via the ```` ``` ````
logit collapsing live (15.94) vs clean (26.60) while the `Let` logit is matched (FR13_NODE5_LADDER).

**This is an AMPLIFIER, not the origin** — the deep full-attn layers (and the gate 1/rms ~32×) magnify the
L0-born GDN state-feed ULP into a margin-crossing flip. It contributes to BOTH our 5 and native's 3 (native's
deep full-attn amplifies native's smaller L0 gap to 3); it is NOT a separate +2 seam to fix. Do NOT chase a
per-layer L60/L61 patch ([[reference_diffuse_gdn_accumulation_explained]]: amplifier not source; the only
truly-irreducible floor is the ~1e-12 MMA grouping, far below the 1.1–9.75-nat flips). The FA2-fork full-attn
2-ULP floor is downstream and non-compounding — it amplifies, does not originate (FR13_FA2_CARRIER_OVERTURNED).

---

## 4. Ranked non-WY levers to close the +2 (spine 5 → native 3), each with a cheap GPU test

1. **Recurrent-vs-recurrent scan op alignment (HIGHEST — this is where the +2 lives).** Drive the fr10
   tree-scan to byte-exact vs native's `fused_sigmoid_gating_delta_rule_update` **recurrent** verify on the
   SAME spine prefix + h0 (class-10 codegen: num_warps / static_range-unroll / tl.load order / cast
   boundaries). Non-WY (this is the sequential rank-1 path, NOT chunked-WY). **Cheap GPU test:** in-process
   decoherence-free A/B (like the FA2 MAB / GDN sub-op MAB at `fr10_phase4_patch_vllm_tree_gdn.py:1276+`):
   capture deep-spine row scan_out, ours-recurrent vs native-recurrent on identical input → first-nonzero
   sub-op + magnitude. If 0.0 → the +2 is the chunked-oracle frame (see lever 4); if nonzero → that op is
   the seam, align it. ~1 boot.

2. **Tree-reshape to cut depth-accumulation (DEPLOYABLE lever, [[project_fr13_tree_reshape_unifying_lever]]).**
   The +2 is NOT depth-reducible on a PURE spine (chain3=chain5=5), but the e2e GOAL is cat9/cat3w (22), and
   reshape removes the +17 leaf co-residency AND, per the state-feed-scales-with-accept-depth model, a
   shallower committed spine + root-sibling width recovers accept with LESS deep state-feed accumulation. Does
   not by itself reach 3 on a spine, but is the only lever that touches the deployable 22→~5 AND the speed tax
   (fewer verify rows). **Cheap GPU test:** the cat3w boot already running + a cat4w/shallow-root-sibling
   shape; flip-count vs own oracle + accept vs native.

3. **Spine-only-FLASH dispatch (LEGITIMATE but ~0 yield).** Route pure-spine forwards (no branches present)
   through stock causal FA2 instead of the tree-bias fork. Clean (the bias is inert on a spine, proven §2),
   NOT a reward-hack. **But expected yield ≈ 0 flips** (spine attention already FLASH-identical within the
   15×-below-floor 2-ULP). Worth it ONLY as a confirmatory null / to remove the 2 single-ULP residuals;
   do NOT expect it to move the +2. **Cheap GPU test:** chain5 with spine-FLASH dispatch ON vs OFF, same
   oracle — predict 5→5.

4. **Re-frame the oracle as recurrent (MEASUREMENT lever, not a kernel change).** The spine arms are scored
   vs a CHUNKED-prefill oracle; the TRUE per-path reference is the RECURRENT non-MTP oracle (9.14× smaller L0
   gap). If the +2 is largely the chunked-vs-recurrent oracle frame (lever 1 returns ~0.0), then the deploy
   gate should score vs the recurrent oracle (or accept that native suffers the same chunk-vs-recurrent frame
   and the real bar is e2e-vs-E5 within-floor, not vs a chunked teacher-force). **Cheap GPU test:** re-run the
   spine flip probe vs the recurrent non-MTP oracle (`nospec/recur` path in fr13_verify_decisive) — predict
   the count drops toward native's 3.

5. **WY (PARKED — explicitly NOT this).** Recorded for completeness: WY = the batched chunked scan, was
   pursued to align to a chunked verify readout that DOESN'T EXIST ([[reference_gdn_verify_sequential_dispatch]]
   falsified it). It is the WRONG tool for the +2 (the verify path is recurrent). Not revived without the user.

---

## 5. Bottom line for the decision (do NOT conclude the wall)
- **+2 carrier = GDN recurrent state-feed (chunk-vs-recurrent realization), NOT attention.** Proven: L0 GDN
  9.14× larger vs chunked than vs recurrent oracle; FA2 tree-bias is arithmetically inert on a pure spine
  (bias == causal, `+0.0` no-op); the 2-ULP FA2 floor is 15× below the noise floor and non-compounding.
- **Candidate 2 crux answer: a branchless spine CAN legitimately use a FLASH-identical path (not a hack) —
  but it ALREADY effectively does (inert bias), so that lever yields ~0.** Branch rows genuinely need the
  tree-bias and cannot route through plain FLASH.
- **Non-WY closability: OPEN, not walled.** The in-kernel non-WY lever is **recurrent-vs-recurrent scan op
  alignment** (class-10 codegen, the family that cracked conv + the prior scan). fp32-state is exhausted
  (already done). The decisive cheap GPU test (lever 1) tells us whether there is a localizable codegen seam
  or whether the +2 is the chunked-oracle measurement frame. WY is the WRONG tool and stays parked.
- **The +2 is the GOAL-blocking gap ON TOP of the leaf co-residency (+17).** Reshape alone floors at ~5;
  reaching native 3 needs lever 1 (or the lever-4 re-frame). Surface to the user AFTER cat3w + lever-1 A/B.
