# FR13 — Suffix+MTP hybrid drafter design (CPU design, no GPU)

Workflow `wf_7b180c80-f83` (4 agents, CPU-only, source-cited). Raw:
`research/fr13_workflows/suffix_mtp_hybrid_design_wf_7b180c80.raw.json`. Red-team
**holds=True** (lossless-clean) but with a strong speed caveat. User idea (2026-06-13):
combine our lossless drafter-agnostic tree verifier with Snowflake SuffixDecoding —
narrow MTP caterpillar at root/1/2, suffix-decode guess tokens grafted on the branches.

## Snowflake SuffixDecoding (the drafter)
Model-free, NeurIPS 2025 (arXiv 2411.04975). Drafts from **suffix-tree lookup** (no draft
model, no head): a per-request tree (prompt + generated) + a global tree (prior request
outputs). MatchPattern on the last p tokens → greedy best-first speculation tree scored by
empirical path frequency D(N); adaptive length MAX_SPEC=α·p. **Apache-2.0**, already a
vLLM v1 spec method (`speculative_config={"method":"suffix"}`, lib `arctic-inference`).
Accept profile: **wins big on code/agentic/verbatim-echo** (SWE-Bench 7.8 mean accepted/step,
AgenticSQL 6.3→7.5; SWE-Bench 2.5× wall-clock) — **loses to EAGLE on novel prose**
(Spec-Bench ~1.6–1.8), which is why the paper's headline mode is a hybrid fallback.

## The design (recommended: HYBRID n=2, 3-deep suffix tail on the spine tip)
`speculative_token_tree = [(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0),(0,0,0,0,0,0)]`
`parent[] = [-1,0,0,1,1,3,3,5,7,8]` — 9 draft nodes, n_pad=16, **9 verify rows = same budget as cat9**.
- nodes 1–6 = MTP head: root + d0/d1/d2 top-2 alts (node 2 `(1,)` = the cat10 d0-rescue sibling), spine to d3.
- nodes 7–9 = a single **model-free suffix continuation chain** grafted on the spine tip (5→7→8→9).
- **Merge** at the existing injection seam (`fr10_phase4_patch_vllm_tree_gdn.py:215-241` static topology auto-derives parents/masks/row-map with zero source knowledge; per-step the MTP head fills the near slots via FIX-1 single-logits top-k read-only, arctic `SuffixDecodingCache.speculate` fills the tail slots). q=None greedy; MTP-softmax + suffix freq C(N) for temp>0.

## Lossless — clean by construction, but the LIVE build isn't lossless yet
The committer is drafter-agnostic (per-edge SpecInfer/STree rejection, `:6638-6700`, reads only
parents/drafts/logits/q). A suffix token is one more candidate: a miss is **rejected (costs a
verify row, never a wrong commit)**. Deep grafts are in-scope for the per-path branch oracle
(SpecInfer Def 4.1 / STree). **No new lossy mechanism, no copy/dense/splice/multispine** = not a
reward-hack. **BUT** the hybrid does NOT relax the standing cat9 committer defect (the ~4.6%
deep-row non-argmax flips, the 22-flip chase) — suffix tails live at exactly those deep rows, so
it **inherits** that open gap. Re-gate on the per-token argmax-vs-clean-oracle probe, never scalar
accept/event.

## Speed — NOT established (GPU arm needed), but there is NO real cat10 dilution to beat
- lm-head row tax ≈ free (one batched GEMM, ~2.9 ms/row at B=1). **The binding cost is GDN
  per-node state traffic** (+2.9 ms/node on the replay route; +42–46 ms/node legacy). The suffix
  tail SWAPS cat9's deep spine rows (d3-d5) rather than adding any, so net-new node count ≈ 0.
- **CORRECTION (user ruling f0bf9e0e — supersedes the original cat10 "dilution" read):** the
  cat10 −0.27 accept/event is an **ACCOUNTING ARTIFACT, not real dilution** — (a) trajectory
  confound (cat9 [98,128,128,128] vs cat10 [73,128,128,128] are different streams ⇒ whole-window
  accept/event is a DRAW), (b) sibling-stop denominator, (c) m1 structurally ruled out: strict_mask
  makes the root sibling attention-invisible to the spine, so it CANNOT dilute spine acceptance.
  So earlier text framing the cat10 −0.27 as a "dilution trap to beat" was based on the overturned
  verdict. The suffix tail's merit stands on its own measured numbers (below); and unlike cat10's
  added `(1,)` sibling, the suffix tail is part of the spine path (extends d3→d4→d5), not a separate
  co-resident node — so the (already-non-real) cat10 mechanism does not even apply.

## VERDICT + the cheap gate (cost-gate before building)
**Qualified yes, narrow form only, and NOT a GPU prototype yet.** Highest-risk unmeasured
assumption: that the suffix tail's conditional accept gain on the agentic-SWE stream **exceeds the
co-residency dilution cost** (escapes the cat10 −0.27 trap). Validate FIRST, **CPU-only, no GPU**:
replay banked SWE-Verified token streams against a warmed suffix tree and measure, per accept
event at the spine tip, the suffix hit-rate + expected-accepted-tokens vs the cat9 leaf accepts
those rows would displace, net of the per-node GDN tax. Only if expected suffix accepts clearly
beat the displaced leaf accepts does a single GPU arm (n=2, 3-deep tip tail, replay route, global
tree warmed from prior SWE outputs, gated on the per-token-argmax probe + paired teacher-forced
accept) earn a slot. Practicality: the shipped vLLM `suffix` proposer returns a single CHAIN (not
a tree) — grafting needs the arctic cache underneath; cold-start = empty tree on the first task.

**Ordering:** suffix-fusion is downstream of the cat9 lossless chase regardless (speed-first-
lossless-gate). The CPU hit-rate study can run now (offline, doesn't need cat9 lossless) to decide
whether it ever earns a GPU slot. Pairs with [[project_fr13_speed_first_lossless_gate]],
[[feedback_speed_is_the_goal_cost_gate]], [[project_fr13_tree_reshape_unifying_lever]],
[[reference_multispine_not_lossless_closed_nonship]], [[project_fr10_drafter_verifier_interface]].

---

## CPU HIT-RATE GATE RESULT (wf_facb0c38, 2026-06-13) — CONDITIONAL-GO

Replayed the suffix tail against REAL banked SWE-Verified streams (4540 decode positions,
68 requests: 64 deliverable astropy SWE + 4 acceptance-ladder), no-leakage verified 3 ways
(incremental==from-scratch rebuild 0 mismatches on all 64; future-corruption collapses
capped-mean to exactly 0.0; global-build skips the target). Red-team **holds=True**.

**The load-bearing assumption HOLDS — and the tail is a SWAP not an ADD (the key insight):**
- Expected suffix accepts at the spine tip A = E[min(CORRECT_LEN,3)] = **2.32** (as-deployed,
  global tree warm) / **1.90** (strict cross-task leakage bound) / 0.82 (cold per-request-only).
  Bimodal: 69.9% reach the full 3-cap, 15% hit 0; uncapped echo mean ~9, p90 22, max 51.
- Displaced cat9 deep MTP (d3-d5 spine) B = **0.721** sequential-expected (0.483 + 0.483·0.379 +
  0.483·0.379·0.30).
- Margin A−B = **+1.18 (strict) to +1.60 (leaky)**, positive even cold (+0.82).
- **The 3-deep tail SWAPS cat9's deep spine rows, it does not ADD a 4th** → net-new node count
  ~0 → the +2.9 ms/node GDN tax is not net-new. Break-even reduces to A−B>0, cleared in every
  variant. (NB: the cat10 −0.27 it was framed as "beating" is itself an accounting artifact per
  f0bf9e0e — see the Speed-section correction — so there was no real dilution to beat; the margin
  stands on its own.)

**Why CONDITIONAL not GO — data thinness (the only real caveat):** only **4 independent
astropy tasks** (16 temp-varied samples each → effective task-N≈4, not 4540); streams are short
(64-128 tok) so the per-request tree contributes ~nothing — **the entire win rides on the GLOBAL
cross-request tree** (shared agentic boilerplate). The margin DIRECTION is robust (positive even
under the strict cross-task bound), but the magnitude needs a real multi-task SWE corpus.

**Conditions for the single GPU prototype arm:** (1) GLOBAL corpus warm is a hard precondition
(cold/per-request-only is the +0.82 thin-margin worst case); (2) the arm must be a pure SWAP of
the cat9 deep spine (d3-d5) for the suffix tail, NOT additive (additive = the cat10 trap);
(3) gate on the per-token argmax-vs-clean-oracle probe (suffix tails sit at the same deep rows as
the open 22-flip defect — so this is **downstream of the cat9 lossless chase**, not before it).

---

## LEG-GROW n-SWEEP RESULT (wf_82b46622, CPU, 2026-06-13) — legs do NOT beat tip

User refinement: MTP builds the cat10-shape caterpillar (spine + (n-1) legs/depth + root
sibling), suffix GROWS THE LEGS; sweep n=1..5 under the asymmetric cost (MTP depth = head
forwards; suffix growth = free draft, verify rows only). Red-team **holds=True**, no leakage
(3 adversarial corruption probes → 0.0), reward-hack-free, modeling flaw is conservative
(biases toward legs, the losing arm).

**VERDICT: growing the LEGS does NOT beat growing the TIP.** Two EXACT (not modeled) signals:
1. **Suffix payoff is FLAT across graft depth** — E[min(CL,3)] ≈ 1.43–1.70 at d0…d5. A chain
   grafted on a shallow leg predicts ~the same #tokens as one on the deep tip → a leg has no
   intrinsic payoff edge. Its only possible advantage is catching divergences the tip misses.
2. **At matched MTP-head forwards (spine=3), TIP wins every extra-node budget 1–8** (TIP A/E
   2.33→2.94 vs LEGS 2.17→2.55), top-2 modeled (no geometric approx).
The modeled leg-rescue is only ~1.08 tokens (reach-corrected ~0.76) and ~63% of it is the
single **unreliable capped d0 catch** (modeled 0.95, raw 1.76 = the cat9-vs-native trajectory
confound, NOT a real rank-2 recovery). Under the asymmetric cost the optimizer never selects a
leg at any n (best-with-legs A/cost strictly falls 0.323→0.241 as n 2→5). MECHANISM: the spine
survives early (cat9 d0/d1 accept 0.87/0.83), so the tip is reached by the large surviving mass
and the suffix continues from the tip as well as from any leg; the leg only fires on the small,
modeled, unreliable early-divergence mass.

**The sobering part (tempers the gate's optimism):** the **EXACT tip-grow arm reaches only
A/E ≈ 2.94**, BELOW cat9 3.18 and native E5 3.16. The earlier gate's "+1.18–1.60" was a
*portion* comparison (the 3-deep tail vs the displaced d3-d5 nodes, 2.32 vs 0.721) — the FULL
config accept/event is ~2.94, because shortening the MTP spine to fund the suffix tail trades
reliable deep-MTP accepts for echo-dependent suffix ones. The ONLY native-beating number (raw
3.43, n=2 budget-fill) lives entirely inside the modeled leg-rescue and is NOT claimable on CPU.

**The decisive open number is GPU-only:** no per-depth MTP **top-k LEG drafts** exist anywhere
on CPU (`fr10_mtp_draft_trace.jsonl` holds only the width-1 MTP-5 spine draft; no top-k in any
cat9/cat10/native capture). So the leg verdict is a **MODELED "no"** — the true per-depth
leg-catch (especially d0) and the leg token identities are unknowable offline. RECOMMENDATION:
**do not fund the leg-grow GPU arm** until the true d0 leg-catch is measured; if a suffix arm is
built, build the **TIP-grow** (n=1 most cost-efficient, or spineD2-3+tip4 for max raw accept ~2.94).
But note even tip-grow doesn't clear native on exact evidence — the suffix lever is weaker than
the gate first suggested. All of this stays **downstream of the cat9 lossless chase**.
