# FR13 suffix⊕MTP frontier research — SOTA catalog + a REFUTED "beats-native" claim

Workflow `wf_daf5b395-3df` (5 agents, CPU, online+git+synth). Raw:
`research/fr13_workflows/suffix_mtp_frontier_wf_daf5b395.raw.json`. Red-team **holds=FALSE**
(it refuted the synthesis's headline). Bank the SURVEYED facts + the refutation; do NOT carry
the "Design A beats native" claim. 2026-06-14.

## Frontier SOTA for combining model-free + model-based drafters in ONE tree (factual, citable)
- **Graft** (arXiv 2605.20104, "Draft Less, Retrieve More") — the closest SOTA to MTP⊕suffix:
  **prune-then-graft** — confidence-prune low-conf EAGLE branches at calibrated depth, then
  GRAFT retrieved (model-free) candidates into the freed topological gaps → ONE hybrid tree at
  fixed budget K_max, single target pass.
- **Sequoia** (arXiv 2402.12374) — **DP-optimal tree shape under a hardware cost model**. Node
  score f(v)=∏ accept-probs on its path; F(T)=Σf(v); DP picks size+depth to MAXIMIZE
  G(n,d)/(t(n)+d·c) from the device's own latency curve t(n). On steep-t(n) hardware (OUR
  regime: each row = an lm-head GEMV + GDN traffic) the optimum is SMALL — add a node only while
  its marginal P[b]·T_max[m] > dt/dn. This is the principled allocator for our per-row tax.
- **Hybrid Verified Decoding** (arXiv 2606.01019) — trains a light MLP to PREDICT a cache/suffix
  draft's accepted-length from pre-verify features; verify the cache draft only if predicted ≥ τ
  else fall back to EAGLE-3. Finds the 4.8-8.9% high-payoff states at 78-88% precision; 2.73× avg
  on agentic. (The learned version of SuffixDecoding's fixed-τ gate.)
- **SuffixDecoding τ-gate** (2411.04975) — per-step PICK-ONE: if suffix tree score > τ use suffix,
  else fall back to the model drafter. Adaptive MAX_SPEC=α·p (speculate more when accept-likely).
- **Not-a-Bandit** (2510.20064) — online no-regret (full-information) per-step drafter SELECTION.
- Model-free family that all plug into our drafter-agnostic verifier: **REST** (retrieval+trie),
  **Token Recycling** (vocab adjacency matrix), **Lookahead** (Jacobi n-gram pool), **Ouroboros**,
  **PLD**. All are lossless candidates for our per-path rejection verifier.
- **Our own origin** (FR13_DIRECTION_AND_NUMBERS.md:5-31): FR13 IS the suffix-fusion verifier
  prototype — MTP-3 short anchor + suffix composing branches/tail; superset target = the 9-node
  caterpillar beats the 5-node native-E5 spine (~3.076); lossless = B=1 greedy byte-exact spine
  AND branch (branch vs native-on-path oracle). The concept is already banked.

## The synthesis claimed Design A beats native — REFUTED (holds=FALSE), and why
Synthesis "best bet" = Design A = **ADD-NOT-SWAP**: keep the full 9-node cat9 + graft a free
suffix chain (T=3-5) on the d4 spine tip; projected A/E 3.34-3.47 > native. The red-team killed it:
1. **On EXACT evidence it does NOT clear native.** Its measured analog (the leg-sweep's tip-grow,
   keep-spine + add-tip) saturates at **A/E 2.94 < native 3.16** — same wall as before. The
   ">native" projection needs THREE optimistic assumptions at once: the f0bf9e0e no-dilution read
   + the *leaky* (not strict) suffix number + an *unconfirmed* 3.18 cat9 baseline.
2. **The "ADD = zero dilution" premise is contradicted by the live cat10 ADD experiment** (adding
   one node dropped accept/event 3.198→2.932). Whether that −0.27 is artifact (f0bf9e0e) or partly
   real is genuinely UNSETTLED and is the entire basis for "suffix is free" — a CPU model can't
   decide co-residency dilution. If any fraction is real, Design A's +0.16-0.19 is wiped → sub-native.
3. **Fabricated accounting**: "cat9 3.18 = spine 2.359 + branch bonus 0.821" is in no committed doc
   (2.359 = spine_accept_expected(5); 0.821 = a backed-out residual). Baseline confusion (3.076 vs
   3.16 used interchangeably); cat9's only >native number (3.1789) is a single-draw DRAW.
4. **Lossless gate currently FAILS** (gold gate 2026-06-13, B=1 not graduated) and every design
   inherits that defect on the exact deep rows suffix tails occupy.
5. **Mostly re-derived**, not novel: A = tip-grow generalized; B = the sub-native swap arm; D
   (Sequoia) fed the same payoff table reproduces A's 2.94 saturation.

## The ONE live idea that survived: Design C = ADAPTIVE confidence-gated tail length
Novel vs our history AND corroborated by two real papers (SuffixDecoding adaptive MAX_SPEC=α·p +
Hybrid-Verified-Decoding's learned accept-length predictor). It attacks Design A's actual
weakness: the suffix CORRECT_LEN is **bimodal** (69.9% hit the 3-cap, frac≥5 uncapped 0.582,
p90=22, max 51) — a FIXED short tail leaves the long-echo mass on the table while still paying rows
on the dead 15%. Spend deep rows ONLY on high-confidence echo states (gate tail length on the
suffix match-score). Doesn't shorten the MTP spine (sidesteps swap-loss + the shared-machinery
scope constraint); only changes which candidates are in the descriptor (lossless-clean).

## Cheapest next step (CPU, no GPU) + sequencing
Extend the existing `leg_suffix_replay.py` to test an ADAPTIVE-length policy (T = f(match_len,
confidence threshold), n_pad-capped) vs fixed-T=3 (Design A) vs cat9 baseline, under BOTH the
strict cross-task bound AND cold-start — does variance-capture beat flat +0.19 net of the per-row
tax? Pure code extension, 3-no-leakage-test-passing harness, zero GPU. Either kills C cheaply or
earns it a slot. **ALL suffix work stays downstream of the cat9 lossless chase** (the L56 fix);
no GPU slot now — and a suffix GPU arm is blocked behind the open greedy-committer-defect fix,
since suffix tails land on exactly the defective deep rows. Do NOT fund any leg-grow arm
(modeled "no"). Pairs with [[project_fr13_suffix_fusion_prototype]],
[[feedback_speed_is_the_goal_cost_gate]], [[reference_scalar_metric_per_token_blindspot]],
[[project_fr13_speed_first_lossless_gate]].
