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

## Speed — NOT established; the cat10 dilution trap is the load-bearing risk
- lm-head row tax ≈ free (one batched GEMM, ~2.9 ms/row at B=1). **The binding cost is GDN
  per-node state traffic** (+2.9 ms/node on the replay route; +42–46 ms/node legacy). A 3-deep
  tail = ~3 nodes' tax on **every** forward, paid even on suffix misses.
- **cat10 precedent (direct refutation to beat):** +1 co-resident row dropped accept/event
  3.198→2.932 (−0.27) and regressed TPS, even though its targeted d0 rescue was real. Suffix
  tails add MORE rows, DEEPER, with a workload-conditional payoff, and the root-concentrated d0
  deficit (0.585 conditional) means the spine often dies before reaching the tip where the tail
  sits. On prose the tail is pure row-waste.

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
