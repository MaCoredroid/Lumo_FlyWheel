# FR13 Acceptance Ladder — CPU Localization Bind (2026-06-10)

**Substrate**: post cc008587, fix flags `FR13_TREE_PER_REQ_GEN`/`FR13_TREE_REQKEY`/`FR13_TREE_REMAP_SEQ` = 1 (defaults ON).
Boot T = TREE_ATTN forked-FA2 + tree GDN, BI=1, CUDA graphs ON; Boot N = FLASH_ATTN naive MTP-5, BI=0, graphs ON.
Run dir: `output/fr13_acceptance_ladder` (W1 greedy is the primary window; N5 spine oracle = `branch_oracle_spine.json`).
B=1 strictly sequential, 4 pinned SWE-4 prompts, max_tokens=128, both arms bit-identical run-to-run (r0 substrate checks).
**Caveat carried**: BI asymmetry across arms (tree BI=1, native BI=0) — flips on near-ties cannot be fully separated from backend
divergence without a BI-equalized third boot (plan problem #5, still open).

Topology (committer/BFS node ids): n0=d0; (n1 spine, n2 alt)=d1; (n3,n4)=d2; (n5,n6)=d3; (n7,n8)=d4.
Spine = [0,1,3,5,7]; alts (2,4,6,8) are leaves. Paths: [0,2] [0,1,4] [0,1,3,6] [0,1,3,5,7] [0,1,3,5,8].

---

## R1 — POST-FIX ROOT-REJECT: REQKEY did NOT kill it

| basis | greedy | t06 |
|---|---|---|
| committer rows (`accepted_len==0`) | **56/163 = 34.4%** | 71/173 = 41.0% |
| scheduler per-pos metrics (1 − pos0 marginal) | 33.3% (159 events) | 41.5% (171) |

vs pre-fix 36.6%. The stale-slot attribution is dead — root-reject survives the slot/REQKEY/remap fixes essentially unchanged.
Its anatomy is now bound below (R3): drafter-flip dominated, not stale state slots.

## R2 — PER-DEPTH ACCEPT TABLE (same prompts, both arms)

Marginal = P(accepted_len > d) per scheduler event ( /metrics per-pos delta ÷ drafts). Conditional = marginal_d / marginal_{d-1}.

GREEDY (tree 159 events / native 123):

| depth | tree marg | native marg | tree cond | native cond | cond deficit |
|---|---|---|---|---|---|
| 0 | 0.667 | 0.894 | 0.667 | 0.894 | **−0.227** |
| 1 | 0.547 | 0.756 | 0.821 | 0.845 | −0.024 |
| 2 | 0.384 | 0.650 | 0.701 | 0.860 | **−0.159** |
| 3 | 0.258 | 0.463 | 0.672 | 0.713 | −0.041 |
| 4 | 0.226 | 0.390 | 0.878 | 0.842 | +0.036 |
| sum | **2.082** | **3.154** | | | |

T06: tree d0 cond 0.585 vs native 0.886 (−0.301); d1+ deficits ≤0.11. Same shape: **the deficit is root-concentrated**
(d0 alone ≈ 0.23–0.30 of conditional; the d0 marginal gap then multiplies through every deeper depth).

N5 oracle (native-on-path re-run of 157 aligned greedy tree events; per-check, tree-stream prefixes):

| depth | tree accepts | native-on-path accepts | both | neither | tree-only | native-only | argmax match |
|---|---|---|---|---|---|---|---|
| 0 | 0.662 | 0.694 | 99 | 43 | 5 | 10 | 0.892 |
| 1 | 0.650 | 0.573 | 87 | 52 | 15 | 3 | 0.828 |
| 2 | 0.688 | 0.675 | 103 | 46 | 5 | 3 | 0.904 |
| 3 | 0.764 | 0.739 | 111 | 32 | 9 | 5 | 0.898 |
| 4 | 0.694 | 0.701 | 107 | 45 | 2 | 3 | 0.917 |

KEY READING: on the tree's own stream, tree-accept ≈ native-on-path-accept at every depth (within a few points). The verify is
NOT systematically rejecting good drafts — **the drafts themselves are worse on the tree stream** (native itself only accepts
them 69% at d0, vs 89% for its own drafter's proposals). The deficit is upstream of the verify.

**Oracle noise caveat (measured)**: the oracle's native-on-path runs are prefill-shaped; at event-1 d3 it says argmax 3655 while
the LIVE native decode forward (call1 logits) says 2781 — identical to the tree. So oracle argmax-mismatch rates (~8–17%) are an
UPPER bound containing prefill-vs-decode near-tie noise; live-vs-live evidence below is the binding signal.

## R3 — REJECT ANATOMY AT THE MODAL DEPTH (d0) + seam classification

Committer math: **CLEARED**. Re-derived acceptance from `draft==parent_target` runs over all 5 paths: **163/163 events exactly
match logged `accepted_len`** (greedy argmax rule, longest path wins). The 35 events where accepted_len exceeds the spine run are
genuine branch acceptances via alt nodes (+35 tokens = the realized tree upside).

d0 root-rejects among 157 oracle-aligned greedy events = 53. Decomposition:
- **~10 (19%) verify-divergence flips**: native-on-path ACCEPTS the same draft; tree target argmax differs (kernel/wiring class).
- **~43 rejected by both targets** — but native's own arm root-accept is 0.894, so only ≈17 of 157 events should honestly
  root-reject. The ≈26 excess = **tree drafter proposed a different token than the native drafter chain** (rejected by both
  targets). Cross-check: lockstep drafter d0 flip rate 4/20 = 20% → predicts ≈31. Consistent.
- Net split of the 53: ≈17 expected-honest / ≈10 verify-flip / ≈26 drafter-flip. **Drafter-flip is the dominant class.**

### Seam S1 — COMMITTER BONUS-ROW BUG (wiring, exact, fix specified) — lossless violation
Audit of the bonus rule over all 163 rows (expected bonus = `self_target[last accepted node]`, or `pt[n0]` at acc=0):
**149 OK, 14 violations — ALL of them winner_path [0,2] (d1-alt leaf accept, acc=2), all `bonus_source='path0_native_bonus'`,
and in every case the served token == `native_bonus_token` == argmax of node 8's row (st[8], the LAST verify row)** instead of
st[2] (the accepted leaf's row). 15 `path0_native_bonus` events total; 1 coincidentally correct. The 20 deeper branch-accepts
([0,1,4]/[0,1,3,6]/[0,1,3,5,8]) all used `tree_self_target` and are correct.

ROOT CAUSE (read, `scripts/fr10_phase4_patch_vllm_tree_gdn.py:3505-3522`): leaves are enumerated in node order (L3468), so
`leaves=[2,4,6,7,8]` and **path_idx 0 = [0,2] (the d1 ALT), not the spine** (the spine [0,1,3,5,7] is path_idx 3 — confirmed by
oracle `path_index: 3`). The `best_path_idx == 0` special case (L3514) — intended "fully-accepted native chain → reuse vLLM's
native bonus" — therefore fires exactly on [0,2]-winners and commits `bonus_token_ids[req]`, which vLLM samples from the LAST
row of the forward = node 8's row. Wrong row, wrong token, served. NOTE: `path0_lcp` (L3505) and `superset_violation` (L3539)
reference the same wrong path 0 = [0,2], so the superset diagnostic is currently checked against the d1-alt path, not the spine.

IMPACT: 14 wrong tokens served / 163 events (8.6% of events; ~2.9% of tokens). Prompt 3's stream fork at pos 14 is exactly this
bug (expected st[2]=3418 == the token native committed — no fork with the right row). Measured cascade on acceptance: NONE
(events 1–2 after a wrong bonus: mean acc 2.00 / root-reject 28.6% vs 2.10 / 34.4% elsewhere). **Pure lossless violation.**

### Seam S2 — EPISODIC VERIFY-FORWARD CORRUPTION (live, gross) — lossless violation
fp32 final-logit captures (16 calls, prompt 0, both arms) at lockstep events with identical prefixes AND identical row inputs:
- Events 0–6 (24 comparable row-pairs): max|Δ| 0.6–5.8, mean|Δ| 0.09–0.73, **argmax identical in all 24 rows** (BI=1-vs-BI=0 +
  backend numerics visible but benign at these events).
- Event 7 (tree call7 vs native call6, ctx token 1970 @pos16): **whole-forward corruption** — mean|Δ| 1.95–4.49 (10–25× the
  baseline), max|Δ| 15–16. Root row argmax: tree 369 vs native 3051 with native margin 7.25 — NOT a near-tie; the next row
  flips too (369 vs 5759). Tree commits 369 → prompt-0 fork. Identical drafts that native accepts (3051...) are rejected at root.
- Trigger NOT bound: the one logit-observed instance follows the first acc=2 spine commit, but the oracle-wide correlation of
  argmax-mismatch with previous-event accepted_len is weak (prev_acc=2: 16.9% vs prev_acc=0: 9.0% of checks); no mod-16/mod-64
  absolute-position alignment across the four forks; no draft-token echo fingerprint. Corruption appears transient (later events
  recover argmax agreement), pointing at a per-event read path (state/window/row gather) rather than persistent state poisoning.
  This is the live multi-event residual anticipated after the conv prior-window fix (3a9039cc verified single-event only).

### Seam S3 — DRAFTER SPINE NOT BYTE-IDENTICAL (accept-deficit dominator; q-side, not a p-lossless violation)
Lockstep (identical committed prefixes, both arms' drafter anchored at the same position; 20 common-start events):
- Spine-vs-native-chain token flips per depth: **d0 4/20, d1 2/20, d2 2/20, d3 6/20, d4 7/20**.
- 3 of 4 prompts flip at the VERY FIRST event after prefill (d3/d4) — the divergence is inside the multi-depth tree-draft
  rollout, not (only) post-commit state rebuild.
- 5 of 7 first-flips at d≥1 are exact top1/top2 SWAPS: the alt slot holds native's chain token. The caterpillar then dead-ends
  (alts are leaves), capping the event at the swap depth — matching the pre-fix "90.8% of rejects had branch co-located dead".
- Direct deficit on identical prefixes: tree 1.55 vs native 2.05 accepted/event over those 20 events.
- Candidate mechanisms (UNBOUND, needs GPU discriminator): alt co-residency contaminating drafter GDN/conv state across the
  2-row depth batches; BI=1-vs-BI=0 near-tie flips; drafter state rebuild after commit (d0 flips). The "drafter byte-identical,
  gate-verified" premise is REFUTED at token level — the topology check verified structure, not tokens.

## R4 — FIRST DIVERGENT EVENTS

Greedy outputs are NOT identical across arms (lossless fails at B=1 greedy on this boot pair). Fork per prompt (token_ids index):

| prompt | fork pos | tree event | class |
|---|---|---|---|
| 0 | 16 | ev#7 (acc=0, bonus) | **S2** gross verify corruption (369 vs 3051, native margin 7.25, fp32-bound) |
| 1 | 24 | ev#9 (acc=0, bonus) | **S2** verify argmax flip (tree 44675 vs native 198; drafter symmetric-swapped at same event) |
| 2 | 21 | ev#6 (d3 row, accepted) | **S2** verify argmax flip (tree 1970 == its d3 draft, accepted; native says 3425 = the tree's own d3-ALT) |
| 3 | 14 | ev#5 (acc=2 via [0,2], bonus) | **S1** committer bonus bug (served st[8]=1970; correct row st[2]=3418 == native's token) |

Earliest latent (uncommitted) divergence: drafter spine flips at event 0 d3/d4 (3 of 4 prompts) — strictly before any verify or
commit divergence. First verify divergence live-vs-live: prompt-0 event 7 (all 24 earlier compared rows argmax-identical; the
oracle's event-1 "mismatch" is demonstrated prefill-noise, see R2).

## R5 — FIX (described, NOT implemented; bind-first per task scope)

**S1 (small wiring fix, `scripts/fr10_phase4_patch_vllm_tree_gdn.py`)**:
1. Compute the true spine path index by following first-children from the root (`children[-1][0] → children[n][0] …`) instead of
   assuming `path_idx 0`; or simplest: DELETE the `path0_native_bonus` special case (L3513-3516) and always use
   `tree_self_target` (`self_targets[best_path[best_lcp-1]]`) for fully-accepted paths — at greedy this row IS the verify
   argmax bonus for the accepted path, for every path including the spine. vLLM's precomputed `bonus_token_ids` comes from the
   last row (node 8) and is the right row for NO path in this topology except [0,1,3,5,8].
2. Fix `path0_lcp`/`superset_violation` (L3505, L3539) to use the spine's path index (currently they reference [0,2]).
3. Audit the SAMPLED (t06) committer path for the same wrong-row bonus reuse before any temp>0 comparison.

**S2 next (GPU)**: re-run greedy B=1 with `FR13_FINAL_LOGIT_CAPTURE` budget concentrated on events following acc≥1 commits of
varying lengths + capture the verify-side state-advance inputs (conv window rows, GDN checkpoint indices) at the corrupt event;
upstream-first read of the state-advance/row-gather consumers for the acc=2 commit shape. The corruption is episodic and gross
(10–25× logit shift) — it will localize in one or two captured events.

**S3 after S1/S2 (accept/event will not reach 3.15 without it)**: discriminators — (a) BI-equalized boots (tree BI=0 or native
BI=1) to price the BI-asymmetry share; (b) drafter A/B: spine-only proposal (alts masked out of the drafter rollout, keeping the
9-node verify) — if spine tokens snap to native chain, the alt co-residency in the drafter is the seam; (c) capture drafter
q-margins at flip depths to separate near-tie instability from gross state divergence.

Priority per lossless policy: S1 (exact, tiny) → S2 (lossless gap) → S3 (accept/speed; q-side does not violate p-lossless but
the superset accept≥native bar is impossible while the spine ≠ native chain).

---
Method: all numbers reproducible CPU-only from the run dir (event walks assert emitted_tokens == served token_ids; native
draft/event pairing validated by greedy accepted-prefix == draft-prefix on all 123 events; logit pairs compared only at
identical-prefix identical-input rows). Flag state stated above; all pre-fix binds (36.6% etc.) describe the pre-cc008587 path.
