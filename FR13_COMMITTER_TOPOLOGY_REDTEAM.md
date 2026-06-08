# FR13 committer / topology red-team — the 56% step-0 reject / 1.199 is VERIFY-FORWARD DRIFT, NOT committer or topology wiring

Monitor red-team synthesis, 2026-06-08. Run under test: `output/fr13_wy_b4_e2e_20260608T183138Z`
(WY kernel, B=4, accept/event 1.1989, 56% step-0 reject). Read-only, no GPU/docker.

## VERDICT (decisive, evidence-grounded)

**ROOT CAUSE = verify-forward divergence from native (the GDN/WY recurrent-state amplification), already
measured on this exact WY build by the live Gate-A ladder. The committer mapping is correct and the
drafter topology IS the intended caterpillar. NO committer or topology fix is the right next action.**

The static prediction in `FR13_WY_B4_REDTEAM.md` ("the WY scan is ~1 ULP / batch-invariant ⇒ the verify
is exonerated ⇒ the 1.199 is committer/topology wiring") is **REFUTED by a GPU ladder that was already
run on this same WY build** (`output/fr13_wy_live_ladder_20260608T162920Z`, `FR13_LADDER_LOG.md:289–322`):

| ladder stage (WY live, spine vs native) | max_abs |
| --- | ---: |
| input_hidden | 0.0 |
| **layer 1 linear_attention hidden** | **0.0001220703125  ← first nonzero (1 bf16 ULP)** |
| final_norm_hidden | **5.294921875** |
| final logits | **3.3203125** |

The "~1 bf16 ULP" cited as evidence the verify is lossless is **only the L1 scan OUTPUT in isolation.**
The same L1 step's scan **STATE** is already 1.66e-3 (`FR13_LADDER_LOG.md:430` table), and that recurrent
state compounds through 64 layers to **final-logit max_abs 3.32** — which flips argmaxes wholesale. A
verify whose final logits differ from native by 3.32 is **lossy at the root**; it is not within the E5
floor and cannot accept the (native-correct) drafter token. Gate A has never passed on any WY build
(`FR13_LADDER_LOG.md` GATE A entries are all FAIL; `FR13_GATEA_DEEP_DIVERGENCE.md:101–102` "GATE A is NOT
passed"). The e2e was explicitly retained "as a symptom only" (`FR13_LADDER_LOG.md:476`).

## Settling the two-camp split in the findings

The FINDINGS contain two contradictory root causes. They cannot both be the cause; the trace + ladder
decide it:

- **Camp "verify-forward drift"** (claims by the deep-divergence agent): CORRECT. The tree verify logits
  at the root are wrong; driving input→logits drift to 0 restores accept/event. Matches the ladder.
- **Camp "root-state-mismatch / drafter-reads-wrong-root-state"** (claims proposing to re-wire the
  drafter's `sample_hidden_states` ↔ accepted-leaf handoff): the **observation is right (the root is
  worst), but the proposed mechanism and fix are wrong.** The drafter is NOT reading a different/stale
  state buffer; it is reading the *correct* native `sample_hidden_states`, and the verify is what drifted.
  Re-wiring the drafter handoff would not move the number and risks breaking the native-identical drafter.

### Why drift, not a state-buffer mismatch — the decisive trace discriminator

Conditional argmax-match by spine depth, conditioned on whether the root matched
(`tree_sampler_debug.jsonl`, 1111 events):

| spine depth | match-rate GIVEN root matched | match-rate GIVEN root did NOT match |
| --- | ---: | ---: |
| d0 (root)   | 1.000 (485/485) | 0.000 (0/626) |
| d1          | **0.635** | 0.510 |
| d2          | 0.610 | 0.484 |
| d3          | 0.627 | 0.548 |
| d4          | 0.577 | 0.532 |

If the drafter and verify shared the same state AND the verify were native-exact, then **conditioning on
a matched root** would force depth-1 to ~1.0 (a greedy drafter chained on the *same* model). It is 0.635.
So even when drafter and verify agree at the root, the verify's depth-1 argmax disagrees with the
native-correct drafter 37% of the time. **The verify forward is producing non-native logits at every
spine position** — that is forward drift, not a handoff/buffer selection bug. (A wrong-root-state buffer
would instead show near-100% downstream agreement once the root is forced to match.)

The root being the *single worst* position (0.441 vs ~0.53 deeper, mean raw target-prob 0.298 vs 0.38–0.41)
is also explained by drift, not buffer mismatch: per the spine-only ladder, **row 0 (the carried/anchor
token) is the first to diverge** (`FR13_GATEA_DEEP_DIVERGENCE.md:59–60`) because its recurrent h0 carries
the most accumulated state error into this event; depths 1–4 are fresh draft rows. First-event evidence
clinches it: the *very first* spec forward of req 0 already rejects the root (`tree_path_lcp.jsonl`
events: `(req0, root_accepted=False, len=0)` is event #1), so this is a single-forward divergence, not a
multi-step accumulation — exactly the spine-only ladder's "final logits 3.32 on one forward."

## Lead (1) drafter-topology mismatch — REFUTED for this run

The stock-`propose_tree` `child_drafts=[1,2,1,1,1]` two-parallel-chains bug is patched out and NOT active.

- Runtime parents from the live gather are `[-1,0,0,1,1,3,3,5,5]` = a clean caterpillar (spine
  0→1→3→5→7, leaves 2,4,6,8 as second children). Source: `tree_sampler_debug.jsonl` `tree_logit_gather`
  `parent_node_id` per node.
- The FR10 caterpillar native-spine drafter packs slots as
  `[spine0, spine1, leaf0, spine2, leaf1, spine3, leaf2, spine4, leaf3]`
  (`scripts/fr10_phase4_patch_vllm_tree_gdn.py:5463–5476`), which is exactly that parent tree.
- The packed spine is **byte-identical to native MTP-5**: tree idx-0 draft
  `[71093,12305,198,198,271,727,1005,9637,26548]` unpacks to spine `[71093,12305,198,727,9637]` =
  native idx-0 `[71093,12305,198,727,9637]` (`fr10_mtp_draft_trace.jsonl` tree vs native). Drafter is
  native-correct; do NOT touch the caterpillar packing.

The prompt's `parent=[-1,0,1,1,2,2,4,4,6,6]` is `target_logits_indices` (the parent-position row index),
not topological parents; both reconstruct from the same caterpillar. No mismatch.

## Lead (2) committer (canonical_multidraft) wiring — RULED OUT

The committer faithfully applies the rejection rule to the verify logits it is handed; it rejects the
root because the **verify** assigns p(71093)=0 at position 0, not because of any mapping error.

- Correct stochastic path fired: `policy='canonical_multidraft'`, `all_greedy=false` on every event
  (`tree_path_lcp.jsonl`, `tree_sampler_debug.jsonl`). The prior `FR13_COMMITTER_FINDING.md` greedy-dispatch
  hypothesis is REFUTED (native MTP-5 is also `all_greedy=false` and hits 3.076).
- Draft→verify-row mapping is correct: committer indexes `_target_row = start + children[0]`
  (`fr10_phase4_patch_vllm_tree_gdn.py:3743`) on the per-node-gathered `target_probs`; committer node-0
  argmax 248068 == gather node-0 argmax 248068, committer node-3 argmax 271 == gather node-3 argmax 271.
- Single-child root accept rule is mathematically native-identical: `accept_prob = min(1, p[draft]/1) =
  p[draft]`, equal to stock `rejection_sampler` with `NO_DRAFT_PROBS=True`. With the drifted verify giving
  `p[71093]=0`, accept_prob=0 → correct reject given that distribution.
- Deeper committer steps accept at native-level rates (step1 76.8%, step2 75.9%, step3 84.8%, step4 80.9%;
  among root-accept events mean accepted_len 2.77 ≈ native 3.076), proving the rejection-sampler step rule
  is fine. The defect is upstream of the committer: the verify logits.

## Why bag-TV proves "lossy," not merely "slow"

`tree_wy_seed1313_vs_e5_seed1313_compare.json`: `first_token_tv=0.0` (prefill→first-decode handoff
byte-exact), but `emitted_token_bag_tv=0.487` (3.3× the E5 self-floor 0.149) and
`exact_token_sequence_match_rate=0.0`, first diff at position 9. A lossless-but-low-accept run would emit
the SAME tokens (bag-TV ≈ floor); 0.487 ≫ 0.149 means the tree run emits DIFFERENT tokens — the verify is
lossy. The drift enters inside the spec loop (position 9), not at the prompt boundary.

## THE EXACT FIX (file:line + change) — it is the open Gate-A grind, not a committer/topology edit

There is no committer or topology change to make. The single open bug is verify-forward drift, localized
by the prior agents to the **WY GDN tree scan recurrent STATE** carried across the recurrent layer stack:

- First nonzero on the live WY ladder = **L1 `linear_attention`**, scan OUTPUT 1.22e-4, scan STATE
  1.66e-3 (`FR13_LADDER_LOG.md:289–322, 430`). The kernel under fix is
  `scripts/fr10_phase4_patch_vllm_tree_gdn.py` `_tree_gdn_wy_kernel` (the WY tree scan;
  `fr10_gdn_tree_kernel.py` `_tree_gdn_kernel`/ancestor-replay).
- The OUTPUT is already at the ~1-bf16-ULP floor; the **STATE handoff (1.66e-3) is ~13× larger and is what
  compounds** to final-logit 3.32. The remaining bit-exact seam is the WY scan **state write** vs native
  FLA's materialized recurrent state, not the per-node output readout. Per `FR13_WY_CASCADE_MAP.md` /
  `FR13_WY_RESIDUAL_CLOSURE.md`, the open lever is the scan state-update reduction/rounding boundary
  (taps #4/#5 `tv_i`/`beta*k*exp(cum_g)` and the state outer-product accumulation order), measured against
  the live `fused_sigmoid_gating_delta_rule_update` STATE surface — drive **scan-state**, not just
  scan-output, to the bf16 floor.

This is the correct continuation of the standing "grind all GDN fronts to lossless" policy
(`FR13_GATEA_DEEP_DIVERGENCE.md:207–217`). Conv is bit-exact (ex2 replica done); FA2 prefill made native
(`FR13_FA2_PREFILL_NATIVE=1`); the live remaining front is the WY **scan-state** amplification.

## SINGLE NEXT ACTION for codex — AFTER the B=4 Gate-A ladder

The B=4 Gate-A ladder is running now (`output/fr13_wy_b4_gateA_20260608T191457Z`, captures present, ladder
json not yet written; its eager-B4 quick already shows accept/event 0.79 — same root-reject signature).

1. **Read the B=4 Gate-A `gateA_spine_ladder.json` first-nonzero layer + final-logit max_abs.** Do NOT
   touch the committer or drafter. Expected (per the B=1 WY ladder): first nonzero ≈ L1 GDN scan, final
   logits ≫ E5 floor. Confirm spine AND branch rows (branch oracle = native-on-branch-path per
   `reference_gdn_tree_branch_oracle_losslessness`).
2. **Grind the WY scan STATE write to the bf16 floor vs native FLA's materialized state surface**
   (`_tree_gdn_wy_kernel` state-update path; verify with the boot-free L1 + a deep-layer clean-input
   payload, then ONE live full-ladder). Target: final-logit drift within the E5 floor across all 64 layers
   so the recurrent amplification dies. This is the same op-by-op bit-exact discipline used for conv.
3. **Only after Gate-A final-logit drift is within floor, re-run the B=4 CUDA-graph e2e.** The expected
   outcome of fixing the verify is that root argmax-match rises from 44% toward native's ~84% and
   accept/event rises from 1.199 toward native 3.076 — with NO committer/topology edit.

If, contrary to the B=1 ladder, the B=4 Gate-A ladder comes back with final-logit drift **already within
the E5 floor** (verify lossless at B=4) yet the e2e still shows 56% step-0 reject, THEN — and only then —
re-open the committer/root-handoff hypothesis with a fresh trace (re-check `_target_row` gather vs
`target_logits_indices`, and the `sample_hidden_states`→accepted-leaf handoff at
`fr10_phase4_patch_vllm_tree_gdn.py:704–712, 5143–5200`). On all current evidence this branch is not
expected to fire.
