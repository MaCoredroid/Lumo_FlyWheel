# FR13 — Is the leaf win/lose comparison APPLE-TO-APPLE? (committer parent_target row-wiring + W/C/R classification of the 23 forks)

Date 2026-06-15 (CPU-only, READ-ONLY). Banked inputs (timestamps Jun 15 04:25–04:36, this session):
`output/fr13_fork_margin_probe/logs/{fr13_fork_margin_dump.jsonl, rescore_cat9_K1_forkmargin.json,
probe_us_k1_forkmargin.json, fork_margin_classify.json}`. vLLM source read fresh from the pinned image via
`scripts/vllm_src.sh` (sha 3dbe092e = 0.19.2rc1.dev134). User directive: do NOT propose a spine-bonus / margin-damp
(rejected); investigate the COMPARISON itself. WY parked (not proposed). No copy/dense/reward-hack.

Quoting **FR13_BUG_CLASS_PLAYBOOK**: **#10** shared-source≠shared-identity / row-identity (byte/int-view, never atol);
**#11** batch-composition / near-ties flip on sub-ULP shifts; **#12** measurement traps (per-pos counters indexing the
wrong length; "branches added 0" artifact). This investigation finds a **class-#12 measurement trap inside the
classify reducer**, not a committer bug.

---

## HEADLINE VERDICT

**The COMMITTER's win/lose comparison is apple-to-apple and its row-wiring is correct (no committer bug, no bonus
needed).** But the **fork-margin classify REDUCER is NOT apple-to-apple for 7 of 23 forks** (a class-#12 position/row
join misalignment in `scripts/fr13_fork_margin_classify.py::_deciding_margin`): it labels a fork "confident A" using
the verify top-2 margin of a tree node that was **served at a DIFFERENT position than the oracle flip**, and for
fully-accepted (bonus) flips it reads the **parent-edge `target_logits` row** while the flipping token came from the
**`self_logits` (tree-self bonus) row whose margin was NEVER measured**. This inflated the "13 confident A" tail —
including the 7.125 / 8.5 / 9.125-nat headline cases. The **genuine** confident leaf wins are **8, not 13**.

W/C/R over 23 (and over the original 13 confident):
- **W (analysis row/position misalignment → margin not apple-to-apple → fix the INSTRUMENT, lossless, NO bonus): 7**
  forks (idx 0,4,9,10,13,15,22). Of the 13 confident: **5** (idx 0,4,10,13,22) — including the whole 7–9-nat tail.
- **C (co-residency-perturbed near-tie, right row, apple-to-apple, verify margin <1 nat): 8** forks (idx 2,3,8,11,16,17,18,20). All were already B.
- **R (genuine verify-vs-decode realization gap, right row, apple-to-apple, verify confidently ≥1 nat prefers a token decode rejects): 8** forks (idx 1,5,6,7,12,14,19,21). These are the *real* confident A.

So the original `n_A_fundamental=13` overstates the genuine confident-win count: **8 genuine (R) + 5 instrument-misaligned (W)**. The accept-vs-lossless tension is REAL but **smaller than banked** (8, not 11–13).

---

## 1. COMMITTER parent_target ROW-WIRING (CODE-READ, cited; MEASURED clean)

Two distinct verify-logit row-sets, both gathered from the same `logits` at DIFFERENT indices, built in the patch
(NOT native — `tree_self_logits_indices` / `tree_parent_indices` are lumo-added; native only has
`target_logits_indices`, confirmed via `scripts/vllm_src.sh v1/sample/rejection_sampler.py` L94/120/182 and
`v1/spec_decode/metadata.py` L20):

`scripts/fr10_phase4_patch_vllm_tree_gdn.py` L9162–9168 (per tree node `_node_idx`, parent `_parent`):
```
_parent_local = 0 if _parent < 0 else int(_parent) + 1
_target.append(_sampled_start + _parent_local)      # target_logits row = the PARENT's output position
_self.append(_sampled_start + _node_idx + 1)         # self_logits row   = the NODE's OWN output position
_draft.append(_sampled_start + _node_idx + 1)
```
- `parent_targets[node]` = argmax(`target_logits[node]`) = `target_logits.argmax` at L8620 → the verify model's
  prediction for *this node's token given the path up to its PARENT* (the edge dist). This is what the accept test
  compares the draft against (L6909 `drafts[node] != parent_targets[node]`). **It is the same conditional the decode
  oracle evaluates at that served position → apple-to-apple by construction.**
- `self_targets[node]` = argmax(`tree_self_logits[node]`) at L8622 → the prediction for the token AFTER node. Used
  ONLY for the full-accept bonus (L6973 `self_targets[best_path[best_lcp-1]]`, bonus_source `tree_self_target`).

Bonus selection (L6956–6976): reject → `parent_targets[best_path[best_lcp]]` (rejected node's parent-edge row);
full-accept → `self_targets[leaf]` (leaf's self row); zero-accept → `parent_targets[root]`. The legacy
`path0_native_bonus` last-row bug is gated OFF (FR13_TREE_BONUS_SELF=1 default; L6961–6970).

**Each node is matched against the row that predicts ITS position along ITS path, consistently for leaf AND spine**
(the row map is per-node, not per-leaf; same indices for spine and leaf). No off-by-one, no wrong-parent, no leaf
reading a spine row. **MEASURED clean** by FR13_COMMIT_ARGMAX_BIND (944 served records, **0 clear-margin channel-1
violations**; the 10 `ch1_match=false` are all exact 0.0-nat ties = gate self-noise): at every served position
`committed_token_id == argmax(verify_logits[the exact flat row the committer indexed])`. I re-confirmed served==
verify_argmax in **16/16** of the apple-to-apple (SAME_POS) forks below. **Committer-W = 0.** (Cross-check
FR13_COMMIT_ARGMAX_GATE channel-1 / playbook class 5.)

---

## 2. THE APPLE-TO-APPLE TEST (MEASURED, no GPU) — join dump deciding-node ↔ decode-oracle argmax at the SAME served pos

For each fork I reconstructed the committed_row→served-position map (same head-skip/offset alignment the classify
uses), located the served position of the classify's DECIDING node, and compared it to the oracle FLIP position.

**Decisive split — is the verify margin measured at the position that flipped?**

| | forks | meaning |
|---|---|---|
| **SAME_POS** (deciding node IS at the flip pos) | 16: 1,2,3,5,6,7,8,11,12,14,16,17,18,19,20,21 | apple-to-apple: served token == verify parent-edge argmax at the flip pos; verify-vs-decode is the real comparison |
| **DIFFERENT_POS** (deciding node at a NON-flip pos) | 7: 0,4,9,10,13,15,22 | NOT apple-to-apple: margin from a neighbor row; flip token came from a different row |

MEASURED detail for the DIFFERENT_POS forks (the analysis bug, `oracle_dev@deciding-node-pos ≈ 0`, flip elsewhere):

| idx | cls | flip_pos | flip-token row kind | reported margin (wrong row) | oracle_dev @ deciding-node pos | oracle_dev @ flip pos |
|---|---|---|---|---|---|---|
| 0 | A | 35 | **BONUS tree_self_target** (leaf self row, margin never measured) | 9.125 (leaf parent-edge, pos 34) | **0.00** | 6.94 |
| 13 | A | 36 | accepted_draft node0 | 8.50 (split node, pos 40) | **0.00** | 4.38 |
| 10 | A | 25 | **BONUS tree_self_target** | 7.125 (split node, pos 24) | **0.00** | 1.75 |
| 22 | A | 122 | accepted_draft node1 | 3.625 (split node, pos 123) | 0.00 | 1.00 |
| 4 | A | 30 | accepted_draft node1 | 1.50 (leaf parent-edge, pos 33) | 0.00 | 1.25 |
| 15 | B | 93 | accepted_draft node0 | 0.25 (node2, pos 94) | 6.75 | 9.12 |
| 9 | B | 97 | accepted_draft node0 | 0.875 (leaf parent-edge, pos 101) | 0.00 | 1.25 |

The entire 7–9-nat "confident" tail (0, 10, 13) is DIFFERENT_POS: the big margin is the verify model being (correctly)
confident at a **non-flipping neighbor** (dev 0.00 there), while the actual flip a few positions away has a small or
unmeasured margin. **This is exactly the user's predicted signature** — "a 7–9 nat confident match that disagrees with
decode by 7–9 nat is the signature of matching against a confidently-wrong (mis-indexed) row" — except the mis-indexing
is in the **classify reducer**, not the committer.

ROOT CODE for the instrument bug: `scripts/fr13_fork_margin_classify.py::_deciding_margin` (L138–157) returns the
**winner lcp-divergence node** margin (or split-node fallback), but `main()` (L213–252) joins that margin to the
**oracle FLIP position** `pos` (L215) regardless of whether the deciding node is served at `pos`. A committed_row spans
several served positions, so the deciding node and the flip can be different positions. Additionally the dump's margin
probe reads ONLY `_fr13_cag_target_logits` (patch L7244) — it **never reads the `self_logits` (tree_self) row**, so for
the 147/250 fully-accepted records whose bonus is `tree_self_target`, the flipping bonus token's own row-margin is
structurally absent (it reports the leaf's parent-edge margin instead — measured: fork0 flat_row=7=node7 parent-edge).

---

## 3. W / C / R CLASSIFICATION (MEASURED from dump+rescore)

For the **16 SAME_POS (apple-to-apple)** forks, served token == verify parent-edge argmax (16/16) and the verify
margin IS at the flip position → split by verify confidence:

- **R — genuine verify-vs-decode realization gap (verify ≥1 nat, decode disagrees): 8** — idx 1,5,6,7,12,14,19,21
  (verify margins 1.25–3.62 nat; oracle deviations 1.12–8.12). Right row, committer faithful, isolated-from-instrument
  apple-to-apple; the **tree verify forward confidently prefers a token the decode oracle rejects**. These ARE the
  genuine confident leaf wins / the accept-edge-vs-lossless tension. Mechanism = CHANNEL 2 (FR13_COMMIT_ARGMAX_BIND):
  diffuse GDN-scan + TREE_ATTN deep-row accumulation flipping argmax at structural boundaries (playbook
  "diffuse GDN accumulation").
- **C — co-residency-perturbed near-tie (verify <1 nat): 8** — idx 2,3,8,11,16,17,18,20 (verify margins 0.12–0.62).
  Right row, but the verify forward is nearly indifferent; the flip is within the realization floor. (All were already
  the B set.) Apple-to-apple fix here = verify-row co-residency isolation (the no-copy direction), NOT WY.

For the **7 DIFFERENT_POS** forks → **W (instrument row/position misalignment)**: idx 0,4,9,10,13,15,22. The leaf does
NOT "win against a wrong committer row" — the **reducer** compared the wrong row's margin. The committer served the
correct argmax of the correct row at each of these positions too (channel-1 clean). Fixing the reducer re-labels these
losslessly; **NO bonus, NO committer change.**

**Counts over 23:** W=7, C=8, R=8.
**Counts over the original 13 confident (margin≥1 nat):** of {0,1,4,5,6,7,10,12,13,14,19,21,22}: **W=5** (0,4,10,13,22),
**R=8** (1,5,6,7,12,14,19,21), C=0. → the genuine confident-win count is **8, not 13**.

---

## 4. VERDICT + the lossless fix (NO bonus)

1. **Committer comparison IS apple-to-apple and correct** — proven by code-read (L9162–9168, L6909, L6954–6976) +
   MEASURED channel-1 clean (944 records, 0 clear-margin violations) + served==verify_argmax 16/16 here. **No committer
   wiring/row bug; no off-by-one; no bonus is warranted or needed.**
2. **The fork-margin CLASSIFY is NOT apple-to-apple for 7/23 forks** (class-#12). The lossless fix is to the
   **instrument**, not the model: in `fr13_fork_margin_classify.py`, join the margin **at the oracle FLIP position's own
   committed row** — i.e. take the verify margin of the node actually served at `pos` (and, when that node's served
   token is a `tree_self_target` bonus, read the **`self_logits` row** margin, which the dump must additionally emit).
   This is a read-only reducer/probe change; it costs nothing and re-classifies the 5 inflated confident-A's correctly
   (they LOSE the "confident" label). **No spine-bonus, no margin-damp, no force-spine.**
3. **The genuine tension is R = 8 confident verify-vs-decode flips** (down from the banked 11–13). These are the
   accept-edge wins that are simultaneously the lossy-ness — the CHANNEL-2 verify-forward divergence
   (FR13_COMMIT_ARGMAX_BIND named seam: TREE_ATTN deep-row attention + conv-fusion committed-prior window + GDN scan at
   the deepest path row). This is the diffuse-GDN front, not a committer fix.
4. **The 8 C near-ties** are within the realization floor; the apple-to-apple lever there is verify-row co-residency
   isolation (no-copy survey, commit 6112a403), explicitly NOT WY.

**Distinguishing MEASURED vs INFERRED:** MEASURED = all row/position joins, served==verify_argmax (16/16), the
SAME/DIFFERENT_POS split, the W/C/R counts, the dump-never-reads-self-row fact, channel-1 cleanliness (from
FR13_COMMIT_ARGMAX_BIND). INFERRED = that the 8 R flips are *irreducible* (they are the named CHANNEL-2 seam, not
re-derived here) and the mechanistic labels C=co-residency / R=realization (consistent with prior binds, not
independently isolated at native-on-path here).

---

## 5. MINIMAL GPU RE-DERIVE (only if disambiguation needed)

The banked dump+rescore DISAMBIGUATE W from C/R fully (the W cases are a pure offline row/position join error; no GPU
needed to fix or confirm them). To separate **C (co-residency-perturbed) from R (genuine realization)** for the 8
SAME_POS confident flips — only if the user wants to attack R/C beyond accept/event parity — the minimal GPU re-derive
is: at the 8 deciding nodes, capture the **isolated native-on-path verify logits** (the node's path-to-root re-run as a
single non-co-resident forward = the SpecInfer/STree branch oracle, per
[[reference_gdn_tree_branch_oracle_losslessness]]) and compare that isolated verify argmax to (a) the co-resident
tree-batched verify argmax and (b) the decode oracle. If isolated==tree-batched but ≠decode → R (genuine, hard); if
isolated==decode but ≠tree-batched → C (co-residency, fix = verify-row isolation). This is ~8 single-path forwards on
the 4 pinned prompts, eager, no copy/dense — a tiny capture, NOT a campaign. **Do not run unless the user moves past
accept/event-parity.** Also worth doing first (free, CPU): re-run the fixed classify reducer to bank the corrected
W=7 / R=8 / C=8 split.

Links: [[reference_scalar_metric_per_token_blindspot]] (scalar A/B count hid the row-misalignment),
[[feedback_check_artifact_before_concluding]] (confounded "13 confident" taken at face value — exactly this),
[[feedback_math_correct_vs_bitexact]] (the 8 R = ℝ-correct-but-not-bit-exact deep-row flips),
FR13_COMMIT_ARGMAX_BIND (channel-1 clean / channel-2 named seam), FR13_FORK_MARGIN_PROBE_BIND (the source bind, now
corrected: 13 confident → 8 genuine + 5 instrument-misaligned).
