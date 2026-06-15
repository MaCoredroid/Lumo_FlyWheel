# FR13 — Are the cat9 flips a SYSTEMATIC "accept-leaf-over-spine" bias, or near-tie symmetric noise?

Date 2026-06-15 (CPU-only, READ-ONLY; a big-denominator GPU run is concurrent — no code/boot edits).
Banked inputs (this session, Jun 15 04:25–04:36):
`output/fr13_fork_margin_probe/logs/{fr13_fork_margin_dump.jsonl, rescore_cat9_K1_forkmargin.json,
fork_margin_classify.json}` (250 spec-steps, 23 clear-margin flips, FULLY JOINED dump↔oracle) and
`output/fr13_scan_align_rerun/logs/{off_recur_flips.json, native_recur_flips.json}` (a SEPARATE cat9-OFF boot
+ the native E5 floor). Committer = `scripts/fr10_phase4_patch_vllm_tree_gdn.py::_lumo_tree_path_lcp_max_greedy_sample`
(L6680, scoring/tie-break L6898–6976). FA2 tree-bias = `scripts/fr13_patch_fa2_tree_bias.py`. vLLM source read
fresh via `scripts/vllm_src.sh` (pinned 3dbe092e). Model vocab = **248320** (`/models/qwen3.6-27b-fp8/config.json`).

Quoting **FR13_BUG_CLASS_PLAYBOOK**: **#12** measurement traps (per-pos counters indexing the wrong length; the
"loud-fork vs quiet-spine" selection/trajectory effect) and **#9** vacuous/non-vacuous (categories re-derived from
the ACTUAL banked dump join, not asserted). int-view equality, never atol.

---

## HEADLINE VERDICT — **SYSTEMATIC directional leaf-bias, small-N (4:1), at #1/#2 near-ties — NOT diffuse random**

Two independent legs both reject "diffuse random noise":

1. **NOT uniform-random** (decisive, large effect): of all 23 clear-margin flips, **23/23 served (verify-argmax)
   tokens are tree-DRAFTED / on the committed path** and **18/23 are in the DECODE oracle's own top-5**. A uniform
   verify-argmax perturbation over the 248320-vocab would land on a drafted token with P ≈ 9/248320 = 3.6e-5
   (expected ~0.0008 of 23) and in the oracle top-5 with P ≈ 2e-5. **Observed category-C "random non-drafted" = 0.**
   The drift is CONCENTRATED at #1/#2 near-ties, not diffuse.

2. **DIRECTIONAL toward #2/leaf, survives the visibility control** (small N): in the apple-to-apple subset where
   verify's pick is observable AT the flip position (n=10), verify picks the **#2/leaf draft 6× vs the #1/spine
   draft 1×**, and the symmetric error test gives **P(verify=#2 & decode=#1) = 4 vs P(verify=#1 & decode=#2) = 1**
   → **4:1 skew toward the leaf**. Both error directions ARE observable in this set (idx 12 is the reverse), so the
   skew is not purely a "loud-fork" selection artifact.

**Caveat (honest): N is small (10 clean / 23 total).** 4-vs-1 is suggestive, not statistically airtight (a binomial
sign test on 5 directional events gives p≈0.19). The mechanism code-read found **NO leaf-favoring math/rounding seam**
in the committer or the forwards — the committer LOGIC is biased *against* the leaf (strict `>`), and FA2 tree-bias /
GDN row-mapping are leaf/spine-symmetric. So the directional tip, where real, is a **verify-FORWARD realization
effect at the divergent (shallow-leaf vs deep-spine) row**, not a fixable scalar. **The single cheap test that would
upgrade 4:1-suggestive → systematic-confirmed is the queued isolated-fork (native-on-path branch oracle) capture.**

---

## 1. CATEGORY COUNTS (A/B/C/D) — re-derived from the banked join (MEASURED, non-vacuous)

The dump records, per fork's divergence (split) node: `spine_token_at_split` (= the #1/spine draft),
`winner_token_at_split` (= the #2/leaf draft), `committed_row` (the served tokens), `bonus_source`. `served_token_id`
(the committer's commit = the verify-forward argmax of the indexed row, **channel-1 clean per FR13_APPLE_TO_APPLE_FORK**)
and `oracle_argmax_id` (the no-spec recurrent DECODE oracle argmax) come from the joined rescore.

**Whole-set A/B/C/D (all 23):** A=6, B=3, C=11, D=3. But **C and the DIFFERENT_POS subset are inflated by the
class-#12 instrument join** (FR13_APPLE_TO_APPLE_FORK §2): for 7 of 23 the recorded split #1/#2 is at a DIFFERENT
depth than the flipped served position, so "served ≠ either recorded sibling draft" is partly a join artifact, not a
genuine both-rejected. The trustworthy cut is the **apple-to-apple SAME_POS subset** (served == split verify-argmax,
both drafts present at the divergence), n=10:

| schema cat | meaning | clean apple-to-apple count | idxs |
|---|---|---|---|
| **A accept-leaf** | is_fork, served == #2/leaf draft (verify picked the leaf) | **6** | 5,6,8,14,17,21 |
| **B accept-spine-wrong** | served == #1/spine draft, decode disagrees (the quiet reverse) | **1** | 12 |
| **C both-rejected / verify-correction** | verify rejected BOTH siblings, committed its own `reject_parent_target` bonus (≠ #1,#2) — still a verify-argmax, NOT random | **3** | 1,16,19 |
| **D other** | — | 0 | — |

Remaining 13 of 23 = **W (instrument, not a model effect)**: 7 DIFFERENT_POS (idx 0,4,9,10,13,15,22 — the reducer
joined a non-flip neighbor row, FR13_APPLE_TO_APPLE_FORK) + 6 SAME_POS-but-served≠split_va (idx 2,3,7,11,18,20 — the
served/flip node is deeper than the recorded split, or a non-fork spine flip). These are NOT apple-to-apple
direction-readable and are excluded from the directional test (NOT counted as evidence either way).

**Is the served (verify-argmax) token DRAFTED or random non-drafted? MEASURED: 23/23 are in `committed_row`
(drafted/accepted/bonus); 0 random non-drafted. 18/23 are in the oracle's own top-5** (near-tie). The 3 "C" served
tokens are `reject_parent_target` corrections = the verify model's edge argmax (its own prediction), not noise.

---

## 2. SYMMETRIC DIRECTION TEST with the VISIBILITY CONTROL (MEASURED)

**Selection trap (the user's caveat, confirmed):** a "fork" is DEFINED by the leaf achieving `lcp > spine_lcp`
(strict, committer L6919), which *requires* the verify-argmax at the split to equal the leaf draft. So the
unconditioned "verify-picks-#2" rate over all 154 fork-steps (63 leaf / 26 spine / 65 third) is **tautologically**
leaf-skewed and cannot be used. Quoting playbook **#12**: counting only the LOUD forks over-states the asymmetry.

**The non-confounded test** restricts to clear-margin FLIPS (verify was wrong) where BOTH a #1-spine and a #2-leaf
draft exist at the divergence AND verify's pick is at the flip position (SAME_POS) — so BOTH error directions are
observable. Crosstab (verify-pick, decode-pick), n=10:

| | decode=#1 spine | decode=#2 leaf | decode=3rd |
|---|---|---|---|
| **verify=#2 leaf** | **4** | 0 | 2 |
| **verify=#1 spine** | 0 | **1** | 0 |
| verify=3rd | 1 | 0 | 2 |

- **P(verify=#2 & decode=#1) = 4** (verify tipped the parent near-tie to the LEAF; decode/no-spec wanted the spine).
- **P(verify=#1 & decode=#2) = 1** (the reverse — verify stayed on spine; decode wanted the leaf).
- **Asymmetry ratio = 4 : 1 toward the leaf.** Verify favors the #2/leaf draft 6× total vs #1/spine 1×.
- **Survives the visibility control:** the reverse direction (idx 12: verify=#1, decode=#2) IS present in the same
  set, so the 4:1 is not because the spine-error is unobservable — it is observable and rare.

**The "quiet reverse" the user warned about is genuinely rare here, not just invisible.** The 4 non-fork
(spine-commit) flips (idx 0,4,9,18) are NOT "verify=#1-spine but decode wanted #2-leaf" — at all 4, best_leaf ==
spine_leaf == 7 (no competing leaf out-scored the spine) and the DECODE oracle wanted a 3rd/non-drafted token
(44675 / 4577 / 1423 / 1901), not a suppressed leaf. So they are spine-realization-vs-3rd flips, not the symmetric
counterpart. This **strengthens** the asymmetry: the spine→wrong flips do not point at a leaf.

---

## 3. RANDOM-vs-SYSTEMATIC ESTIMATE (MEASURED observed vs INFERRED random baseline)

| quantity | random-diffuse prediction (INFERRED) | OBSERVED (MEASURED) |
|---|---|---|
| served token is a tree-drafted token | ~9/248320 = 3.6e-5/flip → ~0.0008 of 23 | **23/23** |
| served token in oracle top-5 (near-tie) | ~5/248320 = 2e-5/flip → ~0 | **18/23** |
| category-C random non-drafted flips | ~23/23 should be non-drafted | **0** |
| direction (#2 vs #1) | symmetric (≈50/50) | **4:1 toward #2/leaf** |

A uniform verify-argmax perturbation predicts essentially ALL flips land on random non-drafted vocab tokens and a
symmetric #1/#2 split. The observation is the **opposite on both axes**: every flip is a near-tie reordering among
drafted tokens, and the direction skews to the leaf. **The drift is NOT uniform-random; it is concentrated at #1/#2
near-ties and is directionally leaf-leaning.** (Native E5 floor confirms the near-tie nature is intrinsic: its 3
clear-margin flips also land in the oracle top-3 — `native_recur_flips.json`; near-ties flip at the realization floor
for native too. The cat9 EXCESS over native, ~20 flips, is what carries the direction.)

De-cascade (playbook FR13_PLUS2): the 23 flips collapse to **14 independent clusters** (adjacent positions are
downstream of one upstream flip). So the directional signal rests on ~14 independent events, of which ~5 are clean
direction-readable — small, hence "suggestive-systematic," not "proven."

---

## 4. MECHANISM CODE-READ — leaf-favoring math/rounding asymmetry? **NONE FOUND** (the committer is biased AGAINST the leaf)

Per the user's hypothesis (a leaf-favoring math/rounding seam would make this FIXABLE at the kernel/committer):

**(i) Committer `_lumo_tree_path_lcp_max_greedy_sample` — biased AGAINST the leaf, NO leaf-favoring tie-break.**
- L6919 `if lcp > best_lcp` is **STRICT** `>`: a leaf path only overtakes on STRICTLY longer accepted prefix. On a
  TIE the incumbent (earliest-enumerated leaf) keeps it; the spine (path_idx 3 in the 9-node caterpillar) overtakes
  shallow alts via strict `>`. So a leaf fork **requires** `leaf_lcp > spine_lcp` strictly — the committer logic
  *suppresses* leaf wins. No `>=` that would tip ties to the leaf.
- Bonus path (L6956–6976): `reject_parent_target` (rejected node's parent-edge row), `tree_self_target` (accepted
  leaf's self row), `root_parent_target` — each indexes the row predicting ITS node's position along ITS path,
  **identical indexing for spine and leaf nodes** (L7025–7054 row map; CAG channel-1 clean, 944 records, 0 violations
  per FR13_APPLE_TO_APPLE_FORK). No `>=` vs `>` or rounding in any lcp/margin compare. **Committer-W = 0.**

**(ii) Verify-forward leaf rows — symmetric, no per-row leaf scale.**
- FA2 tree-bias (`fr13_patch_fa2_tree_bias.py` L40–66): adds a dense ancestry bias `tree_bias[q_rel,k_rel]` (0 for
  ancestors, `-INFINITY` for non-ancestors) `/ params.scale_softmax` after QK, before softmax. The `/scale_softmax`
  is applied to **every** row identically; the bias is a pure ancestry mask. **No asymmetric leaf-vs-spine scale or
  rounding.**
- GDN tree-scan leaf-branch state (`fr10_gdn_tree_kernel.py`) + fp8 GEMM/o_proj per-row scale: not a *favoring*
  asymmetry — the leaf branch state and spine state are computed by the same op-order; the spine state is
  co-residency-INVARIANT (N_PAD test, banked). A leaf-row realization that *differs* from an isolated leaf forward is
  the candidate (see §5), but it is a DIFFUSE deep-row realization, not a directional scalar that "rounds toward #2".
- The committer counter-clue is decisive: **the committer already favors the spine (strict `>`), yet the leaf still
  wins systematically at near-ties → the forward tip must be REAL** (the leaf's divergent row gets a verify argmax ==
  the #2 draft more often than chance), not a committer artifact.

**(iii) Drafter MTP #2 over-representation:** the #2/leaf draft IS systematically the verify-argmax at the split for
forks — but that is the fork *definition* (selection-confounded, §2), so this is not separable as a pure
drafter-quality effect from the dump alone. The 4:1 in the FLIP set (where verify is wrong) is the part that is NOT
explained by drafter quality: verify confidently realizes the #2 token that decode rejects.

**Verdict on mechanism: NO leaf-favoring math/rounding seam in committer or forwards (cited above).** The directional
tip, where real, is the **verify-forward shallow-leaf-row realization at #1/#2 near-ties** — the named CHANNEL-2
diffuse GDN-scan + TREE_ATTN deep-row seam (FR13_COMMIT_ARGMAX_BIND), expressed *directionally* because the leaf's
divergent row sits at a different (shallower) depth than the spine's. Not a scalar fix.

---

## 5. CANDIDATE MECHANISM (if real + directional) + the cheap test

**Candidate:** the **leaf co-residency tipping the parent near-tie toward #2.** At a fork, the #1-spine and #2-leaf
drafts share the parent; the verify forward computes their divergent rows co-resident in the same tree-batched
forward. The leaf's divergent row (shallow node, different ancestry mask / GDN branch state) is realized with a
slightly different op-order than an isolated single-path forward, and at the #1/#2 near-tie this realization
systematically lands on the #2/leaf token (margin <1 nat for the 8 C-class near-ties; 4 of the 6 clean leaf-wins have
decode preferring the spine by 1.1–9.0 nat — the realization is confident, the disagreement real).

**Cheap test (queued isolated-fork / branch oracle, ~8 single-path eager forwards on the 4 pinned prompts, no
copy/dense/reward-hack — per [[reference_gdn_tree_branch_oracle_losslessness]] and FR13_APPLE_TO_APPLE_FORK §5):**
at the 6 A-class clean leaf-win nodes, capture the **isolated native-on-path verify logits** (the leaf's path-to-root
re-run as a single non-co-resident forward) and compare its argmax to (a) the co-resident tree-batched verify argmax
and (b) the decode oracle. Decision:
- **isolated == decode but ≠ tree-batched** (for most/all 6) → the leaf-tip is a **co-residency** effect →
  **FIXABLE / directional** (verify-row isolation, the no-copy direction) → systematic leaf-bias confirmed.
- **isolated == tree-batched but ≠ decode** → the tip is in the path-rerun itself (genuine deep-row realization),
  directionally-neutral-but-confident → near-tie realization, relax to accept/event-parity.

This 8-forward capture is the decisive disambiguator between "fixable directional co-residency bias" and "irreducible
near-tie realization that happens to lean leaf." **Do not run unless the user moves past accept/event-parity.**

---

## DISTINGUISH MEASURED vs INFERRED
- **MEASURED:** all category counts (A/B/C/D and the clean-subset A=6/B=1/C=3); 23/23 served drafted; 18/23 in oracle
  top-5; the symmetric crosstab and P(verify=#2&decode=#1)=4 vs P(verify=#1&decode=#2)=1; the 14-cluster de-cascade;
  the committer strict-`>` tie-break; FA2 bias symmetry; the native floor near-tie nature.
- **INFERRED:** the random-diffuse baseline (9/vocab, 5/vocab — a model, not a measured null); that the 4:1 is a
  *true* directional bias rather than small-N fluctuation (binomial p≈0.19 on 5 events — suggestive, not airtight);
  that the mechanism is co-residency vs path-rerun realization (resolved only by §5's capture, not by this dump).

## VERDICT
**SYSTEMATIC, NOT diffuse-random** on the random-vs-systematic axis (23/23 drafted, 0 random-C, near-ties — decisive).
**DIRECTIONAL toward #2/leaf at 4:1, surviving the visibility control, but small-N (suggestive not proven).** The
mechanism is a verify-FORWARD near-tie realization effect (NO leaf-favoring committer/kernel math/rounding seam — the
committer is biased *against* the leaf), candidate = leaf co-residency tipping the parent near-tie. The one cheap test
that converts "4:1 suggestive" → "fixable directional co-residency bias" vs "irreducible leaf-leaning realization" is
the queued isolated-fork branch-oracle capture (§5). This **does not support "diffuse, relax"** on the random axis;
it leaves the *fixability* open pending the §5 capture.

Links: FR13_APPLE_TO_APPLE_FORK.md (the SAME/DIFFERENT_POS control + W/C/R; channel-1 clean),
FR13_COMMIT_ARGMAX_BIND (channel-2 named seam), [[reference_scalar_metric_per_token_blindspot]] (scalar A/B count hid
the direction), [[feedback_check_artifact_before_concluding]] (the selection-confound on fork-step counts checked
BEFORE concluding), [[reference_gdn_tree_branch_oracle_losslessness]] (the §5 isolated-fork oracle).
