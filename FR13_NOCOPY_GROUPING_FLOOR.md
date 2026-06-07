# FR-13 finding — the no-copy tree FA2 fork hits a single-ULP reduction-grouping FLOOR (not a bug)

> **CORRECTION (workflow `w86uygp1x`, see `FR13_FLOOR_WORKFLOW_VERDICT.md`):** the numbers below from the v1 reducer OVERSTATE the residual. Accurate (v2 oracle, raw `.pt`): **14/16 calls byte-exact 0.0 on the whole tree; 15/16 byte-exact spine; exactly 2 single-bf16-ULP elements total across 16 events (~2e-6 of ~983k comparisons)**. row-0's nonzero is **0.0078/1037** (v1's "0.125/512" superseded) and is **definitively** a single-query-vs-stacked FA2 harness artifact (tree[0] == stacked oracle bit-for-bit). It is a ~2e-6 **probabilistic** single-ULP rounding event, **not** a deterministic per-row floor; **no impossibility theorem exists**. Strongest no-copy losslessness evidence to date. Read the verdict doc, not the v1 numbers below.

Monitor red-team, 2026-06-07, from strict run `output/fr13_verify_strict_tree_20260607T091935Z`
(`--attention-backend TREE_ATTN`, `FR13_FA2_TREE_BIAS=1`, `FR10_ALLOW_LINEAR_FALLBACK` UNSET, GDN tree branch active). Bound to commit `fe21cb73`.

## What we built (works)
Forked vLLM FA2 (CUTLASS) carries an additive ancestry bias: `bias[q,k]=0` if k is an ancestor of q, else `-inf`, added to `acc_s` post-QK pre-softmax. Whole tree (spine + all branches) in ONE FA2 call. Smoke vs FA2-on-path oracle was byte-exact 0.0. Gate-2 (regular decode, no bias) is byte-exact 0.0 vs pristine stock FA2 — the fork does not touch regular decode.

## What the strict model-level run shows
- **Bias is true `-inf`** (capture `tree_attn_bias.uniq_nonzero=[-inf]`). Masked keys contribute `exp2(-inf)=0` **exactly**; adding 0.0 is exact in IEEE fp. The masking is correct.
- **Residual is a single ULP, on a single element**, not a pervasive divergence:
  - stacked-spine packed oracle (`tree_vs_fa2_spine`): `max_abs 0.00390625`, **`nonzero: 1`**.
  - branch row 6 / call10 branch row 9: 1 nonzero element, 0.0039 / 0.00098.
  - per-row oracle row-0 `0.125/512-nonzero` is a single-query-harness artifact (row 0 is exact inside the stacked-spine oracle) — to confirm.

## Root cause: MMA fragment reduction-grouping over scattered no-copy keys
In the shared **no-copy** tree KV, a row's ancestor keys are **scattered** across slots (branch nodes interleaved between spine nodes: `TREE_PARENT=[-1,0,1,1,2,2,4,4,6,6]`). FA2's tensor-core PV reduction assigns each key-slot to a fixed accumulation lane/fragment. The packed oracle places the same ancestor keys at **contiguous** slots → different lane assignment → the non-zero path-key contributions are summed in a different fp32 grouping → one-ULP non-associativity drift. Holes are exact 0, so they don't add error; the grouping of the **non-zero** terms is what differs.

**This is the exact same irreducibility `FR13_FA2_TREE_BIAS_FORK_RESEARCH.md` attributed to Triton-vs-CUTLASS — but it is intrinsic to no-copy shared-KV itself, independent of which kernel runs.** The fork removed the kernel-mismatch source (Triton vs CUTLASS); it cannot remove the layout-mismatch source (scattered vs packed), because under no-copy you cannot make every branch path contiguous simultaneously in one shared KV.

## Consequence
- **Literal byte-exact 0.0 on scattered rows is unreachable** without per-row contiguous repacking (a design change / arguably a copy). The spine could be made 0.0 only if path0 is laid out contiguous-first (no branch nodes interleaved); branches inherently cannot all be contiguous at once.
- **But the floor is ~1 ULP (0.0039) ≪ the E5 self-noise floor (~0.059)** ⟹ the tree verify is **distributionally / argmax lossless**, which is the theorem-backed correct gate for branch nodes (`reference_gdn_tree_branch_oracle_losslessness`: per-depth argmax vs path-rerun, NOT max_abs).

## Decision for the user (gate definition / pass-fail)
1. **Accept the 1-ULP grouping floor** as the no-copy regime and move to the **e2e deliverable gate**: lossless within E5 floor (bag-TV) + superset accept/event ≥ E5. (Recommended — matches the deliverable definition and the theorem-backed branch gate.)
2. **Insist on literal 0.0** → requires per-row ancestor-gather packed FA2 (byte-exact by construction) — more expensive, not "one additive-bias call", and may count as copy/repack. Needs an explicit ruling on whether KV-gather is a banned copy.

New-direction workflow launched to: (a) adversarially confirm the 1-ULP is the irreducible grouping floor (not a fixable wiring bug), (b) confirm the row-0 0.125 is a harness artifact, (c) produce the ONE-GPU e2e lossless+superset measurement plan vs E5.
