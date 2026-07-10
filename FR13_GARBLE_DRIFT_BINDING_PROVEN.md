# FR13 garble — the drift→misspell join, MEASURED (answers "is the 1-ULP the cause?")

**Date:** 2026-07-10. Same-boot (cat8 EAGER, :9950), committer accept-trace
(`commit_trace`, commit 8471ce32) + localizer no-spec teacher-force, joined. This
closes the ONE unmeasured link flagged in FR13_GARBLE_COMMITTER_CLEARED.md: the
tree-verify prob AT the actual misspell commit node.

## The join (two independent instruments, same token)

| identifier misspell | no-spec P(garble) [localizer] | tree-verify P(garble) the committer USED [commit_trace] | correct token |
|---|---|---|---|
| `expected_row_count`→`expected_rows_count` (`_row`→`_rows`) | **0.000000** (rank 3) | **0.0809** (in nucleus) | tree `_row`=0.919 / no-spec 0.999997 |
| `..._boolean`→`..._bool` | ~0 (masked) | **0.0809** | tree `_boolean`=0.919 |

The committer is PROVEN to commit token t at exactly its post-constraint tree-verify
prob (FR13_GARBLE_COMMITTER_CLEARED.md, offline gate 22/22). So:

- no-spec (true model) scores the garble at **~1e-6** (masked, outside top-p).
- the tree-verify FORWARD inflates it **~5 orders** to **~0.08** (into the top-p nucleus).
- the correct committer commits it at ~0.08 → **~8% garble rate** on that identifier.

**⇒ the misspell is tree-verify FORWARD DRIFT, faithfully committed. MEASURED, joined,
no inference gap.** This is the concrete confirmation the user asked for ("do you have
evidence the drift is the reason for the misspell").

## Magnitude nuance (refines the ladder story)

The drift here is NOT a collapse of the CORRECT token's logit (tree still ranks it #1 at
0.92). It is an **inflation of the GARBLE's tail** by ~11 nats (ln(0.08/1e-6)≈11.3), which
COMPRESSES the correct-vs-garble margin from no-spec's ~15 nats to tree-verify's ~2.4 nats
(ln(0.92/0.08)) — just enough to pull the garble INTO the nucleus so the committer emits it
~8% of the time. Same ~11-nat magnitude as the node5 code-fence flip; different manifestation
(tail inflation vs argmax collapse). Consistent with the live astropy pin (correct spelling
dominates 51/59, garble ~13%).

## Population (localizer, N=60 gens this boot)

16 near-neighbor garbles localized: **WRONG_ACCEPT 13, MODEL_TAIL 2, OTHER 1.** Every
WRONG_ACCEPT: no-spec gives the correct token 0.94–0.9999 (rank 0), the garble 0.000000–0.002
(rank 1–15) → the model itself would not emit it → genuine spec wrong-accept. The 2 MODEL_TAIL
(e.g. `expected_row_count` at seed s7: no-spec 0.53/0.47) are genuine model ambiguity, NOT
fixable by a drift fix. So a drift fix targets the ~80% WRONG_ACCEPT class.

commit_trace: 9409 commits, 98.9% argmax (temp-0.6 peaked), 105 non-argmax, 4 confident
wrong-accepts (committed_prob 0.059–0.081) — 2 spelling-match the localizer's named garbles.

## Fix implication (from the amplification workflow wf_f295db1b, adjudication CONDITIONAL)

- Amplification-DAMPING is mathematically dead: the 1.166x/layer growth is the trained
  network's Lipschitz/Jacobian (function property) — no lossless post-hoc knob (literature:
  attention not globally Lipschitz, only training-time regularization reduces it). All 4
  zero-HBM fp32-residual/attn dampers only remove *fresh per-layer rounding* → k≈1.1x, far
  below the needed reduction. Corroborated by the QPAD measured null.
- The "shared-with-E5 → dead" dismissal has a **confirmed threshold hole**: native sits
  ~1.23x under the margin; a uniform perturbation reduction of k>~5.7x pushes the tree below
  the cut while native only gets safer → a shared scaling IS a differential fix.
- The ONLY route delivering k>5.7x is **SEED precision** (fp32 the recurrent-state seed):
  bf16→fp32 shrinks the ULP ~1e4x → clears the margin by 3–4 orders. `MAMBA_SSM_CACHE_DTYPE=
  float32` is ALREADY set (SSM state h fp32), so the DOMINANT residual seed is the **conv
  anchor** (conv1d_out 9.77e-4 = 1 bf16-ULP). Candidate fix = **fp32 conv-state bank**; cost =
  the persistent fp32 state HBM (conv state is small → likely << the workflow's ~sub-1%
  estimate). **MUST be measured**, but ~sub-1% HBM << the tree-reshape's measured 7–9% =
  a much cheaper cost-gate.
- Committer-side (user's idea): calibrated drift-band accept test — the garble sits at
  committed_prob ~0.08 vs genuine model-tail at ~0.5; a boundary-conservative accept test could
  suppress the ~0.08 boundary-crossers. Principled if calibrated to this measured band, no-op on
  native. Not free-lossless (small distribution shift), but a lever.

## Next

1. Pin the conv-state bank dtype + whether an fp32-conv flag exists / its exact decode HBM cost.
2. Per-layer ladder A/B: fp32 conv state → does conv1d_out drop from 9.77e-4 toward 0 (kill the
   seed)? + does the garble's tree-verify committed_prob drop out of the nucleus?
3. Cost-gate to user: fp32-seed (~sub-1% HBM, measured) vs reshape (7–9%) vs accept within-floor.
