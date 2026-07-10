# FR13 — the multidraft committer is CLEARED; the garble is forward drift (supersedes 36f645d0)

**Date:** 2026-07-10. **Supersedes:** commit `36f645d0` ("DECIDING RESULT: NOT drift —
… multidraft ACCEPT-LOGIC over-commits"). That conclusion was an **alignment artifact**;
this doc corrects it from a code read + the banked gather data, no new GPU boot.

## The claim being overturned

`36f645d0` banked: the tree-verify forward scores the garble `_idx` at `1.99e-6`
(== no-spec), yet the multidraft committer commits it at ~13% → therefore the bug is an
**accept-logic over-commit in our committer** (`fr13_device_multidraft_commit` /
`_lumo_tree_canonical_multidraft_sample`), fix locus = accept / q_mix.

## Why that is wrong — the committer is provably distribution-lossless

Reading the production device committer end-to-end
(`scripts/fr13_device_multidraft_kernel.py`, baked default-ON @patcher:10466), the
deterministic (`draft_probs=None`, the deployment MTP path) per-node rule is:

```
overlaps[i]   = p[tᵢ]                       # p = softmax(post-constraint target logits)
overlap_mass  = Σ overlaps
weights[i]    = overlaps[i] / overlap_mass  # source ~ Categorical(weights)  ∝ target prob
q_mix_token   = weights[selected]           # = p[t]/overlap_mass  (unique token)
accept_prob   = min(1, p[t] / q_mix_token)  # = min(1, overlap_mass)
residual      = max(p − q_mix_vocab, 0)/Z   # garble gets q_mix_vocab[g] ≥ p[g] ⇒ 0 mass
```

So **P(commit token t) = p[t]** — exactly the target probability. (SpecInfer guarantee;
empirically confirmed by `scripts/fr13_device_multidraft_offline_gate.py` 22/22, freqs 6σ.)
A token the committer sees at prob `q` commits at `q`, no more.

## The decisive detail from the gather data — top-p/top-k masks near-neighbors to 0

`apply_sampling_constraints` (@patcher:11136) applies **temperature AND top-p/top-k
masking** to `target_logits` *before* it reaches the committer (the dispatch @11596 passes
that same post-constraint tensor). In the banked gather rows
(`output/fr13_dbg/dbg_snapshot.jsonl`), the near-neighbor's post-constraint
`target_prob_draft` is **exactly `0.0`** — top-p/top-k set it to −∞:

```
node 1: draft=9834  target_raw_prob_draft=1.37e-6  target_temp_prob_draft=1.69e-10  target_prob_draft=0.0
```

`target_prob_draft = 0.0` (not `1.69e-10`) ⇒ **masked, not merely temperature-shrunk**. A
masked token has commit prob 0. **The committer literally cannot emit it.**

## The tight chain: it is forward drift

1. Committer commits t at exactly `target_prob_draft[t]` (post-constraint). *[proof + gate]*
2. The truncation garble `_idx` **was** committed (localizer: `applied_entry_idx` 21/21). *[data]*
3. ⇒ at the commit node, `target_prob_draft(_idx) > 0` — i.e. the tree-verify target put
   `_idx` **inside** the top-p nucleus. *[1 ∧ 2]*
4. The **true (no-spec)** model scores `_idx` ~1e-6, **below top-20** → it would be
   top-p-**masked** (prob 0). *[localizer teacher-force]*
5. ⇒ tree-verify inflated `_idx` from ~1e-6 (masked) into the nucleus = **forward drift at
   that node**; the correct committer faithfully committed the drifted target. *[3 ∧ 4]*

## Why 36f645d0 read 2e-6

That `1.99e-6` was **`target_raw_prob_draft`** — the *raw, pre-constraint, temp-1.0* value —
read at a **gather** row and joined to a commit it never observed. The committer uses
**`target_prob_draft`** (post temp + top-p). The `_idx` token is drafted at multiple nodes;
the gathered 2e-6 node is a *non-commit* node (correctly masked). The commit happened at a
*different* node whose post-constraint prob was drift-inflated. Classic wa_capture
alignment trap — the same one flagged in `project_fr13_garble_wrongaccept_padblock`.

## Consequence

- **There is no committer fix to make.** The committer is not the bug; a committer edit
  (reject-near-neighbors gate) would be reward-hacking — the offline gate is its negative
  control.
- This **restores** the rigorously-pinned 2026-07-09 mechanism: near-neighbor garble =
  tree-verify **forward drift** (co-resident-branch M-dependence at L0 GDN) inflating a
  near-neighbor past the top-p cut. = the user's **PATH A** (spine M-invariance,
  compute-only, no-HBM-tax).
- Garble is intermittent (~13%) and low-margin-only precisely because drift must inflate the
  near-neighbor **past the top-p nucleus boundary** — a high bar met only when the correct
  token's margin is thin (novel identifiers), rarely on confident known APIs. Matches the
  localizer's low-margin-flip finding.

## Instrument note (if ever double-confirming by boot)

The device committer emits **no** accept trace today; only the forward `tree_logit_gather`
is logged. To directly A/B drift-vs-committer at the true commit node without a join, add a
measurement-only per-committed-token log of the committer's OWN `target_prob_draft` inside
`fr13_device_multidraft_commit` (EAGER, flag-gated, default byte-identical). But the chain
above already forces the answer — a boot would only re-confirm forward drift.
