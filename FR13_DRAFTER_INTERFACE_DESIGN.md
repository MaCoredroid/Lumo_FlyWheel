# FR13 General Drafter Interface — MTP head / suffix decoding / merged

Design 2026-07-14. Owner: correctness-critical (committer contract). Grounded in the
measured reality of this session, not aspiration.

## Measured constraints that shape the design
1. **Drafter is HOST-orchestration-bound, not compute-bound** (split10: 23ms GPU compute —
   8.3ms model fwd + 15.1ms lm_head — of a 140ms/step drafter; 117ms is host). => a new source
   MUST NOT add host Python overhead; and avoiding the draft *forward* (suffix) saves only the
   23ms GPU slice unless it also cuts host orchestration.
2. **Verify scales ~linearly ~2ms/row (B=1), accept RISES with tree width** (Front 1:
   cat8 9-row 131ms/3.69 -> t33333 16-row 145ms/3.96). The 16 cap is SOFT (gdn_attn.py:294
   pad<=16 warmup). => the high-value use of a cheap draft source is **WIDENING the tree with
   extra branch candidates** (free-in-GPU tokens that keep lifting accept).
3. **The committer already verifies arbitrary per-node candidates — PROVEN committer-transparent**
   (Gate 1: scripts/fr13_suffix_committer_contract_gate.py, 32/32 vs the REAL rule
   host_multidraft_accept_probs). The deployed multidraft rejection rule (draft_probs=None path)
   computes overlaps=p[cands], weights=overlaps/overlap_mass, accept_prob(s)=min(1,p[t]/q_mix[t]),
   residual=max(p-q_mix,0). The exact math (analytic + Monte-Carlo confirmed):
   - **OUTPUT is EXACTLY p, for ANY candidate set** (P(out=t)=min(q_mix,p)+max(p-q_mix,0)=p[t]).
     Adding suffix/garbage candidates NEVER changes the output distribution => LOSSLESS,
     UNCONDITIONAL. (NB: the earlier draft's "accept *distribution* unchanged" was WRONG — the
     per-source weights and accept RATE do change; what's invariant is the OUTPUT.)
   - **ACCEPT RATE = p(S)** = target mass on the DISTINCT candidate set; adding a candidate with
     p>0 RAISES it (monotone). This is the exact speed-win: suffix candidates are a pure monotone
     accept lever. (Requires MergedSource to DEDUP candidates for the clean identity; lossless
     holds even without dedup.)
   - **ZERO garble**: a garbage token g (p[g]~0) is committed at rate EXACTLY p[g]~0 — the safety
     is the SOURCE-SELECTION weight (=p[g]/overlap_mass ~0), NOT accept_prob (which = overlap_mass,
     not small). So a bad suffix candidate is almost never even selected; if selected it still
     commits only at rate p[g]. Suffix decoding carries ZERO garble risk BY THE COMMITTER MATH.

## The interface

```
class DraftSource(Protocol):
    def propose(self, ctx: DraftContext) -> NodeCandidates: ...
    # ctx: running token ids (committed prefix per req), last hidden states,
    #      the tree_spec (per-node parent + which nodes need candidates),
    #      batch_size, generators.
    # returns: per (req, tree_node) -> candidate token id(s). One-per-node for a
    #      fixed tree; the committer picks/verifies. draft_probs stays None
    #      (deployment path) so no [nodes x vocab] q is materialised.
```

Backends:
- **MtpHeadSource** (the current path): the FR13_DRAFTER_SINGLE_LOGITS eager MTP forward +
  compute_logits; argmax spine + topk(root/level) branches. Unchanged; it IS the baseline.
- **SuffixSource** (new): a per-request suffix automaton / hash-of-last-k-gram over the request's
  OWN committed token stream (code editing repeats identifiers/paths/boilerplate). Match the
  current suffix (last k tokens) -> propose the historical continuation as the draft chain.
  ZERO forward, ZERO lm_head. Host cost = an O(1)-amortised dict lookup per req (NOT a Python
  per-node loop — that's the host-overhead trap; keep it a vectorised gather off a prebuilt
  index). Feeds candidates for as many tree nodes as the match length covers.
- **MergedSource** (the goal): compose. Two compose modes, both committer-transparent:
  (M1) **spine-quality + suffix-width**: MTP provides the spine chain (depth, the accept
       backbone); suffix provides EXTRA branch candidates at each depth at ~zero GPU cost ->
       widens the tree past what MTP topk cheaply gives, lifting accept per Front-2 constraint.
  (M2) **confidence pick**: per node, prefer the source with higher agreement; falls back to
       MTP when suffix has no match (cold context). Deterministic, committer-verified either way.

## Correctness gates (mandatory, in order)
1. **Committer contract proof (CPU)**: extend fr13_dm_depthsync_byte_gate / the multidraft offline
   gate — feed suffix-sourced candidates as extra children; assert the accept DISTRIBUTION is
   unchanged for MTP-only nodes and that suffix candidates are verified by the same rule (a
   never-in-target suffix token must REJECT, never wrong-accept). This is the losslessness proof.
2. **Garble gate** (fr13_garble_gate.py, temp-0.6 matrix): suffix must not raise undefined-name
   rate above native — it structurally cannot (committer-verified), but PROVE it live.
3. **Engagement / non-vacuous**: SuffixSource ENGAGED log + match-rate needle (how often suffix
   supplies a candidate; 0% match = vacuous = the source is dead weight).
4. **Live B=4 A/B vs cat8+SLOT_REORDER baseline**: accept, resolve, give-ups, garble, deploy-speed.
   DELIVERY = merged >= baseline on quality AND (accept-per-forward up OR wall/tok down).

## Why this is the RIGHT next lever (given the campaign's cost-gate)
The speed campaign concluded the tree's remaining gap is structural drafter host overhead.
Suffix decoding attacks a DIFFERENT axis: it doesn't make the existing drafter faster — it adds
ACCEPT (free wide branches + free spine-extension on repetitive context) which is the HBM-bound
lever (accept-per-forward), the ONE thing that isn't dead. On agentic code editing (high repeat:
re-reading files, repeated identifiers, boilerplate), suffix match rates are typically high ->
free accepted tokens. And it composes with a wider tree (Front 1: raise the pad-16 cap) since the
extra branch candidates come free from suffix. Net: a plausible accept win with ZERO garble risk
(committer-gated) and ZERO GPU compute add — the profile the measurements point to.

## Impl order (each independently gated; delegate only mechanical/isolated pieces)
S-A. SuffixSource module (isolated new file): per-req rolling k-gram index + match -> candidates.
     CPU-testable in isolation (feed a token stream, assert matches). DELEGABLE (isolated file).
S-B. Wire the DraftSource seam at the propose() call (I own — drafter edit): flag
     FR13_DRAFT_SOURCE=mtp|suffix|merged, default mtp (byte-identical). MtpHeadSource = current.
S-C. MergedSource compose (I own): suffix candidates -> extra tree children; committer-transparent.
S-D. (optional, NOT cheap) widen past n_pad=16 for wide merged trees. NOTE: the 16 cap is a REAL
     register wall, not buffer sizing — the FR10 tree kernel (fr10_gdn_tree_kernel.py:408)
     holds h_cache=[N_PAD,BLOCK_V=16,DIM_K=128] fp32 register-resident (128 KB/CTA at 16 = half
     the SM reg file; 32 spills to HBM). Widening REQUIRES re-tiling BLOCK_V 16->8 (+2x V-grid,
     redundant q/k reloads) — a kernel rewrite. So prefer keeping merged trees <=16 nodes and
     spending the free suffix candidates on BRANCH DENSITY within 16, not on going wider.
Gates 1->4 per step. Suffix is losslessness-safe by the committer, so the risk is SPEED/ACCEPT
(does it help), not CORRECTNESS (it can't garble).

## S-B/S-C SEAM LOCATED (2026-07-14) — small + localized, NOT a drafter rewrite
The tree branch leaf tokens are built at patcher :13361 (FR13_DRAFTER_SINGLE_LOGITS):
  _fr10_logits = self.model.compute_logits(sample_hidden_states)
  draft_token_ids = _fr10_logits.argmax(-1)                    # spine token (per depth)
  _fr10_root_topk = torch.topk(_fr10_logits, k, -1).indices
  _fr10_root_leaf_token  = _fr10_root_topk[:, 1]               # branch rank-2  <-- SEAM
  _fr10_root_leaf2_token = _fr10_root_topk[:, 2]               # branch rank-3  <-- SEAM
The tree GEOMETRY is fixed (tree_choices, n_pad<=16 register wall); the branch LEAF TOKENS are the
free variable. Suffix improves WHICH token fills each branch slot (Gate 1: replacing a low-p
drafter rank-3 with a target-plausible suffix token raises p(S) -> accept, losslessly). NOT adding
nodes (register wall forbids) -- filling fixed slots better ("branch density within 16").

GRAPH CONSTRAINT (must respect): the drafter runs in a PIECEWISE cudagraph; the suffix dict lookup
is a HOST op that CANNOT run in-graph. => S-B wires a PERSISTENT device buffer
(suffix_branch_tokens + suffix_valid mask) at captured addresses + a per-step host->device copy
(host: SuffixSource.propose fills the buffer, like slot_mapping staging). S-C does the in-graph
blend: `_fr10_root_leaf_token = torch.where(suffix_valid, suffix_branch_tokens, drafter_topk_rank2)`
(device torch.where, graph-safe). Flag FR13_DRAFT_SOURCE=mtp|merged, default mtp = byte-identical
(don't touch the tokens); MUST dedup vs the spine argmax (Gate 1: duplicate adds nothing).

COST-GATE (honest): (+) host cost negligible -- SuffixSource.propose ~7us (50k-tok test 0.35s) vs
140ms/step host-bound drafter (<0.01%). (+) DOWNSIDE ZERO -- lossless (Gate 1), can't regress
correctness; a bad suffix guess is just rejected. (?) UPSIDE is workload-dependent -- accept gain
only materializes if suffix guesses carry real target-p on agentic code-editing (repetitive:
re-read files, repeated identifiers). That's the LIVE B=4 A/B (Gate 4) -- unknowable until measured,
but it's a PURE-UPSIDE bet (cheap + can't regress + plausible). Cost-gate PASSES.

NEXT (mine, correctness-critical, next cycle): S-B buffer+staging, then S-C in-graph blend; gate
each with the CPU committer contract (Gate 1 extended to the blend), garble gate, then live B=4 A/B
vs cat8+SLOT_REORDER.
