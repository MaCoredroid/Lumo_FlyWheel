# FR13 — QUEUED ACTION (user 2026-06-14): general config-driven drafter packer for ANY tree (width/depth ≤ 16)

**Priority: AFTER the 22-drift-in-width fix lands** (user: "lets fix 22drift in width or any cat first").
This is the enabler for arbitrary-shape sweeps once the leaf co-residency is fixed/isolated — NOT a blocker
for the current 22-drift work (which uses the already-wired cat9 + the FR13_GDN_SUBOP_MAB A/B).

## Why it's needed (the cat8 failure, root-caused)
The drafter shape dispatch in `scripts/fr10_phase4_patch_vllm_tree_gdn.py:10079-10119` is an EXACT-MATCH `if`
over 4 hardcoded `tree_choices` lists (cat9 / chain5 / chain3 / cat3w). Any other shape (cat8, cat4w, wider
trees) → all `_fr10_is_*` flags False → the custom packing branch is skipped → no packed tree → class-9
fail-loud disengagement (engagement gate `tok/draft==len(TREE)` fails; FR10_ALLOW_LINEAR_FALLBACK banned, so
no silent lossy fallback). It is NOT a kernel crash — it is the guard refusing to run a shape it can't pack.
Per-shape leaf-packing slot order is the ONLY hand-rolled piece; downstream (parent/ancestry masks, committer
path enum, eager-pack replay, conv-fusion prior windows) already auto-adapts off `tree_choices`.

## Scope of the general packer
Replace the per-shape `if`/`torch.stack` branches with ONE packer that, given any `tree_choices` (list of
(len,path) tuples, the vLLM canonical SORTED order), num_spec=len(tree_choices) ≤ 16:
1. **Spine vs leaf classification per node:** a node is a "spine" node if it is the deepest at its depth on
   path0 (the committed chain); else it is a leaf branching off some ancestor at depth d with sibling-rank r.
   Derive `_fr10_spine_steps` (max committed depth) and, per leaf, (parent depth d, rank r) from the tuple
   structure — generalizing the current `_fr10_leaf_steps` frozenset.
2. **Token source per slot:** spine slot ← spine argmax (the native causal MTP step); leaf slot ← the
   depth-d drafter step's rank-r runner-up.
   - **BLOCKER for "all width": rank≥2 not captured.** Today only the rank-1 runner-up is taken
     (`torch.topk(_fr10_step_logits, 2).indices[:, 1]`). Width with rank≥2 siblings (top-3+ at a depth)
     needs widening the topk capture to `topk(K)` where K = max sibling-rank in the tree (+ extra slots).
3. **Pack into the sorted (len,path) slot index** (the canonical order the masks/committer expect) — a
   structural map from each tuple to its slot, not a bespoke stack.
4. **Drive from config:** read the tree from `config/fr13_config.yaml` (the reader is built; launcher WIRING
   DEFERRED per the directive) so a shape change is config-only, no code edit.

## Constraints / envelope
- Width/depth ≤ 16 nodes. **N_PAD=16 h_cache spill** ([FR13_CACHE_SCALING_FUTURE], project_fr13_active_worker_codex_fr15):
  at the deployed N_PAD=16 the GDN h_cache spills → num_warps=8 interim or recompute-from-spine. The general
  packer must co-scale with this (a 16-node tree hits the spill regime).
- Keep the class-9 fail-loud (raise on a shape that violates the envelope, e.g. rank > captured-K or depth >
  N_PAD) — NEVER silently degrade; FR10_ALLOW_LINEAR_FALLBACK stays banned.
- Default path (cat9 LOCKED) must stay byte-identical (additive, flag-gated; the general packer must
  reproduce cat9's exact packing bit-for-bit — gate it the same way the chain3/cat3w branches were verified:
  reduced-NEW == HEAD-OLD token-identical for cat9).

## Status
QUEUED. Active 22-drift-in-width fix first: Front A (oracle frame re-score, the +2 spine) + Front B (the
FR13_GDN_SUBOP_MAB A/B fixed, ready to localize the +17 leaf GDN co-residency on the next GPU slot). The
general packer is built once the co-residency is understood (so the right wider shapes are worth testing).
Pairs with [[project_fr13_tree_reshape_unifying_lever]], [[reference_multispine_not_lossless_closed_nonship]],
[[feedback_fail_loud_assert_engagement]].
