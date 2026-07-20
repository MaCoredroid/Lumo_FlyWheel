# FR13_PB_BASE_COL_INVARIANT — fix pb accept to non-pb parity

## Problem (evidence-complete, 2026-07-20)
tail6_pb accept collapses **−1.5** on deep-tail-heavy tasks (14539/14598/14995: pb ~3.65 vs
non-pb rg1 ~5.12), reproducibly, cache-independent (collapse3 cache-OFF 3.5 == cache-ON 3.6),
task-selective (13/16 tasks neutral). Prior "inherent numerics, net-neutral, no fix" was WRONG.

## Mechanism (root → amplifier)
1. **Root — FA2 base-column layout.** pb packs the 8-slot chain at attention cols 0–7 + subtree
   root (0)^8 at col 8, pushing the base subtree to cols 8–29 (22 nodes). non-pb base occupies
   cols 0–21. The single fused FA2 verify reduction over base rows is a butterfly/online-softmax
   whose FMA association is keyed by PHYSICAL score-tile column (the exact class FR13_SLOT_REORDER
   fixed for the spine, +0.166). Base at cols 8–29 vs 0–21 ⇒ different tile grouping ⇒ tiny
   per-logit diffs ⇒ tips temp-0.6 near-ties in the HEAD verify. Measured per-position: pb head
   pos-1 0.63–0.90 vs non-pb 1.0.
2. **Amplifier — Arctic decide_tail.** The deep tail (depths 6–11) is retrieved by
   `fr13_merged_drafter.decide_tail` (patcher 13902) matching THIS step's head tokens (depths 0–4)
   against the SuffixDecodingCache. A tie-tipped head token → a DIFFERENT suffix match → the whole
   deep tail degrades. Deep tail lifts non-pb +1.82 (3.30→5.12) but pb only +0.58 (3.07→3.647): pb
   captures ~1/3 of the deep-tail value. cat9pb (head only, no Arctic) = −0.23; the Arctic turns
   that −0.23 into −1.5. Deep-tail-heavy tasks live entirely on the amplified path ⇒ task-selective.

## Two contributing kernels (both N_PAD-dependent)
- **FA2 attention** (main lever): base at non-canonical columns. Fix = canonical base columns.
- **GDN scan**: FR13_NPAD_INVARIANT (kernel:185, `_FR13_N_FIXED=16`, default OFF) already exists to
  canonicalize the scan FMA order across tree sizes — CONFIRMING the GDN reduction order is
  N_PAD-dependent too. But N_FIXED=16 < tail6_pb N_PAD=32, so it does not cover tail6_pb as-is.
  GDN handoff VALUE is exact (masked-to-parent); only the reduction FMA ORDER varies.

## Fix design (SLOT_REORDER class, flag FR13_PB_BASE_COL_INVARIANT, default OFF byte-identical)
Decouple GDN order (packed, chain-first — REQUIRED: GDN scan needs parents before children in row
order) from the ATTENTION slot layout (SLOT_REORDER already decouples these; GDN excluded from the
permute because its address space is spec_state_indices, not paged KV).

**Target attention layout:** base subtree at physical cols 0–21 (EXACT match to non-pb: root→phys 0,
base nodes→phys 1–21), chain at physical cols 22–29. Then the base FA2 reduction is byte-identical
to non-pb by construction (identical physical columns), and the chain (phys 22–29) is causally
future + fully ghosted for base queries ⇒ its tiles are no-ops on the base online softmax (fully
-inf tile ⇒ max unchanged, sum unchanged).

**Edits:**
1. Define pb permutation pi (base packed 8–29 → phys 0–21; chain packed 0–7 → phys 22–29), applied
   in the SLOT_REORDER PERMUTE seam for pb spec rows (span==tree_n && drafts==tree_n-1, tree_n=30).
2. Re-derive the pb ATTN mask + tree_bias in phys space (tree_attn.py): base rows use the non-pb
   base tree mask over phys 0–21; chain rows (22–29) attend paged + prior chain, GHOST base (0–21).
3. ATTN_KV_REMAP dst (kernel:479) in phys space (committed KV → base linear slots).
4. RESTORE at propose (un-permute before drafter reads slot_mapping) — SLOT_REORDER already does this.

## OPEN CRUX to resolve BEFORE implementing (col-0 safety)
Moving the chain's ATTENTION position could perturb col-0 (the committed-state export the next
forward's GDN reads) IF the chain's GDN scan consumes the chain's RE-COMPUTED attention hidden.
- If chain GDN k,v,a,b come from the committed KV rings (like the replay) → moving is FREE.
- If from re-computed attention (re-fed tokens through attention) → moving risks a tiny col-0 drift
  (same tie-tipping class), which would trade the base fix for a col-0 regression (worse: col-0
  affects ALL downstream). RESOLVE by reading where the fused-forward chain nodes' GDN inputs come
  from (fr10_gdn_tree_kernel around the node-step k/v load, ~900–990). col-0 is currently PROVEN
  export==replay bit-identical, so whatever the source, the fix MUST keep that invariant.
  Mitigation if re-computed: keep the chain at packed cols 0–7 (col-0 safe) and instead PLACE THE
  BASE at the SAME EXACT columns as non-pb via a fresh tile-aligned block — requires the base's
  intra-tile positions to match non-pb, i.e., base start ≡ non-pb base start mod nothing: the
  online softmax is position-exact (not just tile-aligned), so the base must be at cols 0–21
  literally. That conflicts with chain@0–7 ⇒ the chain MUST move ⇒ crux must be resolved.

## Gates (in order)
1. flag-OFF byte-identical (py source unchanged when env unset).
2. same-seed determinism (4/4) flag-ON.
3. non-spec / prefill byte-identical flag-ON (no chain ⇒ no permute ⇒ no change).
4. col-0 export==replay STILL bit-identical flag-ON (the crux invariant).
5. deep-task accept 14539/14598/14995: expect head pos-1 → 1.0, deep tail → ~5, accept → ~5.
   LIVE qwen-code B=4 temp 0.6 nudge-free only.

## CRUX RESOLVED (2026-07-20): chain GDN inputs are RE-COMPUTED → prefer canonicalization, not permute
Read fr10_gdn_tree_kernel node-step (940–990): the fused-forward chain nodes' GDN k/v/q/beta/g are
loaded from the forward's PROJECTION tensors (re-computed from the attention hidden), NOT the
committed KV rings (rings feed only launch_tree_gdn_replay). ⇒ moving the chain's attention column
perturbs the intra-chain attention → the chain hidden → col-0 (which is currently export==replay
bit-identical). So a column PERMUTE trades the base fix for a col-0 regression (col-0 hurts ALL
downstream). Naive "chain-after-base" is OUT.

### PREFERRED FIX = canonicalize the FA2 tree-attention reduction order (FR13_NPAD_INVARIANT analog)
Instead of moving nodes, make the tree-attention reduction order POSITION/TREE-SIZE INVARIANT so the
base logits don't depend on whether the chain occupies cols 0–7. This is exactly what
FR13_NPAD_INVARIANT does for the GDN scan (kernel:185, pin the scan span to N_FIXED so the FMA order
is canonical). Two coordinated pieces:
  (A) GDN: extend FR13_NPAD_INVARIANT to cover tail6_pb — set N_FIXED >= 32 (currently 16) OR make it
      per-tree max, so tail6_pb (N_PAD=32) AND the non-pb reference reduce over the SAME fixed span.
  (B) FA2 tree attention: canonicalize the online-softmax key reduction so the base rows' result is
      invariant to the presence/count of chain columns. Investigate whether padding the tree key
      span to a fixed N + deterministic split (num_splits=1) + fixed tile walk achieves it; the base
      rows only attend {paged + base cols}, so the requirement is that the chain cols contribute
      as canonical no-op tiles at a FIXED position relative to the base (fully-masked trailing tiles
      are online-softmax no-ops — so if the chain is FORCED to the reduction TAIL logically, even
      without physically moving KV, the base result is chain-invariant). This may be achievable in
      the mask/tile-walk ONLY (no KV move) — the key open question for the next session.
Gate col-0 export==replay bit-identity FIRST on any FA2 change (piggyback must stay lossless).

## Status
Design + crux resolved. Implementation is deep-kernel. NEXT session FIRST action: prototype (B) —
determine whether the FA2 tree-attn base-row reduction can be made chain-column-invariant via the
mask/tile-walk alone (no KV move), preserving col-0. If yes → cheapest correct fix. Then (A) for GDN.
