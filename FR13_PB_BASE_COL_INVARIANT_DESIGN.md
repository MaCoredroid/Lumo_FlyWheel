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

## COL-0 CRUX **RESOLVED SAFE** (2026-07-20) — the permute IS the fix, and it's clean
The pb ATTN mask docstring (patcher 15392–15409) settles it:
- `[1]` NO row attends chain cols 1..7 → the chain COLUMNS are dead (never attended by anyone).
- `[2]` chain rows 1..7 attend NO tree col — ONLY the paged context. The committed chain tokens are
  already durable in the paged KV (stream-0 root write + FR13_ATTN_KV_REMAP), so the chain rows read
  them position-addressed, independent of their tree-block column.
⇒ Moving the chain's physical columns changes NOTHING about the chain's attention → col-0 (exported
from stream 7's GDN state, which is fed by the chain's PAGED attention) is PRESERVED. The earlier
col-0 worry (permute perturbs col-0) is REFUTED. The GDN scan uses packed order (spec_state_indices,
separate address space) — untouched by an attention-column permute.

### THE FIX (concrete, col-0-safe, SLOT_REORDER class)
Definitive pb layout (patcher 4521): col 0 = pos-0 root (fully ghosted, dead), cols 1–7 = chain
(dead columns), col 8 = subtree root (0)^8, cols 9–29 = base subtree. Base = 22 nodes (root+21),
shifted +8 vs non-pb (base at 0–21). The +8 shift misaligns the base's FA2 online-softmax tiles
(context_len-variable, so no fixed padding aligns it) → the base verify perturbation.
Permutation pi on the TREE-BLOCK attention columns ONLY:
  base packed 8–29 → phys 0–21 (root→phys 0, base→phys 1–21) — EXACT non-pb columns
  chain packed 1–7 → phys 22–28 ; pos-0 packed 0 → phys 29  (dead cols → trailing no-op tiles)
Apply pi to: KV slot_mapping (attn), tree_bias BOTH axes, the pb ghost mask, ATTN_KV_REMAP dst.
GDN spec_state_indices UNCHANGED. RESTORE at propose (un-permute before drafter reads slot_mapping)
— the SLOT_REORDER seam already does slot_mapping-permute + bias-permute + restore; supply a
pb-specific base-first pi and engage it for pb trees (span==tree_n && drafts==tree_n-1, tree_n=30).
Base rows then reduce over {paged + base cols 0–21} == non-pb (chain cols 22–29 ghosted+future ⇒
fully-masked trailing tiles ⇒ online-softmax no-ops) ⇒ base verify byte-canonical ⇒ head accepts
like non-pb ⇒ the multiplicative deep-tail collapse lifts.

## IMPLEMENTATION SEAM (located) — reuse SLOT_REORDER machinery with a pb base-first pi
Seam: `_patch_gpu_model_runner_slot_reorder` (patcher ~19185). PERMUTE block 19246–19336 derives
`_sr_pi` and permutes `common_attn_metadata.slot_mapping` per spec row (span==tree_n &&
num_decode_draft_tokens==tree_n-1); RESTORE 19349–19361 un-permutes at propose; bias permute (edit 2)
+ remap dst (edit 3, ~19452) ride the same pi. So the ENTIRE machinery is pi-generic — only the pi
DERIVATION must change for pb.

**pb node indexing** (tree_n=30): node 0 = pos-0 root (ghosted/dead); nodes 1–7 = chain slots
(dead columns); node 8 = subtree root (0)^8; nodes 9–29 = base subtree. (mask asserts chain cols
[1..8] via len(choice)<=8; node 8 = [0]*8 = subtree root.)

**pb base-first pi** (pi[k] = node at physical col k):
    pi_pb = [8, 9, 10, ..., 29]  +  [1, 2, ..., 7]  +  [0]
          = base subtree (packed order → phys 0–21, == non-pb node-for-node since same 22-node
            subtree sorted by (len,tuple))  +  chain (→ phys 22–28)  +  pos-0 (→ phys 29)
The base subtree at phys 0–21 in packed order matches non-pb's base at cols 0–21 ⇒ byte-canonical
base reduction. Dead chain/pos-0 at phys 22–29 = ghosted trailing tiles = online-softmax no-ops.

**Edits (flag FR13_PB_BASE_COL_INVARIANT, default OFF byte-identical):**
1. PERMUTE block: engage on `FR13_SLOT_REORDER==1 OR (FR13_PB_BASE_COL_INVARIANT==1 AND pb-shaped
   tree)`. When pb, set `_sr_pi = pi_pb` (skip the all-zeros/branch derivation).
2. RELAX the assert `_sr_pi[0] == 0` (19297): pb pi has node 8 at phys 0, node 0 at phys 29. Keep
   the permutation-validity check `sorted(_sr_pi) == range(len)`.
3. Verify the bias permute (edit 2, fr13_patch_fa2_tree_bias.py) applies pi to the ALREADY-pb-masked
   bias (pb mask built at patcher 15392 BEFORE slot-reorder) → ghost -inf entries move consistently.
   ORDER CHECK REQUIRED: confirm pb-mask-then-permute (not permute-then-mask).
4. Verify remap dst (edit 3, ~19452) is correct under pi_pb (committed KV → base linear slots).

**Interaction risks to check before/at first boot:** (a) pb-mask vs bias-permute order; (b) the
ATTN_KV_REMAP dst under pi_pb; (c) does the base-root (node 8→phys 0) attention still get the bonus
exactly once (mask [3] dropped col 0 for rows 8–17; after permute those rows move — the mask must
follow via the bias permute).

## Status
Crux RESOLVED SAFE. Fix + pi + seam + edits fully specified. Remaining = a focused, careful build
(intricate multi-seam consistency + garble risk if wrong) + gates: (1) flag-OFF byte-identical,
(2) same-seed determinism 4/4, (3) col-0 export==replay bit-identical (expected pass by construction),
(4) deep-task accept 14539/14598/14995 LIVE qwen-code B=4 temp 0.6 (expect head pos-1→1.0, tail→~5).
Do the build with fresh focus — NOT at the tail of an analysis pass (a wrong multi-seam edit garbles
+ wastes GPU boots). First boot after edits: gate (1)+(3) before any accept run.
