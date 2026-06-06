# FR12 RED-TEAM — the argmax "flip" is a one-depth LAG (structural), not residual rounding

**Claude red-team, 2026-06-06.** Re: `FR12_PARITY_RESULTS.md` §"Token-Level Argmax Gate With Conv Splice" (L387-445).

## The observation
That argmax gate ran with **the conv splice ON** (`FR12_TREE_CONV_NATIVE_SPINE=1`). Its per-depth argmax tables show a systematic pattern, not drift:

Call 2 (spine rows `[0,1,2,4,6]`, native rows `[0,1,2,3,4]`):
- d0 tree 248068 == native 248068 ✓
- d1 tree 12305 == native 12305 ✓
- d2 tree **12305** vs native 198   → tree[2] == native[1]
- d3 tree **198**   vs native 1005  → tree[3] == native[2]
- d4 tree **1005**  vs native 9637  → tree[4] == native[3]

Call 3:
- d1 tree 248069 == native 248069 ✓
- d2 tree **248069** vs native 271   → tree[2] == native[1]
- d3 tree **271**    vs native 71093 → tree[3] == native[2]

**tree_argmax[d] == native_argmax[d−1] for d ≥ 2, in BOTH calls.** Rounding drift cannot reproduce "exactly native's previous-depth distribution." This is a **one-depth lag**, and it begins **exactly at the first branch-row gap** (spine rows are contiguous 0,1,2 then jump to 4 — depth 2→3 is the first place the tree row index stops equaling the depth).

## Why this matters
1. The verdict "remove the remaining post-core/logit residual that changes argmax" assumes the flips are **numeric residual**. They are not — at least depths 2–4 are a structural lag. Chasing more rounding alignment will not fix a row/position misalignment.
2. The run had the **splice ON**. The splice `index_select`s path0 rows, calls native conv, and `index_copy_`s back. If its path0 row identification assumes contiguous rows `[0,1,2,3,4]` but the real spine rows are `[0,1,2,4,6]`, the splice itself can shift downstream state by one at the first gap. **So the lag may be a splice artifact, not a property of our real kernel.**

## Required next measurement (before any more rounding work)
Re-run the per-depth argmax gate with the **REAL kernel**: splice **OFF** (`FR12_NATIVE_SPINE_ORACLE=0`), `FR12_TREE_CONV_NATIVE_BF16_TAPS=1` ON, same matched event, per-depth argmax tree vs native. Two outcomes:
- **Lag vanishes** → it was a splice `index_copy` artifact; trust the splice-OFF argmax as the real gate; conv bf16-taps may already lift L0; continue numerics alignment on the next real seam.
- **Lag persists** → it is a real tree-row→depth alignment bug (spine logit capture row mapping, or position-ids / causal mask at branch boundaries). Fix THAT before any rounding — it is a far bigger lossless lever than 1-ULP residuals, and it would explain the whole accept deficit better than diffuse drift.

Either way: **the honest gate is splice-OFF per-depth argmax on the same event.** Do not conclude "argmax still flips, keep aligning rounding" from a splice-ON table.

## Theory backs this read (online research, 2026-06-06, primary-source cited)
The branch-losslessness research independently predicts this exact signature. SpecInfer (arXiv:2305.09781 Def 4.1) + STree (arXiv:2505.14969 §3 Eq.4-6): a node's verify output equals the target run on its path-to-root **only if** the ancestor mask / state-accumulation folds in **exactly that node's ancestors and no others**. The named failure mode (research caveat 2): *"any cross-branch sharing below the fork (state from a sibling/non-ancestor branch bleeding in) violates Def 4.1 and silently corrupts the oracle — detectable as a per-node argmax/logit mismatch against the path-rerun, NOT a fundamental limit."* A **one-depth lag that begins at the first branch-row gap** is precisely a path/ancestor-set construction error (the spine node at depth d accumulating the wrong ancestor count), i.e. a mask/row-mapping/position bug — **categorically different from 1-ULP rounding residual.** Rounding cannot yield "exactly native's previous-depth distribution."

Corollary: the correct gate is **per-depth argmax / distributional equivalence vs the recurrent path-rerun** (research confirms: gate on argmax, not bit-exact max_abs). And STree's diagonal-A shortcut does NOT apply to our non-diagonal `(I−βkkᵀ)` term — so there is no shared-accumulator excuse; each spine/branch node needs its correct ancestor-ordered operator. If the lag is real (splice-OFF), it is the single biggest lossless lever on the board.

## Also pending
- conv bf16-taps = 0.0 was shown **offline (boot-free replay)** only. Confirm in-server: splice OFF, bf16-taps ON ⇒ `conv1d_out` max_abs == 0.0 live.
- Branch-path oracle (per `FR12_LOSSLESS_PLAN.md`): off-spine branch nodes have no native MTP-5 counterpart; validate each branch's logits against **no-MTP native run on that branch's linear ancestor-path** (depth-based RoPE). Add to the parity harness once the spine argmax gate is clean splice-OFF.
