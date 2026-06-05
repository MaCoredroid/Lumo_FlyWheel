# FR10 — Drafter-side findings (code-verified on LIVE eagle.py, 2026-06-04)

## ⟢ ARBITER RESULT (draft-token parity, decisive) — THE CLIFF IS DRAFT-SIDE

`output/fr10_draft_parity_tree_20260604T194024Z/draft_parity_compare.json`, 32 teacher-forced
rows, tree path0 draft vs native mtp5 draft:
- depth0: 32/32 match. depth1: 32/32 match. **depth2: 0/32 match** (cascades to depth3,4).
- **`tree_path0[2] == native[3]` in 32/32 rows; `tree_path0[2] == native[2]` in 0/32.**
  Clearest on the rare token: pos17 native `[4145,11,632,95449,11]` vs tree `[4145,11,95449,321,10548]`
  — tree's depth-2 is native's depth-3 (95449), native's depth-2 (632) is SKIPPED.

INTERPRETATION (deterministic, not contamination): the spine tokens are drafted CORRECTLY
(native d0,d1,d2,d3… all appear) but **mapped to the wrong tree slots.** The drafter places
the spine into CONSECUTIVE slots while the verify path0 topology reads the caterpillar spine
at NON-CONSECUTIVE slots `[0,1,3,5,7]` (tree_choices sorted). So path0 reads every-other-spine
token (skips the real d2 at slot2, reads d3 at slot3). The verifier verifies that shifted spine
against the target → rejects at depth-2 → the `{0:210,1:96,2:0}` cliff. This is **BUG 1 (the
uniform-propose_tree topology mismatch) hitting path0 directly** — NOT a deep GDN-recurrence
problem (BUG 2 is real but not what bites here). My earlier "spine draft is clean, cliff is
verify-side" hand-trace was WRONG — the parity overturned it. The user's draft-side hypothesis
is confirmed.

FIX = the same pure-causal-spine + read-only top-2-leaf below, with the spine placed at the
SAME slots the verify topology uses (`[0,1,3,5,7]`) and leaves at `[2,4,6,8]`. Honors the
read-only gate (spine tokens stay native; only the slot mapping + leaf read change). Good news:
no tree-aware GDN drafter kernel needed — it's a placement/topology fix.

---


Source: `/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py` inside
`fr10-speed-start` — read directly (not the stale /tmp copy). Two DISTINCT drafter bugs,
both independent of the verify-side scan/commit. Neither is fixed by any conv/scan/commit fix.

## BUG 1 (decisive) — stock `propose_tree` CANNOT realize the non-uniform caterpillar

`propose_tree` precomputes a SINGLE SCALAR children-count per level (eagle.py:289-301):
```
num_drafts_per_level[d]   = count of tree_choices with len==d+1
child_drafts_per_level[d] = num_drafts_per_level[d] // num_drafts_per_level[d-1]   # UNIFORM, per level
```
For the caterpillar `[(0,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0),(0,1),(0,0,1),(0,0,0,1),(0,0,0,0,1)]`:
- `num_drafts_per_level   = [1,2,2,2,2]`
- `child_drafts_per_level = [1, 2//1, 2//2, 2//2, 2//2] = [1,2,1,1,1]`

`[1,2,1,1,1]` = "branch into 2 at depth-2, then give EVERY node exactly 1 child." So the
drafter actually generates **TWO PARALLEL CHAINS**, not a caterpillar:
- spine:   `(0,)→(0,0)→(0,0,0)→(0,0,0,0)→(0,0,0,0,0)`
- chain-2: `(0,)→(0,1)→(0,1,0)→(0,1,0,0)→(0,1,0,0,0)`

But the VERIFY tree (tree_choices, sorted by `(len,path)`) labels the slots as a caterpillar.
Draft flat order vs verify slot order:

| slot | verify tree_choices | draft places (token's true path) | match |
|------|---------------------|----------------------------------|-------|
| 0 | (0,)        | (0,)                | ✓ spine |
| 1 | (0,0)       | (0,0)               | ✓ spine |
| 2 | (0,1)       | (0,1)               | ✓ leaf  |
| 3 | (0,0,0)     | (0,0,0)             | ✓ spine |
| 4 | (0,0,1)     | (0,1,0)  ← chain-2  | ✗ WRONG |
| 5 | (0,0,0,0)   | (0,0,0,0)           | ✓ spine |
| 6 | (0,0,0,1)   | (0,1,0,0) ← chain-2 | ✗ WRONG |
| 7 | (0,0,0,0,0) | (0,0,0,0,0)         | ✓ spine |
| 8 | (0,0,0,0,1) | (0,1,0,0,0) ← chain-2 | ✗ WRONG |

=> The SPINE/path0 slots (0,1,3,5,7) are placed correctly token-wise (each = its spine
parent's argmax). The LEAF/branch slots (4,6,8) get **chain-2's continuation**, NOT the
spine nodes' top-2. The "record the MTP top-2 at each spine depth as a leaf" design in the
brief is **NOT what stock propose_tree implements.** `_fr10_record_tree_consumption`'s
`placement_ok` flag measures exactly this — and it is currently silently swallowed by its
`except Exception: pass` guard (eagle.py:1126). GET THAT LOG WORKING — it is the direct
instrument for BUG 1.

Impact: branch recovery / acceptance-superset is measuring GARBAGE branches (chain-2, not
spine top-2). This corrupts the superset gate, NOT path0.

## BUG 2 — deep-spine draft contaminated by chain-2 IF the drafter GDN forward is flat

The drafter re-forwards the WHOLE flattened tree each level (eagle.py:1147-1238), and the
flat order interleaves chain-2 between spine nodes: `(0,),(0,0),(0,1),(0,0,0),(0,1,0),...`.
- A spine node's OUTPUT used to draft its child is computed at the level BEFORE its sibling
  leaf is appended, so its hidden state precedes the leaf in flat order → spine drafts are
  CLEAN through depth-3 (slot3=(0,0,0) = (0,0)'s clean argmax).
- From depth-4 (slot5,7): the producing forward contains chain-2 (e.g. (0,1)@idx2) BEFORE
  the spine node (0,0,0)@idx3. If the drafter's GDN recurrence is FLAT (not tree-aware),
  (0,0,0)'s state includes (0,1) → slot5/slot7 spine draft CONTAMINATED. Verify-side
  conv/scan/commit fixes are gated on `num_spec_decodes>0`+`fr10_tree_parent` (the VERIFY
  forward); the drafter's `build_for_drafting` forward makes only the FULL-ATTENTION layers
  tree-aware — GDN layers very likely run flat. Untouched by every fix so far.

## What this means for the depth-3 path0 cliff (survival {0:210,1:96,2:0})

slot3=(0,0,0) draft is CLEAN (hand-traced above). So a depth-3 cliff most likely is
**VERIFY/COMMIT-side**, NOT draft. The in-flight draft-parity capture is the arbiter:
- spine draft IDs (slots 0,1,3,5,7) == native mtp5 through depth-3 → confirms draft clean →
  cliff is verify-side (scan-output or ssm-commit at the first branch).
- diverge at depth-4 (slot5) → BUG 2 confirmed (GDN-flat deep spine).

## The clean fix that closes BUG 1 + BUG 2 at once (the brief's true intent)

Do NOT walk the stock uniform tree for this shape, and do NOT build a tree-aware GDN
*drafter* kernel. Instead implement the caterpillar drafter as:
> draft the spine EXACTLY like native mtp5 — pure causal, k autoregressive steps, ONE chain,
> no tree in the recurrence — and at each spine step record the **top-2** token of that
> step's logits as the leaf for that depth. The leaf is a logits side-output; it is NEVER
> fed into any forward.

This makes the spine byte-identical to native (BUG 2 gone) AND places the correct per-depth
top-2 at leaf slots 4,6,8 (BUG 1 gone). Leaves stay genuinely free (no recurrent-state copy,
no leaf re-forward). The verify side (already tree-aware) handles the tree.

## HARD GATE — any drafter change must be MATHEMATICALLY READ-ONLY (user 2026-06-04)

If we touch the drafter, the change must NOT alter the native drafter's math in any way. It is
ONLY allowed to ADD a read-only extraction of the runner-up token. Enforce as a GATE with a
powered negative control:

- **GATE-D1 (spine draft unchanged):** our drafter's spine token IDs == native mtp5 spine token
  IDs, BYTE-EXACT, at every forced position and every depth. The way each position is drafted
  must be identical to native. (We already have the native per-position chains captured in
  `fr10_draft_parity_native_*/native_draft_measure.json` — diff against ours.)
- **GATE-D2 (per-position distribution unchanged):** the logits / top-p output at each position
  == native mtp5's per-position output, byte-exact (FP-deterministic, fixed teacher-forced
  context). The runner-up leaf is READ from these SAME native logits.
- **Read-only rule:** the top-2 (branch#2) extraction must NOT feed back into any forward, must
  NOT mutate the recurrent (conv/ssm) state, and must NOT change the spine top-1 or the
  per-position distribution. It is a pure side-read of the native distribution's 2nd mode.

So the sanctioned drafter = "native mtp5 drafter, UNCHANGED, + record the 2nd-ranked token of
each step's native logits." We REMOVE the contaminating extra tree-forward (chain-2); we do NOT
add any new math to the spine path. Negative control (proves the gate has power): the CURRENT
stock 2-parallel-chain `propose_tree` FAILS GATE-D2 — it emits contaminated logits at the leaf
slots (chain-2's forward), so a flat diff vs native must show those slots differ. A passing
implementation differs from native ONLY by the (read-only) presence of the runner-up leaves.

## Action items for codex (in order)
1. Finish the in-flight spine draft-parity capture; report spine-slot draft IDs vs native
   per depth. This decides draft-vs-verify for the depth-3 cliff.
2. Repair `_fr10_record_tree_consumption` (drop/loosen the bare except) so `placement_ok`
   emits — quantify BUG 1 (expect slots 4,6,8 placement_ok=False).
3. If the cliff is verify-side: continue the scan-output / ssm-commit captured-replay.
   If draft-side at depth-4: implement the pure-causal-spine + top-2-leaf drafter above.
4. BUG 1 fix (pure-causal-spine + top-2-leaf) is required regardless for a valid SUPERSET
   gate — the current branches are chain-2 garbage.
