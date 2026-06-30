# FR13 — Tree shape: why cat6root is the B=1 sweet spot (and cat10 can't beat it)

The speculative trees (each tuple is a path from the root; `0` = top-1 child, `1` = rank-2 sibling, etc.):

```
chain5 (E5 spine):  (0,) (0,0) (0,0,0) (0,0,0,0) (0,0,0,0,0)                       = 5 nodes, pure spine
cat6root         :  (0,) (0,0) (0,0,0) (0,0,0,0) (0,0,0,0,0)  + (1,)              = spine + 1 sibling AT THE ROOT
cat10            :  (0,) (0,0) (0,0,0) (0,0,0,0) (0,0,0,0,0)
                       + (1,) (0,1) (0,0,1) (0,0,0,1) (0,0,0,0,1)                 = spine + 1 sibling AT EVERY depth
```
(defs: `scripts/fr13_bigdenom_swe_serve_variant.sh:51,55,56`)

So **cat10 is NOT "cat9 + a root sibling"** — it is the *same spine as cat6root* plus a rank-2 rescue
sibling at **every** depth (5 of them), where cat6root adds **one** sibling, at the root only.

## The mechanism: acceptance is DEPTH-limited, not WIDTH-limited

Realized acceptance ceiling on this model at B=1 ≈ **3.5 tokens/event** — verification walks ~3-4 levels down
the spine before a draft mismatches, then stops. Measured directly (config-diff workflow, spec-accept agent):
**tree positions ≥5 accept exactly ZERO** (cat6root depth-5 node = 0; cat10 positions 5-9 = 0). The deep
nodes sit *past the frontier acceptance ever reaches*.

A depth-*k* rescue sibling only fires if acceptance (a) walks past depths 1…k−1 all-correct, **then** (b) the
top-1 at depth *k* is wrong but the rank-2 is right. Two facts make all but the root sibling dead weight:

1. **Rescue value is concentrated at the root (depth-1).** That's the **~27% d0-rescue** — when the root
   top-1 misses, the rank-2 catches it. cat6root captures this with its *single* `(1,)` sibling. Depth-1 is
   always reached, so this rescue always has a chance to pay.
2. **Depths 4-5 are rarely reached** (accept ceiling ~3.5), so cat10's deep siblings `(0,0,0,1)`,
   `(0,0,0,0,1)` are beyond the frontier and never get accepted.

## The result
| tree | nodes drafted/step | realized accept/event | decode @ B=1 |
|---|---|---|---|
| chain5 (spine) | 5 | ~3.5 (no rescue) | 18.80 (historic ref) |
| **cat6root** | **6** | **~3.5 + the 27% d0-rescue** | best of the three |
| cat10 | 10 | **~3.5 (same as cat6root — the 5 siblings add ≈0)** | worse: 14.1 (4-task TW) |

cat10 accepts the **same ~3.5** as cat6root but **drafts 10 nodes vs 6** → ~67% more verification compute
per forward pass for **zero extra acceptance** → strictly slower at B=1.

**Principle:** spend exactly one extra node on the one rescue that pays (depth-1). Widening the tree only
helps if you can also push the accept *depth* past ~3.5 — and at B=1 with this drafter you can't. cat6root's
"spine + root sibling" is the efficient shape.

**Caveat (not yet tested):** this is the **B=1** regime. Wider trees can pay off at higher concurrency, or
with a stronger drafter (higher per-depth top-1 accuracy → acceptance reaches deeper → deep siblings become
live). Both are open follow-ups; for the B=1 deployment regime cat6root wins.
