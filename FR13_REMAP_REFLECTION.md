# Reflection: the spec-decode tree KV garble hunt (5-Whys retrospective)

## TL;DR
The branched-tree spec-decode "garble" (committing near-neighbor / truncated identifiers — `_row→_rows`,
`wcs_wcs_hdr`, `lon_`) was **the missing attention-KV re-linearization**. After a tree accept, the GDN/conv
recurrent state was re-linearized to the committed path (`launch_tree_state_linear_remap`), but the
**attention KV cache was not**. A branching tree's accepted path is *non-contiguous* in the flat verify-slot
order, so the next forward read a **sibling branch node's foreign K/V** at the committed positions →
compounding attention drift → gross near-neighbor wrong-accepts. Fix = `launch_attn_kv_linear_remap`: copy
each committed node's K/V from its flat verify slot to the linear committed slot, per full-attn layer, in
`sample_tokens`. Result: token-level gate **15/15 → 0/15**; agentic gate **~40% → 0%**; branches kept;
speed tax −0.7% (noise). It was a **data-movement/wiring bug, not a kernel bug** — the branch KV was computed
correctly all along; it was read from the wrong slot on the next step.

---

## Part 1 — Why did the bug exist? (5 Whys, technical root cause)

**Why 1 — Why did the branched tree commit wrong (near-neighbor) tokens?**
The verify forward's logits were drifted: the correct next-token was depressed and a near-neighbor inflated,
and both the greedy and temp-0.6 accept rules committed the near-neighbor.

**Why 2 — Why were the verify logits drifted?**
The forward read the wrong K/V for the committed positions — a *sibling branch node's* K/V instead of the
accepted node's — so attention over the recent context was contaminated.

**Why 3 — Why did it read a sibling's K/V?**
Each tree node writes its K/V to a **flat, unique verify slot** in `sorted((len,path))` order. For a
branching tree the accepted path is **non-contiguous**: cat8's spine sits at flat slots `1,2,4,6,8` (branches
interleave at `3,5,7,9`). But after accept, `seq_len` advances and the next forward reads the committed
tokens at **linear** positions `1,2,3,4,5` → position 3 reads flat-slot-3, which is a *sibling branch node*,
not the accepted spine node.

**Why 4 — Why wasn't the attention KV re-linearized to the accepted path?**
Because **no attention-KV remap existed.** The GDN/conv recurrent state *was* re-linearized post-accept
(`launch_tree_state_linear_remap` moves node columns → linear accepted positions), but the attention KV
cache had **no equivalent** — nothing moved the accepted node's K/V into the linear committed slot.

**Why 5 — Why did GDN/conv get a remap but attention didn't? (the true root)**
An **asymmetry born of how the tree was layered onto vLLM's chain/eagle spec-decode.** GDN/conv state lives
in a *per-request bank* the tree code owns outright — it was self-evident that accepted node-columns had to
be re-linearized, so a remap was written. The attention KV lives in vLLM's **sequential paged cache**, which
the native block-table/slot mechanism re-uses — and that mechanism is **correct for chains** (contiguous
accepted path = linear slots) but **silently mis-addresses branching trees** (non-contiguous). The implicit
assumption "the framework handles the KV" held for the chain case and was never revisited for branches.

> **Root cause:** two parallel post-accept state stores (GDN/conv bank + attention KV) both need
> re-linearization for a *branching* tree; only one got it, because the other was assumed handled by the
> framework — true for chains, false for branches. A symmetric gap, invisible because its twin was solved.

---

## Part 2 — Why didn't we get there earlier? (5 Whys, the investigation)

**Why 1 — Why did this take many sessions?**
The hunt spent a long time on **compute** — GEMM batch-invariance, GDN scan/state nativeness, attention
compute, in_proj — before turning to **data routing**.

**Why 2 — Why the compute focus?**
The **symptom mis-led us.** Near-neighbor identifiers and small per-token differences *look like* numerical /
seed nondeterminism (a compute drift), so we pattern-matched the symptom to the wrong bug class. (Ironically,
a foreign-KV read produces *exactly* this signature: the foreign K/V is a *sibling draft*, a near-neighbor,
so the drift is small-and-near-neighbor — consistent with routing all along.)

**Why 3 — Why did the compute hypothesis survive so many refutations?**
Each compute fix was tested **in isolation** (localize one op). When garble persisted, the conclusion was
"not *this* op" — which leaves "some *other* compute op" alive, rather than "**not compute at all**." The
class was never falsified, only its members one at a time.

**Why 4 — Why wasn't the decisive test run earlier?**
The default mode was **bisect-one-op-at-a-time**. The decisive experiment is the opposite: the **kitchen-sink**
— arm *every* suspected compute/state fix simultaneously and see if garble survives. That **rules out the
whole class in one shot.** When finally run, it took ~15 min and immediately returned 15/15 → compute
exonerated → it must be KV content/routing. That single experiment was the turning point; run early it would
have saved the isolation grind.

**Why 5 — Why was the KV-routing blind spot so persistent? (three compounding masks)**
- **(a) The solved twin hid the gap.** The GDN/conv remap existed and worked, so "state re-linearization
  felt solved" — the *symmetric* missing attention remap was invisible precisely because its analog was done.
- **(b) `chain5` clean masked the branch-specificity.** Spine-only chains were garble-free, so garble read as
  "branches are just harder" instead of the structural fact: **chains are contiguous → no foreign KV; branches
  are non-contiguous → foreign KV.** We didn't convert "chain clean / tree garbled" into "non-contiguity."
- **(c) The near-neighbor symptom kept pulling back to compute** every time (Why 2).

---

## Part 3 — What would have found it faster
1. **Run the kitchen-sink (all-suspected-fixes-at-once) EARLY.** Class-elimination-by-combination beats
   member-elimination-by-bisection when you suspect a whole category. It's cheap and decisive.
2. **Treat "chain clean / branched garbled" as a structural clue, not a difficulty gradient.** The single
   difference between them is *contiguity of the accepted path* — which points straight at slot addressing.
3. **Audit symmetry.** Any per-request state re-linearized post-accept (GDN/conv) should trigger: "what is the
   attention-KV analog, and does it exist?" The working GDN remap was a **signpost to the missing twin**.
4. **Distrust the symptom's surface.** "Small near-neighbor drift" felt like compute, but a foreign-KV read
   (a sibling draft's K/V) produces exactly that — the symptom was never evidence against routing.
5. **Instrument the accept criterion, not the rate.** Rate stats and a greedy gate (which had false negatives
   — a near-tie flipping "clean" under BATCH_INVARIANT nearly declared a false win) wasted cycles; the honest
   per-token temp-0.6 gate was the instrument that held.

## Part 4 — Meta-lessons (portable)
- **Elimination-by-combination > bisection** for ruling out a *class*; bisection is for finding the *member*.
- **A working analog is a map to the missing symmetric piece** — asymmetry between parallel subsystems is a
  high-yield place to look.
- **Framework assumptions that hold for the common case silently break the edge case** — audit "the framework
  handles X" *at the edge* (here: branching, non-contiguity), not the happy path (chains).
- **Match the instrument to the failure mode.** A scalar rate is blind to a small per-token defect; the gate
  must probe the exact thing that fails (per-token argmax at temp 0.6, not a greedy near-tie).
- The fix, once localized, was **obvious and cheap** (mirror the GDN remap; small KV copy, no HBM tax) —
  virtually all the cost was in *localization*, and localization was slow because we fought the symptom's
  disguise instead of the structure.
