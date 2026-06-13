# FR13 cat10 root-branch investigation — the -28 is artifact-dominated; the root branch is real

Workflow `wf_59bf2440-2bf` (CPU, 4 agents). Raw:
`research/fr13_workflows/cat10_investigate_wf_59bf2440.raw.json`. Adversarial verify **holds=FALSE**
— it CORRECTED the readers (they over-attributed the d0->d1 drop to m1; it is the sibling-stop
denominator artifact). Act on the verify's corrected conclusion. PREP for the deferred post-22-flip
cat10 work. HEAD 80ebee8c.

## The -28 (cat9 3.198 -> cat10 2.932) is ARTIFACT-DOMINATED, not a real co-residency/handoff bug
1. **Trajectory/denominator confound (dominant):** cat9 & cat10 generate different greedy streams
   (diverge early p0@17/p1@11/p2@21/p3@61); cat10 p0 hit EOS 25 tokens sooner (73 vs 98) -> 25 fewer
   tokens in 1 MORE event -> mechanically lower accept/event (class-12 whole-window confound).
2. **The d0->d1 sharp drop (-0.149) is the SIBLING-STOP DENOMINATOR ARTIFACT, NOT m1** (verify
   correction): a sibling-leaf [0,2] win is `accepted_len=1` (caps at d0), swelling per_pos[0] but
   contributing 0 to d1+, which deflates the d1|d0 conditional. De-confounding pos0 RECOVERS d1|d0
   to ~0.84+. d2/d3/d4 conditionals are FLAT cat9-vs-cat10.
3. **m1 (verify co-residency) is structurally ruled out:** the verify strict_mask walks parents to
   root, and node 2 (1,) is never a spine ancestor, so NO spine row has strict[spine,2]=1 — the
   root-sibling row is attention-invisible to every spine row, and the GDN tree-scan uses the same
   mask. Only residual = a sub-ULP fp reduction-order leak (the extra real row in the shared 16-pad
   tile), not -0.27.
4. **m3 (commit handoff) is inert:** sibling commits are accepted_len=1, so they do NOT trip the
   num_accepted>1 conv prior-window bug.
5. **The -28 and the 22-flip are SEPARATE:** the cat9 flip set [6,6,4,6] and cat10 [2,6,8,6] are
   100% DISJOINT positions (total flat 22=22) — the 22-flip is the same channel-2 defect repositioned
   by the trajectory. Fixing the 22-flip will NOT remove the cat10 dilution, and vice-versa.

## The root branch IS valuable (user's instinct vindicated)
- **d0-rescue rate = P(target == root-rank-2 | root-rank-1 missed) = 0.273 (27%)** — when the
  drafter's root top-1 misses, the runner-up is the true token 27% of the time (a "2-horse-race"
  signature; random rank-2 would be ~0%). The root sibling captures exactly that: d0-reject rate
  0.129 -> 0.094, +0.035 d0 accept (+~21/boot).
- COST = +1 verify row paid 100% of events (+2.9 ms/fwd, ~1.013x), rescue fires only ~3.5% of events.

## THE LEVER (post-22-flip): the CONFIDENCE-GATED root branch (shape-true)
Emit the `(1,)` root sibling row ONLY when the root is a near-tie (margin g = logit[rank1] -
logit[rank2] < tau, or a softmax prob gap); else serve the clean cat9 9-node spine. The entire 0.273
rescue lives in near-tie events; on confident-rank-1 events the runner-up is ~never the target, so
the row there is pure cost. **The gate is FREE** — top2 = torch.topk(_fr10_logits,2) is already
materialized (archive :9665); one scalar compare per event, zero extra forward. Shape FIXED:
caterpillar + at most ONE single root node, never deeper. At greedy, the gated cat10 pushes d0
accept above cat9's spine-only d0 (+0.035, the legit greedy branch rescue 86a255a4) WITHOUT the
(mostly-artifact) d1-d4 dilution.

## DECISIVE post-22-flip-fix test (needs a per-node counter — ABSENT from saved data)
The saved artifacts lack the per-event winner/bonus_source log (sibling-win vs spine-win), so the
residual real dilution (if any) after de-confounding is UNMEASURED. Post-fix: BOOT-A re-run cat9 on
the FIXED spine (the NEW baseline — accept should RISE toward native ~3.16 as the 22-flip clears),
BOOT-B re-run cat10 (full tree) WITH a per-node accept counter (FR10_METRICS=1 + a sibling-vs-spine
d0 tag) -> recompute d0->d1 AFTER removing sibling-stop events from the per_pos[0] denominator. The
load-bearing unknown: is there ANY residual spine d1-d4 dilution once the sibling-stop artifact is
stripped? Then the confidence-gated cat10 superset verdict at temp>0.

## STATUS (user 2026-06-13): cat10 PARKED with this note
User ruling: "cat10 is more like accounting, not a real (drafter-side) issue — park cat10 for now
with a note; finish the 22-flip chase ALL THE WAY to NATIVE LEVEL and use cat9." So:
- **cat10 = PARKED** (not abandoned). The −28 is mostly a measurement artifact (above); there is
  no real drafter-side bug to fix. The drafter is alt-free; m1 ruled out; m3 inert.
- **The note to carry forward:** the root branch's d0 rescue (27% on near-tie roots) is REAL and
  worth it; the future lever is the CONFIDENCE-GATED root branch (emit (1,) only when root is a
  near-tie, FREE top2-margin gate, shape-true caterpillar+1 root node). Revisit ONLY after cat9
  is at native level, with a per-node sibling-vs-spine counter to de-confound.
- **ACTIVE GOAL = drive the cat9 22-flip down to NATIVE LEVEL (native's 3 clear-margin flips vs
  the same oracle).** That is the lossless bar. The node7-ladder localizes the carrier; then align
  the carrier sub-op bit-exact to native and re-gate the per-token argmax probe to ~3.
