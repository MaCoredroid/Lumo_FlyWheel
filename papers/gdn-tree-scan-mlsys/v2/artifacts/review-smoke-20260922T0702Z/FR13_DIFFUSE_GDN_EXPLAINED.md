# What "diffuse L0–L58 GDN accumulation" actually is — and whether our kernel can fix it

(User asked 2026-06-13: "I'm not saying it's wrong, but what is it? is it something our kernel
cannot fix?" — this is the honest answer.)

## What it is (mechanically)
The model is 64 layers (48 GDN/linear-attn + 16 full-attn). The tree verifies a token by running
all 64; the **argmax of the final logit** is the served token. A "flip" = the tree's final argmax
≠ the clean (non-MTP oracle) argmax.

The decisive workflow localized it with the **substitution test** (FR13_HIDDEN_SUBSTITUTE):
overwrite the tree's hidden at layer N with the oracle's, run the rest, see if the flip reverts.
- splice oracle@**L58 → flip REVERTS** (clean again);
- splice oracle@**L0 only → still flips**.
⇒ the divergence is **present at every GDN layer as a tiny amount and ADDS UP across depth**. At
each GDN layer our kernel computes the *same math* as native but in a slightly different
finite-precision realization (fp add is non-associative; different reduction order; a cast
rounded one step earlier; a different Triton tiling). Each is ~**1 bf16 ULP** (~1 part in 128) —
individually below the noise floor. But they are **correlated** (lean the same way), so over ~48
GDN layers — amplified by the gate's `1/rms` and the deep full-attn layers near the end (an
**amplifier**, not a source) — the accumulated difference grows until it crosses the margin and
flips the final argmax. **Death by a thousand cuts; no single broken op.**

## Is it something our kernel cannot fix? — NO.
Load-bearing fact: **native E5 runs the SAME model, SAME fp8, SAME 64 layers, and drifts only 3
flips vs our 22.** Same physics, 7× less drift ⇒ a clean version provably exists *at the same
precision*. "Diffuse" is **not a law of nature** — it means **our per-layer realization differs
from native's in a way that accumulates**. If our kernel matched native's realization bit-for-bit
at each op (op order, cast boundaries, reduction order, the Triton `num_warps`/`BV` codegen), we'd
accumulate exactly like native = land at native's 3-flip floor (the accepted bar).

History backs this: the same "diffuse wall" decomposed before into a handful of nameable seams
once someone drove each to literal 0.0 (conv bf16-tap rounding, scan `static_range` unroll,
intra-chunk A cast). **"Diffuse" usually = "a few un-aligned seams nobody drove to zero," not a
thousand independent ones.** ([[feedback_math_correct_vs_bitexact]])

**The one honest caveat:** a truly irreducible floor exists — ~2-ULP MMA-grouping at ~1e-12 — but
that's a *do-not-grind* floor far below an argmax flip. Our 22 flips are at **1.1–6.0 nats**,
enormously above it. They are alignment differences, not the noise floor.

## So the real question is FEW seams vs MANY — and that's what the chase resolves
- drift-localize ruled out fp8 / conv-tap / conv-window and left **one un-closed seam: the GDN
  scan `num_warps=8/BV=16` codegen** (vs native 4/BV=32), only ever atol=1e-3-gated. **Boot 2
  (BV=8) directly tests it** — if the 22 flips drop, "diffuse" was mostly that one codegen seam
  (cheap, fixable, the user's read); if not, more seams to align (a short grind) or the attention
  path.
- The "diffuse L0–L58" was characterized on the **SPINE**. If the flips turn out to be at deep
  **BRANCH** nodes (Boot 1 pins this), the mechanism is **co-residency** (the branch's deep state
  contaminated by sharing the batched forward) — *more* localizable than diffuse spine
  accumulation, pointing at the isolation fix, not the grind.

**Verdict:** fixable in principle (native is the existence proof). The open question we're
resolving is which of {one codegen seam / a short multi-seam grind / branch co-residency} — none
of which is "our kernel cannot touch it."
