# K64 qrow16 + SFWD physical32 B1 partial-stack readiness

Status: production-gated and statically verified; GPU timing not run in this
worktree.

Source commit `f60f3fc9904290b7ef00bddf573f34d013be8612` composes the
byte-qualified SFWD state-fusion kernel with the qualified qrow16 FA2 kernel in
the exact K64, root-reduced, sequential B1 runtime. Production is allowed for
both physical32 lineages:

- Tail23: `tail6_fixed32`, logical mask `0x7a9ce7ff`.
- Hydra27: `hydra27_fixed32`, logical mask `0x7abdffff`.

Both modes launch 31 physical draft rows plus one root row. Logical trees can
therefore be smaller than 32 while retaining the physical32 kernel geometry;
inactive nodes are masked, not used as accepted drafts.

This runner does **not** serve the source-v7 all-parent committer candidate:
`FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0` and
`FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0` are deliberate. The exact4
B4 shadow gate must first publish a source-v7 production PASS, and a combined
co-candidate gate must bind that PASS before all-parent can join this timing
stack. Until then, this is a qrow16 + SFWD measurement only.

The historical SFWD byte pass was collected with the full-vocabulary draft
head, but the qualified executable closure ends before draft-head logits and
contains no vocabulary or root-reduction access. The production gate binds
that exact AST closure and separately requires `FR13_DRAFT_VOCAB_ROOT=1` and
`FR13_DRAFT_VOCAB_K=65536`. The timing runner contains no full-vocabulary or
synthetic/probe route.

Run the real SWE-Verified exact-four Hydra27 B1 arm with:

```bash
RUNROOT=output/fr13_k64_qrow16_sfwd_exact4_b1_<tag> \
TAG=<tag> \
QROW16_FA2_SO=/absolute/path/to/the/pinned/qrow16.abi3.so \
bash scripts/fr13_run_b1_k64_qrow16_sfwd_stack_timing.sh
```

The supplied FA2 binary must be exactly 299,507,792 bytes with SHA256
`1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86`.
The runner authenticates the fixed four-task subset before Docker launch and
reports full wall TPS/ms plus SFWD, DFWD, CFWD, GPU-total, and other-wall phase
times. A single candidate run is timing evidence only; it cannot establish the
one-sided U95 hardware-floor acceptance claim.

No GPU task was launched while preparing this artifact. The most recent valid
K64 qrow16 Hydra27 comparator remains 232.779790071 ms/step, 24.718146718 full
wall TPS, 159.619263244 ms SFWD, 36.813368134 ms DFWD, 20.677390557 ms CFWD,
and 4.753885004 accepted drafts/event. Those are prior measurements, not
results for the integrated SFWD candidate.
