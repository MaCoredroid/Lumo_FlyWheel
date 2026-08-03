# Fixed32 CFWD logit-direct source candidate

Status: `SOURCE_ONLY`, `DEFAULT_OFF`, and not wired into the served arm.

This artifact binds the fixed32 CFWD logit-direct decision candidate to source
commit `20eb33a6508ccf2996b73d59e55da65fae48a06f`. The specialization is limited
to K64/root1, physical32, Tail23 (`tail6_fixed32`) or Hydra27
(`hydra27_fixed32`), B1 or B4, verifier vocabulary 248,320, fanout three, and
walk cap 12. It is bound only to the exact Tail23 mask `0x7a9ce7ff` and Hydra27
mask `0x7abdffff`; it makes no arbitrary-tree claim.

## Structural work removal

The served all-parent path gathers and softmaxes 13 self and 17 target FP32
logit rows per request, then writes 132 more FP32 full-vocabulary rows while
building normalized probabilities, q_mix, and residuals. The candidate reads
the source logits directly, writes only max/sum-exp block statistics, and
performs source selection, acceptance, sparse residual correction, and token
selection in a second Triton launch site.

| Batch | Incumbent full-vocab writes | Active block-stat writes | Writes removed | Persistent block-stat workspace |
| --- | ---: | ---: | ---: | ---: |
| B1 | 190,709,760 B | 14,640 B | 190,695,120 B (181.86 MiB) | 15,360 B |
| B4 | 762,839,040 B | 58,560 B | 762,780,480 B (727.44 MiB) | 61,440 B |

The incumbent probability producer has four source-level full-vocabulary
tensor operations: two indexed gathers and two softmaxes. The candidate has
two explicit Triton launch sites, so two source dispatch sites are removed.
The later dense decision operations are also absent but are deliberately not
assigned a static launch count. Exact physical kernel-launch removal is
`PENDING_GPU_TRACE`, because backend implementation of the large-vocabulary
PyTorch operators must be measured. The existing integer committer remains
one launch.

## Host verification

The source, artifact-integrity, incumbent TAW, exact committer, and GDN
committer contract suite passed with `126 passed, 1 skipped` after excluding
one unrelated pre-existing B1 runner-string assertion. That baseline test
expects an obsolete inline full-vocabulary conditional; the current runner
uses a workload-profile case instead. No harness file was changed.

The focused tests cover exact B1/B4 byte accounting, physical32 geometry
rejection, persistent workspace dimensions, logit-space versus dense
probability algebra, duplicate sibling tokens, strict inverse-CDF boundaries,
zero-residual fallback, source/accept/residual uniform ordering, exact immutable
metadata values and pointer versions, writable-buffer disjointness, source-only
isolation, sticky device-domain guards, and AST-exact launch arity.

Both kernels compiled offline for SM121a with the pinned CUDA 13.0, Triton 3.6,
and PyTorch 2.10 toolchain. The block-stat and direct-decision kernels both use
80 registers per thread, with zero stack, local memory,
spills, or calls. The code-object hashes and exact B1/B4 grids are recorded in
`codegen_summary.json`.

No Docker, GPU execution, real SWE-Verified product A/B, Nsight trace, or
timing run was performed. This artifact does not claim a speedup or update the
latest valid B1 Hydra result of 232.78 ms/step.

## Required runtime gates

1. Offline SM121a codegen: `PASS`. Live occupancy and graph-capture behavior
   remain part of the product gate.
2. Wire this exact source behind a default-off, no-fallback selector and
   allocate a distinct fixed workspace per concurrently replayable graph.
   Exact immutable metadata binding must run once before capture, and the
   sticky invalid-domain scalar must remain zero. The reference path must
   remain served until all product gates pass.
3. On authenticated real SWE-Verified tasks, run fixed-uniform product A/B for
   Tail23 and Hydra27 at B1 and B4. Compare all five decision products and the
   final integer-walk products byte for byte. A mismatch is a failure; the
   mathematical distribution identity does not silently waive the byte gate.
4. Capture Nsight traces for both arms and both batches. Report actual kernel
   launches, DRAM bytes, achieved bandwidth, stage-one block work, stage-two
   scan occupancy, and the complete CFWD event time. This closes the currently
   unmeasured physical launch count.
5. Only after product PASS, run the standing real four-task B1 and B4 full-wall
   campaign with K64/root1, physical32, and clean DFWD/SFWD/CFWD breakdowns.
   The B1 cap remains one-sided U95 at or below 137.61 ms/step.
6. Run the standing 16-task confirmation only if the four-task U95 clears the
   cap.
