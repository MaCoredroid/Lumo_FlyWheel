# Fixed32 GDN GQA-group3 source candidate

Status: `SOURCE_ONLY`, `DEFAULT_OFF`, and not wired into the served arm.

This artifact binds the fixed32 GDN GQA-group3 candidate to source commit
`fadd12d33ca5962baadf42ac2cc77b9179758b26`.  The candidate is restricted to
K64/root1, physical32, Tail23 (`tail6_fixed32`) or Hydra27
(`hydra27_fixed32`), B1 or B4, and the exact Qwen3.6-27B GDN geometry:
16 key heads, 48 value heads, Dk=Dv=128, BV8, and 48 GDN layers.

## Static work removal

The incumbent fixed32 single-launch kernel assigns one CTA to one value-head
tile.  Three value heads share each key head, so it redundantly loads and
normalizes the same q/k vectors three times.  This candidate assigns one CTA
to the three sibling value heads and performs their three independent
recurrences explicitly after one shared q/k load and normalization.

| Batch | Reference CTAs/layer | Candidate CTAs/layer | CTAs removed/event | Redundant q/k bytes removed/event | q/k norm reductions removed/event |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | 768 | 256 | 24,576 | 402,653,184 (384 MiB) | 1,572,864 |
| B4 | 3,072 | 1,024 | 98,304 | 1,610,612,736 (1.5 GiB) | 6,291,456 |

Physical launches remain one per GDN layer.  State/V work, the ordered
physical32 recurrence, output stores, and committer export surfaces are not
removed.  Fixed physical work is unchanged for any logical tree with at most
32 nodes.

## Host verification

The candidate, artifact-integrity, and incumbent contract suite passed with
`46 passed, 1 skipped`.  It covers exact B1/B4 work accounting, geometry rejection,
single-writer value-tile coverage, grouped-versus-independent recurrence
equivalence on CPU, source-only isolation, ordered single-launch structure,
fixed32 exact I/O, and K-norm/gate/decay committer contracts.

Triton and a GPU were unavailable in this host environment.  No container,
GPU compile, real SWE-Verified task, byte A/B, or timing run was performed.
Consequently this artifact does not claim a speedup or acceptance progress.

## Required runtime gates

1. Wire this exact source candidate behind a default-off, no-fallback selector
   in the production patch path.
2. Compile the exact B1 and B4 specializations for SM121a.  Reject on spills,
   local memory, stack growth, or an occupancy/resource regression that
   invalidates the CTA-reduction premise.
3. Run authenticated real SWE-Verified byte A/B gates for both Tail23 and
   Hydra27 at B1 and B4.  Compare all 48 GDN layers: output, K/V/A/B rings,
   K-norm ring, gate/decay ring, flags, and invocation counter.  Reference
   must remain served on any mismatch.
4. Only after byte PASS, run the standing real four-task B1 and B4 full-wall
   campaign with fixed32 K64/root1 and clean phase breakdowns.  The current
   B1 acceptance target remains one-sided U95 at or below 137.61 ms/step.
5. Run the 16-task confirmation only if the four-task U95 clears the cap.
